import copy
import utils.comm_meter as _CM
import os
import gc
import random

import numpy as np
import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from train import task_loss as _TL

from data.dataset import DatasetSplit


def _dl_kwargs():
    _nw = int(os.environ.get('DL_WORKERS', '4'))
    if _nw <= 0:
        return {}
    return dict(num_workers=_nw, pin_memory=True, persistent_workers=True,
                prefetch_factor=int(os.environ.get('DL_PREFETCH', '2')))


def server_grad_query(net_server_frozen, smashed, label, args, mask=None, train_mode=False):

    (net_server_frozen.train() if train_mode else net_server_frozen.eval())
    for p in net_server_frozen.parameters():
        p.requires_grad_(False)
    criterion = _TL.GlobalCriterion()

    s = smashed.detach().clone().requires_grad_(True)
    if args.model_name in ('RoBerta', 'RoBerta_large', 'DistilRoBerta', 'DistilBert', 'GPT2', 'Qwen'):
        assert mask is not None, "RoBerta query needs ext_mask"
        logits, _ = net_server_frozen(s, mask)
    else:
        logits, _ = net_server_frozen(s)
    loss = criterion(logits, label)
    grad = torch.autograd.grad(loss, s, create_graph=False, retain_graph=False)[0]
    return grad.detach()


_MSYNC_CUR = {'seed': None, 'devs': []}


def _rng_snapshot():
    return (torch.get_rng_state(), (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None))


def _rng_restore(snap):
    torch.set_rng_state(snap[0])
    if snap[1] is not None:
        torch.cuda.set_rng_state_all(snap[1])


def mask_sync_on(args):
    return bool(getattr(args, 'hd_mask_sync', False))


def mask_seed(args, rnd, cid, bidx):
    base = int(getattr(args, 'seed', 123))
    h = (base * 1000003) ^ (int(rnd) * 1000033) ^ (int(cid) * 1009) ^ (int(bidx) * 97)
    return h & 0x7FFFFFFF


@contextlib.contextmanager
def masked_rng(args, rnd, cid, bidx, device=None):
    if not mask_sync_on(args) or rnd is None or cid is None or bidx is None:
        yield None
        return
    dev = device if device is not None else getattr(args, 'device', 'cpu')
    devs = []
    try:
        if torch.cuda.is_available() and str(dev).startswith('cuda'):
            devs = [torch.cuda.current_device()]
    except Exception:
        devs = []
    sd = mask_seed(args, rnd, cid, bidx)
    _prev = dict(_MSYNC_CUR)
    with torch.random.fork_rng(devices=devs):
        torch.manual_seed(sd)
        if devs:
            torch.cuda.manual_seed_all(sd)
        _MSYNC_CUR['seed'], _MSYNC_CUR['devs'] = sd, devs
        try:
            yield sd
        finally:
            _MSYNC_CUR.update(_prev)


def _cache_on_cpu(args):
    v = str(getattr(args, 'round_cache_device', 'auto')).lower()
    if v == 'cpu':
        return True
    if v == 'gpu':
        return False
    return args.model_name not in ('RoBerta', 'RoBerta_large', 'DistilRoBerta', 'DistilBert', 'GPT2', 'Qwen')


def _g0_and_logits_for_omega(net_server, s0, y, m, is_llm, criterion, args):
    if bool(getattr(args, 'hd_omega_recompute', False)):
        assert bool(getattr(args, 'hd_eval_exchange', False)), '[OMEGA-RECOMPUTE] bit-identical only with eval exchange (--hd_eval_exchange); train mode not allowed'
        logits, _ = (net_server(s0, m) if is_llm else net_server(s0))
        g = torch.autograd.grad(criterion(logits, y), s0, create_graph=False, retain_graph=False)[0]
        del logits
        logits, _ = (net_server(s0, m) if is_llm else net_server(s0))
        return g, logits
    logits, _ = (net_server(s0, m) if is_llm else net_server(s0))
    g = torch.autograd.grad(criterion(logits, y), s0, create_graph=False, retain_graph=True)[0]
    return g, logits

def get_round_initial_cache(net_server, smashed_data_list, label_list, args,
                            mask_list=None, rnd=None, cid=None, want_lowrank=False, want_target=False, want_omega=False):
    if mask_sync_on(args):
        net_server.train()
    else:

        (net_server.eval() if getattr(args, 'hd_eval_exchange', False) else net_server.train())
    criterion = _TL.GlobalCriterion()
    g_list, U_list, th_list = [], [], []
    a_star_list = []
    assert not (want_lowrank and want_target), '[TARGET] cannot be combined with lowrank'
    assert not (want_omega and (want_lowrank or want_target)), '[OMEGA] cannot be combined with lowrank/target'
    omega_list, omega_alt_list = [], []
    _om_mode = str(getattr(args, 'hd_mode', ''))
    _om_both = bool(int(getattr(args, 'hd_omega_log_both', 0) or 0))
    _om_taylor = want_omega and (str(getattr(args, 'hd_omega_scale', 'taylor')) in ('taylor', 'radius'))
    cbar_list = []
    _eta_t = float(getattr(args, 'hd_target_eta', 0.0) or 0.0) if want_target else 0.0
    is_llm = args.model_name in ('RoBerta', 'RoBerta_large', 'DistilRoBerta', 'DistilBert', 'GPT2', 'Qwen')
    _oc = _cache_on_cpu(args)
    _k = int(getattr(args, 'hd_lanczos_k', 4))
    _m = int(getattr(args, 'hd_lanczos_iters', 8))
    _bds = []

    for i in range(len(smashed_data_list)):
        s0 = smashed_data_list[i].to(args.device).detach().requires_grad_(True)
        y  = label_list[i].to(args.device)
        m  = mask_list[i].to(args.device) if is_llm else None
        with masked_rng(args, rnd, cid, i, args.device):
            if bool(getattr(args, 'instr_q0', True)):
                from train.hd_instr import q0_capture as _q0c
                if i == 0:
                    args._q0_ref = {}
                args._q0_ref[i] = _q0c(net_server, args.device)
            if want_lowrank and str(getattr(args, 'hd_lowrank_src', 'fisher')) == 'subhvp':

                _hv = make_hvp_cached(net_server, s0, y, m, is_llm, args=args, tag='subhvp')
                try:
                    g = _hv.g1.detach()
                    U, th, _Mb = subhvp_lowrank(_hv, s0, _k)
                finally:
                    _hv.close()
                U_list.append(U.detach().cpu() if _oc else U.detach())
                th_list.append(th.detach().cpu() if _oc else th.detach())
                _CM.down(_Mb, msgs=0)
                _bds.append(int(_k))
                if i == 0 and (rnd is None or int(rnd) % 10 == 1):
                    print('[SUBHVP] r=%s cid=%s k=%d n=%d | eig(M) sample0=%s | hvp_calls=%d (batch)' % (
                        rnd, cid, int(_k), int(U.shape[1]), ' '.join('%.3e' % float(x) for x in th[0].tolist()), int(_k)), flush=True)
                del U, th, _Mb
            elif want_lowrank:
                _gvp = make_gvp_cached(net_server, s0, m, is_llm, y, args=args, tag='fisher')
                try:
                    g = _gvp.g0
                    U, th, _bd = lanczos_lowrank(_gvp, s0, g, _m, _k, return_bd=True)
                finally:
                    _gvp.close()
                U_list.append(U.detach().cpu() if _oc else U.detach())
                th_list.append(th.detach().cpu() if _oc else th.detach())
                _CM.down(U, th, msgs=0)
                _bds.append(int(_bd))
                del U, th
            elif want_omega and _om_mode == 'omegaMpJ':

                from train.hd_omega import compute_J_blocks, rslocM_task, M_ELL, G_ELL, rslocMp_m, rslocMpJ_pack, compute_J_blocks_lm, rslocMp_m_lm, rslocMpJ_pack_lm, compute_J_sample_lm, rslocMp_m_lm_sample, rslocMpJ_pack_lm1
                g, logits = _g0_and_logits_for_omega(net_server, s0, y, m, is_llm, criterion, args)
                _lt, _kb = rslocM_task(args)
                if _kb == 'lm' and str(getattr(args, 'rslocMpJ_lm_block', 'sample')) == 'sample':
                    _J_blk, _jinfo, _wk, _Gk = compute_J_sample_lm(logits, s0, int(s0.size(0)), y, int(getattr(args, 'rslocMpJ_probes', 1)))
                    _mn = rslocMp_m_lm_sample(logits, int(s0.size(0)), y, M_ELL[_lt])
                    _pk = rslocMpJ_pack_lm1(_J_blk, _mn, _wk, _Gk)
                elif _kb == 'lm':
                    _J_blk, _jinfo, _pos, _wk = compute_J_blocks_lm(logits, s0, int(s0.size(0)), y, int(getattr(args, 'rslocMpJ_probes', 1)))
                    _mn = rslocMp_m_lm(logits, int(s0.size(0)), _pos, M_ELL[_lt])
                    _pk = rslocMpJ_pack_lm(_J_blk, _mn, _wk)
                else:
                    _J_blk, _jinfo = compute_J_blocks(logits, s0, int(s0.size(0)), str(getattr(args, 'rslocM_exact_omega', 'auto')), int(getattr(args, 'rslocMpJ_probes', 1)))
                    _mn = rslocMp_m(logits, int(s0.size(0)), M_ELL[_lt], str(getattr(args, 'rslocMp_m', 'trace')), loss_kind=_lt)
                    _pk = rslocMpJ_pack(_J_blk, _mn)
                args._rslocMpJ_mode = _jinfo['mode']

                with torch.no_grad():
                    _Jf = _J_blk.reshape(_J_blk.size(0), _J_blk.size(1), _J_blk.size(2), -1).float()
                    _JF2 = (_Jf * _Jf).sum((0, 2, 3)) * (1.0 if _jinfo['mode'] == 'E' else 1.0 / float(_jinfo['R']))
                    _r0 = _r0_per_sample(logits, y, int(s0.size(0)), _kb)
                    if i == 0:
                        args._stepcond_acc = []
                    args._stepcond_acc.append((_JF2.cpu(), _r0.cpu()))
                omega_list.append(_pk.cpu() if _oc else _pk)
                _CM.down(_pk, msgs=0)
                if not getattr(args, '_rslocMpJ_logged', False):
                    args._rslocMpJ_logged = True
                    print(f"[RSLOCMPJ] w_n={('1/N_valid (sample blocks)' if _jinfo['mode']=='S' else 'mask/N_valid') if _kb=='lm' else '1/('+str(_kb)+'*N)'} M_ell={M_ELL[_lt]} G_ell={G_ELL[_lt]:.7g}{'·√n_t' if _jinfo['mode']=='S' else ''} L_S=3 m_mode={getattr(args,'rslocMp_m','trace')} loss={_lt} C={_jinfo['C']} blocks={_jinfo['K']} mode={_jinfo['mode']} R={_jinfo['R']} n_backward/batch={_jinfo['n_backward']} (N={int(s0.size(0))}{', T1='+str(_jinfo['T1'])+', n_valid='+str(_jinfo['n_valid']) if _kb=='lm' else ''}) | m_n med={float(_mn.median()):.4f} max={float(_mn.max()):.4f}| {tuple(_pk.shape)} {_pk.numel()*4/1e6:.1f}MB/batch", flush=True)
                del logits, _J_blk, _mn, _pk
            elif want_omega and _om_mode in ('omegaMp', 'omegaMpG'):

                from train.hd_omega import compute_omega_M, rslocM_task, M_ELL, rslocMp_m, rslocMp_pack
                g, logits = _g0_and_logits_for_omega(net_server, s0, y, m, is_llm, criterion, args)
                _lt, _kb = rslocM_task(args)
                _om_blk, _oinfo = compute_omega_M(logits, s0, int(s0.size(0)), str(getattr(args, 'rslocM_exact_omega', 'auto')), return_blocks=True)
                _mn = rslocMp_m(logits, int(s0.size(0)), M_ELL[_lt], str(getattr(args, 'rslocMp_m', 'trace')), loss_kind=_lt)
                _pk = rslocMp_pack(_om_blk, _mn)
                omega_list.append(_pk.cpu() if _oc else _pk)
                _CM.down(_pk, msgs=0)
                if not getattr(args, '_rslocMp_logged', False):
                    args._rslocMp_logged = True
                    from train.hd_omega import G_ELL as _GE
                    print(f"[RSLOCMP] arm={'rsloc_MpG' if _om_mode == 'omegaMpG' else 'rsloc_Mp'} G_ell={_GE[_lt] if _om_mode == 'omegaMpG' else '-'} w_n=1/({_kb}*N) M_ell={M_ELL[_lt]} L_S=3 m_mode={getattr(args,'rslocMp_m','trace')} loss={_lt} C={_oinfo['C']} blocks={_oinfo['K']} exact_omega={_oinfo['exact']} n_backward/batch={_oinfo['n_backward']} (N={int(s0.size(0))}) | m_n med={float(_mn.median()):.4f} max={float(_mn.max()):.4f}", flush=True)
                del logits, _om_blk, _mn, _pk
            elif want_omega and _om_mode == 'omegaM':

                from train.hd_omega import compute_omega_M, rslocM_task, M_ELL
                g, logits = _g0_and_logits_for_omega(net_server, s0, y, m, is_llm, criterion, args)
                _lt, _kb = rslocM_task(args)
                _om_raw, _oinfo = compute_omega_M(logits, s0, int(s0.size(0)), str(getattr(args, 'rslocM_exact_omega', 'auto')))
                omega_list.append(_om_raw.cpu() if _oc else _om_raw)
                _CM.down(_om_raw, msgs=0)
                if not getattr(args, '_rslocM_logged', False):
                    args._rslocM_logged = True
                    print(f"[RSLOCM] w_n=1/({_kb}*N) M_ell={M_ELL[_lt]} loss={_lt} C={_oinfo['C']} blocks={_oinfo['K']} exact={_oinfo['exact']} n_backward/batch={_oinfo['n_backward']} (N={int(s0.size(0))})", flush=True)
                del logits, _om_raw
            elif want_omega:

                from train.hd_omega import compute_omega, normalize_omega
                g, logits = _g0_and_logits_for_omega(net_server, s0, y, m, is_llm, criterion, args)
                _om = compute_omega(logits.float() if logits.dtype != torch.float32 else logits, s0, _om_mode,
                                    int(getattr(args, 'hd_omega_m', 1)), _om_both)
                _key = '3b' if _om_mode == 'omega3b' else '3a'
                _oh = normalize_omega(_om[_key].float(), str(getattr(args, 'hd_omega_reduce', 'none')))
                omega_list.append(_oh.cpu() if _oc else _oh)
                _CM.down(_oh, msgs=0)
                if _om_both:
                    _alt = normalize_omega(_om['3a' if _key == '3b' else '3b'].float(), str(getattr(args, 'hd_omega_reduce', 'none')))
                    omega_alt_list.append(_alt.cpu() if _oc else _alt)
                if _om_taylor:
                    from train.hd_omega import compute_cbar
                    _cb = compute_cbar(logits, s0, retain_graph=False)
                    cbar_list.append(_cb)
                    _CM.down(torch.zeros(1), msgs=0)
                    if i < 4:
                        print(f'[OMEGA] cbar r={rnd} cid={cid} j={i} v={_cb:.6e}', flush=True)
                del logits, _om, _oh
            else:
                logits, _ = (net_server(s0, m) if is_llm else net_server(s0))
                _gadj = getattr(args, '_gas_adj', {}).get(int(cid) if cid is not None else -1) if bool(getattr(args, 'gas_lf', False)) else None
                if _gadj is not None:
                    logits = logits + _gadj.to(logits.device, logits.dtype)[None, :]
                g = torch.autograd.grad(criterion(logits, y), s0, create_graph=False)[0]
        g_list.append(g.detach().cpu() if _oc else g.detach())
        if want_target:
            from train.hd_target import make_target
            _ast = make_target(s0.detach(), g, _eta_t)
            a_star_list.append(_ast.cpu() if _oc else _ast)
            _CM.down(_ast, msgs=0)
            del _ast
        else:
            _CM.down(g, msgs=0)
        net_server.zero_grad()
        del s0, g

    if want_lowrank and _PAY_LOG:
        _n = int(U_list[0].shape[1]) if U_list else 0
        _kk = int(U_list[0].shape[2]) if U_list else 0

        _t0 = th_list[0]
        _live = int((_t0.abs() > 0).sum(dim=1).max().item()) if _t0.dim() == 2 else -1
        print('[PAY-build] r=%s cli=%s B=%d n=%d K=%d th_live=%d/%d bd=%s'
              % (rnd, cid, len(g_list), _n, _kk, _live, _kk, _bds[:4]), flush=True)
    if want_target:
        return g_list, a_star_list
    if want_omega:
        return g_list, omega_list, (omega_alt_list if _om_both else None), (cbar_list if _om_taylor else None)
    return (g_list, U_list, th_list) if want_lowrank else g_list

def client_ntk_seed(net_c, batch_input, g0, args, is_llm):
    was_training = net_c.training
    net_c.eval()
    net_c.zero_grad()
    a = (net_c(*batch_input)[0] if is_llm else net_c(batch_input))
    a1 = a.detach().clone()
    a.backward(g0)
    saved = []
    with torch.no_grad():
        for p in net_c.parameters():
            if p.grad is not None:
                d = p.grad.detach().clone(); saved.append((p, d)); p.add_(d, alpha=-1.0)
        a2 = (net_c(*batch_input)[0] if is_llm else net_c(batch_input)).detach()
        for p, d in saved:
            p.add_(d, alpha=1.0)
    net_c.zero_grad()
    if was_training:
        net_c.train()
    seed = (a2 - a1).detach()
    if float(seed.reshape(seed.size(0), -1).norm(dim=1).mean()) < 1e-9:
        return g0.detach()
    return seed


class _HDClientBase(object):

    def __init__(self, args, dataset=None, idxs=None, wandb=None, model_idx=None):
        self.args = args
        collator = (getattr(args, "sst2_collator", None)
                    if args.model_name in ('RoBerta', 'RoBerta_large', 'DistilRoBerta', 'DistilBert', 'GPT2', 'Qwen') else None)
        DS_alter = DatasetSplit(dataset, idxs)

        if not bool(getattr(args, 'fd_fixed_order', False)):
            random.shuffle(DS_alter.idxs)
        self.ldr_train = DataLoader(DS_alter, batch_size=args.local_bs,
                                    shuffle=False, collate_fn=collator,
                                    **_dl_kwargs())
        self.wandb = wandb
        self.model_idx = model_idx

    def get_smashed_data(self, net, rnd=None, cid=None):
        smashed_data, label, cached_inputs = [], [], []

        if mask_sync_on(self.args):
            net.train()
        else:
            (net.eval() if getattr(self.args, 'hd_eval_exchange', False) else net.train())
        with torch.no_grad():
            for _bi, (data, labels) in enumerate(self.ldr_train):
                data   = data.to(self.args.device)
                labels = labels.to(self.args.device)
                with masked_rng(self.args, rnd, cid, _bi, self.args.device):
                    fx = net(data)

                _oc = _cache_on_cpu(self.args)
                _a0 = fx.clone().detach()
                _in = data.clone().detach()
                if _oc:
                    _a0 = _a0.cpu(); _in = _in.cpu()
                smashed_data.append(_a0.requires_grad_(True))
                _CM.up(_a0, msgs=0)
                label.append(labels)
                cached_inputs.append(_in)
        return smashed_data, label, cached_inputs

    def get_smashed_data_llm(self, net, rnd=None, cid=None):
        smashed_data, mask_list, label_list = [], [], []
        if mask_sync_on(self.args):
            net.train()
        else:
            (net.eval() if getattr(self.args, 'hd_eval_exchange', False) else net.train())
        _oc = _cache_on_cpu(self.args)
        with torch.no_grad():
            for _bi, batch in enumerate(self.ldr_train):
                data  = batch['input_ids'].to(self.args.device)
                mask  = batch['attention_mask'].to(self.args.device)
                label = batch['labels'].to(self.args.device)
                with masked_rng(self.args, rnd, cid, _bi, self.args.device):
                    fx, ext_mask = net(data, mask)
                _a0 = fx.clone().detach()
                if _oc:
                    _a0 = _a0.cpu()
                smashed_data.append(_a0.requires_grad_(True))
                _CM.up(_a0, msgs=0)
                mask_list.append(ext_mask.clone().detach())
                label_list.append(label)
        return smashed_data, mask_list, label_list


_PREV_A0_STORE = {}

_ORACLE_1C_ROUNDS = set()


def diag_hessian_ggn(net_server, a, label, args, mask=None, rng_ctx=None, stepcond=None):


    is_llm = args.model_name in ('RoBerta', 'RoBerta_large', 'DistilRoBerta', 'DistilBert', 'GPT2', 'Qwen')

    _ctx = rng_ctx if rng_ctx is not None else contextlib.nullcontext
    (net_server.train() if rng_ctx is not None else net_server.eval())
    s = a.detach().clone().requires_grad_(True)
    _t = _TL.current_task()
    if _t == 'lm':
        raise RuntimeError('diag_hessian_ggn: the C=V class loop is impractical for lm; use hd_diag_method=hutchinson or the lowrank path')
    if _t == 'reg':
        with _ctx():
            _z = (net_server(s, mask)[0] if is_llm else net_server(s)[0])
        _J = torch.autograd.grad(_z.sum(), s)[0]
        return ((2.0 * _J * _J) / float(s.size(0))).detach()
    with _ctx():
        logits = (net_server(s, mask)[0] if is_llm else net_server(s)[0])
    C = logits.size(1)

    if C < 2:
        raise NotImplementedError(
            '[HD] diag_hessian_ggn is softmax-CE only (%d logit columns); for squared loss use --hd_mode lowrank.' % C)
    p = torch.softmax(logits, dim=1)
    S1 = torch.zeros_like(s)
    S2 = torch.zeros_like(s)
    JF2 = torch.zeros(s.size(0), device=s.device, dtype=s.dtype)
    for c in range(C):
        Jc = torch.autograd.grad(logits[:, c].sum(), s, retain_graph=(c < C - 1))[0]
        pc = p[:, c].view(-1, *([1] * (s.dim() - 1)))
        S1 = S1 + pc * (Jc * Jc)
        S2 = S2 + pc * Jc
        JF2 = JF2 + (Jc * Jc).reshape(s.size(0), -1).sum(1)
    if isinstance(stepcond, dict):

        _r = p.detach().clone()
        if label is not None and label.numel() == p.size(0):
            _r[torch.arange(p.size(0), device=p.device), label.reshape(-1).long()] -= 1.0
        stepcond['JF2'] = JF2.detach(); stepcond['r0'] = _r.norm(dim=1).detach()
    return ((S1 - S2 * S2) / float(s.size(0))).detach()


def diag_hessian_hutchinson(net_server, a, label, args, mask=None, m_diag=8, eps_rel=1e-2, rng_ctx=None, stats=None):
    a = a.detach()
    an = a.reshape(a.size(0), -1).norm()
    diag = torch.zeros_like(a)
    for _ in range(int(m_diag)):
        z = (torch.randint(0, 2, a.shape, device=a.device, dtype=a.dtype) * 2 - 1)
        vn = z.reshape(z.size(0), -1).norm()
        eps = float(eps_rel) * float((an / (vn + 1e-12)).item())
        if rng_ctx is not None:

            with rng_ctx():
                gp = server_grad_query(net_server, a + eps * z, label, args, mask, train_mode=True)
            with rng_ctx():
                gm = server_grad_query(net_server, a - eps * z, label, args, mask, train_mode=True)
        else:
            gp = server_grad_query(net_server, a + eps * z, label, args, mask)
            gm = server_grad_query(net_server, a - eps * z, label, args, mask)
        _est = z * ((gp - gm) / (2.0 * eps))
        if isinstance(stats, dict):
            stats.setdefault('probes', []).append(_est.detach())
        diag = diag + _est
    return (diag / float(m_diag)).detach()


def _r0_per_sample(logits, y, N, kind):
    with torch.no_grad():
        z = logits.detach().float()
        if z.size(1) == 1 and y.dtype.is_floating_point:
            return (2.0 * (z.reshape(-1) - y.reshape(-1).float())).abs()
        yy = y.reshape(-1).long()
        p = torch.softmax(z, dim=1)
        valid = (yy != -100)
        idx = torch.arange(z.size(0), device=z.device)
        r = p.clone()
        r[idx[valid], yy[valid]] -= 1.0
        r[~valid] = 0.0
        rn2 = (r * r).sum(1)
        R = z.size(0)
        if kind == 'lm':
            return rn2.reshape(N, R // N).sum(1).sqrt()
        if R == N:
            return rn2.sqrt()
        return rn2.reshape(R // N, N).sum(0).sqrt()


def _spath_collect(diag, z0, zb, label, taus=(0.0, 0.5, 1.0)):
    with torch.no_grad():
        z0 = z0.detach().float().reshape(-1, z0.size(-1)); zb = zb.detach().float().reshape(-1, zb.size(-1))
        if z0.size(0) != zb.size(0):
            return
        yy = label.reshape(-1) if label is not None else None
        keep = (yy != -100) if (yy is not None and yy.numel() == z0.size(0)) else torch.ones(z0.size(0), dtype=torch.bool, device=z0.device)
        if not bool(keep.any()):
            return
        z0 = z0[keep]; zb = zb[keep]; dz = zb - z0
        p0 = torch.softmax(z0, 1).max(1).values
        pm = torch.stack([torch.softmax(z0 + t * dz, 1).max(1).values for t in taus], 0)
        pmax_path = pm.max(0).values
        ratio = pmax_path / p0.clamp_min(1e-12)
        diag.setdefault('sp_p0', []).append(float(p0.median())); diag.setdefault('sp_pb', []).append(float(pm[-1].median()))
        diag.setdefault('sp_pp', []).append(float(pmax_path.median()))
        diag.setdefault('sp_ratio_med', []).append(float(ratio.median())); diag.setdefault('sp_ratio_max', []).append(float(ratio.max()))
        diag.setdefault('sp_ratio_gt3', []).append(float((ratio > 3.0).float().mean()))


def _stepcond_collect(diag, args, JF2, r0, m_ell=0.5):
    import math as _m
    eta = float(getattr(args, 'lr', 0.0)); JF2 = JF2.float(); r0 = r0.float()
    eta_max = 2.0 / (float(m_ell) * JF2.clamp_min(1e-30))
    diag.setdefault('sc_JF2_med', []).append(float(JF2.median())); diag.setdefault('sc_JF2_max', []).append(float(JF2.max()))
    diag.setdefault('sc_r0_med', []).append(float(r0.median())); diag.setdefault('sc_r0_max', []).append(float(r0.max()))
    diag.setdefault('sc_eta_med', []).append(float(eta_max.median())); diag.setdefault('sc_eta_min', []).append(float(eta_max.min()))
    diag.setdefault('sc_viol', []).append(float((eta > eta_max).float().mean()))


def compute_diag(net_server, a, label, args, mask=None, rng_ctx=None, stepcond=None, stats=None):
    if str(getattr(args, 'hd_diag_method', 'ggn')) == 'hutchinson':
        return diag_hessian_hutchinson(
            net_server, a, label, args, mask,
            m_diag=int(getattr(args, 'hd_mdiag', 8)),
            eps_rel=float(getattr(args, 'hd_eps', 1e-2)), rng_ctx=rng_ctx, stats=stats)
    return diag_hessian_ggn(net_server, a, label, args, mask, rng_ctx=rng_ctx, stepcond=stepcond)


def build_krylov_basis(net_server, x2, g0, label, args, mask, m, eps_rel, seed=None):
    B = x2.size(0)
    flat = lambda t: t.reshape(B, -1)
    an = flat(x2).norm()

    def _hvp(v):
        vn = flat(v).norm()
        eps = float(eps_rel) * float((an / (vn + 1e-12)).item())
        gp = server_grad_query(net_server, (x2 + eps * v).detach(), label, args, mask)
        gm = server_grad_query(net_server, (x2 - eps * v).detach(), label, args, mask)
        return ((gp - gm) / (2.0 * eps)).detach()

    def _normrows(t):
        n = flat(t).norm(dim=1).clamp(min=1e-12)
        return t / n.view(-1, *([1] * (t.dim() - 1)))

    U, HU = [], []
    u = _normrows(seed if seed is not None else g0)
    for j in range(int(m)):
        U.append(u)
        Hu = _hvp(u)
        HU.append(Hu)
        if j < int(m) - 1:
            w = Hu.clone()
            for uk in U:
                coef = (flat(w) * flat(uk)).sum(dim=1)
                w = w - coef.view(-1, *([1] * (w.dim() - 1))) * uk
            u = _normrows(w)
    return U, HU


def correct_hybrid(delta, U, HU, diagH, g0):
    B = delta.size(0)
    flat = lambda t: t.reshape(B, -1)
    df = flat(delta)
    corr_lr = torch.zeros_like(g0)
    d_par = torch.zeros_like(delta)
    for u, Hu in zip(U, HU):
        c = (df * flat(u)).sum(dim=1).view(-1, *([1] * (delta.dim() - 1)))
        corr_lr = corr_lr + c * Hu
        d_par = d_par + c * u
    d_perp = delta - d_par
    return g0 + corr_lr + diagH * d_perp


def correct_diag_only(delta, diagH, g0):
    return g0 + diagH * delta


def ggn_lowrank(net_server, a, args, mask=None, rank=None, eps_floor=1e-12):
    is_llm = args.model_name in ('RoBerta', 'RoBerta_large', 'DistilRoBerta', 'DistilBert', 'GPT2', 'Qwen')
    net_server.eval()
    s = a.detach().clone().requires_grad_(True)
    logits = (net_server(s, mask)[0] if is_llm else net_server(s)[0])
    B, C = logits.size(0), logits.size(1)
    p = torch.softmax(logits, dim=1)
    bshape = [1] * (s.dim() - 1)
    Js = []
    for c in range(C):
        Jc = torch.autograd.grad(logits[:, c].sum(), s, retain_graph=(c < C - 1))[0]
        Js.append(Jc.detach())
    J = torch.stack(Js, dim=1)
    pj = p.view(B, C, *bshape)
    mean = (pj * J).sum(dim=1, keepdim=True)
    Vcol = torch.sqrt(p.clamp(min=0)).view(B, C, *bshape) * (J - mean)
    Vf = Vcol.reshape(B, C, -1)
    M = torch.einsum('bcn,bdn->bcd', Vf, Vf)
    evals, evecs = torch.linalg.eigh(M)
    r = (C - 1) if rank is None else max(1, min(int(rank), C - 1))
    idx = torch.argsort(evals, dim=1, descending=True)[:, :r]
    theta = torch.gather(evals, 1, idx).clamp(min=0.0)
    sel = torch.gather(evecs, 2, idx.unsqueeze(1).expand(B, C, r))
    U = torch.einsum('bcr,bcn->bnr', sel, Vf)
    inv = torch.rsqrt(theta.clamp(min=eps_floor)).unsqueeze(1)
    valid = (theta > eps_floor).to(U.dtype).unsqueeze(1)
    U = U * inv * valid
    return U.detach(), (theta / float(B)).detach()


_ACT_ON    = bool(int(__import__('os').environ.get('HD_ACT_LOG', '1')))
_ACT_EVERY = max(1, int(__import__('os').environ.get('HD_ACT_EVERY', '10')))
_ACT_QS    = (1, 2, 4, 8)
_PAY_LOG   = bool(int(__import__('os').environ.get('HD_PAY_LOG', '1')))


def subhvp_lowrank(hvp_fn, x2, k):
    B = x2.size(0); n = x2[0].numel(); k_t = max(1, int(k))
    A = x2.detach().reshape(B, -1)
    Vh = torch.linalg.svd(A, full_matrices=False).Vh
    kk = min(k_t, Vh.size(0))
    U = Vh[:kk].t().contiguous()
    HU = []
    for j in range(kk):
        v = U[:, j].reshape(1, *x2.shape[1:]).expand_as(x2).contiguous()
        HU.append(hvp_fn(v).reshape(B, -1))
    HU = torch.stack(HU, dim=2)
    M = torch.einsum('nk,bnl->bkl', U, HU)
    M = 0.5 * (M + M.transpose(1, 2))
    lam, V = torch.linalg.eigh(M)
    Ub = torch.einsum('nk,bkl->bnl', U, V)
    if kk < k_t:
        pad = k_t - kk
        lam = torch.cat([lam, lam.new_zeros(B, pad)], dim=1); Ub = torch.cat([Ub, Ub.new_zeros(B, n, pad)], dim=2)
    return Ub.detach(), lam.detach(), M.detach()


def lanczos_lowrank(hvp_fn, x2, seed, m_iters, k, reorth=True, return_bd=False):
    B = x2.size(0)
    bshape = [1] * (x2.dim() - 1)
    flat = lambda t: t.reshape(B, -1)
    def _normrows(t):
        nn = flat(t).norm(dim=1).clamp(min=1e-12)
        return t / nn.view(B, *bshape)
    Q, alphas, betas = [], [], []
    q_prev = torch.zeros_like(x2)
    beta = torch.zeros(B, device=x2.device, dtype=x2.dtype)
    q = _normrows(seed)
    m_iters = max(1, int(m_iters))
    _bd_step = m_iters
    scale = 0.0
    for j in range(m_iters):
        Q.append(q)
        w = hvp_fn(q)
        alpha = (flat(w) * flat(q)).sum(dim=1)
        alphas.append(alpha)
        scale = max(scale, float(alpha.abs().max()))
        w = w - alpha.view(B, *bshape) * q - beta.view(B, *bshape) * q_prev
        if reorth:
            for qk in Q:
                c = (flat(w) * flat(qk)).sum(dim=1)
                w = w - c.view(B, *bshape) * qk
        beta = flat(w).norm(dim=1)

        if j >= 1 and float(beta.max()) < 1e-6 * max(scale, 1e-30):
            _bd_step = j
            break
        if j < m_iters - 1:
            betas.append(beta)
            q_prev = q
            q = w / beta.clamp(min=1e-12).view(B, *bshape)
    m = len(alphas)
    A = torch.stack(alphas, dim=1)
    T = torch.diag_embed(A)
    if m > 1:
        Bd = torch.stack(betas, dim=1)
        T = T + torch.diag_embed(Bd, offset=1) + torch.diag_embed(Bd, offset=-1)
    evals, evecs = torch.linalg.eigh(T)
    kk = min(int(k), m)
    idx = torch.argsort(evals.abs(), dim=1, descending=True)[:, :kk]
    theta = torch.gather(evals, 1, idx)
    sel = torch.gather(evecs, 2, idx.unsqueeze(1).expand(B, m, kk))
    Qf = torch.stack(Q, dim=1).reshape(B, m, -1)
    U = torch.einsum('bmk,bmn->bnk', sel, Qf)

    k_t = max(1, int(k))
    if kk < k_t:
        pad = k_t - kk
        theta = torch.cat([theta, theta.new_zeros(B, pad)], dim=1)
        U = torch.cat([U, U.new_zeros(B, U.size(1), pad)], dim=2)
    if return_bd:
        return U.detach(), theta.detach(), _bd_step
    return U.detach(), theta.detach()


def ggn_vec_product(net_server, a, mask, v, is_llm, label, loss_fn=None):
    fwd = (lambda x: net_server(x, mask)[0]) if is_llm else (lambda x: net_server(x)[0])

    a1 = a.detach().clone().requires_grad_(True)
    z1 = fwd(a1)
    L = (loss_fn(z1, label) if loss_fn is not None
         else _TL.gnll(z1, label))
    gz = torch.autograd.grad(L, z1, create_graph=True)[0]

    r = torch.zeros_like(z1, requires_grad=True)
    JTr = torch.autograd.grad(z1, a1, grad_outputs=r, create_graph=True, retain_graph=True)[0]
    Jv = torch.autograd.grad((JTr * v.detach()).sum(), r, retain_graph=True)[0]

    w = torch.autograd.grad((gz * Jv.detach()).sum(), z1, retain_graph=True)[0].detach()

    Gv = torch.autograd.grad((z1 * w).sum(), a1)[0]
    return Gv.detach()


_GVPC_STAT = {'obj': 0, 'matvec': 0, 'fwd': 0, 'fwd_naive': 0,
              'logn': 0, 'maxrel': 0.0, 'verified': 0, 'kind': {}}


def _gvpc_env(name, default):
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return int(default)


class _GVPCacheBase(object):
    kind = 'base'

    def _init_common(self, a, verify, tag):
        self.tag = tag
        self.maxrel, self.n_matvec, self.n_verified = 0.0, 0, 0
        self._verify_left = int(verify)
        self._cuda = bool(getattr(a, 'is_cuda', False))
        self._m0 = 0
        if self._cuda:
            try:
                import torch as _t
                self._m0 = _t.cuda.max_memory_allocated()
            except Exception:
                self._cuda = False
        _GVPC_STAT['obj'] += 1
        _GVPC_STAT['fwd'] += 1
        _GVPC_STAT['kind'][self.kind] = _GVPC_STAT['kind'].get(self.kind, 0) + 1

    def __call__(self, v):
        out = self._matvec(v)
        self.n_matvec += 1
        _GVPC_STAT['matvec'] += 1
        _GVPC_STAT['fwd_naive'] += 1
        if self._verify_left > 0:
            self._verify_left -= 1
            ref = self._reference(v)
            den = float(ref.abs().max()) + 1e-30
            rel = float((out - ref).abs().max()) / den
            self.maxrel = max(self.maxrel, rel)
            self.n_verified += 1
            _GVPC_STAT['maxrel'] = max(_GVPC_STAT['maxrel'], rel)
            _GVPC_STAT['verified'] += 1
            del ref
            if rel > 1e-6:
                raise RuntimeError('[GVP-CACHE] cached/uncached mismatch maxrel=%.3e (>1e-6) kind=%s tag=%s; fall back to HD_GVP_CACHE=0'
                                   % (rel, self.kind, self.tag))
        return out

    def close(self):
        dm = 0.0
        if self._cuda:
            try:
                import torch as _t
                dm = (_t.cuda.max_memory_allocated() - self._m0) / float(2 ** 20)
            except Exception:
                dm = 0.0
        m = max(1, self.n_matvec)
        cap = _gvpc_env('HD_GVP_LOGCAP', 8)
        if _GVPC_STAT['logn'] < cap:
            _GVPC_STAT['logn'] += 1
            tot_c, tot_n = _GVPC_STAT['matvec'], _GVPC_STAT['fwd_naive']
            print('[GVP-CACHE] kind=%s tag=%s m=%d | fwd 1/%d  jtr 1/%d  passes %d/%d | verify n=%d maxrel=%.3e bitexact=%s | mem_peak_delta=%+.1fMB | cumulative obj=%d matvec=%d fwd %d/%d (saved %.1f%%)'
                  % (self.kind, self.tag, m, m, m,
                     self._pass_cached(m), self._pass_naive(m),
                     self.n_verified, self.maxrel,
                     'T' if (self.n_verified and self.maxrel < 1e-9) else
                     ('~' if self.n_verified else '-'),
                     dm, _GVPC_STAT['obj'], tot_c,
                     _GVPC_STAT['fwd'], tot_n,
                     100.0 * (1.0 - _GVPC_STAT['fwd'] / float(max(1, tot_n)))),
                  flush=True)
        for k in ('JTr', 'g1', 'z1', 'r', 'a1', 'gz'):
            if hasattr(self, k):
                try:
                    delattr(self, k)
                except Exception:
                    pass


class _GVPCache(_GVPCacheBase):
    kind = 'ggn'

    def __init__(self, net_server, a, mask, is_llm, label, loss_fn=None, verify=0, tag=''):
        self._net, self._a, self._mask, self._is_llm = net_server, a, mask, is_llm
        self._label, self._loss_fn = label, loss_fn
        self.B = float(a.size(0))
        fwd = (lambda x: net_server(x, mask)[0]) if is_llm else (lambda x: net_server(x)[0])
        self.a1 = a.detach().clone().requires_grad_(True)
        self.z1 = fwd(self.a1)
        _L = (loss_fn(self.z1, label) if loss_fn is not None
              else _TL.gnll(self.z1, label))
        self.gz = torch.autograd.grad(_L, self.z1, create_graph=True)[0]
        self.r = torch.zeros_like(self.z1, requires_grad=True)
        self.JTr = torch.autograd.grad(self.z1, self.a1, grad_outputs=self.r,
                                       create_graph=True, retain_graph=True)[0]

        self.g0 = torch.autograd.grad(_L, self.a1, retain_graph=True)[0].detach()

        self._msync_seed = _MSYNC_CUR.get('seed')
        self._msync_devs = list(_MSYNC_CUR.get('devs') or [])
        self._init_common(a, verify, tag)

    def _matvec(self, v):
        Jv = torch.autograd.grad((self.JTr * v.detach()).sum(), self.r, retain_graph=True)[0]

        w = torch.autograd.grad((self.gz * Jv.detach()).sum(), self.z1,
                                retain_graph=True)[0].detach()
        Gv = torch.autograd.grad((self.z1 * w).sum(), self.a1, retain_graph=True)[0]
        return Gv.detach()

    def _reference(self, v):

        _sd = getattr(self, '_msync_seed', None)
        if _sd is None:
            return ggn_vec_product(self._net, self._a, self._mask, v, self._is_llm,
                                   self._label, self._loss_fn)
        with torch.random.fork_rng(devices=self._msync_devs):
            torch.manual_seed(_sd)
            if self._msync_devs:
                torch.cuda.manual_seed_all(_sd)
            return ggn_vec_product(self._net, self._a, self._mask, v, self._is_llm,
                                   self._label, self._loss_fn)

    @staticmethod
    def _pass_naive(m):

        return 4 * m

    @staticmethod
    def _pass_cached(m):
        return 2 * m + 2


class _HVPCache(_GVPCacheBase):
    kind = 'hvp'

    def __init__(self, net_server, a, lab, mask, is_llm, verify=0, tag=''):
        self._net, self._a, self._lab = net_server, a, lab
        self._mask, self._is_llm = mask, is_llm
        self.a1 = a.detach().clone().requires_grad_(True)
        z1 = (net_server(self.a1, mask)[0] if is_llm else net_server(self.a1)[0])
        self.g1 = torch.autograd.grad(_TL.gnll(z1, lab), self.a1,
                                      create_graph=True, retain_graph=True)[0]

        self._msync_seed = _MSYNC_CUR.get('seed')
        self._msync_devs = list(_MSYNC_CUR.get('devs') or [])
        self._init_common(a, verify, tag)

    def _matvec(self, v):
        hv = torch.autograd.grad((self.g1 * v.detach()).sum(), self.a1, retain_graph=True)[0]
        return hv.detach()

    def _reference(self, v):
        def _plain():
            _a1 = self._a.detach().clone().requires_grad_(True)
            _z1 = (self._net(_a1, self._mask)[0] if self._is_llm else self._net(_a1)[0])
            _g1 = torch.autograd.grad(_TL.gnll(_z1, self._lab), _a1, create_graph=True)[0]
            return torch.autograd.grad((_g1 * v.detach()).sum(), _a1)[0].detach()
        _sd = getattr(self, '_msync_seed', None)
        if _sd is None:
            return _plain()
        with torch.random.fork_rng(devices=self._msync_devs):
            torch.manual_seed(_sd)
            if self._msync_devs:
                torch.cuda.manual_seed_all(_sd)
            return _plain()

    @staticmethod
    def _pass_naive(m):
        return 3 * m

    @staticmethod
    def _pass_cached(m):
        return m + 2


class _NoCacheShim(object):

    def __init__(self, fn):
        self._fn = fn
        self.n_matvec, self.maxrel = 0, 0.0

    def __call__(self, v):
        self.n_matvec += 1
        return self._fn(v)

    def close(self):
        return None


def _gvpc_verify_budget():
    n = _gvpc_env('HD_GVP_VERIFY', 1)
    if n <= 0:
        return 0
    return n if _GVPC_STAT['obj'] < _gvpc_env('HD_GVP_VERIFY_OBJ', 4) else 0


def make_gvp_cached(net_server, a, mask, is_llm, label, loss_fn=None, args=None, tag=''):


    if args is not None and mask_sync_on(args) and not _gvpc_env('HD_GVP_CACHE', 1):
        raise RuntimeError(
            '[abort] HD_GVP_CACHE=0 with --hd_mask_sync is not allowed (mask changes per matvec); enable the cache or drop --hd_mask_sync.')
    on = _gvpc_env('HD_GVP_CACHE', int(getattr(args, 'hd_gvp_cache', 1)) if args is not None else 1)
    if not on:
        return _NoCacheShim(lambda v: ggn_vec_product(net_server, a, mask, v, is_llm, label, loss_fn))
    return _GVPCache(net_server, a, mask, is_llm, label, loss_fn,
                     verify=_gvpc_verify_budget(), tag=tag)


def make_hvp_cached(net_server, a, lab, mask, is_llm, args=None, tag=''):
    on = _gvpc_env('HD_GVP_CACHE', int(getattr(args, 'hd_gvp_cache', 1)) if args is not None else 1)
    if not on:
        def _plain(v):
            _a1 = a.detach().clone().requires_grad_(True)
            _z1 = (net_server(_a1, mask)[0] if is_llm else net_server(_a1)[0])
            _g1 = torch.autograd.grad(_TL.gnll(_z1, lab), _a1, create_graph=True)[0]
            return torch.autograd.grad((_g1 * v.detach()).sum(), _a1)[0].detach()
        return _NoCacheShim(_plain)
    return _HVPCache(net_server, a, lab, mask, is_llm, verify=_gvpc_verify_budget(), tag=tag)


def _server_last_hidden(net_server, a, ext_mask, is_llm):
    with torch.no_grad():
        hs = a
        for key in sorted(net_server.layers.keys(), key=int):
            hs = (net_server.layers[key](hs, attention_mask=ext_mask)[0] if is_llm
                  else net_server.layers[key](hs)[0])
    return hs.detach()


def _pd_params(net):
    return [p for _, p in net.named_parameters() if p.requires_grad]


def _pd_snapshot(net):
    return [(None if p.grad is None else p.grad.detach().clone()) for p in _pd_params(net)]


def _pd_flat(snap):
    import torch as _t
    vs = [(g.reshape(-1) if g is not None else None) for g in snap]
    ref = next((v for v in vs if v is not None), None)
    if ref is None:
        return None
    return _t.cat([(v if v is not None else _t.zeros(1, device=ref.device, dtype=ref.dtype))
                   for v in vs])


def _pd_restore(net, snap):
    for p, g in zip(_pd_params(net), snap):
        p.grad = (None if g is None else g)


def correct_lowrank(delta, U, theta, g0, gate_neg=False, cap_tau=0.0, cap_unit='sample'):
    B = delta.size(0)
    df = delta.reshape(B, -1)
    c = torch.einsum('bnr,bn->br', U, df)
    th = theta.clamp(min=0.0) if gate_neg else theta
    corr = torch.einsum('bnr,br->bn', U, th * c)
    if cap_tau and cap_tau > 0:
        g0f = g0.reshape(B, -1)
        if str(cap_unit) == 'batch':
            cn = corr.norm() + 1e-12
            scale = torch.clamp(cap_tau * g0f.norm() / cn, max=1.0)
        else:
            cn = corr.norm(dim=1, keepdim=True) + 1e-12
            scale = torch.clamp(cap_tau * g0f.norm(dim=1, keepdim=True) / cn, max=1.0)
        corr = corr * scale
    return g0 + corr.reshape_as(g0)


class Localupdate_hess_diag_client(_HDClientBase):

    def train_hess_diag(self,
                        net_client,
                        net_server_frozen,
                        cached_x2_list,
                        cached_g_list,
                        cached_label_list,
                        cached_mask_list=None,
                        cached_inputs=None,
                        cached_labels=None,
                        cached_U_list=None,
                        cached_th_list=None,
                        cached_kappa_list=None,
                        cached_a_star_list=None,
                        cached_omega_list=None,
                        cached_omega_alt_list=None,
                        cached_cbar_list=None,
                        cur_epoch=None,
                        user_idx=None):
        args = self.args
        dev = args.device
        is_llm = args.model_name in ('RoBerta', 'RoBerta_large', 'DistilRoBerta', 'DistilBert', 'GPT2', 'Qwen')
        if is_llm:
            assert cached_mask_list is not None, 'the LLM path requires a mask'

        mode      = str(getattr(args, 'hd_mode', 'hybrid'))
        m_basis   = int(getattr(args, 'hd_basis_m', 4))
        eps_rel   = float(getattr(args, 'hd_eps', 1e-2))
        is_hybrid = (mode == 'hybrid')

        _needs_dhat = mode in ('secant', 'midfisher')
        _prev_a0 = None
        if _needs_dhat and (user_idx is not None):
            _prev_a0 = _PREV_A0_STORE.pop(user_idx, None)

            _need_b = sum(int(t.numel()) for t in cached_x2_list) * 2
            _avail_b = -1
            try:
                with open('/proc/meminfo') as _mf:
                    for _ln in _mf:
                        if _ln.startswith('MemAvailable'):
                            _avail_b = int(_ln.split()[1]) * 1024
                            break
            except Exception:
                pass
            if 0 <= _avail_b < 2 * _need_b:
                raise RuntimeError(
                    f"[HD-B1B4] CPU RAM: δ̂ {_need_b/1e9:.2f}GB(fp16), {_avail_b/1e9:.2f}GB(<2x). "
                    f" mnli secant/midfisher RAM.")

            _PREV_A0_STORE[user_idx] = [None] * len(cached_x2_list)
            if _prev_a0 is None:
                print(f"[HD-B1B4] user {user_idx}: a0 → (secant=stale/midfisher=fisher) ", flush=True)
        _dhat_miss = 0

        _oracle_1c_skip = False
        if bool(getattr(args, 'hd_oracle_hess', False)) and bool(getattr(args, 'hd_oracle_single_client', False)):
            _rid = -1 if cur_epoch is None else int(cur_epoch)
            if _rid in _ORACLE_1C_ROUNDS:
                _oracle_1c_skip = True
                print(f"[HD-ORACLE-1C] user {user_idx}: → stale ", flush=True)
            else:
                _ORACLE_1C_ROUNDS.add(_rid)
                print(f"[HD-ORACLE-1C] user {user_idx}: H δ + ", flush=True)

        net_c = net_client.to(dev)

        if getattr(args, 'hd_train_forward', False):
            net_c.train()
        else:
            (net_c.eval() if getattr(args, 'hd_eval_exchange', False) else net_c.train())

        net_c_init = copy.deepcopy(net_c); net_c_init.eval()

        net_server_probe = (copy.deepcopy(net_server_frozen).to(dev) if mask_sync_on(args)
                            else net_server_frozen.to(dev))
        (net_server_probe.train() if mask_sync_on(args) else net_server_probe.eval())
        for p in net_server_probe.parameters():
            p.requires_grad_(False)

        net_server_eval = copy.deepcopy(net_server_frozen).to(dev)
        (net_server_eval.train() if mask_sync_on(args) else net_server_eval.eval())
        for p in net_server_eval.parameters():
            p.requires_grad_(False)

        optimizer_client = torch.optim.AdamW(
            net_c.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=1e-8)
        criterion = _TL.GlobalCriterion()

        _CAP_FORM = str(getattr(args, 'cap_form', 'none')); _CAP_RHO = float(getattr(args, 'cap_rho', 0.12)); _CAP_RHOT = float(getattr(args, 'cap_rho_theta', 0.12))
        _CAP_MAXBT = int(getattr(args, 'cap_max_bt', 3)); _CAP_LRMULT = float(getattr(args, 'cap_lr_mult', 0.5)); _CAP_LRM = 1.0; _CAP_PREV = None
        _CAP_TH0 = ([p.detach().clone() for p in net_c.parameters()] if _CAP_FORM != 'none' else None)
        _CAP_ST = {'pre': [], 'post': [], 'bt': [], 'capped': [], 'dtheta': float('nan')}
        _INSTR_R = [int(t) for t in str(getattr(args, 'instr_rounds', '') or '').replace(',', ' ').split()]
        _INSTR_ON = (cur_epoch is not None and int(cur_epoch) in _INSTR_R)
        _nb_i = len(cached_x2_list); _INSTR_BIDX = set()
        for _tok in str(getattr(args, 'instr_steps', 'mid,last')).replace(' ', ',').split(','):
            if _tok == 'mid': _INSTR_BIDX.add(max(0, _nb_i // 2 - 1))
            elif _tok == 'last': _INSTR_BIDX.add(_nb_i - 1)
            elif _tok.isdigit(): _INSTR_BIDX.add(int(_tok) - 1)
        _INSTR_DP = bool(getattr(args, 'instr_dpersist', False)); _DP_D = []; _DP_G = []; _U_eo = None
        _L2_ON = bool(_INSTR_ON) and bool(getattr(args, 'instr_l2', True)); _L2_IN = None
        _DP_NB = int(getattr(args, 'instr_dpersist_nb', 0)); _DP_SET = (set(int(round(i * (_nb_i - 1) / max(1, _DP_NB - 1))) for i in range(_DP_NB)) if _DP_NB > 0 else set())
        assert not (_CAP_FORM != 'none' and bool(getattr(args, 'hd_param_diag', False))), '[CAP] cannot be combined with hd_param_diag'
        if not getattr(args, '_q0_logged', False):
            from train.hd_instr import q0_setting_log as _q0l
            _q0l(net_c, net_server_frozen, args); args._q0_logged = True

        def _fetch(t, cast=False):
            if t is None:
                return None
            if t.device == dev:
                return t.float() if (cast and t.dtype != torch.float32) else t
            _rg = bool(t.requires_grad)
            u = t.detach().to(dev)
            if cast and u.dtype != torch.float32:
                u = u.float()
            return u.requires_grad_(True) if _rg else u

        _cpu_cache = bool(len(cached_x2_list)) and (cached_x2_list[0].device.type == 'cpu')
        if _cpu_cache:
            x2_gpu, g_gpu = cached_x2_list, cached_g_list
        else:
            x2_gpu  = [s.to(dev).float() for s in cached_x2_list]
            g_gpu   = [g.to(dev).float() for g in cached_g_list]
        lab_gpu = [l.to(dev) for l in cached_label_list]

        _LAM0 = None
        if mode == 'kprobe' and str(getattr(args, 'hd_kprobe_lambda_mode', 'abs')) == 'lam0':
            from train.hd_instr import dca_lam0 as _dl0
            with torch.no_grad():
                _l0m, _l0a, _l0b = _dl0([_fetch(g_gpu[b], cast=True) for b in range(len(g_gpu))], [_fetch(x2_gpu[b], cast=True) for b in range(len(x2_gpu))], float(getattr(args, 'dca_rho_t', 0.1)))
            _LAM0 = _l0m
            print('[DCA0] Epoch %s user %s | lam0(sum) med/p10/p90=%.4e/%.4e/%.4e | mult=%g | lam_used(sum)=%.4e | rho_t=%g | B=%d' % (
                cur_epoch, user_idx, _l0m, _l0a, _l0b, float(getattr(args, 'hd_kprobe_lambda_mult', 1.0)), _l0m * float(getattr(args, 'hd_kprobe_lambda_mult', 1.0)), float(getattr(args, 'dca_rho_t', 0.1)), len(g_gpu)), flush=True)
        if mode == 'kprobe' and not getattr(args, '_dca_cfg_logged', False):
            print('[DCA-CFG] loss_reduction=mean (F.cross_entropy; g0=(1/B_b) sum_i grad l_i) | lambda_unit=%s | lambda_mode=%s | g_sum=B_b*g0, D=g_sum^2, client mean-unit correction lambda_mean=lambda_sum*B_b (B_b = actual batch size incl. the last partial batch)' % (
                str(getattr(args, 'hd_kprobe_lambda_unit', 'mean')), str(getattr(args, 'hd_kprobe_lambda_mode', 'abs'))), flush=True)
            args._dca_cfg_logged = True
        mask_gpu = ([m.to(dev) for m in cached_mask_list] if is_llm else None)

        use_cached_inputs = (not is_llm) and (cached_inputs is not None) and (cached_labels is not None)
        if use_cached_inputs:
            cached_inputs_gpu = (cached_inputs if _cpu_cache else [c.to(dev) for c in cached_inputs])
            cached_labels_gpu = [l.to(dev) for l in cached_labels]
        if _cpu_cache:
            print('[HESS-DIAG] round-cache=CPU (per-batch upload) | batches=%d' % len(cached_x2_list),
                  flush=True)

        seed_inputs = []
        if is_hybrid:
            if use_cached_inputs:
                seed_inputs = [_fetch(cached_inputs_gpu[b]) for b in range(len(cached_inputs_gpu))]
            elif is_llm:
                for batch in self.ldr_train:
                    seed_inputs.append((batch['input_ids'].to(dev),
                                        batch['attention_mask'].to(dev)))
            else:
                for batch in self.ldr_train:
                    seed_inputs.append(batch[0].to(dev))

        _arm = ('stale' if bool(getattr(args, 'hd_stale_mode', False))
                else 'upperdiag' if bool(getattr(args, 'hd_upper_mode', False)) else mode)
        _uses_lr = _arm not in ('stale', 'upperdiag', 'kprobe', 'target', 'omega3b', 'omega3a', 'omegaM', 'omegaMp', 'omegaMpG', 'omegaMpJ')
        _m_shown = (int(getattr(args, 'hd_lanczos_iters', 0)) if (_uses_lr and mode == 'lowrank')
                    else (m_basis if (_uses_lr and is_hybrid) else 0))
        _k_shown = int(getattr(args, 'hd_lanczos_k', 0)) if (_uses_lr and mode == 'lowrank') else 0
        _src = getattr(args, 'hd_lowrank_src', '-') if _uses_lr else '-'
        print(f"[HESS-DIAG] arm={_arm} diag={getattr(args,'hd_diag_method','ggn')} "
              f"src={_src} m={_m_shown} k={_k_shown} lowrank={_uses_lr} "
              f"mask_sync={mask_sync_on(args)} | batches={len(x2_gpu)} "
              f"(round-initial 1, 0)", flush=True)
        if mode == 'omegaMp':
            assert cached_omega_list is not None, '[RSLOCMP] cached_omega_list missing; the driver must request want_omega'
            from train.hd_omega import rslocM_task, M_ELL, L_S
            _lt_h, _kb_h = rslocM_task(args)
            print(f"[HESS-DIAG] arm=rsloc_Mp | g̃=g0 + Σ_k w_n·c_k(δ)·(ω_k⊙δ), c_k=min(M_ℓ, m_k + (L_S/2)·sqrt(Σω_kδ²)), w_n=1/({_kb_h}N), M_ℓ={M_ELL[_lt_h]}, L_S={L_S:g}| 0", flush=True)
        if mode == 'omegaMpG':
            assert cached_omega_list is not None, '[RSLOCMPG] cached_omega_list missing; the driver must request want_omega'
            from train.hd_omega import rslocM_task, M_ELL, G_ELL, L_S
            _lt_h, _kb_h = rslocM_task(args)
            print(f"[HESS-DIAG] arm=rsloc_MpG | g̃=g0 + Σ_k w_n·c^G_k(δ)·(ω_k⊙δ), c^G_k=min(M_ℓ, m_k + (L_S/2)·Δ_k, G_ℓ/Δ_k), Δ_k=sqrt(Σω_kδ²), w_n=1/({_kb_h}N), M_ℓ={M_ELL[_lt_h]}, G_ℓ={G_ELL[_lt_h]:.7g}, L_S={L_S:g}| 0", flush=True)
        if mode == 'omegaMpJ':
            assert cached_omega_list is not None, '[RSLOCMPJ] cached_omega_list missing; the driver must request want_omega'
            from train.hd_omega import rslocM_task, M_ELL, G_ELL, L_S
            _lt_h, _kb_h = rslocM_task(args)
            print(f"[HESS-DIAG] arm=rsloc_MpJ | g̃=g0 + Σ_k w_n·c^G_k(Δ̂_k)·s_k(δ), s_k=Σ_c J_c⟨J_c,δ⟩ (E) / (1/k)Σ_j u_j⟨u_j,δ⟩ (P), Δ̂_k=‖J₀δ‖ (E) / sqrt(1/kΣ⟨u_j,δ⟩²) (P), c^G=min(M_ℓ, m+(L_S/2)Δ̂, G_ℓ/Δ̂), w_n=1/({_kb_h}N), M_ℓ={M_ELL[_lt_h]}, G_ℓ={G_ELL[_lt_h]:.7g}, L_S={L_S:g}, k={int(getattr(args,'rslocMpJ_probes',1))}| (E_diag=0 / O(1/k)) 0 rank-1 ", flush=True)
        if mode == 'omegaM':
            assert cached_omega_list is not None, '[RSLOCM] cached_omega_list missing; the driver must request want_omega'
            from train.hd_omega import rslocM_task, M_ELL
            _lt_h, _kb_h = rslocM_task(args)
            print(f"[HESS-DIAG] arm=rsloc_M | g̃=g0 + w_n·M_ℓ·(ω⊙δ), w_n=1/({_kb_h}N), M_ℓ={M_ELL[_lt_h]}, ω=JᵀvJᵀv(/,)| 0", flush=True)
        if mode in ('omega3b', 'omega3a'):
            assert cached_omega_list is not None, '[OMEGA] cached_omega_list missing; the driver must request want_omega'
            _om_lam = getattr(args, '_omega_lam', None)
            _om_scale = str(getattr(args, 'hd_omega_scale', 'taylor'))
            if _om_scale == 'radius':
                if getattr(args, '_omega_kappa_r', None) is None:
                    args._omega_kappa_r = float(getattr(args, 'hd_kappa', 1.0))
                assert cached_cbar_list is not None, '[OMEGA] radius rule: cached_cbar_list missing'
                print(f"[HESS-DIAG] arm={mode} scale=radius kappa_r={args._omega_kappa_r:.4g} R*={float(getattr(args,'hd_radius_target',0.05)):g} gamma={float(getattr(args,'hd_radius_gamma',0.5)):g} "
                      f"|: corr=κ_r c̄_j ω̂^p⊙δ, κ_r←κ_r (R_obs/R*)^γ, R_obs=max_b‖δ_b‖/‖a0‖ (, 3000)", flush=True)
            if _om_scale == 'rspring' and str(getattr(args, 'hd_rstar_mode', 'warmup')) == 'local':
                print(f"[HESS-DIAG] arm={mode} scale=rspring mode=local(rsloc) rloc_steps={int(getattr(args,'hd_rloc_steps',1))}| R_loc=r_2 √B, b=1 2, corr_b=‖g0‖ (r_b/R_loc) û_b ( 0)", flush=True)
            elif _om_scale == 'rspring':
                _rs = getattr(args, '_omega_rstar', None)
                if _rs is None:
                    assert str(getattr(args, 'hd_rstar_mode', 'warmup')) == 'fixed', '[OMEGA] rspring: warm-up R* missing (no warm-up or measurement failed); abort unless --hd_rstar_mode fixed'
                    _rs = float(getattr(args, 'hd_radius_target', 0.05)); args._omega_rstar = _rs
                print(f"[HESS-DIAG] arm={mode} scale=rspring R*={_rs:.4f} ({getattr(args,'hd_rstar_mode','warmup')})| corr_b=‖g0‖ (‖δ_b‖/(R*‖a0‖)) (ω̂^p⊙δ_b)/‖ω̂^p⊙δ_b‖ (, 0)", flush=True)
            if _om_scale == 'steplock':
                print(f"[HESS-DIAG] arm={mode} scale=steplock kappa={float(getattr(args,'hd_kappa',1.0)):g}|: λ_b=κ 3000 ‖g0²δ_b‖/‖ω̂^pδ_b‖ ( = dcasgd, = ω̂;)", flush=True)
            if _om_scale == 'taylor':
                assert cached_cbar_list is not None, '[OMEGA] taylor rule: cached_cbar_list missing'
                if int(getattr(args, 'hd_lambda_warm', 0) or 0) == 1 and not getattr(args, '_om_warn_lw', False):
                    args._om_warn_lw = True; print('[OMEGA] warning: --hd_lambda_warm is ignored with --hd_omega_scale taylor (kappa*cbar applied from the first corrected round)', flush=True)
                print(f"[HESS-DIAG] arm={mode} scale=taylor kappa={float(getattr(args,'hd_kappa',1.0)):g}| g̃=g0+κ c̄ ω̂^p⊙δ, c̄=tr(GGN)/(N d_a) 1 (,)", flush=True)
            elif _om_scale == 'dc' and _om_lam is None and not str(getattr(args, 'hd_delta_ref', '') or '') and int(getattr(args, 'hd_lambda_warm', 0) or 0) != 1:
                raise ValueError('omega mode(dc) requires --hd_delta_ref or --hd_lambda_warm 1')
            print(f"[HESS-DIAG] arm={mode} m={int(getattr(args,'hd_omega_m',1))} p={float(getattr(args,'hd_omega_p',1.0)):g} "
                  f"reduce={getattr(args,'hd_omega_reduce','none')} rule={'kappa=%g' % float(args.hd_omega_kappa) if float(getattr(args,'hd_omega_kappa',0) or 0) > 0 else 'rho=%g' % float(getattr(args,'hd_rho',0.05))} "
                  f"lambda={'not calibrated (lambda=0 this round; calibrated from the last-step delta_B)' if _om_lam is None else '%.6e' % _om_lam} "
                  f"| g̃=g0+λ ω̂^p⊙δ (3,; //EMA)", flush=True)
        if mode == 'target':
            assert getattr(args, 'hd_target_eta', None) is not None, '[TARGET] --hd_target_eta is required'
            assert cached_a_star_list is not None, '[TARGET] cached_a_star_list missing; the driver must request want_target'
            print(f"[HESS-DIAG] arm=target eta_a={float(args.hd_target_eta):g} shape={getattr(args,'hd_target_shape','iso')} "
                  f"| g_train=(a_b−a*)/η_a, a*=a0−η_a g0 ( λ=1/η_a;)", flush=True)
        if mode == 'kprobe':
            print(f"[HESS-DIAG] arm=kprobe form={getattr(args,'hd_kprobe_form','diag')} "
                  f"alpha={float(getattr(args,'hd_kprobe_alpha',1.0)):g} lambda={float(getattr(args,'hd_kprobe_lambda',3000)):g} "
                  f"gate_neg={bool(getattr(args,'hd_kprobe_gate_neg',True))} gnorm_tau={float(getattr(args,'hd_kprobe_gnorm_tau',1e-8)):g} "
                  f"| kappa_cached={cached_kappa_list is not None} (Pearlmutter HVP 1f+2b/,)", flush=True)

        diag = {'cos_corr': [], 'cos_base': [], 'rel_err_corr': [], 'rel_err_base': [],
                'delta_norm': [], 'corr_norm': [], 'total_steps': 0,
                'delta_cos': [], 'cos_corr_gap': [], 'staleness': [], 'corr_gap_ratio': [],
                'mag_ratio': [], 'scale_frac': [], 'mag_err_base': [], 'mag_err_corr': [],
                'residual_ratio': [],
                'g0_norm': [], 'gt_norm': [], 'gtil_norm': [],
                'alpha_signed': [], 'scale_err': [], 'dir_err': [],
                'l1_cos_dhat': [], 'tau': [], 'l2_cos_negg0': [],
                'rho_a': [], 'rho_g': [], 'cos_au': [],
                'kappa_ap': [], 'negfrac_ap': [],
                'rho_h': [], 'cos_ha': [],
                'taylor2_frac': [], 'cos_hd_gap': [], 'dpers': [],
                'corr_par_frac': [], 'cos_corr_g0': [],
                'u1_cos_g0': [], 'th1': [],
                'eff_rank': [], 'bd_step': [],
                'eig_cos_g0': [], 'eig_theta': [], 'eig_share': [],
                'pearl_fd_relerr': [],
                'cos_hess': [], 'rel_err_hess': [], 'hvp_cos': [], 'hvp_relerr': [], 'hvp_magratio': [],
                'tc_cos_25': [], 'tc_mag_25': [], 'tc_cos_50': [], 'tc_mag_50': [],
                'tc_cos_75': [], 'tc_mag_75': []}
        _last_lr_rank = 0
        _first_pay = True
        _prev_delta = None
        _DELTA_ACC = []

        _DEC_ACC = {k: [] for k in ('e_base', 'e_full', 'e_par', 'e_perp', 'e_orc',
                                    'cos_perp', 'mag_perp', 'c_opt', 'stale')}
        _DEC_BIDX = []
        _PAR_ACC = {'cos': [], 'mag': [], 'nb': 0, 'bwd': 0}
        _PAR_BIDX = []
        _pd_cap = int(getattr(self.args, 'hd_param_diag_max_batches', 8))

        for it in range(args.local_ep):
            batch_iter = (range(len(cached_inputs_gpu)) if use_cached_inputs
                          else enumerate(self.ldr_train))
            for it_val in batch_iter:

                if use_cached_inputs:
                    batch_idx = it_val
                    images = _fetch(cached_inputs_gpu[batch_idx]); label = cached_labels_gpu[batch_idx]
                    with masked_rng(args, cur_epoch, user_idx, batch_idx, dev):
                        fx = net_c(images)
                    ext_mask = None
                else:
                    batch_idx, batch = it_val
                    if is_llm:
                        ids = batch['input_ids'].to(dev); bmask = batch['attention_mask'].to(dev)
                        label = batch['labels'].to(dev)
                        with masked_rng(args, cur_epoch, user_idx, batch_idx, dev):
                            fx, ext_mask = net_c(ids, bmask)
                    else:
                        images, label = batch[0].to(dev), batch[1].to(dev)
                        with masked_rng(args, cur_epoch, user_idx, batch_idx, dev):
                            fx = net_c(images)
                        ext_mask = None

                if _L2_ON and batch_idx == 0 and _L2_IN is None:
                    _L2_IN = ((ids, bmask) if is_llm else (images, None))
                net_c.zero_grad(); optimizer_client.zero_grad()
                x2 = _fetch(x2_gpu[batch_idx], cast=True); g0 = _fetch(g_gpu[batch_idx], cast=True)
                lab0 = lab_gpu[batch_idx]; msk0 = mask_gpu[batch_idx] if is_llm else None

                delta = (fx - x2).detach()
                if _CAP_FORM == 'a2' and _CAP_PREV is not None:
                    import train.hd_instr as _Ic
                    _dr_prev = _Ic.drift_ratio(fx.detach(), x2)
                    _CAP_ST['pre'].append(_dr_prev)
                    if _dr_prev > _CAP_RHO:
                        _Ic.restore_client(net_c, optimizer_client, _CAP_PREV); _CAP_LRM *= _CAP_LRMULT; _Ic.set_lr(optimizer_client, float(args.lr) * _CAP_LRM)
                        net_c.zero_grad(); optimizer_client.zero_grad()
                        with masked_rng(args, cur_epoch, user_idx, batch_idx, dev):
                            if is_llm:
                                fx, ext_mask = net_c(ids, bmask)
                            else:
                                fx = net_c(images)
                        delta = (fx - x2).detach()
                        _CAP_ST['capped'].append(1.0); _CAP_ST['bt'].append(1)
                    else:
                        _CAP_ST['capped'].append(0.0); _CAP_ST['bt'].append(0)
                    _CAP_ST['post'].append(_Ic.drift_ratio(fx.detach(), x2))
                    _CAP_PREV = None
                dn = delta.reshape(delta.size(0), -1).norm(dim=1).mean().item()

                with torch.no_grad():
                    _a0n = x2.reshape(x2.size(0), -1).norm(dim=1).mean().item()
                    if _a0n > 0:
                        _DELTA_ACC.append(dn / _a0n)

                if _needs_dhat and (user_idx is not None) and (it == 0):
                    _PREV_A0_STORE[user_idx][batch_idx] = (
                        x2.detach().to('cpu', dtype=torch.float16), lab0.detach().to('cpu'))
                diagH = None
                if bool(getattr(args, 'hd_stale_mode', False)):

                    g_tilde = g0
                elif bool(getattr(args, 'hd_upper_mode', False)):

                    with torch.enable_grad():
                        _fxu = fx.detach().clone().requires_grad_(True)
                        with masked_rng(args, cur_epoch, user_idx, batch_idx, dev):
                            _outu, _ = (net_server_eval(_fxu, ext_mask) if is_llm
                                        else net_server_eval(_fxu))
                        g_tilde = torch.autograd.grad(criterion(_outu, label), _fxu)[0].detach()
                    del _fxu, _outu
                    _CM.up(fx, label, msgs=1); _CM.down(g_tilde, msgs=1)
                elif mode == 'secant':

                    _slot = _prev_a0[batch_idx] if (_prev_a0 is not None and batch_idx < len(_prev_a0)) else None
                    if (_slot is not None) and (
                            tuple(_slot[0].shape) != tuple(x2.shape)
                            or not torch.equal(_slot[1], lab0.detach().cpu())):

                        _dhat_miss += 1
                        _slot = None
                    if _slot is None:
                        g_tilde = g0
                    else:
                        _pa = _slot[0].to(dev, dtype=x2.dtype)
                        if it == int(args.local_ep) - 1:
                            _prev_a0[batch_idx] = None
                        _dhat = (x2 - _pa).detach()
                        _q = server_grad_query(net_server_probe, (x2 + _dhat).detach(), lab0, args, msk0)
                        _D1 = (_q.reshape_as(g0) - g0).detach()
                        _df = delta.reshape(delta.size(0), -1)
                        _hf = _dhat.reshape(_dhat.size(0), -1)
                        _tau = (_df * _hf).sum(dim=1) / (_hf.norm(dim=1).pow(2) + 1e-12)
                        g_tilde = g0 + _tau.view(-1, *([1] * (g0.dim() - 1))) * _D1
                        diag['l1_cos_dhat'].append(F.cosine_similarity(_df, _hf, dim=1).mean().item())
                        diag['tau'].append(_tau.mean().item())
                        del _pa, _dhat, _q, _D1, _df, _hf, _tau
                elif mode == 'midfisher':

                    _k = int(getattr(args, 'hd_lanczos_k', 4))
                    _m = int(getattr(args, 'hd_lanczos_iters', 16))
                    _slot_m = _prev_a0[batch_idx] if (_prev_a0 is not None and batch_idx < len(_prev_a0)) else None
                    if (_slot_m is not None) and (
                            tuple(_slot_m[0].shape) != tuple(x2.shape)
                            or not torch.equal(_slot_m[1], lab0.detach().cpu())):
                        _dhat_miss += 1
                        _slot_m = None
                    if _slot_m is not None:
                        _pa = _slot_m[0].to(dev, dtype=x2.dtype)
                        if it == int(args.local_ep) - 1:
                            _prev_a0[batch_idx] = None
                        _dhat_m = (x2 - _pa).detach()
                        _mid = (x2 + 0.5 * _dhat_m).detach()
                        diag['l1_cos_dhat'].append(F.cosine_similarity(
                            delta.reshape(delta.size(0), -1), _dhat_m.reshape(_dhat_m.size(0), -1), dim=1).mean().item())
                        del _pa, _dhat_m
                    else:
                        _mid = x2
                    _gvp_mid = make_gvp_cached(net_server_probe, _mid, msk0, is_llm, lab0,
                                               args=args, tag='midfisher')
                    try:
                        U_lr, th_lr = lanczos_lowrank(_gvp_mid, x2, g0, _m, _k)
                    finally:
                        _gvp_mid.close()
                    g_tilde = correct_lowrank(delta, U_lr, th_lr, g0,
                                              gate_neg=bool(getattr(args, 'hd_gate_neg', False)),
                                              cap_tau=float(getattr(args, 'hd_corr_cap', 0.0)),
                                              cap_unit=str(getattr(args, 'hd_corr_cap_unit', 'sample')))
                    _last_lr_rank = int(th_lr.shape[-1])
                    del U_lr, th_lr, _mid
                elif bool(getattr(args, 'hd_oracle_hess', False)):
                    if _oracle_1c_skip:
                        g_tilde = g0
                    elif bool(getattr(args, 'hd_oracle_exact', False)):

                        _a1 = x2.detach().clone().requires_grad_(True)
                        _z1 = (net_server_probe(_a1, msk0)[0] if is_llm else net_server_probe(_a1)[0])
                        _g1 = torch.autograd.grad(_TL.gnll(_z1, lab0), _a1, create_graph=True)[0]
                        _Hd = torch.autograd.grad((_g1 * delta.detach()).sum(), _a1)[0].detach()
                        g_tilde = g0 + _Hd.reshape_as(g0)
                        if bool(getattr(args, 'hd_oracle_single_client', False)):

                            _an_o = x2.reshape(x2.size(0), -1).norm()
                            _vn_o = delta.reshape(delta.size(0), -1).norm() + 1e-12
                            _ep_o = eps_rel * float((_an_o / _vn_o).item())
                            _gp_o = server_grad_query(net_server_probe, (x2 + _ep_o * delta).detach(), lab0, args, msk0)
                            _gm_o = server_grad_query(net_server_probe, (x2 - _ep_o * delta).detach(), lab0, args, msk0)
                            _Hd_fd = ((_gp_o - _gm_o) / (2.0 * _ep_o)).reshape_as(g0)
                            _num = (_Hd.reshape_as(g0) - _Hd_fd).reshape(g0.size(0), -1).norm(dim=1)
                            _den = _Hd.reshape(g0.size(0), -1).norm(dim=1) + 1e-12
                            diag['pearl_fd_relerr'].append((_num / _den).mean().item())
                            del _gp_o, _gm_o, _Hd_fd, _num, _den
                        del _a1, _z1, _g1, _Hd
                    else:

                        _an_o = x2.reshape(x2.size(0), -1).norm()
                        _vn_o = delta.reshape(delta.size(0), -1).norm() + 1e-12
                        _ep_o = eps_rel * float((_an_o / _vn_o).item())
                        _gp_o = server_grad_query(net_server_probe, (x2 + _ep_o * delta).detach(), lab0, args, msk0)
                        _gm_o = server_grad_query(net_server_probe, (x2 - _ep_o * delta).detach(), lab0, args, msk0)
                        g_tilde = g0 + ((_gp_o - _gm_o) / (2.0 * _ep_o)).reshape_as(g0)
                elif mode == 'aprobe':

                    _hyb_ap = bool(getattr(args, 'hd_aprobe_hybrid', False))
                    _op_ap = str(getattr(args, 'hd_aprobe_operator', 'hess'))
                    _src_ap = str(getattr(args, 'hd_aprobe_src', 'a0'))
                    _Bs = x2.size(0)
                    if _src_ap == 'hlast':

                        _h0 = _server_last_hidden(net_server_probe, x2, msk0, is_llm).reshape(_Bs, -1)
                        _vp = _h0 / (_h0.norm(dim=1, keepdim=True) + 1e-12)
                        _a0n = x2.detach().reshape(_Bs, -1)
                        _a0n = _a0n / (_a0n.norm(dim=1, keepdim=True) + 1e-12)
                        diag['cos_ha'].append((_vp * _a0n).sum(dim=1).mean().item())
                        if bool(getattr(args, 'hd_hprobe_ortho_a', False)):
                            _pcah = (_vp * _a0n).sum(dim=1, keepdim=True)
                            _vp = _vp - _pcah * _a0n
                            _vp = _vp / (_vp.norm(dim=1, keepdim=True) + 1e-12)
                            del _pcah
                        del _h0, _a0n
                    else:
                        _vp = x2.detach().reshape(_Bs, -1)
                        _vp = _vp / (_vp.norm(dim=1, keepdim=True) + 1e-12)
                    U_ap = th_ap = None
                    if _hyb_ap:
                        U_ap, th_ap = ggn_lowrank(net_server_probe, x2, args, msk0, rank=None)
                        _pc = torch.einsum('bnr,bn->br', U_ap.reshape(_Bs, -1, U_ap.shape[-1]), _vp)
                        _vp = _vp - torch.einsum('bnr,br->bn', U_ap.reshape(_Bs, -1, U_ap.shape[-1]), _pc)
                        _vp = _vp / (_vp.norm(dim=1, keepdim=True) + 1e-12)
                        del _pc
                    _vfull = _vp.reshape_as(x2)
                    if _op_ap == 'ggn':
                        _Hv = ggn_vec_product(net_server_probe, x2, msk0, _vfull, is_llm,
                                              lab0).reshape(_Bs, -1)
                    else:
                        _a1p = x2.detach().clone().requires_grad_(True)
                        _z1p = (net_server_probe(_a1p, msk0)[0] if is_llm else net_server_probe(_a1p)[0])
                        _g1p = torch.autograd.grad(_TL.gnll(_z1p, lab0), _a1p, create_graph=True)[0]
                        _Hv = torch.autograd.grad((_g1p * _vfull.detach()).sum(), _a1p)[0].detach().reshape(_Bs, -1)
                        del _a1p, _z1p, _g1p
                    _cd = (delta.reshape(_Bs, -1) * _vp).sum(dim=1)
                    _kap = (_vp * _Hv).sum(dim=1)
                    diag['kappa_ap'].append(_kap.mean().item())
                    diag['negfrac_ap'].append((_kap < 0).float().mean().item())
                    if _src_ap == 'hlast':

                        _dns = delta.reshape(_Bs, -1).norm(dim=1) + 1e-12
                        diag['rho_h'].append((_cd / _dns).mean().item())
                        del _dns
                    if bool(getattr(args, 'hd_aprobe_gate_neg', True)) and _op_ap != 'ggn':
                        _cd = torch.where(_kap >= 0, _cd, torch.zeros_like(_cd))
                    _corr_ap = (_cd.unsqueeze(1) * _Hv).reshape_as(g0)
                    if _hyb_ap:
                        g_tilde = correct_lowrank(delta, U_ap, th_ap, g0,
                                                  gate_neg=bool(getattr(args, 'hd_gate_neg', False)),
                                                  cap_tau=float(getattr(args, 'hd_corr_cap', 0.0)),
                                                  cap_unit=str(getattr(args, 'hd_corr_cap_unit', 'sample'))) + _corr_ap
                        _last_lr_rank = int(th_ap.shape[-1]) + 1
                        del U_ap, th_ap
                    else:
                        g_tilde = g0 + _corr_ap
                        _last_lr_rank = 1
                    del _vp, _vfull, _Hv, _cd, _kap, _corr_ap
                elif mode == 'target':

                    from train.hd_target import target_signal, target_log_perbatch
                    _ast = _fetch(cached_a_star_list[batch_idx], cast=True)
                    _eta = float(args.hd_target_eta)
                    g_tilde = target_signal(fx, _ast, _eta, g0=g0, a0=x2, shape=str(getattr(args, 'hd_target_shape', 'iso')))
                    target_log_perbatch(diag, fx, x2, _ast, g0, g_tilde, _eta)
                    _last_lr_rank = 1
                    del _ast
                elif mode == 'omegaMpG':

                    from train.hd_omega import rslocMp_unpack, rslocMpG_correction, rslocM_weight, rslocM_task, rslocMpG_line, M_ELL, G_ELL
                    _pk = _fetch(cached_omega_list[batch_idx], cast=True)
                    _ob, _mb = rslocMp_unpack(_pk, delta.shape)
                    _lt_c, _ = rslocM_task(args)
                    _corr, _mst = rslocMpG_correction(delta, _ob, _mb, rslocM_weight(args, int(delta.size(0))), M_ELL[_lt_c], G_ELL[_lt_c])
                    g_tilde = g0 + _corr
                    with torch.no_grad():
                        diag.setdefault('om_cnr', []).append(float(_corr.reshape(g0.size(0), -1).norm() / (g0.reshape(g0.size(0), -1).norm() + 1e-30)))
                        diag.setdefault('mp_frac_cap', []).append(_mst['frac_cap']); diag.setdefault('mp_c_med', []).append(_mst['c_med'])
                        diag.setdefault('mp_frac_G', []).append(_mst['frac_G']); diag.setdefault('mp_cD_med', []).append(_mst['cD_med'])
                        _nb_all = len(cached_x2_list)
                        if batch_idx == 0 or batch_idx == _nb_all - 1:
                            print(rslocMpG_line(cur_epoch, user_idx, batch_idx + 1, _mst), flush=True)
                    _last_lr_rank = 1
                    del _pk, _ob, _mb, _corr
                elif mode == 'omegaMpJ':

                    from train.hd_omega import rslocMpJ_unpack, rslocMpJ_correction, rslocM_weight, rslocM_task, rslocMpJ_line, M_ELL, G_ELL, rslocMpJ_unpack_lm, rslocMpJ_unpack_lm1
                    _pk = _fetch(cached_omega_list[batch_idx], cast=True)
                    _lt_c, _kb_c = rslocM_task(args)
                    _jmode = str(getattr(args, '_rslocMpJ_mode', ''))
                    assert _jmode in ('E', 'P', 'L', 'S'), '[RSLOCMPJ] server mode not recorded'
                    _guse = G_ELL[_lt_c]
                    if _jmode == 'S':
                        _Jb, _mb, _wb, _Gb = rslocMpJ_unpack_lm1(_pk, delta.shape); _wuse = _wb; _guse = _Gb
                    elif _jmode == 'L':
                        _Jb, _mb, _wb = rslocMpJ_unpack_lm(_pk, delta.shape); _wuse = _wb
                    else:
                        _Jb, _mb = rslocMpJ_unpack(_pk, delta.shape); _wuse = rslocM_weight(args, int(delta.size(0)))
                    _nb_all = len(cached_x2_list); _do_log = (batch_idx == 0 or batch_idx == _nb_all - 1)
                    _corr, _mst = rslocMpJ_correction(delta, _Jb, _mb, _wuse, M_ELL[_lt_c], _guse, _jmode, with_diag=_do_log, coef=str(getattr(args, 'rslocMpJ_coef', 'cG')))
                    g_tilde = g0 + _corr
                    _sca = getattr(args, '_stepcond_acc', None)
                    if _sca is not None and batch_idx < len(_sca):
                        _stepcond_collect(diag, args, _sca[batch_idx][0], _sca[batch_idx][1])
                    with torch.no_grad():
                        diag.setdefault('om_cnr', []).append(float(_corr.reshape(g0.size(0), -1).norm() / (g0.reshape(g0.size(0), -1).norm() + 1e-30)))
                        diag.setdefault('mp_frac_cap', []).append(_mst['frac_cap']); diag.setdefault('mp_c_med', []).append(_mst['c_med'])
                        diag.setdefault('mp_frac_G', []).append(_mst['frac_G']); diag.setdefault('mp_cD_med', []).append(_mst['cD_med'])
                        _sh = float(_corr.reshape(-1).norm() / (g_tilde.reshape(-1).norm() + 1e-30)); diag.setdefault('mpj_share', []).append(_sh)
                        if _do_log:
                            diag.setdefault('mpj_ratio_diag', []).append(_mst['ratio_diag'])
                            print(rslocMpJ_line(cur_epoch, user_idx, batch_idx + 1, _jmode, int(_Jb.size(2)), _mst, _sh), flush=True)
                    _last_lr_rank = 1
                    del _pk, _Jb, _mb, _corr
                elif mode == 'omegaMp':

                    from train.hd_omega import rslocMp_unpack, rslocMp_correction, rslocM_weight, rslocM_task, rslocMp_line, M_ELL
                    _pk = _fetch(cached_omega_list[batch_idx], cast=True)
                    _ob, _mb = rslocMp_unpack(_pk, delta.shape)
                    _lt_c, _ = rslocM_task(args)
                    _corr, _mst = rslocMp_correction(delta, _ob, _mb, rslocM_weight(args, int(delta.size(0))), M_ELL[_lt_c])
                    g_tilde = g0 + _corr
                    with torch.no_grad():
                        diag.setdefault('om_cnr', []).append(float(_corr.reshape(g0.size(0), -1).norm() / (g0.reshape(g0.size(0), -1).norm() + 1e-30)))
                        diag.setdefault('mp_frac_cap', []).append(_mst['frac_cap']); diag.setdefault('mp_c_med', []).append(_mst['c_med'])
                        _nb_all = len(cached_x2_list)
                        if batch_idx == 0 or batch_idx == _nb_all - 1:
                            print(rslocMp_line(cur_epoch, user_idx, batch_idx + 1, _mst), flush=True)
                    _last_lr_rank = 1
                    del _pk, _ob, _mb, _corr
                elif mode == 'omegaM':

                    from train.hd_omega import rslocM_correction, rslocM_weight, rslocM_task, rslocM_line, M_ELL
                    _oh = _fetch(cached_omega_list[batch_idx], cast=True)
                    _lt_c, _ = rslocM_task(args)
                    _corr = rslocM_correction(delta, _oh, rslocM_weight(args, int(delta.size(0))), M_ELL[_lt_c])
                    g_tilde = g0 + _corr
                    with torch.no_grad():
                        diag.setdefault('om_cnr', []).append(float(_corr.reshape(g0.size(0), -1).norm() / (g0.reshape(g0.size(0), -1).norm() + 1e-30)))
                        _nb_all = len(cached_x2_list)
                        if batch_idx == 0 or batch_idx == _nb_all - 1:
                            print(rslocM_line(cur_epoch, user_idx, batch_idx + 1, _oh, _corr, g_tilde), flush=True)
                    _last_lr_rank = 1
                    del _oh, _corr
                elif mode in ('omega3b', 'omega3a'):

                    from train.hd_omega import omega_correction, calibrate_lambda, load_delta_ref, omega_stats, omega_line
                    _oh = _fetch(cached_omega_list[batch_idx], cast=True)
                    _p_om = float(getattr(args, 'hd_omega_p', 1.0)); _rho = float(getattr(args, 'hd_rho', 0.05))
                    _lam = getattr(args, '_omega_lam', None)
                    if _lam is None and str(getattr(args, 'hd_delta_ref', '') or ''):
                        _dref = load_delta_ref(str(args.hd_delta_ref), f'u{user_idx}_b{batch_idx}').to(delta.device)
                        _lam = calibrate_lambda(g0, _oh, _dref, _p_om, _rho); args._omega_lam = _lam
                        print(f'[OMEGA] lambda={_lam:.6e} rho={_rho:.3f} (delta_ref={args.hd_delta_ref}, r={cur_epoch} user={user_idx} b={batch_idx})', flush=True)
                    _om_scale = str(getattr(args, 'hd_omega_scale', 'taylor'))
                    if _om_scale == 'taylor':
                        from train.hd_omega import omega_correction_taylor
                        _cb_j = float(cached_cbar_list[batch_idx]); _kap_t = float(getattr(args, 'hd_kappa', 1.0))
                        _lam_eff = _kap_t * _cb_j
                        _corr = omega_correction_taylor(delta, _oh, _kap_t, _cb_j, _p_om)
                        diag.setdefault('om_cbar', []).append(_cb_j)

                        if batch_idx == len(cached_x2_list) - 1 and not getattr(args, '_om_kdc_done', False):
                            args._om_kdc_done = True
                            from train.hd_omega import calibrate_lambda_kappa
                            _ldc, _cq = calibrate_lambda_kappa(g0, _oh, delta, _p_om, 1.0)
                            print(f"[OMEGA] calib scale=taylor kappa={_kap_t:g} cbar={_cb_j:.6e} lam_dc_equiv={_ldc:.6e} kappa_dc={_ldc/(_cb_j+1e-30):.4f} "
                                  f"|g0|={_cq['g0']:.4e} |delta_B|={_cq['delta']:.4e} |omega^p*delta_B|={_cq['om_delta']:.4e} |g0^2*delta_B|={_cq['g2_delta']:.4e} | r={cur_epoch} user={user_idx} b={batch_idx+1}", flush=True)
                    elif _om_scale == 'radius':
                        from train.hd_omega import omega_correction_taylor
                        _cb_j = float(cached_cbar_list[batch_idx]); _kap_r = float(args._omega_kappa_r)
                        _lam_eff = _kap_r * _cb_j
                        _corr = omega_correction_taylor(delta, _oh, _kap_r, _cb_j, _p_om)
                        diag.setdefault('om_cbar', []).append(_cb_j)
                        with torch.no_grad():
                            _rr = float(delta.reshape(g0.size(0), -1).norm(dim=1).mean() / (x2.reshape(g0.size(0), -1).norm(dim=1).mean() + 1e-30))
                        diag.setdefault('om_rel_r', []).append(_rr)
                    elif _om_scale == 'rspring' and str(getattr(args, 'hd_rstar_mode', 'warmup')) == 'local':
                        import math as _math
                        from train.hd_omega import omega_correction_rspring
                        _Bn = len(cached_x2_list) * int(getattr(args, 'local_ep', 1))
                        with torch.no_grad():
                            _rb = float(delta.reshape(g0.size(0), -1).norm(dim=1).mean() / (x2.reshape(g0.size(0), -1).norm(dim=1).mean() + 1e-30))
                        _ks = int(getattr(args, 'hd_rloc_steps', 1))

                        _gstep = it * len(cached_x2_list) + batch_idx
                        if _gstep == 0:
                            diag['_rloc'] = None; diag['_rloc_acc'] = []
                            _corr = torch.zeros_like(delta); _lam_eff = 0.0
                        elif _gstep <= _ks:
                            if _rb <= 0.0:
                                raise SystemExit(f'[OMEGA] rsloc: r_{_gstep+1}=0, (r={cur_epoch} user={user_idx})')
                            diag['_rloc_acc'].append(_rb / _math.sqrt(_gstep))
                            if _gstep == _ks:
                                diag['_rloc'] = float(sum(diag['_rloc_acc']) / len(diag['_rloc_acc'])) * _math.sqrt(_Bn)
                                diag['_r2'] = float(diag['_rloc_acc'][0]); diag['_B'] = _Bn
                                print(f'[OMEGA] rsloc B={_Bn} r2={diag["_r2"]:.5f} R_loc={diag["_rloc"]:.5f} (r={cur_epoch} user={user_idx})', flush=True)
                            _corr = torch.zeros_like(delta); _lam_eff = 0.0
                        else:
                            _corr, _lam_eff, _ = omega_correction_rspring(delta, _oh, g0, x2, float(diag['_rloc']), _p_om)
                            with torch.no_grad():
                                _wdn = float((_oh.reshape(_oh.size(0), -1).pow(_p_om) * delta.reshape(g0.size(0), -1)).norm())
                                if _wdn == 0.0:
                                    print(f'[OMEGA] rsloc warn zero_wd (r={cur_epoch} user={user_idx} b={_gstep+1}) → corr=0', flush=True); _corr = torch.zeros_like(delta); _lam_eff = 0.0
                                _cnr_chk = float(_corr.reshape(g0.size(0), -1).norm() / (g0.reshape(g0.size(0), -1).norm() + 1e-30))
                                if _wdn > 0.0 and abs(_cnr_chk - _rb / diag['_rloc']) > 1e-6:
                                    print(f'[OMEGA] rsloc IDENT-FAIL cnr={_cnr_chk:.8f} r_b/R_loc={_rb/diag["_rloc"]:.8f} (r={cur_epoch} user={user_idx} b={_gstep+1})', flush=True)
                                diag['om_ident_n'] = diag.get('om_ident_n', 0) + 1

                                if _wdn > 0.0:
                                    from train.hd_omega import rsloc_audit as _ra
                                    _au = _ra(_corr, delta, _oh, g0, x2, float(diag['_rloc']), _p_om,
                                              r2=diag.get('_r2'), B=diag.get('_B'))
                                    diag['om_audit_n'] = diag.get('om_audit_n', 0) + 1
                                    if not _au['ok']:
                                        diag['om_audit_fail'] = diag.get('om_audit_fail', 0) + 1
                                        print('[OMEGA] rsloc AUDIT-FAIL cos_dir=%.10f mag_rel=%.3e rloc_rel=%.3e (r=%s user=%s b=%d)'
                                              % (_au['cos_dir'], _au['mag_rel'], _au.get('rloc_rel', float('nan')), cur_epoch, user_idx, _gstep + 1), flush=True)
                                    diag['om_audit_cos'] = min(diag.get('om_audit_cos', 1.0), float(_au['cos_dir']))
                                    diag['om_audit_mag'] = max(diag.get('om_audit_mag', 0.0), float(_au['mag_rel']))
                        diag.setdefault('om_lam_b', []).append(float(_lam_eff)); diag.setdefault('om_rel_r', []).append(float(_rb))
                    elif _om_scale == 'rspring':
                        from train.hd_omega import omega_correction_rspring
                        _corr, _lam_eff, _rb = omega_correction_rspring(delta, _oh, g0, x2, float(args._omega_rstar), _p_om)
                        diag.setdefault('om_lam_b', []).append(float(_lam_eff)); diag.setdefault('om_rel_r', []).append(float(_rb))
                    elif _om_scale == 'steplock':
                        from train.hd_omega import omega_correction_steplock
                        _corr, _lam_eff = omega_correction_steplock(delta, _oh, g0, float(getattr(args, 'hd_kappa', 1.0)), _p_om)
                        diag.setdefault('om_lam_b', []).append(float(_lam_eff))
                    else:
                        _lam_eff = 0.0 if _lam is None else float(_lam)
                        _corr = omega_correction(delta, _oh, _lam_eff, _p_om)
                    g_tilde = g0 + _corr
                    with torch.no_grad():
                        diag.setdefault('om_cnr', []).append(float(_corr.reshape(g0.size(0), -1).norm() / (g0.reshape(g0.size(0), -1).norm() + 1e-30)))
                    _nb_all = len(cached_x2_list)
                    if batch_idx == 0 or batch_idx == _nb_all - 1:
                        _alt = (_fetch(cached_omega_alt_list[batch_idx], cast=True) if cached_omega_alt_list is not None else None)
                        _st = omega_stats(_oh, g0, delta, _corr, omega_alt=_alt)
                        print(omega_line(cur_epoch, batch_idx + 1, _lam_eff, _st), flush=True)
                        for _k, _v in _st.items():
                            diag.setdefault('om_' + _k + ('_first' if batch_idx == 0 else '_last'), []).append(float(_v))
                    if _om_scale != 'taylor' and _lam is None and batch_idx == _nb_all - 1 and int(getattr(args, 'hd_lambda_warm', 0) or 0) == 1:
                        _kap = float(getattr(args, 'hd_omega_kappa', 0.0) or 0.0)
                        if _kap > 0:
                            from train.hd_omega import calibrate_lambda_kappa
                            _lam, _cq = calibrate_lambda_kappa(g0, _oh, delta, _p_om, _kap); args._omega_lam = _lam
                            _chk = _lam * _cq['om_delta'] / (3000.0 * _cq['g2_delta'] + 1e-30)
                            print(f"[OMEGA] calib rule=kappa kappa={_kap:g} lambda={_lam:.6e} |g0|={_cq['g0']:.4e} |delta_B|={_cq['delta']:.4e} "
                                  f"|omega^p*delta_B|={_cq['om_delta']:.4e} |g0^2*delta_B|={_cq['g2_delta']:.4e} | G2 ratio(|λω̂^pδ|/|3000g0²δ|)={_chk:.4f} (=κ±1%: {'PASS' if abs(_chk/_kap-1)<0.01 else 'FAIL'}) "
                                  f"| r={cur_epoch} user={user_idx} b={batch_idx+1}", flush=True)
                        else:
                            _lam = calibrate_lambda(g0, _oh, delta, _p_om, _rho); args._omega_lam = _lam
                            print(f'[OMEGA] calib rule=rho rho={_rho:.3f} lambda={_lam:.6e} |g0|={float(g0.reshape(g0.size(0),-1).norm()):.4e} |delta_B|={float(delta.reshape(g0.size(0),-1).norm()):.4e} '
                                  f'|omega^p*delta_B|={float((_oh.reshape(_oh.size(0),-1).pow(_p_om)*delta.reshape(g0.size(0),-1)).norm()):.4e} '
                                  f'|g0^2*delta_B|={float((g0.reshape(g0.size(0),-1)**2*delta.reshape(g0.size(0),-1)).norm()):.4e} | r={cur_epoch} user={user_idx} b={batch_idx+1} (lambda_warm,)', flush=True)
                    _last_lr_rank = 1
                    del _oh, _corr
                elif mode == 'damp':

                    _mu_s = str(getattr(args, 'hd_damp_mu', '1e-3'))
                    if _mu_s == 'thetaK':
                        assert cached_th_list is not None and batch_idx < len(cached_th_list), '[damp] theta_K requires the lowrank cache (want_lowrank)'
                        _th_d = _fetch(cached_th_list[batch_idx], cast=True).reshape(g0.size(0), -1).float()
                        _live = _th_d.abs() > 1e-8 * _th_d.abs().max(dim=1, keepdim=True).values.clamp_min(1e-30)
                        _mu = torch.where(_live, _th_d, torch.full_like(_th_d, float('inf'))).min(dim=1).values
                        _mu = torch.where(torch.isfinite(_mu), _mu, torch.zeros_like(_mu)).clamp_min(0.0)
                    else:
                        _mu = torch.full((g0.size(0),), float(_mu_s), device=g0.device, dtype=g0.dtype)
                    g_tilde = g0 + _mu.to(g0.dtype).view(-1, *([1] * (g0.dim() - 1))) * delta
                    diag.setdefault('damp_mu', []).append(float(_mu.float().median()))
                    _last_lr_rank = 1
                elif mode == 'kprobe':

                    from train.hd_kprobe import correct_kprobe, kprobe_log_perbatch, round_perm
                    assert cached_kappa_list is not None and batch_idx < len(cached_kappa_list),\
                        '[KPROBE] cached_kappa_list missing; must run through the driver (splitfed_kprobe.py)'
                    _kp2 = _fetch(cached_kappa_list[batch_idx], cast=True)
                    _kp, _kpraw = _kp2[0], _kp2[1]
                    _kform = str(getattr(args, 'hd_kprobe_form', 'diag'))
                    _perm = None
                    if _kform == 'const_perm':
                        _pc = getattr(args, '_dca_perm', None)
                        if _pc is None or _pc[0] != int(cur_epoch) or _pc[1].numel() != g0[0].numel():
                            _pc = (int(cur_epoch), round_perm(g0[0].numel(), int(cur_epoch), int(getattr(args, 'seed', 0)), g0.device)); args._dca_perm = _pc
                        _perm = _pc[1]

                    _lmode = str(getattr(args, 'hd_kprobe_lambda_mode', 'abs')); _lunit = str(getattr(args, 'hd_kprobe_lambda_unit', 'mean'))
                    if _lmode == 'lam0':
                        assert _LAM0 is not None, 'lam0 mode but lambda0 was not computed'
                        _lbase = float(_LAM0) * float(getattr(args, 'hd_kprobe_lambda_mult', 1.0))
                    else:
                        _lbase = float(getattr(args, 'hd_kprobe_lambda', 3000))
                    _lam_mean = (_lbase * float(g0.size(0))) if _lunit == 'sum' else _lbase
                    _lam_sum = (_lbase if _lunit == 'sum' else _lbase / float(g0.size(0)))
                    _r0l = getattr(args, '_dca_r0', None)
                    _r0b = (_r0l[batch_idx] if (_r0l is not None and batch_idx < len(_r0l)) else None)
                    g_tilde = correct_kprobe(delta, g0, _kp,
                                             form=_kform,
                                             alpha=float(getattr(args, 'hd_kprobe_alpha', 1.0)),
                                             lam_const=_lam_mean,
                                             lam0=float(getattr(args, 'hd_kprobe_lambda0', 0.0)),
                                             a_eps=float(getattr(args, 'hd_kprobe_a_eps', 1e-7)), perm=_perm, r0n=_r0b)
                    if _kform.startswith('const'):
                        with torch.no_grad():
                            _g2 = (g0.reshape(-1).float() ** 2); _lam_c = _lam_mean
                            diag.setdefault('dca_lam_eff', []).append(_lam_c * float(_g2.mean()))
                            diag.setdefault('dca_lam_mean', []).append(_lam_c); diag.setdefault('dca_lam_sum', []).append(_lam_sum)
                            diag.setdefault('dca_g2_skew', []).append(float(torch.quantile(_g2, 0.9) / _g2.median().clamp_min(1e-30)))
                            if _r0b is not None:
                                diag.setdefault('dca_r0', []).append(float(_r0b.float().mean()))
                            from train.hd_instr import dca_theory_batch as _dtb
                            _rv = getattr(args, '_dca_r0vec', None); _pv = getattr(args, '_dca_p', None)
                            _rvb = (_rv[batch_idx] if (_rv is not None and batch_idx < len(_rv)) else None)
                            _pvb = (_pv[batch_idx] if (_pv is not None and batch_idx < len(_pv)) else None)
                            _th = _dtb(g0, delta, x2, _lam_sum, r0vec=_rvb, p=_pvb, do_eig=bool(_INSTR_ON))
                            for _kk, _vv in _th.items():
                                diag.setdefault('dcaT_' + _kk, []).append(_vv.detach().float().cpu())
                    kprobe_log_perbatch(diag, _kp, _kpraw, g0, delta, g_tilde - g0, x2,
                                        float(getattr(args, 'hd_kprobe_gnorm_tau', 1e-8)))
                    _last_lr_rank = 1
                    del _kp2, _kp, _kpraw
                elif mode == 'lowrank':

                    _src = str(getattr(args, 'hd_lowrank_src', 'ggn'))
                    _k = int(getattr(args, 'hd_lanczos_k', 8))
                    _m = int(getattr(args, 'hd_lanczos_iters', 20))
                    if _src == 'hvp':
                        if bool(getattr(args, 'hd_hvp_exact', True)):

                            _hvp = make_hvp_cached(net_server_probe, x2, lab0, msk0, is_llm, args, tag='hvp')
                        else:

                            _an = x2.reshape(x2.size(0), -1).norm()
                            def _hvp(v, _x2=x2, _lab0=lab0, _msk0=msk0, _an=_an):
                                vn = v.reshape(v.size(0), -1).norm()
                                ep = eps_rel * float((_an / (vn + 1e-12)).item())
                                gp = server_grad_query(net_server_probe, (_x2 + ep * v).detach(), _lab0, args, _msk0)
                                gm = server_grad_query(net_server_probe, (_x2 - ep * v).detach(), _lab0, args, _msk0)
                                return ((gp - gm) / (2.0 * ep)).detach()
                        try:
                            U_lr, th_lr, _bd = lanczos_lowrank(_hvp, x2, g0, _m, _k, return_bd=True)
                        finally:
                            getattr(_hvp, 'close', lambda: None)()
                    elif _src == 'fisher':

                        if int(getattr(args, 'num_classes', 999)) <= int(getattr(args, 'hd_ggn_exact_maxc', 0)):
                            U_lr, th_lr = ggn_lowrank(net_server_probe, x2, args, msk0, rank=(_k or None))
                            _bd = -1
                        else:

                            if cached_U_list is not None and batch_idx < len(cached_U_list):
                                U_lr  = _fetch(cached_U_list[batch_idx], cast=True)
                                th_lr = _fetch(cached_th_list[batch_idx], cast=True)
                                _bd   = -2
                                if _PAY_LOG and _first_pay:
                                    print('[PAY-use] r=%s cli=%s b=%d/%d n=%d K=%d th=[%s]'
                                          % (cur_epoch, user_idx, batch_idx, len(cached_U_list),
                                             int(U_lr.shape[1]), int(U_lr.shape[2]),
                                             ' '.join('%.3e' % float(v) for v in th_lr[0][:4])), flush=True)
                                    _first_pay = False
                            else:
                                _gvp = make_gvp_cached(net_server_probe, x2, msk0, is_llm, lab0,
                                                       args=args, tag='fisher')
                                try:
                                    U_lr, th_lr, _bd = lanczos_lowrank(_gvp, x2, g0, _m, _k, return_bd=True)
                                finally:
                                    _gvp.close()
                    elif _src == 'subhvp':

                        assert cached_U_list is not None and batch_idx < len(cached_U_list), '[subhvp] round-initial (U, M) cache missing; the driver must request want_lowrank'
                        U_lr  = _fetch(cached_U_list[batch_idx], cast=True)
                        th_lr = _fetch(cached_th_list[batch_idx], cast=True)
                        _bd   = -2
                    else:
                        U_lr, th_lr = ggn_lowrank(
                            net_server_probe, x2, args, msk0,
                            rank=(_k or None))
                        _bd = -1
                    g_tilde = correct_lowrank(delta, U_lr, th_lr, g0,
                                              gate_neg=bool(getattr(args, 'hd_gate_neg', False)),
                                              cap_tau=float(getattr(args, 'hd_corr_cap', 0.0)),
                                              cap_unit=str(getattr(args, 'hd_corr_cap_unit', 'sample')))
                    _U_eo = (U_lr.detach() if (_INSTR_ON and batch_idx in _INSTR_BIDX) else None)
                    _last_lr_rank = int(th_lr.shape[-1])

                    _gbu_l = g0.reshape(g0.size(0), -1)
                    _gbu_l = _gbu_l / (_gbu_l.norm(dim=1, keepdim=True) + 1e-12)
                    _u1_l = U_lr.reshape(g0.size(0), -1, U_lr.shape[-1])[..., 0]
                    diag['u1_cos_g0'].append((_u1_l * _gbu_l).sum(dim=1).abs().mean().item())
                    diag['th1'].append(th_lr.reshape(g0.size(0), -1)[:, 0].mean().item())

                    _th_r = th_lr.reshape(g0.size(0), -1).abs()
                    _th_ref = _th_r[:, :1].clamp(min=1e-30)
                    diag['eff_rank'].append(float((_th_r > 1e-8 * _th_ref).sum(dim=1).float().mean()))
                    diag['bd_step'].append(float(_bd))
                    del _th_r, _th_ref

                    _meas0_e = (cur_epoch is None) or ((int(cur_epoch) + 1) % 10 == 0)
                    if _meas0_e:
                        _Uf_e = U_lr.reshape(g0.size(0), -1, U_lr.shape[-1])
                        _th_e = th_lr.reshape(g0.size(0), -1)
                        _df_e = delta.reshape(g0.size(0), -1)
                        _cos_e = torch.einsum('bnk,bn->bk', _Uf_e, _gbu_l).abs().mean(0)
                        _ci_e = (_th_e * torch.einsum('bnk,bn->bk', _Uf_e, _df_e)).abs().mean(0)
                        _sh_e = _ci_e / (_ci_e.sum() + 1e-12)
                        _kt_e = max(int(_k), int(_cos_e.numel()))
                        def _padk_e(t):
                            _v = [float(x) for x in t.tolist()]
                            return _v + [float('nan')] * (_kt_e - len(_v))
                        diag['eig_cos_g0'].append(_padk_e(_cos_e))
                        diag['eig_theta'].append(_padk_e(_th_e.mean(0)))
                        diag['eig_share'].append(_padk_e(_sh_e))
                        del _Uf_e, _th_e, _df_e, _cos_e, _ci_e, _sh_e
                    del _gbu_l, _u1_l
                    del U_lr, th_lr
                elif is_hybrid:
                    diagH = compute_diag(net_server_probe, x2, lab0, args, msk0,
                                         rng_ctx=((lambda: masked_rng(args, cur_epoch, user_idx, batch_idx, args.device)) if mask_sync_on(args) else None))
                    if it == 0:
                        _CM.down(diagH, msgs=0)
                    seed_b = client_ntk_seed(net_c_init, seed_inputs[batch_idx], g0, args, is_llm)
                    U, HU = build_krylov_basis(net_server_probe, x2, g0, lab0, args, msk0,
                                               m_basis, eps_rel, seed=seed_b)
                    g_tilde = correct_hybrid(delta, U, HU, diagH, g0)
                    del U, HU
                else:
                    _sc = {}; _hs = {}
                    diagH = compute_diag(net_server_probe, x2, lab0, args, msk0,
                                         rng_ctx=((lambda: masked_rng(args, cur_epoch, user_idx, batch_idx, args.device)) if mask_sync_on(args) else None),
                                         stepcond=_sc, stats=_hs)
                    if it == 0:
                        _CM.down(diagH, msgs=0)
                    if bool(getattr(args, 'hd_diag_gate_neg', False)):
                        diagH = diagH.clamp_min(0.0)
                    _dsc = str(getattr(args, 'hd_diag_scale', '1'))
                    if _dsc.startswith('trace:') or _dsc.startswith('trace_sum:'):
                        _lam_t = float(_dsc.split(':', 1)[1]) * (float(g0.size(0)) if _dsc.startswith('trace_sum:') else 1.0)
                        _g2s = (g0.reshape(g0.size(0), -1).float() ** 2).sum(1)
                        _ds = diagH.reshape(diagH.size(0), -1).float().sum(1).clamp_min(1e-30)
                        _kap = (_lam_t * _g2s / _ds).view(-1, *([1] * (diagH.dim() - 1)))
                        diagH = diagH * _kap.to(diagH.dtype)
                        diag.setdefault('dg_kappa', []).append(float(_kap.median()))
                    elif _dsc != '1':
                        diagH = diagH * float(_dsc)
                    g_tilde = correct_diag_only(delta, diagH, g0)
                    if _sc.get('JF2') is not None:
                        _stepcond_collect(diag, args, _sc['JF2'], _sc['r0'])
                    if _hs.get('probes'):
                        from train.hd_instr import hvar_stats as _hvs
                        _hv = _hvs(torch.stack(_hs['probes'], 0), delta)
                        diag.setdefault('hv_snr', []).append(_hv['snr_med']); diag.setdefault('hv_noise', []).append(_hv['frac_noise']); diag.setdefault('hv_rho', []).append(_hv['rho_absdiag_med'])
                        del _hs

                diag['delta_norm'].append(dn); diag['total_steps'] += 1

                if _ACT_ON and (cur_epoch is None or (int(cur_epoch) + 1) % _ACT_EVERY == 0):

                    diag.setdefault('_act_d', []).append(
                        delta.detach().mean(dim=0).float().cpu())
                diag['corr_norm'].append(
                    (g_tilde - g0).reshape(g0.size(0), -1).norm(dim=1).mean().item())

                with torch.no_grad():
                    _cf = (g_tilde - g0).reshape(g0.size(0), -1).float(); _df_r = delta.reshape(g0.size(0), -1).float()
                    _dn2 = (_df_r * _df_r).sum(1); _live = _dn2 > 0
                    if bool(_live.any()):
                        _rho = ((_cf * _df_r).sum(1)[_live] / _dn2[_live])
                        diag.setdefault('restore_rho_med', []).append(float(_rho.median())); diag.setdefault('restore_rho_min', []).append(float(_rho.min()))
                        diag.setdefault('restore_neg', []).append(float((_rho < 0).float().mean()))

                        _g0f_r = g0.reshape(g0.size(0), -1).float()
                        diag.setdefault('rho_cr', []).append(float((_cf.norm(dim=1) / _g0f_r.norm(dim=1).clamp_min(1e-30)).median()))
                        diag.setdefault('rho_drift', []).append(float(_df_r.norm() / x2.reshape(-1).float().norm().clamp_min(1e-30)))
                if _INSTR_DP and (_DP_NB <= 0 or batch_idx in _DP_SET):
                    _DP_D.append(delta.detach().float().cpu()); _DP_G.append(g0.detach().float().cpu())
                if _INSTR_ON and (batch_idx in _INSTR_BIDX) and os.environ.get('INSTR_NO_BLOCK', '0') != '1':
                  import train.hd_instr as _I
                  from train.hd_omega import compute_J_blocks as _cJ
                  _rng_i = _rng_snapshot()
                  try:
                    with torch.enable_grad():
                        with masked_rng(args, cur_epoch, user_idx, batch_idx, dev):
                            if bool(getattr(args, 'instr_q0', True)):
                                _I.q0_check(getattr(args, '_q0_ref', {}).get(batch_idx), _I.q0_capture(net_server_probe, dev), 'INSTR r%s u%s b%d' % (cur_epoch, user_idx, batch_idx))
                            _hvp_i, _lg_i, _gs_i, _sl_i = _I.make_hvp_sum(net_server_probe, x2, lab0, msk0, is_llm, criterion)
                        _C_i = int(_lg_i.size(1)); _J_i = None
                        if _C_i <= int(getattr(args, 'instr_max_c', 256)) and _lg_i.size(0) == x2.size(0):
                            _J_i = _cJ(_lg_i, _sl_i, int(x2.size(0)), 'exact')[0][0].reshape(x2.size(0), _C_i, -1).detach()
                        _r0d = _I.r0_measure(_hvp_i, _J_i, _lg_i, delta.detach(), m=int(getattr(args, 'instr_lanczos_m', 20)))
                        print(_I.r0_line(cur_epoch, user_idx, batch_idx + 1, _r0d), flush=True)
                        if _C_i <= int(getattr(args, 'instr_max_c', 256)) and _lg_i.size(0) == x2.size(0):
                            _lam_i, _kind_i = _I.s0_lam_max(_lg_i)
                            print(_I.s0_line(cur_epoch, user_idx, batch_idx + 1, _lam_i, _kind_i), flush=True)
                        if _J_i is not None:
                            with torch.no_grad():
                                _fr_i, _fd_i = _I.row_projection_fracs(_J_i, (g_tilde - g0).detach(), delta.detach())
                                _eo_i = (_I.e_out(_U_eo.reshape(_U_eo.size(0), -1, _U_eo.size(-1)).float(), delta.detach()) if _U_eo is not None else None)
                                print(_I.range_line(cur_epoch, user_idx, batch_idx + 1, _fr_i, _fd_i, _C_i, int(_J_i.size(-1)), _eo_i), flush=True)
                                if mode == 'kprobe':
                                    _dj = _I.dca_theory_J(g0, _J_i, _lg_i, lab0)
                                    _et = _I._q3(_dj['ef_tr']); _ab = _dj['anchor_bound']; _gr = _dj['g_rel']
                                    print('[DCAJ] Epoch %s user %s b=%d | ef_over_ggn_tr med/p10/p90=%.4f/%.4f/%.4f | anchor_bound max/med=%.4f/%.4f | g_sum=J0^T r0 rel med/max=%.2e/%.2e | rho_G med=%.4e' % (
                                        cur_epoch, user_idx, batch_idx + 1, _et[0], _et[1], _et[2], float(_ab.max()), float(_ab.median()), float(_gr.median()), float(_gr.max()), _r0d['rho_Gfull_med']), flush=True)
                        del _hvp_i, _lg_i, _gs_i, _sl_i, _J_i, _r0d
                  except torch.OutOfMemoryError as _e:
                    print('[INSTR] Epoch %s user %s b=%d | SKIPPED (CUDA OOM during instrumentation; training path unaffected): %s' % (cur_epoch, user_idx, batch_idx + 1, str(_e)[:80]), flush=True)
                    _hvp_i = _lg_i = _gs_i = _sl_i = _J_i = _r0d = None
                    torch.cuda.empty_cache()
                  finally:
                    _rng_restore(_rng_i)

                with torch.no_grad():
                    _cn_b = (g_tilde - g0).reshape(g0.size(0), -1).norm(dim=1)
                    _gn_b = g0.reshape(g0.size(0), -1).norm(dim=1) + 1e-12
                    diag.setdefault('corr_ratio', []).append(float((_cn_b / _gn_b).mean().item()))
                    diag.setdefault('g0_norm_all', []).append(float(_gn_b.mean().item()))

                _meas_round = (cur_epoch is None) or ((int(cur_epoch) + 1) % 10 == 0)

                _diag_base = (not getattr(args, 'hd_diag_at_test', False)) or _meas_round
                _do_diag = _diag_base or (_INSTR_ON and os.environ.get('INSTR_NO_DIAGEXT', '0') != '1')
                _diag_rng = (_rng_snapshot() if (_do_diag and not _diag_base) else None)

                _anch_s = str(getattr(args, 'hd_diag_anchors', '') or '')
                if _anch_s.strip():
                    if not hasattr(args, '_hd_anchor_set'):
                        _pm = max(0, int(getattr(args, 'hd_diag_anchor_pm', 1)))
                        _aset = set()
                        for _tok in _anch_s.split(','):
                            _tok = _tok.strip()
                            if _tok:
                                _ai = int(_tok)
                                for _dd in range(-_pm, _pm + 1):
                                    if _ai + _dd >= 0:
                                        _aset.add(_ai + _dd)
                        args._hd_anchor_set = _aset
                    _do_diag = _do_diag and (batch_idx in args._hd_anchor_set)
                else:
                    _dmb = int(getattr(args, 'hd_diag_max_batches', 0))
                    if _dmb > 0:
                        _do_diag = _do_diag and (batch_idx < _dmb)
                if _oracle_1c_skip:
                    _do_diag = False
                if _do_diag and mode == 'kprobe':

                    from train.hd_omega import flip_stats as _fs
                    diag.setdefault('kp_flip', []).append(_fs(g0, g_tilde - g0))
                if _do_diag:
                    with torch.enable_grad():
                        fx_true = fx.detach().clone().requires_grad_(True)
                        with masked_rng(args, cur_epoch, user_idx, batch_idx, dev):
                            out_true, _ = (net_server_eval(fx_true, ext_mask) if is_llm
                                           else net_server_eval(fx_true))
                        g_true = torch.autograd.grad(criterion(out_true, label), fx_true)[0].detach()

                    with torch.no_grad():
                        with masked_rng(args, cur_epoch, user_idx, batch_idx, dev):
                            _z0_sp, _ = (net_server_eval(x2, ext_mask) if is_llm else net_server_eval(x2))
                        _spath_collect(diag, _z0_sp, out_true.detach(), label)
                        if mode == 'omegaMpJ' and _L2_ON:
                            _L0 = float(criterion(_z0_sp, label)); _Lb = float(criterion(out_true.detach(), label))
                            _Lhat = _L0 + float((g0.float() * delta.float()).sum()) + 0.5 * float(_mst.get('quad2', float('nan')))
                            diag.setdefault('l2_maj_viol', []).append(1.0 if _Lb > _Lhat else 0.0)
                            diag.setdefault('l2_maj_exc', []).append((_Lb - _Lhat) / max(abs(_L0), 1e-30))
                            diag.setdefault('l2_maj_mode', []).append(str(_mst.get('mode', '?')))
                    _lr_rho = float(getattr(args, 'srv_lips_rho', 0.0) or 0.0)
                    if _lr_rho > 0:
                        from train.srv_jitter import lips_estimate, make_gen
                        _lg = make_gen(dev, (int(cur_epoch or 0) * 100003 + int(user_idx or 0) * 1009 + int(batch_idx)) ^ 0x5A5A)
                        diag.setdefault('lips', []).append(lips_estimate(
                            net_server_eval, x2, lab0, criterion, _lr_rho, _lg, ext_mask=(msk0 if is_llm else None), is_llm=is_llm,
                            rng_ctx=lambda: masked_rng(args, cur_epoch, user_idx, batch_idx, dev)))
                        diag.setdefault('lips_b', []).append(int(batch_idx))
                    with torch.no_grad():
                        gt = g_true.reshape(g_true.size(0), -1)
                        gh = g_tilde.reshape(g0.size(0), -1)
                        gb = g0.reshape(g0.size(0), -1)
                        cc = F.cosine_similarity(gh, gt, dim=1)
                        cba = F.cosine_similarity(gb, gt, dim=1)
                        diag['cos_corr'].append(cc.mean().item())
                        diag['cos_base'].append(cba.mean().item())
                        diag['delta_cos'].append((cc - cba).mean().item())
                        gn = gt.norm(dim=1) + 1e-12
                        diag['rel_err_corr'].append(((gt - gh).norm(dim=1) / gn).mean().item())
                        diag['rel_err_base'].append(((gt - gb).norm(dim=1) / gn).mean().item())
                        gap = gt - gb
                        corr = gh - gb
                        gap_n = gap.norm(dim=1) + 1e-12
                        diag['staleness'].append((gap_n / (gb.norm(dim=1) + 1e-12)).mean().item())
                        diag['cos_corr_gap'].append(F.cosine_similarity(corr, gap, dim=1).mean().item())
                        diag['corr_gap_ratio'].append((corr.norm(dim=1) / gap_n).mean().item())

                        _cn_d = corr.norm(dim=1) + 1e-12
                        _gbu_d = gb / (gb.norm(dim=1, keepdim=True) + 1e-12)
                        _cpar_d = (corr * _gbu_d).sum(dim=1)
                        diag['corr_par_frac'].append(((_cpar_d ** 2) / (_cn_d ** 2)).mean().item())
                        diag['cos_corr_g0'].append((_cpar_d / _cn_d).mean().item())
                        del _cn_d, _gbu_d, _cpar_d

                        if bool(getattr(args, 'hd_decomp_diag', False)):
                            _eps_d = 1e-12
                            _g0u = gb / (gb.norm(dim=1, keepdim=True) + _eps_d)
                            _gap_d = gt - gb
                            _cpar = (corr * _g0u).sum(dim=1, keepdim=True) * _g0u
                            _cperp = corr - _cpar
                            _gpar = (_gap_d * _g0u).sum(dim=1, keepdim=True) * _g0u
                            _gperp_d = _gap_d - _gpar
                            _gtn = gt.norm(dim=1) + _eps_d
                            _copt = (gt * gb).sum(dim=1) / (gb.norm(dim=1) ** 2 + _eps_d)
                            _e = {
                                'e_base': (gb - gt).norm(dim=1) / _gtn,
                                'e_full': (gb + corr - gt).norm(dim=1) / _gtn,
                                'e_par' : (gb + _cpar - gt).norm(dim=1) / _gtn,
                                'e_perp': (gb + _cperp - gt).norm(dim=1) / _gtn,
                                'e_orc' : (_copt.unsqueeze(1) * gb - gt).norm(dim=1) / _gtn,
                            }
                            for _k_d, _v_d in _e.items():
                                _DEC_ACC[_k_d].extend(_v_d.detach().cpu().tolist())
                            _DEC_ACC['cos_perp'].extend(
                                F.cosine_similarity(_cperp, _gperp_d, dim=1).detach().cpu().tolist())
                            _DEC_ACC['mag_perp'].extend(
                                (_cperp.norm(dim=1) / (_gperp_d.norm(dim=1) + _eps_d)).detach().cpu().tolist())
                            _DEC_ACC['c_opt'].extend(_copt.detach().cpu().tolist())
                            _DEC_BIDX.append(int(batch_idx))
                            _DEC_ACC['stale'].extend(
                                (_gap_d.norm(dim=1) / (gb.norm(dim=1) + _eps_d)).detach().cpu().tolist())
                            del _g0u, _gap_d, _cpar, _cperp, _gpar, _gperp_d, _gtn, _copt, _e

                        mag0 = gb.norm(dim=1); magt = gt.norm(dim=1) + 1e-12
                        alpha = (gb * gt).sum(dim=1) / (mag0 ** 2 + 1e-12)
                        scale_frac = ((alpha - 1.0) ** 2 * mag0 ** 2) / (gap_n ** 2)
                        diag['mag_ratio'].append((magt / (mag0 + 1e-12)).mean().item())
                        diag['scale_frac'].append(scale_frac.clamp(0.0, 1.0).mean().item())
                        diag['mag_err_base'].append(((mag0 - magt).abs() / magt).mean().item())
                        diag['mag_err_corr'].append(((gh.norm(dim=1) - magt).abs() / magt).mean().item())

                        diag['g0_norm'].append(mag0.mean().item())
                        diag['gt_norm'].append(magt.mean().item())
                        diag['gtil_norm'].append(gh.norm(dim=1).mean().item())
                        diag['alpha_signed'].append(alpha.mean().item())
                        diag['scale_err'].append((alpha - 1.0).abs().mean().item())
                        _gperp = gt - alpha.unsqueeze(1) * gb
                        diag['dir_err'].append((_gperp.norm(dim=1) / (mag0 + 1e-12)).mean().item())

                        diag['l2_cos_negg0'].append(F.cosine_similarity(
                            delta.reshape(delta.size(0), -1), -gb, dim=1).mean().item())

                        _x2d = x2.reshape(x2.size(0), -1)
                        _dfd = delta.reshape(delta.size(0), -1)
                        diag['rho_a'].append(F.cosine_similarity(_dfd, _x2d, dim=1).mean().item())
                        diag['rho_g'].append(F.cosine_similarity(_dfd, gb, dim=1).mean().item())
                        diag['cos_au'].append(F.cosine_similarity(_x2d, gb, dim=1).mean().item())

                        if (_prev_delta is not None) and (_prev_delta.shape == delta.shape):
                            diag['dpers'].append(F.cosine_similarity(
                                delta.reshape(1, -1), _prev_delta.reshape(1, -1), dim=1).item())
                        _reb = (gt - gb).norm(dim=1) / gn; _rec = (gt - gh).norm(dim=1) / gn
                        diag['residual_ratio'].append((_rec / (_reb + 1e-12)).mean().item())

                        if bool(getattr(args, 'hd_compare_hess', False)):
                          with torch.enable_grad():
                            _an = x2.reshape(x2.size(0), -1).norm()
                            _vn = delta.reshape(delta.size(0), -1).norm() + 1e-12
                            _ep = eps_rel * float((_an / _vn).item())
                            _gp = server_grad_query(net_server_probe, (x2 + _ep * delta).detach(), lab0, args, msk0).reshape(g0.size(0), -1)
                            _gm = server_grad_query(net_server_probe, (x2 - _ep * delta).detach(), lab0, args, msk0).reshape(g0.size(0), -1)
                            Hd = (_gp - _gm) / (2.0 * _ep)
                            g_hess = gb + Hd
                            diag['cos_hess'].append(F.cosine_similarity(g_hess, gt, dim=1).mean().item())
                            diag['rel_err_hess'].append(((gt - g_hess).norm(dim=1) / gn).mean().item())
                            _cu = gh - gb
                            diag['hvp_cos'].append(F.cosine_similarity(_cu, Hd, dim=1).mean().item())
                            diag['hvp_relerr'].append(((Hd - _cu).norm(dim=1) / (Hd.norm(dim=1) + 1e-12)).mean().item())
                            diag['hvp_magratio'].append((_cu.norm(dim=1) / (Hd.norm(dim=1) + 1e-12)).mean().item())
                            for _t in (0.25, 0.5, 0.75):
                                _gtt = server_grad_query(net_server_probe, (x2 + _t * delta).detach(), lab0, args, msk0).reshape(g0.size(0), -1)
                                diag[f'tc_cos_{int(_t*100)}'].append(F.cosine_similarity(gb, _gtt, dim=1).mean().item())
                                diag[f'tc_mag_{int(_t*100)}'].append((_gtt.norm(dim=1) / (gb.norm(dim=1) + 1e-12)).mean().item())

                        if bool(getattr(args, 'hd_taylor2_diag', False)):
                          with torch.enable_grad():
                            _a1t = x2.detach().clone().requires_grad_(True)
                            _z1t = (net_server_probe(_a1t, msk0)[0] if is_llm else net_server_probe(_a1t)[0])
                            _g1t = torch.autograd.grad(_TL.gnll(_z1t, lab0), _a1t, create_graph=True)[0]
                            _Hd_t = torch.autograd.grad((_g1t * delta.detach()).sum(), _a1t)[0].detach().reshape(g0.size(0), -1)
                            del _a1t, _z1t, _g1t
                          _gap_t = gt - gb
                          _gapn_t = _gap_t.norm(dim=1) + 1e-12
                          diag['taylor2_frac'].append(((_gap_t - _Hd_t).norm(dim=1) / _gapn_t).mean().item())
                          diag['cos_hd_gap'].append(F.cosine_similarity(_Hd_t, _gap_t, dim=1).mean().item())
                          del _Hd_t, _gap_t, _gapn_t
                    del g_true
                _prev_delta = delta.detach()

                if bool(getattr(args, 'hd_scale_ablation', False)):
                    with torch.enable_grad():
                        _fxt = fx.detach().clone().requires_grad_(True)
                        _outt, _ = (net_server_eval(_fxt, ext_mask) if is_llm else net_server_eval(_fxt))
                        _gtrue_abl = torch.autograd.grad(_TL.gnll(_outt, lab0), _fxt)[0].detach()
                    _g0f = g0.reshape(g0.size(0), -1)
                    _alpha = (_gtrue_abl.reshape(_gtrue_abl.size(0), -1).norm(dim=1, keepdim=True)
                              / (_g0f.norm(dim=1, keepdim=True) + 1e-12))

                    _gamma = float(getattr(args, 'hd_scale_gamma', 1.0))
                    _alpha_g = 1.0 + _gamma * (_alpha - 1.0)
                    g_train = (_alpha_g * _g0f).reshape_as(g0)
                else:
                    g_train = g_tilde

                if _diag_rng is not None:
                    _rng_restore(_diag_rng); _diag_rng = None
                _pd_on = (bool(getattr(args, 'hd_param_diag', False)) and _do_diag
                          and (_PAR_ACC['nb'] < _pd_cap or _pd_cap == 0))
                if _pd_on:
                    fx.backward(g_train, retain_graph=True)
                    _v_snap = _pd_snapshot(net_c)
                    _v_til = _pd_flat(_v_snap)
                    net_c.zero_grad(); optimizer_client.zero_grad()
                    with torch.enable_grad():
                        _fx_p = fx.detach().clone().requires_grad_(True)
                        _out_p, _ = (net_server_eval(_fx_p, ext_mask) if is_llm
                                     else net_server_eval(_fx_p))
                        _gt_p = torch.autograd.grad(criterion(_out_p, label), _fx_p)[0].detach()
                        del _fx_p, _out_p
                    fx.backward(_gt_p)
                    _v_tru = _pd_flat(_pd_snapshot(net_c))
                    _PAR_ACC['bwd'] += 1
                    if _v_til is not None and _v_tru is not None:
                        with torch.no_grad():
                            _PAR_ACC['cos'].append(float(F.cosine_similarity(
                                _v_til.unsqueeze(0), _v_tru.unsqueeze(0), dim=1).item()))
                            _PAR_ACC['mag'].append(float(_v_til.norm().item()
                                                         / (_v_tru.norm().item() + 1e-12)))
                    _PAR_ACC['nb'] += 1; _PAR_BIDX.append(int(batch_idx))
                    net_c.zero_grad(); optimizer_client.zero_grad()
                    _pd_restore(net_c, _v_snap)
                    del _v_snap, _v_til, _v_tru, _gt_p
                    optimizer_client.step()
                elif _CAP_FORM == 'a1':

                    import train.hd_instr as _Ic
                    _snap = _Ic.snapshot_client(net_c, optimizer_client)
                    fx.backward(g_train); optimizer_client.step()
                    _bt = 0; _pre_dr = None
                    while True:
                        with torch.no_grad(), masked_rng(args, cur_epoch, user_idx, batch_idx, dev):
                            _fxn = (net_c(ids, bmask)[0] if is_llm else net_c(images))
                        _dr = _Ic.drift_ratio(_fxn, x2)
                        if _pre_dr is None:
                            _pre_dr = _dr
                        if _dr <= _CAP_RHO or _bt >= _CAP_MAXBT:
                            break
                        _Ic.restore_client(net_c, optimizer_client, _snap); _CAP_LRM *= _CAP_LRMULT; _Ic.set_lr(optimizer_client, float(args.lr) * _CAP_LRM)
                        optimizer_client.step(); _bt += 1
                    _CAP_ST['pre'].append(_pre_dr); _CAP_ST['post'].append(_dr); _CAP_ST['bt'].append(_bt); _CAP_ST['capped'].append(1.0 if _bt > 0 else 0.0)
                    del _snap, _fxn
                elif _CAP_FORM == 'a2':

                    import train.hd_instr as _Ic
                    _CAP_PREV = _Ic.snapshot_client(net_c, optimizer_client)
                    fx.backward(g_train); optimizer_client.step()
                elif _CAP_FORM == 'c':

                    import train.hd_instr as _Ic
                    fx.backward(g_train); optimizer_client.step()
                    _rel, _capd = _Ic.cap_project_theta(net_c, _CAP_TH0, _CAP_RHOT)
                    _CAP_ST['capped'].append(1.0 if _capd else 0.0); _CAP_ST['bt'].append(0); _CAP_ST['dtheta'] = _rel
                else:
                    fx.backward(g_train); optimizer_client.step()
                if _CAP_FORM in ('a1', 'a2', 'c'):
                    with torch.no_grad():
                        _CAP_ST['dtheta'] = float(torch.cat([(p - q).reshape(-1) for p, q in zip(net_c.parameters(), _CAP_TH0)]).norm() / torch.cat([q.reshape(-1) for q in _CAP_TH0]).norm().clamp_min(1e-30))
                del diagH, g_tilde, delta

        if _needs_dhat and _dhat_miss > 0:
            print(f"[HD-B1B4] user {user_idx}: δ̂ {_dhat_miss}/{len(cached_x2_list)} "
                  f"( fd_fixed_order)", flush=True)

        _L2_X0 = (_fetch(x2_gpu[0], cast=True).detach() if (_L2_ON and _L2_IN is not None and len(x2_gpu) > 0) else None)
        _L2_M0 = ((mask_gpu[0] if is_llm else None) if _L2_X0 is not None else None)
        del x2_gpu, g_gpu, lab_gpu, mask_gpu
        if is_hybrid and seed_inputs:
            del seed_inputs
        gc.collect(); torch.cuda.empty_cache()

        def _m(xs): return float(np.mean(xs)) if len(xs) > 0 else 0.0
        def _s(xs): return float(np.std(xs)) if len(xs) > 0 else 0.0
        def _mx(xs): return float(np.max(xs)) if len(xs) > 0 else 0.0
        diag_summary = {
            '[HD] cos(g_tilde, g_true) mean': _m(diag['cos_corr']),
            '[HD] cos(g_tilde, g_true) std' : _s(diag['cos_corr']),
            '[HD] cos(g_0,     g_true) mean': _m(diag['cos_base']),
            '[HD] cos(g_0,     g_true) std' : _s(diag['cos_base']),
            '[HD] delta_cos mean (corr improvement)' : _m(diag['delta_cos']),
            '[HD] delta_cos max'            : _mx(diag['delta_cos']),
            '[HD] cos(corr, gap) mean'      : _m(diag['cos_corr_gap']),
            '[HD] corr_par_frac mean'       : _m(diag['corr_par_frac']),
            '[HD] cos(corr, g0) mean'       : _m(diag['cos_corr_g0']),
            '[HD] |cos(u1,g0)| mean'        : _m(diag['u1_cos_g0']),
            '[HD] theta1 mean'              : _m(diag['th1']),
            '[HD] ||corr||/||gap|| mean'    : _m(diag['corr_gap_ratio']),
            '[HD] staleness ||gap||/||g0||mean': _m(diag['staleness']),
            '[HD] staleness max'            : _mx(diag['staleness']),
            '[HD] rel_err corrected mean'   : _m(diag['rel_err_corr']),
            '[HD] rel_err baseline  mean'   : _m(diag['rel_err_base']),
            '[HD] residual_ratio mean'      : _m(diag['residual_ratio']),
            '[HD] scale_frac mean'          : _m(diag['scale_frac']),
            '[HD] mag_ratio mean'           : _m(diag['mag_ratio']),
            '[HD] ||g0|| mean'              : _m(diag['g0_norm']),
            '[HD] ||g_true|| mean'          : _m(diag['gt_norm']),
            '[HD] ||g_tilde|| mean'         : _m(diag['gtil_norm']),
            '[HD] alpha(signed) mean'       : _m(diag['alpha_signed']),
            '[HD] scale_err |a-1| mean'     : _m(diag['scale_err']),
            '[HD] dir_err perp/g0 mean'     : _m(diag['dir_err']),
            '[HD] mag_err_base mean'        : _m(diag['mag_err_base']),
            '[HD] mag_err_corr mean'        : _m(diag['mag_err_corr']),
            '[HD] L1 cos(delta,dhat) mean'  : _m(diag['l1_cos_dhat']),
            '[HD] tau mean'                 : _m(diag['tau']),
            '[HD] L2 cos(delta,-g0) mean'   : _m(diag['l2_cos_negg0']),
            '[HD] rho_a cos(d,a0) mean'     : _m(diag['rho_a']),
            '[HD] rho_g cos(d,g0) mean'     : _m(diag['rho_g']),
            '[HD] cos(a0,g0) mean'          : _m(diag['cos_au']),
            '[HD] kappa vHv mean'           : _m(diag['kappa_ap']),
            '[HD] kappa neg frac'           : _m(diag['negfrac_ap']),
            '[HD] rho_h cos(d,vhat) mean'   : _m(diag['rho_h']),
            '[HD] cos(hlast,a0) mean'       : _m(diag['cos_ha']),
            '[HD] taylor2_frac mean'        : _m(diag['taylor2_frac']),
            '[HD] cos(Hd,gap) mean'         : _m(diag['cos_hd_gap']),
            '[HD] delta persistence mean'   : _m(diag['dpers']),
            '[HD] pearl-vs-FD relerr mean'  : _m(diag['pearl_fd_relerr']),
            '[HD] cos_hess (upper bound) mean'      : _m(diag['cos_hess']),
            '[HD] rel_err_hess (upper bound) mean'  : _m(diag['rel_err_hess']),
            '[HD] hvp_cos(F·δ vs H·δ) mean' : _m(diag['hvp_cos']),
            '[HD] hvp_relerr mean'          : _m(diag['hvp_relerr']),
            '[HD] hvp_magratio mean'        : _m(diag['hvp_magratio']),
            '[HD] tc_cos@.5 mean'           : _m(diag['tc_cos_50']),
            '[HD] tc_mag@.5 mean'           : _m(diag['tc_mag_50']),
            '[HD] ||corr|| mean'            : _m(diag['corr_norm']),
            '[HD] ||delta_x|| mean'         : _m(diag['delta_norm']),
            '[HD] ||delta_x|| max'          : _mx(diag['delta_norm']),

            '[HD] basis rank (m)'           : float(m_basis if is_hybrid else
                                                    (float(np.median(diag['eff_rank']))
                                                     if diag['eff_rank'] else _last_lr_rank)),
            '[HD] eff_rank median'          : (float(np.median(diag['eff_rank'])) if diag['eff_rank'] else 0.0),
            '[HD] bd_step median'           : (float(np.median(diag['bd_step'])) if diag['bd_step'] else 0.0),
            '[HD] total steps'              : diag['total_steps'],
            '[HD] n_diag'                   : float(len(diag['cos_corr'])),
        }

        if len(diag['eig_cos_g0']) > 0:
            _e_c = np.array(diag['eig_cos_g0'], dtype=float)
            _e_t = np.array(diag['eig_theta'], dtype=float)
            _e_s = np.array(diag['eig_share'], dtype=float)
            for _ii in range(_e_c.shape[1]):
                if np.all(np.isnan(_e_c[:, _ii])):
                    continue
                diag_summary[f'[HD] eig{_ii+1} |cos(u,g0)|'] = float(np.nanmean(_e_c[:, _ii]))
                diag_summary[f'[HD] eig{_ii+1} theta']       = float(np.nanmean(_e_t[:, _ii]))
                diag_summary[f'[HD] eig{_ii+1} corr_share']  = float(np.nanmean(_e_s[:, _ii]))

        _stale_only = bool(getattr(self.args, 'hd_stale_mode', False))

        _ref_only = _stale_only or bool(getattr(self.args, 'hd_upper_mode', False))
        if _ref_only:
            for _k in ['[HD] cos(g_tilde, g_true) mean', '[HD] cos(g_tilde, g_true) std',
                       '[HD] delta_cos mean (corr improvement)', '[HD] delta_cos max',
                       '[HD] cos(corr, gap) mean', '[HD] ||corr||/||gap|| mean',
                       '[HD] rel_err corrected mean', '[HD] residual_ratio mean',
                       '[HD] ||g_tilde|| mean', '[HD] mag_err_corr mean', '[HD] ||corr|| mean',
                       '[HD] corr_par_frac mean', '[HD] cos(corr, g0) mean']:
                diag_summary.pop(_k, None)

        _act_d = diag.pop('_act_d', None) or []

        _act_Lmsg = ''
        if len(_act_d) >= 2:
            _shp = [tuple(t.shape) for t in _act_d]
            if len(set(_shp)) == 1:
                _act_d = [t.reshape(-1) for t in _act_d]
            elif len(set(x[1:] for x in _shp)) == 1:
                _Lm = max(x[0] for x in _shp)
                _act_d = [(t if t.shape[0] == _Lm else
                           F.pad(t, (0, 0) * (t.dim() - 1) + (0, _Lm - t.shape[0]))).reshape(-1)
                          for t in _act_d]
                _Ls = sorted(x[0] for x in _shp)
                _act_Lmsg = ' L=%d~%d(pad→%d)' % (_Ls[0], _Ls[-1], _Lm)
            else:
                _szs = [int(t.numel()) for t in _act_d]
                _sz = max(set(_szs), key=_szs.count)
                _act_d = [t.reshape(-1) for t in _act_d if int(t.numel()) == _sz]
                print('[ACT] mixed shapes; using the most common size %d only, %d/%d batches excluded'
                      % (_sz, len(_szs) - len(_act_d), len(_szs)), flush=True)
        elif len(_act_d) == 1:
            _act_d = [_act_d[0].reshape(-1)]
        if _ACT_ON and len(_act_d) >= 2:
            with torch.no_grad():
                _aM = torch.stack(_act_d, 0)
                _anz = _aM.norm(dim=1)
                _akeep = _anz > 0
                _aM = _aM[_akeep]
                if _aM.shape[0] >= 2:
                    _aA = (_aM / _aM.norm(dim=1, keepdim=True)).double()
                    _aG = _aA @ _aA.t()
                    _aev, _aW = torch.linalg.eigh(_aG)
                    _aev = _aev.flip(0).clamp(min=0.0); _aW = _aW.flip(1)
                    _atot = float(_aev.sum()) + 1e-30
                    _acap = [(q, float(_aev[:q].sum()) / _atot)
                             for q in _ACT_QS if q <= int(_aev.numel())]
                    _aq = min(max(_ACT_QS), int(_aev.numel()))
                    _aV = (_aA.t() @ _aW[:, :_aq]) / _aev[:_aq].clamp(min=1e-30).sqrt().unsqueeze(0)
                    _aix = getattr(self, 'idxs', None)
                    try:
                        _ack = int(user_idx) if user_idx is not None else (int(min(_aix)) if (_aix is not None and len(_aix) > 0) else -1)
                    except Exception:
                        _ack = int(user_idx) if user_idx is not None else -1
                    _ast = getattr(self.args, '_act_prev', None)
                    if _ast is None:
                        _ast = {}
                        setattr(self.args, '_act_prev', _ast)
                    _apv = _ast.get(_ack)
                    _apc = None
                    if _apv is not None and tuple(_apv.shape) == tuple(_aV.shape):
                        _asv = torch.linalg.svdvals(_apv.to(torch.float64).t() @ _aV)
                        _apc = [float(v) for v in _asv.clamp(-1.0, 1.0)]
                    _ast[_ack] = _aV.to(torch.float32)
                    print('[ACT] ckey=%d B_used=%d/%d n=%d%s %s'
                          % (_ack, int(_aM.shape[0]), len(_act_d), int(_aM.shape[1]), _act_Lmsg,
                             ' '.join('cap%d=%.4f' % (q, c) for q, c in _acap)), flush=True)
                    if _apc is not None:
                        print('[ACT] prev_subspace_cos(q=%d)=[%s] mean=%.4f min=%.4f'
                              % (_aq, ' '.join('%.4f' % v for v in _apc),
                                 sum(_apc) / len(_apc), min(_apc)), flush=True)
                    print('[ACT-perbatch] cos(d_b,d_1)=[%s]'
                          % ' '.join('%+.4f' % float(v) for v in _aG[0].clamp(-1.0, 1.0)),
                          flush=True)
                    print('[ACT-perbatch] ||d_b||=[%s]'
                          % ' '.join('%.3e' % float(v) for v in _anz), flush=True)
                del _act_d
        if len(diag['cos_corr']) > 0:
            def _pb(xs, f='{:.4f}'): return " ".join(f.format(v) for v in xs)

            print(f"[HD-perbatch] n={len(diag['cos_base'])} cos_base(g0,gtrue)=[{_pb(diag['cos_base'], '{:.6f}')}]", flush=True)

            print(f"[HD-perbatch] ||dx||=[{_pb(diag['delta_norm'], '{:.2e}')}]", flush=True)
            print(f"[HD-perbatch] rel_err_base(g0)=[{_pb(diag['rel_err_base'])}]", flush=True)
            print(f"[HD-perbatch] scale_frac=[{_pb(diag['scale_frac'])}]", flush=True)
            print(f"[HD-perbatch] mag_ratio(|gt|/|g0|)=[{_pb(diag['mag_ratio'])}]", flush=True)

            print(f"[HD-perbatch] ||g0||=[{_pb(diag['g0_norm'], '{:.3e}')}]", flush=True)

            if diag['eff_rank']:
                print(f"[HD-perbatch] eff_rank(#|th_j|>1e-8|th_1|)=[{_pb(diag['eff_rank'], '{:.2f}')}]", flush=True)
                print(f"[HD-perbatch] bd_step( j; -1=Lanczos)=[{_pb(diag['bd_step'], '{:.0f}')}]", flush=True)
            print(f"[HD-perbatch] ||g_true||=[{_pb(diag['gt_norm'], '{:.3e}')}]", flush=True)
            print(f"[HD-perbatch] alpha(signed)=[{_pb(diag['alpha_signed'], '{:+.3f}')}]", flush=True)
            print(f"[HD-perbatch] scale_err(|a-1|)=[{_pb(diag['scale_err'])}] dir_err=[{_pb(diag['dir_err'])}]", flush=True)

            if len(diag['l2_cos_negg0']) > 0:
                print(f"[HD-perbatch] L2 cos(delta,-g0)=[{_pb(diag['l2_cos_negg0'], '{:+.4f}')}]", flush=True)
            if len(diag['l1_cos_dhat']) > 0:
                print(f"[HD-perbatch] L1 cos(delta,dhat)=[{_pb(diag['l1_cos_dhat'], '{:+.4f}')}]", flush=True)

            if len(diag['rho_a']) > 0:
                print(f"[HD-perbatch] rho_a cos(d,a0)=[{_pb(diag['rho_a'], '{:+.4f}')}]", flush=True)
                print(f"[HD-perbatch] rho_g cos(d,g0)=[{_pb(diag['rho_g'], '{:+.4f}')}]", flush=True)
                print(f"[HD-perbatch] cos(a0,g0)=[{_pb(diag['cos_au'], '{:+.4f}')}]", flush=True)
            if len(diag['kappa_ap']) > 0:
                print(f"[HD-perbatch] kappa(vHv)=[{_pb(diag['kappa_ap'], '{:+.3e}')}]", flush=True)
                print(f"[HD-perbatch] kappa neg_frac=[{_pb(diag['negfrac_ap'], '{:.2f}')}]", flush=True)
            if len(diag['rho_h']) > 0:
                print(f"[HD-perbatch] rho_h cos(d,vhat)=[{_pb(diag['rho_h'], '{:+.4f}')}]", flush=True)
                print(f"[HD-perbatch] cos(hlast,a0)=[{_pb(diag['cos_ha'], '{:+.4f}')}]", flush=True)
            if len(diag['taylor2_frac']) > 0:
                print(f"[HD-perbatch] taylor2_frac(2+)=[{_pb(diag['taylor2_frac'])}]", flush=True)
                print(f"[HD-perbatch] cos(Hd,gap)=[{_pb(diag['cos_hd_gap'], '{:+.4f}')}]", flush=True)
            if len(diag['dpers']) > 0:
                print(f"[HD-perbatch] delta_persistence=[{_pb(diag['dpers'], '{:+.4f}')}]", flush=True)
            if len(diag['tau']) > 0:
                print(f"[HD-perbatch] tau=[{_pb(diag['tau'], '{:+.3f}')}]", flush=True)
            if len(diag['pearl_fd_relerr']) > 0:
                print(f"[HD-perbatch] pearl-vs-FD relerr=[{_pb(diag['pearl_fd_relerr'], '{:.2e}')}]", flush=True)
            if _ref_only:
                print(f"[HD-perbatch] mag_err base=[{_pb(diag['mag_err_base'])}]", flush=True)
            else:

                print(f"[HD-perbatch] cos_corr(gtil,gtrue)=[{_pb(diag['cos_corr'], '{:.6f}')}]", flush=True)
                print(f"[HD-perbatch] delta_cos=[{_pb(diag['delta_cos'], '{:+.6f}')}]", flush=True)
                print(f"[HD-perbatch] cos(corr,gap)=[{_pb(diag['cos_corr_gap'], '{:+.4f}')}]", flush=True)

                print(f"[HD-perbatch] corr_par_frac(g0)=[{_pb(diag['corr_par_frac'], '{:.6f}')}]", flush=True)
                print(f"[HD-perbatch] cos(corr,g0)=[{_pb(diag['cos_corr_g0'], '{:+.6f}')}]", flush=True)
                if len(diag['u1_cos_g0']) > 0:
                    print(f"[HD-perbatch] |cos(u1,g0)|=[{_pb(diag['u1_cos_g0'], '{:.6f}')}]", flush=True)
                    print(f"[HD-perbatch] theta1=[{_pb(diag['th1'], '{:+.3e}')}]", flush=True)

                if len(diag['eig_cos_g0']) > 0:
                    for _ii in range(len(diag['eig_cos_g0'][0])):
                        _row_e = [_r[_ii] for _r in diag['eig_cos_g0']]
                        if all((_x != _x) for _x in _row_e):
                            continue
                        print(f"[HD-eig] i={_ii+1} |cos(u_i,g0)|=[{_pb(_row_e, '{:.6f}')}] "
                              f"theta=[{_pb([_r[_ii] for _r in diag['eig_theta']], '{:+.3e}')}] "
                              f"share=[{_pb([_r[_ii] for _r in diag['eig_share']], '{:.3f}')}]", flush=True)
                print(f"[HD-perbatch] rel_err_corr(gtil)=[{_pb(diag['rel_err_corr'])}]", flush=True)
                print(f"[HD-perbatch] residual_ratio(corr/base,<1=)=[{_pb(diag['residual_ratio'])}]", flush=True)
                print(f"[HD-perbatch] ||g_tilde||=[{_pb(diag['gtil_norm'], '{:.3e}')}]", flush=True)
                print(f"[HD-perbatch] mag_err base=[{_pb(diag['mag_err_base'])}] corr=[{_pb(diag['mag_err_corr'])}]", flush=True)

            if bool(getattr(self.args, 'hd_compare_hess', False)) and len(diag['cos_hess']) > 0:
                print(f"[HD-compare] cos_hess=[{_pb(diag['cos_hess'])}]", flush=True)
                print(f"[HD-compare] rel_err_hess=[{_pb(diag['rel_err_hess'])}]", flush=True)
                print(f"[HD-compare] hvp_cos(F·δ vs H·δ)=[{_pb(diag['hvp_cos'])}]", flush=True)
                print(f"[HD-compare] hvp_relerr=[{_pb(diag['hvp_relerr'])}]", flush=True)
                print(f"[HD-compare] hvp_magratio=[{_pb(diag['hvp_magratio'])}]", flush=True)
                print(f"[HD-compare] tc_cos[.25|.5|.75]=[{_pb(diag['tc_cos_25'])}||{_pb(diag['tc_cos_50'])}||{_pb(diag['tc_cos_75'])}]", flush=True)
                print(f"[HD-compare] tc_mag[.25|.5|.75]=[{_pb(diag['tc_mag_25'])}||{_pb(diag['tc_mag_50'])}||{_pb(diag['tc_mag_75'])}]", flush=True)

        if _DEC_ACC['e_base']:
            import numpy as _np_d
            def _q(xs):
                a_ = _np_d.asarray(xs, dtype=_np_d.float64)
                return (float(a_.mean()), float(_np_d.percentile(a_, 50)),
                        float(_np_d.percentile(a_, 90)), float(_np_d.percentile(a_, 99)),
                        float(a_.max()))
            _eb = _np_d.asarray(_DEC_ACC['e_base']); _ef = _np_d.asarray(_DEC_ACC['e_full'])
            _ep_ = _np_d.asarray(_DEC_ACC['e_par']); _eo = _np_d.asarray(_DEC_ACC['e_orc'])
            _parts = []
            for _k_d in ('e_base', 'e_full', 'e_par', 'e_perp', 'e_orc'):
                _m5 = _q(_DEC_ACC[_k_d])
                _parts.append('%s mean=%.4f p50=%.4f p90=%.4f p99=%.4f max=%.4f' % ((_k_d,) + _m5))
            _cp = _np_d.asarray(_DEC_ACC['cos_perp']); _mp = _np_d.asarray(_DEC_ACC['mag_perp'])
            _co = _np_d.asarray(_DEC_ACC['c_opt'])
            _parts.append('cos_perp mean=%.4f p50=%.4f' % (_cp.mean(), _np_d.percentile(_cp, 50)))
            _parts.append('mag_perp mean=%.4f p50=%.4f' % (_mp.mean(), _np_d.percentile(_mp, 50)))
            _parts.append('c_opt mean=%.4f p50=%.4f p90=%.4f'
                          % (_co.mean(), _np_d.percentile(_co, 50), _np_d.percentile(_co, 90)))
            _parts.append('frac(e_full<e_base)=%.4f' % float((_ef < _eb).mean()))
            _parts.append('frac(e_full<e_orc)=%.4f' % float((_ef < _eo).mean()))
            _parts.append('frac(e_par<e_full)=%.4f' % float((_ep_ < _ef).mean()))
            _parts.append('n_samples=%d' % _eb.size)

            _parts.append('batches=%s' % ','.join(str(b) for b in _DEC_BIDX))
            print('[HD-DECOMP] r=%s user=%s | %s' % (cur_epoch, user_idx, ' | '.join(_parts)), flush=True)

            _st_a = _np_d.asarray(_DEC_ACC['stale'])
            if _st_a.size >= 100:
                _cut = _np_d.percentile(_st_a, 99); _sel = _st_a >= _cut
                if _sel.sum() > 0:
                    print('[HD-DECOMP-TAIL] r=%s user=%s | top 1%% staleness subset (p99 cut) | n=%d e_base mean=%.4f e_full mean=%.4f e_orc mean=%.4f frac(e_full<e_base)=%.4f'
                          % (cur_epoch, user_idx, int(_sel.sum()), float(_eb[_sel].mean()),
                             float(_ef[_sel].mean()), float(_eo[_sel].mean()),
                             float((_ef[_sel] < _eb[_sel]).mean())), flush=True)

        if _PAR_ACC['cos']:
            import numpy as _np_p
            _pc = _np_p.asarray(_PAR_ACC['cos']); _pm = _np_p.asarray(_PAR_ACC['mag'])
            print('[HD-PARAM] r=%s user=%s | n_batches=%d | cos_param mean=%.4f p50=%.4f p90=%.4f '
                  '| mag_param mean=%.4f p50=%.4f | extra_bwd=%d | batches=%s'
                  % (cur_epoch, user_idx, _PAR_ACC['nb'], _pc.mean(), _np_p.percentile(_pc, 50),
                     _np_p.percentile(_pc, 90), _pm.mean(), _np_p.percentile(_pm, 50),
                     _PAR_ACC['bwd'], ','.join(str(b) for b in _PAR_BIDX)), flush=True)

        if _L2_ON and _L2_IN is not None and _L2_X0 is not None:
            import train.hd_instr as _I2
            _rng_l2 = _rng_snapshot()
            try:
                _x0_l2 = _L2_X0; _m0_l2 = _L2_M0
                _ctx = (lambda: masked_rng(args, cur_epoch, user_idx, 0, dev))
                with torch.no_grad():
                    with _ctx():
                        _aB_l2 = (net_c(_L2_IN[0], _L2_IN[1])[0] if is_llm else net_c(_L2_IN[0])).detach()
                _flip, _kinds = _I2.l2_relu_flip(net_server_probe, _x0_l2, _aB_l2, _m0_l2, is_llm, _ctx)
                _gen = _I2.l2_probe_gen(int(getattr(args, 'seed', 0)) * 7919 + int(cur_epoch) * 131 + int(user_idx))
                with torch.enable_grad():
                    with _ctx():
                        _r0_l2, _V_l2, _ = _I2.l2_jacobian_rows(net_server_probe, _x0_l2, _m0_l2, is_llm, k=8, gen=_gen)
                    with _ctx():
                        _rB_l2, _, _ = _I2.l2_jacobian_rows(net_server_probe, _aB_l2, _m0_l2, is_llm, k=8, gen=None, V=_V_l2)
                if _r0_l2 is None:
                    print('[L2] Epoch %s user %s | undefined (LM logits) | act=%s' % (cur_epoch, user_idx, ','.join(_kinds)), flush=True)
                else:
                    _seg = _I2.l2_rel(_r0_l2, _rB_l2); _R_l2 = int(_r0_l2.size(1)); _mode_l2 = ('E' if _V_l2 is None else 'P')
                    _dr_l2 = float((_aB_l2 - _x0_l2).reshape(-1).norm() / _x0_l2.reshape(-1).norm().clamp_min(1e-30))
                    print('[L2] Epoch %s user %s | relu_flip=%s (act=%s) | J_seg_rel=%.4e | drift_b0=%.4f | rows=%s R=%d | n=%d' % (
                        cur_epoch, user_idx, ('%.4e' % _flip) if _flip is not None else 'n/a', ','.join(_kinds), _seg, _dr_l2, _mode_l2, _R_l2, int(_x0_l2.size(0))), flush=True)
                    diag['l2_relu_flip'] = (float(_flip) if _flip is not None else float('nan')); diag['l2_J_seg_rel'] = _seg
                    if not hasattr(args, '_l2_store'):
                        args._l2_store = {}
                    args._l2_store[int(user_idx) if user_idx is not None else -1] = {'rows0': _r0_l2.detach().cpu(), 'V': (_V_l2.detach().cpu() if _V_l2 is not None else None), 'rnd': cur_epoch, 'seg': _seg, 'k': 8}
                del _aB_l2, _r0_l2, _rB_l2
            except torch.OutOfMemoryError as _e:
                print('[L2] Epoch %s user %s | SKIPPED (CUDA OOM): %s' % (cur_epoch, user_idx, str(_e)[:80]), flush=True); torch.cuda.empty_cache()
            finally:
                _rng_restore(_rng_l2)
        if diag.get('l2_maj_viol'):
            import statistics as _stm
            _mv = diag['l2_maj_viol']; _ex = [e for e, v in zip(diag['l2_maj_exc'], _mv) if v > 0]
            print('[L2MAJ] Epoch %s user %s | frac_viol=%.4f | excess med(viol)=%s | frac_G med=%s | mode=%s | n=%d' % (
                cur_epoch, user_idx, _stm.mean(_mv), ('%.4e' % _stm.median(_ex)) if _ex else '-', ('%.4f' % _stm.median(diag['mp_frac_G'])) if diag.get('mp_frac_G') else '-', diag['l2_maj_mode'][-1], len(_mv)), flush=True)
        if _DELTA_ACC:
            import statistics as _st
            _thr = ''
            try:
                if cached_th_list:
                    _t0 = _fetch(cached_th_list[0], cast=True).reshape(-1).abs()
                    if _t0.numel() >= 2 and float(_t0[0]) > 0:
                        _thr = ' | lam_k/lam_1=%.3e' % (float(_t0[-1]) / float(_t0[0]))
            except Exception:
                pass
            print('[DELTA] Epoch %s user %s | ||d||/||a0|| mean=%.4e max=%.4e n=%d%s'
                  % (cur_epoch, user_idx, _st.mean(_DELTA_ACC), max(_DELTA_ACC),
                     len(_DELTA_ACC), _thr), flush=True)
        if diag.get('corr_norm'):
            import statistics as _stc
            _cn = diag['corr_norm']; _cr = diag.get('corr_ratio', [])
            print('[HD-perbatch] ||corr||=[%s]' % ' '.join('%.3e' % v for v in _cn), flush=True)
            print('[HD-perbatch] corr_norm_ratio_persample(mean_n|c_n|/|g0_n|; differs from cnr)=[%s]' % ' '.join('%.4f' % v for v in _cr), flush=True)
            print('[HD-perbatch] ||g0||=[%s]' % ' '.join('%.3e' % v for v in diag.get('g0_norm_all', [])), flush=True)
            print('[HD-CORR] Epoch %s user %s | ||corr|| min/med/max=%.3e/%.3e/%.3e | |c|/|g0| min/med/max=%.4f/%.4f/%.4f | n=%d'
                  % (cur_epoch, user_idx, min(_cn), _stc.median(_cn), max(_cn),
                     (min(_cr) if _cr else float('nan')), (_stc.median(_cr) if _cr else float('nan')), (max(_cr) if _cr else float('nan')), len(_cn)), flush=True)
            diag_summary['[HD] ||corr|| max'] = float(max(_cn))
            if _cr:
                diag_summary['[HD] |c|/|g0| med'] = float(_stc.median(_cr)); diag_summary['[HD] |c|/|g0| max'] = float(max(_cr))
        if diag.get('kp_flip'):
            _fl = diag['kp_flip']
            _md = lambda k: _stc.median([d[k] for d in _fl])
            print('[HD-FLIP] Epoch %s user %s | n=%d | flip_hi=%.4f flip_lo=%.4f | cnr=%.4f cnr_hi=%.4f cnr_lo=%.4f'
                  % (cur_epoch, user_idx, len(_fl), _md('flip_hi'), _md('flip_lo'), _md('cnr'), _md('cnr_hi'), _md('cnr_lo')), flush=True)
        if mode == 'kprobe':
            from train.hd_kprobe import kprobe_log_round
            diag_summary.update(kprobe_log_round(diag, cur_epoch, user_idx, args))
        if diag.get('restore_rho_med'):
            import statistics as _stm
            print('[RESTORE] Epoch %s user %s | rho med/min=%.4e/%.4e | frac_neg=%.3f | n=%d' % (
                cur_epoch, user_idx, _stm.median(diag['restore_rho_med']), min(diag['restore_rho_min']), _stm.mean(diag['restore_neg']), len(diag['restore_rho_med'])), flush=True)
            print('[RHO] Epoch %s user %s | rho_arm med/min=%.4e/%.4e | corr_ratio med=%.4e | drift med/max=%.4f/%.4f | n=%d' % (
                cur_epoch, user_idx, _stm.median(diag['restore_rho_med']), min(diag['restore_rho_min']), _stm.median(diag.get('rho_cr', [float('nan')])),
                _stm.median(diag.get('rho_drift', [float('nan')])), max(diag.get('rho_drift', [float('nan')])), len(diag['restore_rho_med'])), flush=True)
        if diag.get('hv_snr'):
            import statistics as _stm
            print('[HVAR] Epoch %s user %s | snr_med=%.3f | frac_noise=%.3f | rho_absdiag med=%.4e | n=%d' % (cur_epoch, user_idx, _stm.median(diag['hv_snr']), _stm.mean(diag['hv_noise']), _stm.median(diag['hv_rho']), len(diag['hv_snr'])), flush=True)
        if diag.get('dca_lam_eff'):
            import statistics as _stm
            print('[DCA] Epoch %s user %s | lam_eff med=%.4e | r0_norm mean=%.4f | g2_skew med=%.2f | n=%d' % (cur_epoch, user_idx, _stm.median(diag['dca_lam_eff']), (_stm.mean(diag['dca_r0']) if diag.get('dca_r0') else float('nan')), _stm.median(diag['dca_g2_skew']), len(diag['dca_lam_eff'])), flush=True)
            if diag.get('dcaT_rho_A'):
                from train.hd_instr import _q3 as _q3f
                _cat = lambda k: (torch.cat(diag[k]) if diag.get(k) else None)
                _rA = _cat('dcaT_rho_A'); _rn2 = _cat('dcaT_r0_norm2'); _ed = _cat('dcaT_ef_dir'); _ls = _cat('dcaT_lam_max_S0')
                _q = lambda t: (_q3f(t) if t is not None else (float('nan'),) * 3)
                _qa, _qr, _qe, _ql = _q(_rA), _q(_rn2), _q(_ed), _q(_ls)
                _dec = (float((_rA / _rn2.clamp_min(1e-30)).median()) if (_rn2 is not None and _rn2.numel() == _rA.numel()) else float('nan'))
                print('[DCAT] Epoch %s user %s | lam_sum=%.4e lam_mean=%.4e | rho_A med/p10/p90=%.4e/%.4e/%.4e | r0_norm2 med=%.4f | rho_A/r0n2 med=%.4e | lam_max_S0 med=%.4f | ef_dir med/p10/p90=%.3f/%.3f/%.3f | eq_radius/|a0| med=%.4e (p10 %.4e p90 %.4e) | frac_free med=%.4f | drift_free med=%.4f | n=%d' % (
                    cur_epoch, user_idx, _stm.median(diag['dca_lam_sum']), _stm.median(diag['dca_lam_mean']), _qa[0], _qa[1], _qa[2], _qr[0], _dec, _ql[0], _qe[0], _qe[1], _qe[2],
                    float(_cat('dcaT_eq_med').median()), float(_cat('dcaT_eq_p10').median()), float(_cat('dcaT_eq_p90').median()),
                    float(_cat('dcaT_frac_free').median()), float(_cat('dcaT_drift_free').median()), int(_rA.numel())), flush=True)
        if diag.get('damp_mu'):
            import statistics as _stm
            print('[DAMP] Epoch %s user %s | mu med/min/max=%.4e/%.4e/%.4e | n=%d' % (cur_epoch, user_idx, _stm.median(diag['damp_mu']), min(diag['damp_mu']), max(diag['damp_mu']), len(diag['damp_mu'])), flush=True)
        if diag.get('dg_kappa'):
            import statistics as _stm
            print('[DIAGK] Epoch %s user %s | kappa(trace-match) med=%.4e | n=%d' % (cur_epoch, user_idx, _stm.median(diag['dg_kappa']), len(diag['dg_kappa'])), flush=True)
        if _CAP_FORM != 'none' and _CAP_ST['capped']:
            import statistics as _stm, train.hd_instr as _Ic
            print(_Ic.cap_line(cur_epoch, user_idx, {'form': _CAP_FORM, 'rho': (_CAP_RHOT if _CAP_FORM == 'c' else _CAP_RHO), 'bt_tot': int(sum(_CAP_ST['bt'])), 'bt_max': int(max(_CAP_ST['bt'])), 'lr_mult': _CAP_LRM,
                                'pre_med': (_stm.median(_CAP_ST['pre']) if _CAP_ST['pre'] else float('nan')), 'pre_max': (max(_CAP_ST['pre']) if _CAP_ST['pre'] else float('nan')),
                                'post_med': (_stm.median(_CAP_ST['post']) if _CAP_ST['post'] else float('nan')), 'post_max': (max(_CAP_ST['post']) if _CAP_ST['post'] else float('nan')),
                                'frac_capped': _stm.mean(_CAP_ST['capped']), 'dtheta_rel': _CAP_ST['dtheta'], 'n': len(_CAP_ST['capped'])}), flush=True)
        if _INSTR_DP and _DP_D:
            import train.hd_instr as _Ic
            if getattr(args, '_dpersist', None) is None:
                args._dpersist = _Ic.DPersist()
            _cr, _cg, _cs, _nr = args._dpersist.update(int(user_idx) if user_idx is not None else -1, _DP_D, _DP_G)
            print('[DPERSIST] Epoch %s user %s | cos_round med=%.4f (n=%d) | cos_g0 med=%.4f | cos_step med=%.4f | B=%d' % (cur_epoch, user_idx, _cr, _nr, _cg, _cs, len(_DP_D)), flush=True)
            _DP_D.clear(); _DP_G.clear()
            diag_summary['[RESTORE] rho med'] = float(_stm.median(diag['restore_rho_med'])); diag_summary['[RESTORE] frac_neg'] = float(_stm.mean(diag['restore_neg']))
        if diag.get('sp_p0'):
            import statistics as _stm
            print('[SPATH] Epoch %s user %s | pmax(z0) med=%.4f | pmax(z_b) med=%.4f | max_tau pmax med=%.4f | ratio med/max=%.3f/%.3f | frac(ratio>3)=%.3f | n=%d' % (
                cur_epoch, user_idx, _stm.median(diag['sp_p0']), _stm.median(diag['sp_pb']), _stm.median(diag['sp_pp']),
                _stm.median(diag['sp_ratio_med']), max(diag['sp_ratio_max']), _stm.mean(diag['sp_ratio_gt3']), len(diag['sp_p0'])), flush=True)
            diag_summary['[SPATH] ratio med'] = float(_stm.median(diag['sp_ratio_med']))
        if diag.get('sc_JF2_med'):
            import statistics as _stm
            print('[STEPCOND] Epoch %s user %s | JF2 med/max=%.4e/%.4e | r0 med/max=%.4f/%.4f | eta_max med/min=%.4e/%.4e | frac_violate=%.3f (eta=%.1e) | n=%d' % (
                cur_epoch, user_idx, _stm.median(diag['sc_JF2_med']), max(diag['sc_JF2_max']), _stm.median(diag['sc_r0_med']), max(diag['sc_r0_max']),
                _stm.median(diag['sc_eta_med']), min(diag['sc_eta_min']), _stm.mean(diag['sc_viol']), float(getattr(args, 'lr', 0.0)), len(diag['sc_JF2_med'])), flush=True)
            diag_summary['[STEPCOND] frac_violate'] = float(_stm.mean(diag['sc_viol']))
        if mode in ('omegaMpG', 'omegaMpJ') and diag.get('mp_frac_cap'):
            import statistics as _stm
            _tag = 'RSLOCMPJ' if mode == 'omegaMpJ' else 'RSLOCMP'
            _extra = (' | share med=%.4f | ratio_diag med=%.4e' % (_stm.median(diag['mpj_share']), _stm.median(diag['mpj_ratio_diag']))) if mode == 'omegaMpJ' and diag.get('mpj_ratio_diag') else ''
            print('[%s] Epoch %s user %s | frac_cap mean/last=%.3f/%.3f | frac_G mean/last=%.3f/%.3f | c_med mean/last=%.4f/%.4f | cDelta_med mean/max=%.4e/%.4e | n=%d%s' % (
                _tag, cur_epoch, user_idx, _stm.mean(diag['mp_frac_cap']), diag['mp_frac_cap'][-1], _stm.mean(diag['mp_frac_G']), diag['mp_frac_G'][-1],
                _stm.mean(diag['mp_c_med']), diag['mp_c_med'][-1], _stm.mean(diag['mp_cD_med']), max(diag['mp_cD_med']), len(diag['mp_frac_cap']), _extra), flush=True)
            diag_summary['[MP] frac_cap'] = float(_stm.mean(diag['mp_frac_cap'])); diag_summary['[MP] frac_G'] = float(_stm.mean(diag['mp_frac_G']))
            assert all(a + b <= 1.0 + 1e-6 for a, b in zip(diag['mp_frac_cap'], diag['mp_frac_G'])), '[RSLOCMPG] frac_cap+frac_G > 1'
        if mode == 'omegaMp' and diag.get('mp_frac_cap'):
            import statistics as _stm
            print('[RSLOCMP] Epoch %s user %s | frac_cap mean/last=%.3f/%.3f | c_med mean/last=%.4f/%.4f | n=%d' % (cur_epoch, user_idx, _stm.mean(diag['mp_frac_cap']), diag['mp_frac_cap'][-1], _stm.mean(diag['mp_c_med']), diag['mp_c_med'][-1], len(diag['mp_frac_cap'])), flush=True)
            diag_summary['[MP] frac_cap'] = float(_stm.mean(diag['mp_frac_cap']))
        if mode in ('omega3b', 'omega3a', 'omegaM', 'omegaMp', 'omegaMpG', 'omegaMpJ') and diag.get('om_cnr'):
            print('[HD-perbatch] cnr(|λω̂^pδ|/|g0|)=[%s]' % ' '.join('%.4f' % v for v in diag['om_cnr']), flush=True)
            diag_summary['[OM] cnr last'] = float(diag['om_cnr'][-1]); diag_summary['[OM] cnr max'] = float(max(diag['om_cnr']))
            if diag.get('om_cbar'):
                diag_summary['[OM] cbar min'] = float(min(diag['om_cbar'])); diag_summary['[OM] cbar max'] = float(max(diag['om_cbar']))
            if mode in ('omega3b', 'omega3a') and str(getattr(args, 'hd_omega_scale', 'taylor')) == 'radius' and diag.get('om_rel_r'):
                _Robs = float(max(diag['om_rel_r'])); _Rt = float(getattr(args, 'hd_radius_target', 0.05)); _gm = float(getattr(args, 'hd_radius_gamma', 0.5))
                _k_old = float(args._omega_kappa_r); _k_new = _k_old * (_Robs / _Rt) ** _gm
                args._omega_kappa_r = _k_new
                print(f'[OMEGA] radius r={cur_epoch} user={user_idx} R_obs={_Robs:.4f} R*={_Rt:g} kappa {_k_old:.4g} -> {_k_new:.4g} | cbar med={sorted(diag["om_cbar"])[len(diag["om_cbar"])//2]:.3e} lam_eff med={_k_old*sorted(diag["om_cbar"])[len(diag["om_cbar"])//2]:.3e}', flush=True)
                diag_summary['[OM] kappa_r'] = _k_new; diag_summary['[OM] R_obs'] = _Robs
            if diag.get('om_lam_b'):
                print('[HD-perbatch] lam_b(%s)=[%s]' % (str(getattr(args, 'hd_omega_scale', '')), ' '.join('%.3e' % v for v in diag['om_lam_b'])), flush=True)
                if str(getattr(args, 'hd_omega_scale', '')) == 'rspring' and diag.get('om_rel_r'):
                    print('[HD-perbatch] r_b=[%s]' % ' '.join('%.4f' % v for v in diag['om_rel_r']), flush=True)
                    if str(getattr(args, 'hd_rstar_mode', 'warmup')) == 'local':
                        print('[OMEGA] rsloc r=%s user=%s B=%s r2=%.5f R_loc=%.5f | r_b max=%.4f last=%.4f | cnr last=%.4f | cnr_max=%.4f | ident_n=%d' % (
                            cur_epoch, user_idx, diag.get('_B'), diag.get('_r2', float('nan')), diag.get('_rloc') or float('nan'), max(diag['om_rel_r']), diag['om_rel_r'][-1], diag['om_cnr'][-1], max(diag['om_cnr']), diag.get('om_ident_n', 0)), flush=True)
                        diag_summary['[OM] R_loc'] = float(diag.get('_rloc') or float('nan'))
                        if diag.get('om_audit_n'):
                            print('[OMEGA] rsloc AUDIT n=%d fail=%d | min cos_dir=%.10f | max mag_rel=%.3e | rloc=r2*sqrt(B) check=%s'
                                  % (diag['om_audit_n'], diag.get('om_audit_fail', 0), diag.get('om_audit_cos', float('nan')),
                                     diag.get('om_audit_mag', float('nan')), 'on' if diag.get('_r2') is not None else 'off'), flush=True)
                            diag_summary['[OM] audit fail'] = float(diag.get('om_audit_fail', 0))
                    else:
                        print('[OMEGA] rspring r=%s user=%s R*=%.4f | r_b max=%.4f last=%.4f | cnr last=%.4f' % (cur_epoch, user_idx, float(args._omega_rstar), max(diag['om_rel_r']), diag['om_rel_r'][-1], diag['om_cnr'][-1]), flush=True)
                    diag_summary['[OM] r_b max'] = float(max(diag['om_rel_r']))
                diag_summary['[OM] lam_b med'] = float(sorted(diag['om_lam_b'])[len(diag['om_lam_b']) // 2]); diag_summary['[OM] lam_b max'] = float(max(diag['om_lam_b']))
            for _k in ('cov_omega', 'cov_g', 'corr_log', 'leak_frac', 'cnr_hi', 'cnr_lo', 'flip_hi', 'flip_lo', 'omax', 'corr_top1_mass', 'spearman_3a3b'):
                if diag.get('om_' + _k + '_last'):
                    diag_summary['[OM] ' + _k + ' last'] = float(diag['om_' + _k + '_last'][-1])
        if mode == 'target':
            from train.hd_target import target_log_round
            diag_summary.update(target_log_round(diag, cur_epoch, user_idx, args))
        if diag.get('lips'):
            _lv = diag['lips']
            print('[LIPS] r=%s user=%s rho_probe=%g b=[%s] Lhat=[%s] | min/max=%.4e/%.4e n=%d'
                  % (cur_epoch, user_idx, float(getattr(args, 'srv_lips_rho', 0.0)), ' '.join(str(b) for b in diag['lips_b']),
                     ' '.join('%.3e' % v for v in _lv), min(_lv), max(_lv), len(_lv)), flush=True)
            diag_summary['[LIPS] Lhat min'] = float(min(_lv)); diag_summary['[LIPS] Lhat max'] = float(max(_lv))
            diag.pop('lips', None); diag.pop('lips_b', None)
        return net_c.state_dict(), self.args, diag_summary
