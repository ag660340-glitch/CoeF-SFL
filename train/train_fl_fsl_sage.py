import copy
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import utils.comm_meter as _CM
from train import task_loss as _TL
from train.train_fl import _dl_kwargs
from train.train_fl_cse_fsl import _unpack_batch, _client_fwd, _head_fwd, _LLM_SET
from train.train_fl_hess_diag import masked_rng, mask_sync_on, _cache_on_cpu
from data.dataset import DatasetSplit
from utils.utils import calculate_accuracy


class SageAuxState(object):

    def __init__(self, aux_sd, cap, buffer='batch', kind='cls'):
        self.aux_sd = {k: v.detach().cpu().clone() for k, v in aux_sd.items()}
        self.cap = int(cap)
        self.buffer = buffer
        self.kind = kind
        self.ax, self.ay, self.am = [], [], []
        self.n_data = 0
        self.n_align = 0
        self.last_align_round = None
        self.n_part = 0
        self.last_align_loss = float('nan')
        self.opt_sd = None
        self.opt_step = 0

    def add(self, a, y, m):
        self.ax.append(a.detach().cpu().clone())
        self.ay.append(y.detach().cpu().clone())
        self.am.append(m.detach().cpu().clone() if m is not None else None)
        self.n_data += int(a.size(0))
        if self.buffer == 'sample':

            while self.n_data > self.cap and self.ax:
                excess = self.n_data - self.cap
                a0, y0, m0 = self.ax[0], self.ay[0], self.am[0]; B0 = int(a0.size(0))
                if excess >= B0:
                    self.ax.pop(0); self.ay.pop(0); self.am.pop(0); self.n_data -= B0
                else:
                    self.ax[0] = a0[excess:]; self.ay[0] = _slice_y(y0, B0, excess, B0, self.kind)
                    self.am[0] = (m0[excess:] if m0 is not None else None); self.n_data -= excess
            return
        while self.n_data > self.cap and len(self.ax) > 1:
            self.n_data -= int(self.ax[0].size(0))
            self.ax.pop(0); self.ay.pop(0); self.am.pop(0)


def aux_grad_estimate(aux, a, y, ext, is_llm, crit):
    a_ = a.detach().clone().requires_grad_(True)
    logits = _head_fwd(aux, a_, ext, is_llm)
    loss = crit(logits, y)
    g = torch.autograd.grad(loss, a_)[0]
    return g.detach(), loss.detach(), logits.detach()


def _label_kind(args):
    t = _TL.task_type(args)
    return 'qa' if t == 'qa' else ('lm' if t == 'lm' else 'cls')


def _slice_y(y, B, s, e, kind):
    if kind == 'qa' and y.numel() == 2 * B:
        return torch.cat([y[s:e], y[B + s:B + e]], 0)
    if kind == 'lm' and B > 0 and y.numel() != B and y.numel() % B == 0:
        w = y.numel() // B
        return y.reshape(B, w)[s:e].reshape(-1)
    return y[s:e]


def _align_batches(state, args, is_llm, dev):
    if is_llm:

        bs = max(1, int(getattr(args, 'sage_align_bs', 8)))
        out = []

        _kind = _label_kind(args)
        for a, y, m in zip(state.ax, state.ay, state.am):
            B = a.size(0)
            for s in range(0, B, bs):
                e = min(B, s + bs)
                ys = _slice_y(y, B, s, e, _kind)
                out.append((a[s:e].to(dev), ys.to(dev), (m[s:e].to(dev) if m is not None else None)))
        return out
    A = torch.cat(state.ax, 0); Y = torch.cat(state.ay, 0)
    bs = max(2, int(getattr(args, 'sage_align_bs', 1000)))
    out = []
    for s in range(0, A.size(0), bs):
        a, y = A[s:s + bs], Y[s:s + bs]
        if a.size(0) < 2:
            continue
        out.append((a.to(dev), y.to(dev), None))
    return out


def align_aux(aux, net_server, state, args, is_llm):
    dev = args.device
    crit = _TL.GlobalCriterion()
    batches = _align_batches(state, args, is_llm, dev)
    if not batches:
        return float('nan'), 0

    _ws = net_server.training
    net_server.to(dev)
    (net_server.eval() if bool(getattr(args, 'sage_refresh_eval', True)) else net_server.train())
    targets = []
    _ch = int(getattr(args, 'sage_refresh_chunk', 0) or 0)
    _ckind = _label_kind(args)
    def _chunks(a, y, m):
        B = a.size(0)
        if _ch <= 0 or _ch >= B:
            yield 0, B, a, y, m
        else:
            for s in range(0, B, _ch):
                e = min(B, s + _ch)
                yield s, e, a[s:e], _slice_y(y, B, s, e, _ckind), (m[s:e] if m is not None else None)
    for a, y, m in batches:
        B = a.size(0); zs = []
        for s, e, ac, yc, mc in _chunks(a, y, m):
            a_ = ac.clone().requires_grad_(True)
            loss = crit(_head_fwd(net_server, a_, mc, is_llm), yc)
            zs.append(torch.autograd.grad(loss, a_)[0].detach() * (float(e - s) / B))
        targets.append(torch.cat(zs, 0))
    net_server.zero_grad()
    net_server.train(_ws)

    aux.to(dev); aux.train()

    _oname = str(getattr(args, 'sage_align_opt', 'adam'))
    _okw = dict(lr=float(getattr(args, 'sage_align_lr', 1e-3)),
                eps=float(getattr(args, 'sage_align_eps', 1e-8)), betas=(0.9, 0.999))
    opt = (torch.optim.AdamW(aux.parameters(), weight_decay=float(getattr(args, 'sage_align_wd', 0.0)), **_okw)
           if _oname == 'adamw' else
           torch.optim.Adam(aux.parameters(), weight_decay=float(getattr(args, 'sage_align_wd', 0.0)), **_okw))
    if getattr(state, 'opt_sd', None) is not None:
        try:
            opt.load_state_dict(state.opt_sd)
        except Exception as _e:
            print(f'[SAGE] aux optimizer →: {_e!r}', flush=True)
    n_ep = int(getattr(args, 'sage_align_epochs', 100))

    _wu = int(getattr(args, 'sage_align_warmup', 0))
    _base_lr = float(getattr(args, 'sage_align_lr', 1e-3))
    _gstep = int(getattr(state, 'opt_step', 0) or 0)
    order = list(range(len(batches)))
    last = float('nan')
    for _ in range(n_ep):
        random.shuffle(order)
        for j in order:
            if _wu > 0:
                for _g in opt.param_groups:
                    _g['lr'] = _base_lr * min(1.0, (_gstep + 1) / float(_wu))
            _gstep += 1
            a, y, m = batches[j]
            B = a.size(0); opt.zero_grad(); acc_loss = 0.0
            for s, e, ac, yc, mc in _chunks(a, y, m):
                a_ = ac.clone().requires_grad_(True)
                out = crit(_head_fwd(aux, a_, mc, is_llm), yc)
                g_hat = torch.autograd.grad(out, a_, create_graph=True)[0] * (float(e - s) / B)
                loss = F.mse_loss(g_hat, targets[j][s:e], reduction='sum')
                loss.backward(); acc_loss += float(loss.item())
            opt.step()
            last = acc_loss
    aux.zero_grad()
    state.opt_step = _gstep
    state.opt_sd = {k: (v if not torch.is_tensor(v) else v.detach().cpu()) for k, v in opt.state_dict().items()} if False else _opt_sd_cpu(opt)
    n = sum(int(b[0].size(0)) for b in batches)
    return last, n


def _opt_sd_cpu(opt):
    sd = opt.state_dict(); st = {}
    for k, v in sd['state'].items():
        st[k] = {kk: (vv.detach().cpu() if torch.is_tensor(vv) else vv) for kk, vv in v.items()}
    return {'state': st, 'param_groups': sd['param_groups']}


class Localupdate_fsl_sage_client(object):

    def __init__(self, args, dataset=None, idxs=None, wandb=None, model_idx=None):
        self.args = args
        self.is_llm = args.model_name in _LLM_SET
        collator = getattr(args, "sst2_collator", None) if self.is_llm else None
        DS_alter = DatasetSplit(dataset, idxs)
        random.shuffle(DS_alter.idxs)
        self.ldr_train = DataLoader(DS_alter, batch_size=args.local_bs, shuffle=False,
                                    collate_fn=collator, **_dl_kwargs())
        self.wandb = wandb
        self.model_idx = model_idx

    def _need_align(self, state, rnd):
        if state.n_data == 0:
            return False

        _lazy = int(getattr(self.args, 'sage_lazy_until', 0) or 0)
        if _lazy > 0 and int(rnd) > _lazy:
            return False
        l = max(1, int(getattr(self.args, 'sage_align_interval', 10)))
        if str(getattr(self.args, 'sage_align_mode', 'elapsed')) == 'global':

            _W = int(getattr(self.args, 'fd_warmup_epochs', 0) or 0)
            if bool(getattr(self.args, 'sage_align_zero_based', False)):

                return (rnd - 1 > _W) and ((rnd - 1 - _W) % l == 0)
            return (rnd > _W) and ((rnd - _W) % l == 0)
        if state.n_align == 0:
            return True
        return (rnd - int(state.last_align_round)) >= l

    def train(self, net_client, net_server, net_ax, state, rnd, cid):
        args = self.args
        dev = args.device
        Q = max(1, int(getattr(args, 'cse_server_interval', 5)))
        _ri = (str(getattr(args, 'sage_upload', 'round_init')) == 'round_init')

        net_c = net_client.to(dev); net_c.train()
        net_s = net_server.to(dev); net_s.train()
        aux = copy.deepcopy(net_ax); aux.load_state_dict(state.aux_sd); aux.to(dev); aux.train()

        align_loss, n_align = float('nan'), 0
        if self._need_align(state, rnd):
            align_loss, n_align = align_aux(aux, net_s, state, args, self.is_llm)
            state.n_align += 1
            state.last_align_round = int(rnd)
            state.last_align_loss = align_loss
            _CM.down(*[p.detach() for p in aux.parameters()])

        use_cse = (state.n_align == 0 and str(getattr(args, 'sage_bootstrap', 'cse')) == 'cse')

        opt_c = torch.optim.AdamW(net_c.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=1e-8)
        opt_s = torch.optim.AdamW(net_s.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=1e-8)
        opt_a = (torch.optim.AdamW(aux.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=1e-8)
                 if use_cse else None)
        crit = _TL.GlobalCriterion()

        ep_loss_c, ep_acc_c, ep_loss_s, ep_acc_s = [], [], [], []
        n_batches = len(self.ldr_train)
        n_up = 0

        ri_cache = []
        if _ri:
            _oc = _cache_on_cpu(args)
            if mask_sync_on(args):
                net_c.train()
            else:
                (net_c.eval() if getattr(args, 'hd_eval_exchange', False) else net_c.train())
            _CM.up(msgs=1)
            with torch.no_grad():
                for k, batch in enumerate(self.ldr_train):
                    inp, y = _unpack_batch(batch, self.is_llm, dev)
                    with masked_rng(args, rnd, cid, k, dev):
                        fx0, ext0 = _client_fwd(net_c, inp, self.is_llm)
                    a0 = fx0.detach().clone(); e0 = (ext0.detach().clone() if ext0 is not None else None)
                    _CM.up(a0, y, msgs=0)
                    if k % max(1, int(getattr(args, 'sage_store_interval', 5))) == 0:
                        state.add(a0, y, e0)
                    ri_cache.append(((a0.cpu() if _oc else a0), y, e0))
                    n_up += 1
            net_c.train()
        for ep in range(args.local_ep):
            b_loss_c, b_acc_c, b_loss_s, b_acc_s = [], [], [], []
            for k, batch in enumerate(self.ldr_train):
                local_iter = ep * n_batches + k
                inp, y = _unpack_batch(batch, self.is_llm, dev)

                opt_c.zero_grad(); net_c.zero_grad()
                fx, ext = _client_fwd(net_c, inp, self.is_llm)

                if (not _ri) and local_iter % Q == 0:
                    a_srv = fx.clone().detach()
                    ext_srv = ext.clone().detach() if ext is not None else None
                    _CM.up(a_srv, y)
                    opt_s.zero_grad(); net_s.zero_grad()
                    logits_s = _head_fwd(net_s, a_srv, ext_srv, self.is_llm)
                    loss_s = crit(logits_s, y)
                    loss_s.backward()
                    opt_s.step()
                    b_loss_s.append(loss_s.item()); b_acc_s.append(calculate_accuracy(logits_s, y).item())
                    state.add(a_srv, y, ext_srv)
                    n_up += 1

                if use_cse:
                    a_loc = fx.clone().detach().requires_grad_(True)
                    opt_a.zero_grad(); aux.zero_grad()
                    logits_c = _head_fwd(aux, a_loc, ext, self.is_llm)
                    loss_c = crit(logits_c, y)
                    loss_c.backward()
                    opt_a.step()
                    fx.backward(a_loc.grad)
                    acc_c = calculate_accuracy(logits_c, y)
                else:
                    g_hat, loss_c, logits_c = aux_grad_estimate(aux, fx, y, ext, self.is_llm, crit)
                    fx.backward(g_hat)
                    acc_c = calculate_accuracy(logits_c, y)
                opt_c.step()
                b_loss_c.append(float(loss_c.item())); b_acc_c.append(float(acc_c.item()))

            ep_loss_c.append(sum(b_loss_c) / len(b_loss_c)); ep_acc_c.append(sum(b_acc_c) / len(b_acc_c))
            if _ri:
                net_s.train()
                for a0, y0, e0 in ri_cache:
                    a_srv = a0.to(dev)
                    opt_s.zero_grad(); net_s.zero_grad()
                    logits_s = _head_fwd(net_s, a_srv, e0, self.is_llm)
                    loss_s = crit(logits_s, y0)
                    loss_s.backward(); opt_s.step()
                    b_loss_s.append(loss_s.item()); b_acc_s.append(calculate_accuracy(logits_s, y0).item())
            ep_loss_s.append(sum(b_loss_s) / max(1, len(b_loss_s)))
            ep_acc_s.append(sum(b_acc_s) / max(1, len(b_acc_s)))

        state.aux_sd = {k: v.detach().cpu().clone() for k, v in aux.state_dict().items()}
        state.n_part += 1
        stats = dict(loss_c=sum(ep_loss_c) / len(ep_loss_c), acc_c=ep_acc_c[-1],
                     loss_s=sum(ep_loss_s) / len(ep_loss_s), acc_s=ep_acc_s[-1],
                     n_up=n_up, n_batches=n_batches * args.local_ep,
                     aligned=(n_align > 0), align_loss=align_loss, n_align_samples=n_align,
                     n_data=state.n_data, mode=('cse-boot' if use_cse else 'sage'), upload=('round_init' if _ri else 'interval'))
        return net_c.state_dict(), net_s.state_dict(), copy.deepcopy(state.aux_sd), stats
