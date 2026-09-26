import gc
import time
import copy
from utils.sflv import new_locals as _new_locals, sd_keep as _sd_keep, srv_copy as _srv_copy, srv_agg as _srv_agg, srv_agg_sd as _srv_agg_sd, tag as _sflv_tag

import numpy as np
import torch
import wandb

import utils.comm_meter as _CM
from train.train_fl import Localupdate_sfl_client_vanilla, Localupdate_sfl_server, test_img_sfl, train_metric_post
from train.train_fl_hess_diag import Localupdate_hess_diag_client, mask_sync_on, masked_rng, _cache_on_cpu
from train.hd_kprobe import server_kappa_probe
from train.avg import FedAvg
from train.model_assign import SFL_model_assignment_homo
from data.dataset import load_data
from utils.utils import metric_name
from utils.ckpt import maybe_save, maybe_resume
from utils.lr_sched import apply_lr

_LLM_SET = ('RoBerta', 'RoBerta_large', 'DistilRoBerta', 'DistilBert', 'GPT2', 'Qwen')


def kprobe_round_initial(net_server, smashed_data_list, label_list, args, mask_list=None, rnd=None, cid=None):
    if mask_sync_on(args):
        net_server.train()
    else:
        (net_server.eval() if getattr(args, 'hd_eval_exchange', False) else net_server.train())
    is_llm = args.model_name in _LLM_SET
    _oc = _cache_on_cpu(args)
    g_list, k_list = [], []
    n_neg = n_tau = n_tot = 0
    tau = float(getattr(args, 'hd_kprobe_gnorm_tau', 1e-8))
    for i in range(len(smashed_data_list)):
        s0 = smashed_data_list[i].to(args.device).detach()
        y = label_list[i].to(args.device)
        m = mask_list[i].to(args.device) if is_llm else None
        with masked_rng(args, rnd, cid, i, args.device):
            if bool(getattr(args, 'instr_q0', True)):
                from train.hd_instr import q0_capture as _q0c
                if i == 0:
                    args._q0_ref = {}
                args._q0_ref[i] = _q0c(net_server, args.device)
            g, k, kraw = server_kappa_probe(net_server, s0, y, m, is_llm, args, return_raw=True)
        if i == 0:
            args._dca_r0 = []; args._dca_r0vec = []; args._dca_p = []
        args._dca_r0.append(getattr(args, '_dca_r0_last', None))
        args._dca_r0vec.append(getattr(args, '_dca_r0vec_last', None)); args._dca_p.append(getattr(args, '_dca_p_last', None))
        _CM.down(g, msgs=0)
        _CM.down(k, msgs=0)
        n_neg += int((kraw < 0).sum()); n_tot += int(kraw.numel())
        n_tau += int((g.reshape(g.size(0), -1).norm(dim=1) < tau).sum())
        k2 = torch.stack([k, kraw], 0).detach()
        g_list.append(g.detach().cpu() if _oc else g.detach())
        k_list.append(k2.cpu() if _oc else k2)
        net_server.zero_grad()
        del s0, g, k, kraw, k2
    return g_list, k_list, dict(neg=n_neg, tau=n_tau, tot=n_tot)


def main_sfl_kprobe(args):
    args.hd_mode = 'kprobe'
    dataset_train, dataset_test, dict_users, args.num_classes = load_data(args)

    local_cmodels, local_smodels, args.cut_point = SFL_model_assignment_homo(
        args.model_name, args.num_classes, args.cut_point, args.device)
    net_glob_client = copy.deepcopy(local_cmodels[0])
    net_glob_server = copy.deepcopy(local_smodels[0])

    acc_test_total_s = []
    program = args.name
    print(program)

    _warmup_epochs = int(getattr(args, 'fd_warmup_epochs', 0))
    _form = str(getattr(args, 'hd_kprobe_form', 'diag'))
    print(f"[KPROBE] form={_form} alpha={float(getattr(args,'hd_kprobe_alpha',1.0)):g} "
          f"lambda={float(getattr(args,'hd_kprobe_lambda',3000)):g} gate_neg={bool(getattr(args,'hd_kprobe_gate_neg',True))} "
          f"gnorm_tau={float(getattr(args,'hd_kprobe_gnorm_tau',1e-8)):g} hvp_exact={bool(getattr(args,'hd_hvp_exact',True))} "
          f"mask_sync={mask_sync_on(args)} warmup_epochs={_warmup_epochs} (Vanilla SFL during warmup)")

    print("[FLOPS-KPROBE] server round-initial per batch: fwd=1 bwd=2 (g0 backward w/ graph + Pearlmutter HVP) "
          "| stale/fisher-g0 baseline fwd=1 bwd=1 → +1 bwd/batch | client extra ≈ 2 elementwise ops/batch (≈0)", flush=True)

    _start_round = maybe_resume(args, net_glob_client, net_glob_server, aux=None)
    if _start_round > 1:
        print(f'[CKPT] r{_start_round} ~ r{args.epochs} ', flush=True)
    for iter in range(_start_round, args.epochs + 1):

        args._cur_round = iter
        args._srv_jitter_on = (iter > int(getattr(args, 'fd_warmup_epochs', 0) or 0))
        t_epoch = time.time()
        is_phase1 = (_warmup_epochs > 0 and iter <= _warmup_epochs)
        phase_tag = "VANILLA" if is_phase1 else "KPROBE"
        if iter == 1:
            print(f"[Driver] [Round {iter}] PHASE 1 START ({phase_tag}, ~ epoch {_warmup_epochs})")
        elif iter == _warmup_epochs + 1 and _warmup_epochs > 0:
            print(f"[Driver] [Round {iter}] PHASE 2 START (VANILLA → KPROBE)")

        apply_lr(args, iter, warmup_epochs=_warmup_epochs, tag='KP')

        w_locals_c, w_locals_s = _new_locals(args), _new_locals(args)
        w_glob_client = copy.deepcopy(net_glob_client)
        w_glob_server = copy.deepcopy(net_glob_server)

        m = max(int(args.frac * args.num_users), 1)
        _ep = int(__import__('os').environ.get('EXPECT_PART', '0') or 0)
        if not globals().get('_PART_LOGGED', False):
            globals()['_PART_LOGGED'] = True
            print('[PART] N=%d frac=%r m=%d expect=%s epoch_per_round=%.4f'
                  % (args.num_users, args.frac, m, _ep or '-', m / max(args.num_users, 1)), flush=True)
        if _ep and m != _ep:
            raise SystemExit('[PART-GATE] participating clients %d != expected %d (N=%d frac=%r); wrong launch arguments, aborting.' % (m, _ep, args.num_users, args.frac))
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        round_diag_acc = {}
        for idx in idxs_users:
            t_total = time.time()
            is_llm = args.model_name in _LLM_SET
            _mn = metric_name()

            if is_phase1:
                _CM.reset()
                local_c = Localupdate_sfl_client_vanilla(
                    args, dataset=dataset_train, idxs=dict_users[idx], wandb=wandb, model_idx=0)
                _nc_upd = copy.deepcopy(w_glob_client); _ns_upd = _srv_copy(args, w_glob_server)
                weight_c, weight_s, _, loss_s, acc_s = local_c.train_client(net_client=_nc_upd, net_server=_ns_upd)
                loss_s, acc_s = train_metric_post(_nc_upd, _ns_upd, local_c.ldr_train, args, is_llm)
                del _nc_upd, _ns_upd
                t_probe = 0.0
            else:

                _CM.reset(); _CM.up(msgs=1); _CM.down(msgs=1)
                local_c = Localupdate_hess_diag_client(
                    args, dataset=dataset_train, idxs=dict_users[idx], wandb=wandb, model_idx=0)
                local_s = Localupdate_sfl_server(
                    args, dataset=dataset_train, idxs=dict_users[idx], wandb=wandb, model_idx=0)

                if is_llm:
                    smashed_data, mask_list, label = local_c.get_smashed_data_llm(
                        net=copy.deepcopy(w_glob_client), rnd=iter, cid=int(idx))
                    cached_inputs = None
                else:
                    smashed_data, label, cached_inputs = local_c.get_smashed_data(
                        net=copy.deepcopy(w_glob_client), rnd=iter, cid=int(idx))
                    if not bool(getattr(args, 'round_input_cache', True)):
                        cached_inputs = None
                    mask_list = None

                t0 = time.time()
                g_initial_list, kappa_list, _kst = kprobe_round_initial(
                    net_server=copy.deepcopy(w_glob_server), smashed_data_list=smashed_data,
                    label_list=label, args=args, mask_list=mask_list, rnd=iter, cid=int(idx))
                t_probe = time.time() - t0
                print(f'[KPROBE-init] r={iter} cli={int(idx)} B={len(kappa_list)} '
                      f'neg_frac={_kst["neg"]/max(1,_kst["tot"]):.3f} gnorm_tau_frac={_kst["tau"]/max(1,_kst["tot"]):.3f} '
                      f'| probe {t_probe:.1f}s', flush=True)

                frozen_server_snapshot = copy.deepcopy(w_glob_server)
                _nc_upd = copy.deepcopy(w_glob_client)
                weight_c, _, diag = local_c.train_hess_diag(
                    net_client=_nc_upd, net_server_frozen=frozen_server_snapshot,
                    cached_x2_list=smashed_data, cached_g_list=g_initial_list,
                    cached_label_list=label, cached_mask_list=mask_list,
                    cached_inputs=cached_inputs,
                    cached_labels=(label if cached_inputs is not None else None),
                    cached_U_list=None, cached_th_list=None,
                    cached_kappa_list=kappa_list,
                    cur_epoch=iter, user_idx=int(idx))

                _ns_upd = _srv_copy(args, w_glob_server)
                weight_s, _, loss_s, acc_s = local_s.train(
                    net_server=_ns_upd, smashed_data=smashed_data, mask_list=mask_list, label_list=label)
                loss_s, acc_s = train_metric_post(_nc_upd, _ns_upd, local_c.ldr_train, args, is_llm)

                for k, v in diag.items():
                    round_diag_acc.setdefault(k, []).append(v)
                del smashed_data, g_initial_list, kappa_list, label, mask_list, cached_inputs
                del frozen_server_snapshot, _nc_upd, _ns_upd, local_c, local_s, diag
                gc.collect()

            w_locals_c.append([_sd_keep(args, weight_c), 0])
            w_locals_s.append([_sd_keep(args, weight_s), 0])
            del weight_c, weight_s

            _CM.report(iter, phase_tag, user=idx)
            print(f'[{phase_tag}] User {idx} | Total {time.time()-t_total:.1f}s | Probe {t_probe:.1f}s | '
                  f'S_Loss {loss_s:.3f} | S_{_mn} {acc_s:.3f}')
            wandb.log({f'[Train] Server {args.cut_point} loss': loss_s,
                       f'[Train] Server {args.cut_point} {_mn}': acc_s}, step=iter)

        t0_agg = time.time()
        net_glob_client.load_state_dict(FedAvg(w_locals_c))
        _srv_agg(args, net_glob_server, w_locals_s, w_glob_server)
        t_agg = time.time() - t0_agg
        del w_locals_c, w_locals_s, w_glob_client, w_glob_server
        gc.collect(); torch.cuda.empty_cache()

        if round_diag_acc and float(np.sum(round_diag_acc.get('[HD] n_diag', [0]))) > 0:
            diag_log = {f'[Round] {k}': float(np.nanmean(round_diag_acc[k])) for k in round_diag_acc}
            wandb.log(diag_log, step=iter)
            print('[HD-Diag] ' + ' | '.join(f'{k}={v:.4f}' for k, v in diag_log.items()))

        wandb.log({'[Phase] is_phase1': 1.0 if is_phase1 else 0.0,
                   '[Phase] is_phase2': 0.0 if is_phase1 else 1.0}, step=iter)
        print(f'[Time] Phase={phase_tag} | Epoch {iter} | Agg {t_agg:.1f}s | Total {time.time()-t_epoch:.1f}s')

        _ev_p = max(1, int(getattr(args, 'hd_eval_every', 10)))
        _ev_w = int(getattr(args, 'fd_warmup_epochs', 0))
        if (iter <= _ev_w) or (iter % _ev_p == 0) or (iter == int(args.epochs) - 1):
            t0_eval = time.time()
            acc_test_s, loss_test_s = test_img_sfl(net_glob_client, net_glob_server, dataset_test, args)
            print(f'[Time] Epoch {iter} | Eval {time.time()-t0_eval:.1f}s')
            print(f'[Epoch {iter}] ({phase_tag}) Test acc (cut {args.cut_point}): {acc_test_s:.2f}')
            print("[Epoch {}]Testing accuracy with split point {} :  {:.2f}] ".format(iter, args.cut_point, acc_test_s))
            wandb.log({f'[Test] Server {args.cut_point} acc': float(acc_test_s),
                       f'[Test] Server {args.cut_point} loss': float(loss_test_s)}, step=iter)
            acc_test_total_s.append(float(acc_test_s))

        maybe_save(args, iter, net_glob_client, net_glob_server, aux=None, optimizers=None)
        if iter % 50 == 0:
            print(program)

    print('finish')
    return acc_test_total_s
