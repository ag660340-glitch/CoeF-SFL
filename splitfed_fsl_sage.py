import gc
import time
import copy
from utils.sflv import new_locals as _new_locals, sd_keep as _sd_keep, srv_copy as _srv_copy, srv_agg as _srv_agg, srv_agg_sd as _srv_agg_sd, tag as _sflv_tag
import numpy as np
import torch
import wandb

import utils.comm_meter as _CM
from train.train_fl import Localupdate_sfl_client_vanilla, test_img_sfl, train_metric_post
from train.train_fl_fsl_sage import _label_kind, Localupdate_fsl_sage_client, SageAuxState
from train.avg import FedAvg, FedAvg_auxnet
from train.model_assign import SFL_acc_model_assignment_homo
from data.dataset import load_data
from utils.utils import metric_name
from utils.ckpt import maybe_save, maybe_resume
from utils.lr_sched import apply_lr

_LLM_SET = ('RoBerta', 'RoBerta_large', 'DistilRoBerta', 'DistilBert', 'GPT2', 'Qwen')


def _sage_client_test(net_c, aux_proto, sage_states, cids, datatest, args, rnd, acc_s):
    from train.train_fl import eval_client_heads
    auxs = []
    for cid in cids:
        st = sage_states.get(int(cid))
        if st is None:
            continue
        aux = copy.deepcopy(aux_proto); aux.load_state_dict(st.aux_sd); aux.to(args.device); aux.eval()
        auxs.append((int(cid), aux))
    if not auxs:
        return None
    try:
        res = eval_client_heads(net_c, [('cid%d' % c, a) for c, a in auxs], datatest, args)
    finally:
        for _, a in auxs:
            del a
    n = len(res)
    mean_p = sum(d['primary'] for d in res) / n; mean_l = sum(d['loss'] for d in res) / n
    keys = [k for k in res[0] if k not in ('primary', 'loss') and isinstance(res[0][k], (int, float))]
    means = {k: sum(float(d[k]) for d in res) / n for k in keys}
    wandb.log(dict({"[Test] Client {} {}".format(args.cut_point, metric_name()): mean_p,
                    "[Test] Client {} loss".format(args.cut_point): mean_l},
                   **{"[Test] Client {} {}".format(args.cut_point, k): v for k, v in means.items()}), step=rnd)
    per = ' '.join('%d:%.2f' % (c, d['primary']) for (c, _), d in zip(auxs, res))
    extra = ' '.join('%s=%.2f' % (k, means[k]) for k in ('EM', 'F1', 'mismatched', 'mcc', 'f1', 'spearman', 'BLEU', 'PPL') if k in means)
    return ('[Test/SAGE] r=%d client-aux %s=%.2f loss=%.4f%s (mean over %d participating clients, per-client aux | per-client %s) | server=%.2f'
            % (rnd, metric_name(), mean_p, mean_l, (' ' + extra) if extra else '', n, per, float(acc_s)))


def main_sfl_fsl_sage(args):
    dataset_train, dataset_test, dict_users, args.num_classes = load_data(args)

    local_cmodels, local_smodels, auxiliary_models, args.cut_point = SFL_acc_model_assignment_homo(
        args.model_name, args.num_classes, args.cut_point, args.device)
    args.num_models = args.cut_point
    net_glob_client = copy.deepcopy(local_cmodels[0])
    net_glob_server = copy.deepcopy(local_smodels[0])
    from train.train_fl_cse_fsl import build_cmp_aux
    _aux0, _ = build_cmp_aux(args, net_glob_client, net_glob_server, auxiliary_models[0], args.num_classes, tag='SAGE')
    aux_proto = copy.deepcopy(_aux0)

    acc_test_total_s = []
    program = args.name
    print(program)

    _warmup_epochs = int(getattr(args, 'fd_warmup_epochs', 0))
    Q = int(getattr(args, 'cse_server_interval', 5)); L = int(getattr(args, 'sage_align_interval', 10))
    print(f"[FSL-SAGE] upload={getattr(args,'sage_upload','round_init')} (round_init= 1, Q) "
          f"refresh_chunk={int(getattr(args,'sage_refresh_chunk',0) or 0)}", flush=True)
    print(f"[FSL-SAGE] Q={Q} align_interval l={L} align_mode={getattr(args,'sage_align_mode','elapsed')} "
          f"align_epochs={getattr(args,'sage_align_epochs',100)} align_bs={getattr(args,'sage_align_bs',1000)} "
          f"max_data={getattr(args,'sage_max_data',1000)} lazy_until={getattr(args,'sage_lazy_until',0)} bootstrap={getattr(args,'sage_bootstrap','cse')} "
          f"aux_agg={bool(getattr(args,'sage_aux_agg',False))} aux_type={getattr(args,'aux_type','mlp')} "
          f"warmup_epochs={_warmup_epochs}| SFLv1 ( FedAvg)")

    _start_round = maybe_resume(args, net_glob_client, net_glob_server, aux=None)
    if _start_round > 1:
        print(f'[CKPT] r{_start_round} ~ r{args.epochs} ', flush=True)
    _is_llm = args.model_name in _LLM_SET
    sage_states = {}

    for iter in range(_start_round, args.epochs + 1):
        t_epoch = time.time()
        is_phase1 = (_warmup_epochs > 0 and iter <= _warmup_epochs)
        phase_tag = "VANILLA" if is_phase1 else "FSL-SAGE"
        if iter == 1:
            print(f"[Driver] [Round {iter}] PHASE 1 START ({phase_tag}, ~ epoch {_warmup_epochs})")
        elif iter == _warmup_epochs + 1 and _warmup_epochs > 0:
            print(f"[Driver] [Round {iter}] PHASE 2 START (VANILLA → FSL-SAGE)")

        apply_lr(args, iter, warmup_epochs=_warmup_epochs, tag='SAGE')

        w_locals_c, w_locals_s, w_locals_a, part_cids = _new_locals(args), _new_locals(args), [], []
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

        for idx in idxs_users:
            t_total = time.time()
            _mn = metric_name()
            if is_phase1:
                _CM.reset()
                local_c = Localupdate_sfl_client_vanilla(
                    args, dataset=dataset_train, idxs=dict_users[idx], wandb=wandb, model_idx=0)
                _nc_upd = copy.deepcopy(w_glob_client); _ns_upd = _srv_copy(args, w_glob_server)
                weight_c, weight_s, _, loss_s, acc_s = local_c.train_client(net_client=_nc_upd, net_server=_ns_upd)
                loss_s, acc_s = train_metric_post(_nc_upd, _ns_upd, local_c.ldr_train, args, _is_llm)
                w_locals_c.append([_sd_keep(args, weight_c), 0])
                w_locals_s.append([_sd_keep(args, weight_s), 0])
                del _nc_upd, _ns_upd
                _CM.report(iter, phase_tag, user=idx)
                print('[Epoch : {}][User {} VANILLA(warmup)] [S_Loss  {:.3f} | S_{} {:.3f}]'
                      .format(iter, idx, loss_s, _mn, acc_s))
                wandb.log({"[Train] Server {} loss".format(args.cut_point): loss_s,
                           "[Train] Server {} {}".format(args.cut_point, _mn): acc_s}, step=iter)
                continue

            _CM.reset()
            cid = int(idx)
            if cid not in sage_states:
                sage_states[cid] = SageAuxState(aux_proto.state_dict(), int(getattr(args, 'sage_max_data', 1000)), str(getattr(args, 'sage_buffer', 'batch')), _label_kind(args))
            st_obj = sage_states[cid]
            local_c = Localupdate_fsl_sage_client(
                args, dataset=dataset_train, idxs=dict_users[idx], wandb=wandb, model_idx=0)
            _nc_upd = copy.deepcopy(w_glob_client); _ns_upd = _srv_copy(args, w_glob_server)
            t0 = time.time()
            weight_c, weight_s, weight_a, st = local_c.train(
                net_client=_nc_upd, net_server=_ns_upd, net_ax=aux_proto, state=st_obj, rnd=iter, cid=cid)
            t_train = time.time() - t0
            loss_s, acc_s = train_metric_post(_nc_upd, _ns_upd, local_c.ldr_train, args, _is_llm)

            w_locals_c.append([_sd_keep(args, weight_c), 0])
            w_locals_s.append([_sd_keep(args, weight_s), 0])
            w_locals_a.append(weight_a); part_cids.append(cid)
            del _nc_upd, _ns_upd, weight_c, weight_s, local_c
            gc.collect()

            _CM.report(iter, phase_tag, user=idx, batches=st['n_batches'])
            print(f'[Time] User {idx} | Total {time.time()-t_total:.1f}s | Train {t_train:.1f}s '
                  f'| srv_up {st["n_up"]}/{st["n_batches"]} ({"round_init" if st.get("upload")=="round_init" else "Q=%d" % Q})')
            print(f'[SAGE] r={iter} cli={cid} mode={st["mode"]} aligned={st["aligned"]} '
                  f'align_loss={st["align_loss"]:.4e} n_align_samples={st["n_align_samples"]} '
                  f'align_set={st["n_data"]} n_align_total={st_obj.n_align} part={st_obj.n_part}', flush=True)
            print('[Epoch : {}][User {} with cut_point {}] [C_Loss  {:.3f} | C_{} {:.3f}] [S_Loss  {:.3f} | S_{} {:.3f}]'
                  .format(iter, idx, args.cut_point, st['loss_c'], _mn, st['acc_c'], loss_s, _mn, acc_s))
            wandb.log({"[Train] Client {} loss".format(args.cut_point): st['loss_c'],
                       "[Train] Client {} {}".format(args.cut_point, _mn): st['acc_c'],
                       "[Train] Server {} loss".format(args.cut_point): loss_s,
                       "[Train] Server {} {}".format(args.cut_point, _mn): acc_s,
                       "[SAGE] align_loss": (st['align_loss'] if st['aligned'] else float('nan'))}, step=iter)

        t0 = time.time()
        net_glob_client.load_state_dict(FedAvg(w_locals_c))
        _srv_agg(args, net_glob_server, w_locals_s, w_glob_server)
        if (not is_phase1) and bool(getattr(args, 'sage_aux_agg', False)) and w_locals_a:
            _agg = FedAvg_auxnet(w_locals_a, [aux_proto])[0]
            aux_proto.load_state_dict(_agg.state_dict())
            _sd = {k: v.detach().cpu().clone() for k, v in _agg.state_dict().items()}
            for cid in part_cids:
                sage_states[cid].aux_sd = copy.deepcopy(_sd)
        t_agg = time.time() - t0
        del w_locals_c, w_locals_s, w_locals_a, w_glob_client, w_glob_server
        gc.collect(); torch.cuda.empty_cache()
        print(f'[Time] aggregation {t_agg:1f}s |')

        _ev_p = max(1, int(getattr(args, 'hd_eval_every', 10)))
        _ev_w = int(getattr(args, 'fd_warmup_epochs', 0))
        if (iter <= _ev_w) or (iter % _ev_p == 0) or (iter == int(args.epochs) - 1):
            t0 = time.time()
            acc_test_s, loss_test_s = test_img_sfl(net_glob_client, net_glob_server, dataset_test, args)

            _ct_line = (_sage_client_test(net_glob_client, aux_proto, sage_states, part_cids, dataset_test, args, iter, acc_test_s)
                        if not is_phase1 else None)
            print(f'[Time] Epoch {iter} | Split {args.cut_point} | Eval {time.time()-t0:.1f}s | '
                  f'Agg {t_agg:.1f}s | Cumulative Epoch Total {time.time()-t_epoch:.1f}s')
            print("[Epoch {}]Testing accuracy with split point {} :  {:.2f}] ".format(iter, args.cut_point, acc_test_s))
            if _ct_line:
                print(_ct_line, flush=True)
            wandb.log({"[Test] Server {} acc".format(args.cut_point): float(acc_test_s),
                       "[Test] Server {} loss".format(args.cut_point): float(loss_test_s)}, step=iter)
            acc_test_total_s.append(float(acc_test_s))

        maybe_save(args, iter, net_glob_client, net_glob_server, aux=None, optimizers=None)
        if iter % 50 == 0:
            print(program)

    print("finish")
    return acc_test_total_s
