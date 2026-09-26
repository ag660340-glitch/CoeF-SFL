import gc
import time
import copy
from utils.sflv import new_locals as _new_locals, sd_keep as _sd_keep, srv_copy as _srv_copy, srv_agg as _srv_agg, srv_agg_sd as _srv_agg_sd, tag as _sflv_tag
import numpy as np
import torch
import wandb

import utils.comm_meter as _CM
from train.train_fl import Localupdate_sfl_client_vanilla, test_img_sfl, train_metric_post
from train.train_fl_mu_splitfed import Localupdate_mu_splitfed_client
from train.avg import FedAvg
from train.model_assign import SFL_model_assignment_homo
from data.dataset import load_data
from utils.utils import metric_name
from utils.ckpt import maybe_save, maybe_resume
from utils.lr_sched import apply_lr

_LLM_SET = ('RoBerta', 'RoBerta_large', 'DistilRoBerta', 'DistilBert', 'GPT2', 'Qwen')


def _global_step(net_glob, w_avg, eta):
    if float(eta) == 1.0:
        net_glob.load_state_dict(w_avg); return
    sd = net_glob.state_dict(); new = {}
    for k, v in sd.items():
        a = w_avg[k].to(v.device)
        new[k] = (v + float(eta) * (a - v)) if v.is_floating_point() else a
    net_glob.load_state_dict(new)


def main_sfl_mu_splitfed(args):
    dataset_train, dataset_test, dict_users, args.num_classes = load_data(args)

    local_cmodels, local_smodels, args.cut_point = SFL_model_assignment_homo(
        args.model_name, args.num_classes, args.cut_point, args.device)
    net_glob_client = copy.deepcopy(local_cmodels[0])
    net_glob_server = copy.deepcopy(local_smodels[0])

    acc_test_total_s = []
    program = args.name
    print(program)

    _warmup_epochs = int(getattr(args, 'fd_warmup_epochs', 0))
    eta_g = float(getattr(args, 'mu_global_lr', 1.0))
    if bool(getattr(args, 'mu_lowfreq', False)):
        _P = max(1, int(getattr(args, 'mu_num_pert', 1))); _tau = max(1, int(getattr(args, 'mu_tau', 2)))
        print(f"[MU-LOWFREQ]: (1+2P) B (P={_P}) 1 τ={_tau} ZO (τ B) P B 1 ZO 0| "
              f"[FLOPS-MU] forward: client={1+2*_P}(+1) server={2*_P*_tau+2*_P} backward=0 (per-sample fwd FLOPs [FLOPS]/[FLOPS-V])", flush=True)
    print(f"[MU-SPLITFED] tau={getattr(args,'mu_tau',2)} lambda={getattr(args,'mu_mu',5e-3)} "
          f"P={getattr(args,'mu_num_pert',1)} dist={getattr(args,'mu_pert_dist','gauss')} "
          f"lr_s={getattr(args,'mu_lr_s',0.0) or 'args.lr'} lr_c={getattr(args,'mu_lr_c',0.0) or 'args.lr'} "
          f"global_lr={eta_g} joint_final={bool(getattr(args,'mu_joint_final',False))} "
          f"eval_forward={bool(getattr(args,'mu_eval_forward',False))} warmup_epochs={_warmup_epochs} "
          f"| SFLv1 ")

    _start_round = maybe_resume(args, net_glob_client, net_glob_server, aux=None)
    if _start_round > 1:
        print(f'[CKPT] r{_start_round} ~ r{args.epochs} ', flush=True)
    _is_llm = args.model_name in _LLM_SET

    _mu_evals = {}
    for iter in range(_start_round, args.epochs + 1):
        t_epoch = time.time()
        is_phase1 = (_warmup_epochs > 0 and iter <= _warmup_epochs)
        phase_tag = "VANILLA" if is_phase1 else "MU-SPLITFED"
        if iter == 1:
            print(f"[Driver] [Round {iter}] PHASE 1 START ({phase_tag}, ~ epoch {_warmup_epochs})")
        elif iter == _warmup_epochs + 1 and _warmup_epochs > 0:
            print(f"[Driver] [Round {iter}] PHASE 2 START (VANILLA → MU-SPLITFED)")

        apply_lr(args, iter, warmup_epochs=_warmup_epochs, tag='MU')

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
            local_c = Localupdate_mu_splitfed_client(
                args, dataset=dataset_train, idxs=dict_users[idx], wandb=wandb, model_idx=0)
            _nc_upd = copy.deepcopy(w_glob_client); _ns_upd = _srv_copy(args, w_glob_server)
            t0 = time.time()
            weight_c, weight_s, st = local_c.train(net_client=_nc_upd, net_server=_ns_upd, rnd=iter, cid=int(idx))
            t_train = time.time() - t0
            loss_s, acc_s = train_metric_post(_nc_upd, _ns_upd, local_c.ldr_train, args, _is_llm)

            w_locals_c.append([_sd_keep(args, weight_c), 0])
            w_locals_s.append([_sd_keep(args, weight_s), 0])
            del _nc_upd, _ns_upd, weight_c, weight_s, local_c
            gc.collect()

            _CM.report(iter, phase_tag, user=idx, batches=st['n_batches'])
            print(f'[Time] User {idx} | Total {time.time()-t_total:.1f}s | Train {t_train:.1f}s '
                  f'| srv_zo_steps {st["n_srv_steps"]} (tau={st["tau"]}, P={st["P"]}) '
                  f'| zo_loss_proxy {st["loss_s"]:.3f}')
            print('[Epoch : {}][User {} with cut_point {}] [S_Loss  {:.3f} | S_{} {:.3f}]'
                  .format(iter, idx, args.cut_point, loss_s, _mn, acc_s))
            wandb.log({"[Train] Server {} loss".format(args.cut_point): loss_s,
                       "[Train] Server {} {}".format(args.cut_point, _mn): acc_s}, step=iter)

        t0 = time.time()
        if is_phase1:
            net_glob_client.load_state_dict(FedAvg(w_locals_c))
            _srv_agg(args, net_glob_server, w_locals_s, w_glob_server)
        else:
            _global_step(net_glob_client, FedAvg(w_locals_c), eta_g)
            _global_step(net_glob_server, _srv_agg_sd(args, w_locals_s, w_glob_server), eta_g)
        t_agg = time.time() - t0
        del w_locals_c, w_locals_s, w_glob_client, w_glob_server
        gc.collect(); torch.cuda.empty_cache()
        print(f'[Time] aggregation {t_agg:1f}s |')

        _ev_p = max(1, int(getattr(args, 'hd_eval_every', 10)))
        _ev_w = int(getattr(args, 'fd_warmup_epochs', 0))
        if (iter <= _ev_w) or (iter % _ev_p == 0) or (iter == int(args.epochs) - 1):
            t0 = time.time()
            acc_test_s, loss_test_s = test_img_sfl(net_glob_client, net_glob_server, dataset_test, args)
            print(f'[Time] Epoch {iter} | Split {args.cut_point} | Eval {time.time()-t0:.1f}s | '
                  f'Agg {t_agg:.1f}s | Cumulative Epoch Total {time.time()-t_epoch:.1f}s')
            print("[Epoch {}]Testing accuracy with split point {} :  {:.2f}] ".format(iter, args.cut_point, acc_test_s))
            wandb.log({"[Test] Server {} acc".format(args.cut_point): float(acc_test_s),
                       "[Test] Server {} loss".format(args.cut_point): float(loss_test_s)}, step=iter)
            acc_test_total_s.append(float(acc_test_s))
            _mu_evals[iter] = round(float(acc_test_s), 2)

            _ps = int(getattr(args, 'mu_plateau_stop', 0) or 0)
            _rr = [iter - 2 * _ev_p, iter - _ev_p, iter]
            if _ps > 0 and iter >= _ps and iter % _ev_p == 0 and all(r in _mu_evals for r in _rr) and len({_mu_evals[r] for r in _rr}) == 1:
                print(f'[PLATEAU-STOP] r{iter}: test metric identical at r{_rr[0]}, r{_rr[1]}, r{_rr[2]} ({_mu_evals[iter]:.2f}) -> early stop, final value = r{iter}', flush=True)
                break

        maybe_save(args, iter, net_glob_client, net_glob_server, aux=None, optimizers=None)
        if iter % 50 == 0:
            print(program)

    print("finish")
    return acc_test_total_s
