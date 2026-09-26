import gc
import utils.comm_meter as _CM
import time
import copy
from utils.sflv import new_locals as _new_locals, sd_keep as _sd_keep, srv_copy as _srv_copy, srv_agg as _srv_agg, srv_agg_sd as _srv_agg_sd, tag as _sflv_tag

import numpy as np
import torch
import wandb

from train.train_fl import (
    Localupdate_sfl_client_vanilla,
    Localupdate_sfl_server,
    test_img_sfl,
)
from train.train_fl import train_metric_post
from train.train_fl_hess_diag import get_round_initial_cache
from train.train_fl_hess_diag import Localupdate_hess_diag_client
from train.avg import FedAvg
from train.model_assign import SFL_model_assignment_homo
from data.dataset import load_data
from utils.utils import metric_name
from utils.ckpt import maybe_save, maybe_resume
from utils.lr_sched import apply_lr


def main_sfl_hess_diag(args):
    dataset_train, dataset_test, dict_users, args.num_classes = load_data(args)

    local_cmodels, local_smodels, args.cut_point = SFL_model_assignment_homo(
        args.model_name, args.num_classes, args.cut_point, args.device)
    net_glob_client = copy.deepcopy(local_cmodels[0])
    net_glob_server = copy.deepcopy(local_smodels[0])

    acc_test_total_s = []
    program = args.name
    print(program)

    _warmup_epochs = int(getattr(args, 'fd_warmup_epochs', 0))
    print(f"[HESS-DIAG] mode={getattr(args,'hd_mode','hybrid')} "
          f"diag={getattr(args,'hd_diag_method','ggn')} m={getattr(args,'hd_basis_m',4)} "
          f"eps={getattr(args,'hd_eps',1e-2)} fp16={bool(getattr(args,'hd_cache_fp16',False))} "
          f"warmup_epochs={_warmup_epochs} (Vanilla SFL during warmup)")

    _start_round = maybe_resume(args, net_glob_client, net_glob_server, aux=None)
    if _start_round > 1:
        print(f'[CKPT] r{_start_round} ~ r{args.epochs} ', flush=True)

    _gas_on = bool(getattr(args, 'gas_lf', False))
    if _gas_on:
        import train.hd_gas as _gas
        from train.task_loss import task_type as _gtt
        assert _gtt(args) == 'cls', '[GAS-LF] classification tasks only (per-label activation statistics)'
        assert bool(getattr(args, 'hd_stale_mode', False)), '[GAS-LF] must be used with --hd_stale_mode (exchange protocol = stale)'
        from data.dataset import _labels_of as _glab
        _gas_stats = _gas.GasStats(int(args.num_classes), float(getattr(args, 'gas_reg', 1e-5)))
        _all_lab = _glab(dataset_train)
        _gidx = lambda v: (v['idxs'] if isinstance(v, dict) else v)
        args._gas_adj = {int(u): _gas.client_adjust(_all_lab[sorted(int(i) for i in _gidx(dict_users[u]))], int(args.num_classes), float(getattr(args, 'gas_tro', 1.0)))
                         for u in dict_users}
        print('[GAS-CFG] n_min=%d tro=%g reg=%g cov=diag(per-position) C=%d | exchange=stale, statistics from round-initial a0 (no extra communication), generated samples added to server training' % (
            int(getattr(args, 'gas_nmin', 32)), float(getattr(args, 'gas_tro', 1.0)), float(getattr(args, 'gas_reg', 1e-5)), int(args.num_classes)), flush=True)
    else:
        _gas = None; _gas_stats = None
    for iter in range(_start_round, args.epochs + 1):

        args._cur_round = iter
        args._srv_jitter_on = (iter > int(getattr(args, 'fd_warmup_epochs', 0) or 0))
        t_epoch = time.time()
        is_phase1 = (_warmup_epochs > 0 and iter <= _warmup_epochs)
        phase_tag = "VANILLA" if is_phase1 else "HESS-DIAG"

        if iter == 1:
            print(f"[Driver] [Round {iter}] PHASE 1 START ({phase_tag}, ~ epoch {_warmup_epochs})")
        elif iter == _warmup_epochs + 1 and _warmup_epochs > 0:
            print(f"[Driver] [Round {iter}] PHASE 2 START (VANILLA → HESS-DIAG)")

        apply_lr(args, iter, warmup_epochs=_warmup_epochs, tag='HD')

        w_locals_c, w_locals_s = _new_locals(args), _new_locals(args)
        w_glob_client = copy.deepcopy(net_glob_client)
        w_glob_server = copy.deepcopy(net_glob_server)

        m = max(int(args.frac * args.num_users), 1)

        _ep = int(__import__('os').environ.get('EXPECT_PART', '0') or 0)
        if not globals().get('_PART_LOGGED', False):
            globals()['_PART_LOGGED'] = True
            print('[PART] N=%d frac=%r m=%d expect=%s epoch_per_round=%.4f'
                  % (args.num_users, args.frac, m, _ep or '-', m / max(args.num_users, 1)),
                  flush=True)
        if _ep and m != _ep:
            raise SystemExit('[PART-GATE] participating clients %d != expected %d (N=%d frac=%r); wrong launch arguments, aborting.'
                             % (m, _ep, args.num_users, args.frac))
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        round_diag_keys, round_diag_acc = None, {}

        for idx in idxs_users:
            t_total = time.time()
            is_llm = args.model_name in ('RoBerta', 'RoBerta_large', 'DistilRoBerta', 'DistilBert', 'GPT2', 'Qwen')

            if is_phase1:

                _CM.reset()
                local_c = Localupdate_sfl_client_vanilla(
                    args, dataset=dataset_train, idxs=dict_users[idx], wandb=wandb, model_idx=0)

                _nc_upd = copy.deepcopy(w_glob_client); _ns_upd = _srv_copy(args, w_glob_server)

                _rs_on = (str(getattr(args, 'hd_mode', '')) in ('omega3b', 'omega3a') and str(getattr(args, 'hd_omega_scale', '')) == 'rspring'
                          and str(getattr(args, 'hd_rstar_mode', 'warmup')) == 'warmup')
                _nc0 = copy.deepcopy(w_glob_client) if (_rs_on and not is_llm) else None
                if _rs_on and is_llm:
                    raise SystemExit('[RSTAR] warm-up self-measurement is not implemented for the LLM path')
                weight_c, weight_s, _, loss_s, acc_s = local_c.train_client(
                    net_client=_nc_upd, net_server=_ns_upd, net_client0=_nc0)
                if _rs_on and local_c.round_radius is not None:
                    args._rstar_obs = getattr(args, '_rstar_obs', {}); args._rstar_obs.setdefault(int(iter), []).append(float(local_c.round_radius))
                    print('[RSTAR] r=%d user=%d R_obs(max_b)=%.4f series=[%s]' % (iter, idx, local_c.round_radius, ' '.join('%.4f' % v for v in local_c.round_radius_series)), flush=True)
                    del _nc0
                loss_s, acc_s = train_metric_post(_nc_upd, _ns_upd, local_c.ldr_train, args, is_llm)
                del _nc_upd, _ns_upd
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

                _want_lr = (not bool(getattr(args, 'hd_stale_mode', False))
                            and not bool(getattr(args, 'hd_upper_mode', False))
                            and (str(getattr(args, 'hd_mode', 'hybrid')) == 'lowrank'
                                 or (str(getattr(args, 'hd_mode', 'hybrid')) == 'damp' and str(getattr(args, 'hd_damp_mu', '')) == 'thetaK')))
                U_list = th_list = None
                _want_tg = (str(getattr(args, 'hd_mode', 'hybrid')) == 'target'
                            and not bool(getattr(args, 'hd_stale_mode', False)) and not bool(getattr(args, 'hd_upper_mode', False)))
                a_star_list = None
                _want_om = (str(getattr(args, 'hd_mode', 'hybrid')) in ('omega3b', 'omega3a', 'omegaM', 'omegaMp', 'omegaMpG', 'omegaMpJ')
                            and not bool(getattr(args, 'hd_stale_mode', False)) and not bool(getattr(args, 'hd_upper_mode', False)))
                omega_list = omega_alt_list = cbar_list = None
                _rc = get_round_initial_cache(
                    net_server=copy.deepcopy(w_glob_server),
                    smashed_data_list=smashed_data, label_list=label,
                    args=args, mask_list=mask_list,
                    rnd=iter, cid=int(idx), want_lowrank=_want_lr, want_target=_want_tg, want_omega=_want_om)
                if _gas_on:
                    with torch.no_grad():
                        for _b in range(len(smashed_data)):
                            _fx = smashed_data[_b].detach().to(args.device); _yy = label[_b].to(args.device).reshape(-1)
                            _m2 = _gas.mask2d_from_ext(mask_list[_b].to(args.device)) if is_llm else None
                            _gas_stats.update(_fx, _yy, _m2, float(iter))
                            del _fx
                if _want_lr:
                    g_initial_list, U_list, th_list = _rc
                elif _want_tg:
                    g_initial_list, a_star_list = _rc
                elif _want_om:
                    g_initial_list, omega_list, omega_alt_list, cbar_list = _rc
                else:
                    g_initial_list = _rc

                frozen_server_snapshot = copy.deepcopy(w_glob_server)
                if bool(getattr(args, 'hd_upper_mode', False)):
                    _CM._S['dn_b'] = 0; _CM._S['dn_n'] = 0
                _nc_upd = copy.deepcopy(w_glob_client)
                weight_c, _, diag = local_c.train_hess_diag(
                    net_client=_nc_upd,
                    net_server_frozen=frozen_server_snapshot,
                    cached_x2_list=smashed_data,
                    cached_g_list=g_initial_list,
                    cached_label_list=label,
                    cached_mask_list=mask_list,
                    cached_inputs=cached_inputs,
                    cached_labels=(label if cached_inputs is not None else None),
                    cached_U_list=U_list, cached_th_list=th_list,
                    cached_a_star_list=a_star_list,
                    cached_omega_list=omega_list, cached_omega_alt_list=omega_alt_list, cached_cbar_list=cbar_list,
                    cur_epoch=iter,
                    user_idx=int(idx))

                _ns_upd = _srv_copy(args, w_glob_server)
                _sd_tr, _ml_tr, _lb_tr = smashed_data, mask_list, label
                _gen_step = None
                if _gas_on and not bool(getattr(args, 'gas_generate', True)):
                    print('[GAS] r=%d cid=%d generation disabled (--gas_generate False); logit adjustment only' % (iter, int(idx)), flush=True)
                elif _gas_on and _gas_stats.has_all():
                    _lens = None
                    if is_llm:
                        _lens = torch.cat([_gas.mask2d_from_ext(mm).sum(1) for mm in mask_list]).long().tolist()
                    if str(getattr(args, 'gas_mix', 'batch')) == 'step':
                        _gfx, _gy, _gm, _ginfo = _gas.make_generated_batches(
                            _gas_stats, label, mask_list, int(getattr(args, 'gas_nmin', 32)),
                            int(getattr(args, 'gas_gen_bs', 0) or args.local_bs),
                            _gas.gen_for(args, iter, int(idx)), args.device, (mask_list[0] if is_llm else None), is_llm, lengths=_lens)
                        _gen_step = list(zip(_gfx, _gy, _gm)) if _gfx else None
                    else:
                        _sd_tr, _lb_tr, _ml_tr, _ginfo = _gas.mix_generated(
                            _gas_stats, smashed_data, label, mask_list, int(getattr(args, 'gas_nmin', 32)),
                            _gas.gen_for(args, iter, int(idx)), args.device, is_llm, lengths=_lens)
                        if smashed_data[0].device.type == 'cpu':
                            _sd_tr = [x.cpu() for x in _sd_tr]
                    _n_real = sum(int(y.numel()) for y in label)
                    print('[GAS] r=%d cid=%d real=%d gen=%d (labels pad=%d absent=%d) mix=%s %s' % (
                        iter, int(idx), _n_real, _ginfo['n_gen'], _ginfo['labels_pad'], _ginfo['labels_absent'], str(getattr(args, 'gas_mix', 'batch')),
                        ('generated samples in one accumulated step (batch %d)' % len(_gen_step)) if _gen_step else ('mixed batches %d (samples/batch %.0f)' % (len(_sd_tr), (_n_real + _ginfo['n_gen']) / max(1, len(_sd_tr))))), flush=True)
                    args._gas_acc = getattr(args, '_gas_acc', []); args._gas_acc.append((_n_real, _ginfo['n_gen']))
                elif _gas_on:
                    print('[GAS] r=%d cid=%d labels without statistics %d/%d -> no generation' % (iter, int(idx), sum(1 for c in range(_gas_stats.C) if c not in _gas_stats.mean), _gas_stats.C), flush=True)
                weight_s, _, loss_s, acc_s = local_s.train(
                    net_server=_ns_upd,
                    smashed_data=_sd_tr, mask_list=_ml_tr, label_list=_lb_tr, gen=_gen_step)
                del _sd_tr, _ml_tr, _lb_tr

                loss_s, acc_s = train_metric_post(_nc_upd, _ns_upd, local_c.ldr_train, args, is_llm)
                _l2s = getattr(args, '_l2_store', {}).pop(int(idx), None)
                if _l2s is not None and _l2s.get('rnd') == iter:
                    import train.hd_instr as _I2
                    from train.train_fl_hess_diag import masked_rng as _mr, mask_sync_on as _mso, _rng_snapshot as _rs, _rng_restore as _rr
                    _snap = _rs(); _was_tr = _ns_upd.training
                    try:
                        (_ns_upd.train() if _mso(args) else _ns_upd.eval())
                        _a0 = smashed_data[0].to(args.device).detach(); _m0 = (mask_list[0].to(args.device) if is_llm else None)
                        _V = (_l2s['V'].to(args.device) if _l2s['V'] is not None else None)
                        with torch.enable_grad():
                            with _mr(args, iter, int(idx), 0, args.device):
                                _rB, _, _ = _I2.l2_jacobian_rows(_ns_upd, _a0, _m0, is_llm, k=_l2s['k'], gen=None, V=_V)
                        _srv = _I2.l2_rel(_l2s['rows0'].to(args.device), _rB)
                        print('[L2S] Epoch %s user %s | J_srv_rel=%.4e | J_seg_rel=%.4e | srv/seg=%.3f' % (iter, idx, _srv, _l2s['seg'], _srv / max(_l2s['seg'], 1e-30)), flush=True)
                        diag['l2_J_srv_rel'] = _srv
                        del _rB
                    except torch.OutOfMemoryError as _e:
                        print('[L2S] Epoch %s user %s | SKIPPED (CUDA OOM): %s' % (iter, idx, str(_e)[:80]), flush=True); torch.cuda.empty_cache()
                    finally:
                        _ns_upd.train(_was_tr); _rr(_snap)

                if _want_om and omega_list and str(getattr(args, 'hd_mode', '')) not in ('omegaM', 'omegaMp', 'omegaMpG', 'omegaMpJ') and ((not getattr(args, 'hd_diag_at_test', False)) or ((int(iter) + 1) % 10 == 0)):
                    from train.hd_omega import omega_of_batch, _spearman
                    _oe = omega_of_batch(_ns_upd, smashed_data[0], label[0], (mask_list[0] if is_llm else None), args, iter, int(idx), 0)
                    _o0 = omega_list[0].to(_oe.device)
                    _stab = _spearman(_o0.reshape(_o0.size(0), -1).float().expand(_oe.size(0), -1), _oe.reshape(_oe.size(0), -1).float())
                    print(f'[OMEGA-STAB] r={iter} user={idx} omega_stab={_stab:.4f} (θ_s^(0) vs θ_s^(B), 0, a0)', flush=True)
                    diag['[OM] omega_stab'] = float(_stab)
                    del _oe, _o0

                round_diag_keys = True
                for k, v in diag.items():
                    round_diag_acc.setdefault(k, []).append(v)

                del smashed_data, g_initial_list, label, mask_list, cached_inputs
                if U_list is not None:
                    del U_list, th_list
                U_list = th_list = None
                del a_star_list, omega_list, omega_alt_list, cbar_list
                del _rc, frozen_server_snapshot, _nc_upd, _ns_upd, local_c, local_s, diag
                gc.collect()

            w_locals_c.append([_sd_keep(args, weight_c), 0])
            w_locals_s.append([_sd_keep(args, weight_s), 0])
            del weight_c, weight_s

            t_user_total = time.time() - t_total
            _CM.report(iter, phase_tag, user=idx)
            print(f'[{phase_tag}] User {idx} | Total {t_user_total:.1f}s | '
                  f'S_Loss {loss_s:.3f} | S_{metric_name()} {acc_s:.3f}')
            wandb.log({f'[Train] Server {args.cut_point} loss': loss_s,
                       f'[Train] Server {args.cut_point} {metric_name()}': acc_s}, step=iter)

        t0_agg = time.time()
        net_glob_client.load_state_dict(FedAvg(w_locals_c))
        _srv_agg(args, net_glob_server, w_locals_s, w_glob_server)

        if is_phase1 and iter == _warmup_epochs and getattr(args, '_rstar_obs', None):
            import statistics as _st
            _mode_r = str(getattr(args, 'hd_rstar_round', 'last'))
            _vals = (args._rstar_obs.get(int(iter), []) if _mode_r == 'last' else [v for vs in args._rstar_obs.values() for v in vs])
            if _vals:
                args._omega_rstar = float(_st.median(_vals))
                print('[RSTAR] R*=%.4f (%s, n=%d, min/max=%.4f/%.4f) measured on the warm-up trajectory; fixed as the rspring target radius' % (
                    args._omega_rstar, _mode_r, len(_vals), min(_vals), max(_vals)), flush=True)
        t_agg = time.time() - t0_agg

        del w_locals_c, w_locals_s, w_glob_client, w_glob_server
        gc.collect()
        torch.cuda.empty_cache()

        if round_diag_keys and float(np.sum(round_diag_acc.get('[HD] n_diag', [0]))) > 0:
            diag_log = {f'[Round] {k}': float(np.mean(round_diag_acc[k])) for k in round_diag_acc}
            wandb.log(diag_log, step=iter)
            print('[HD-Diag] ' + ' | '.join(f'{k}={v:.4f}' for k, v in diag_log.items()))

        wandb.log({'[Phase] is_phase1': 1.0 if is_phase1 else 0.0,
                   '[Phase] is_phase2': 0.0 if is_phase1 else 1.0}, step=iter)
        print(f'[Time] Phase={phase_tag} | Epoch {iter} | Agg {t_agg:.1f}s '
              f'| Total {time.time()-t_epoch:.1f}s')

        _ev_p = max(1, int(getattr(args, 'hd_eval_every', 10)))
        _ev_w = int(getattr(args, 'fd_warmup_epochs', 0))
        if (iter <= _ev_w) or (iter % _ev_p == 0) or (iter == int(args.epochs) - 1):
            t0_eval = time.time()
            acc_test_s, loss_test_s = test_img_sfl(
                net_glob_client, net_glob_server, dataset_test, args)
            print(f'[Time] Epoch {iter} | Eval {time.time()-t0_eval:.1f}s')
            print(f'[Epoch {iter}] ({phase_tag}) Test acc (cut {args.cut_point}): {acc_test_s:.2f}')
            wandb.log({f'[Test] Server {args.cut_point} acc': float(acc_test_s),
                       f'[Test] Server {args.cut_point} loss': float(loss_test_s)}, step=iter)
            acc_test_total_s.append(float(acc_test_s))

        maybe_save(args, iter, net_glob_client, net_glob_server, aux=None, optimizers=None)

        if iter % 50 == 0:
            print(program)

    print('finish')
    return acc_test_total_s
