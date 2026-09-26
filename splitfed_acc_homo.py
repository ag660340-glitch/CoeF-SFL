from train.extract_weight import extract_submodel_weight_from_global_fjord, extract_submodel_weight_from_global
from train.train_fl import *
from train.train_fl import Localupdate_sfl_client_vanilla
from train.avg import *


from train.model_assign import *
from utils.options import args_parser_main
from utils.utils import  seed_everything, metric_name
from train.train_fl import train_metric_post
_LLM_SET = ('RoBerta', 'RoBerta_large', 'DistilRoBerta', 'DistilBert', 'GPT2', 'Qwen')
from data.dataset import load_data
from utils.ckpt import maybe_save, maybe_resume
from utils.lr_sched import apply_lr

import time
import numpy as np
import time
import wandb
import copy
from utils.sflv import new_locals as _new_locals, sd_keep as _sd_keep, srv_copy as _srv_copy, srv_agg as _srv_agg, srv_agg_sd as _srv_agg_sd, tag as _sflv_tag
import random
import utils.comm_meter as _CM


def main_sfl_acc_homo(args):

    dataset_train, dataset_test, dict_users, args.num_classes = load_data(args)

    local_cmodels, local_smodels, auxiliary_models, args.cut_point = SFL_acc_model_assignment_homo(args.model_name, args.num_classes, args.cut_point, args.device)
    args.num_models = args.cut_point
    net_glob_client = copy.deepcopy(local_cmodels[0])
    net_glob_server = copy.deepcopy(local_smodels[0])

    acc_test_total_c = []
    acc_test_total_s = []

    program = args.name
    print(program)

    _warmup_epochs = int(getattr(args, 'fd_warmup_epochs', 0))
    print(f"[ACC] warmup_epochs={_warmup_epochs} (Vanilla SFL during warmup, ACC after)")

    _start_round = maybe_resume(args, net_glob_client, net_glob_server, aux=auxiliary_models)
    if _start_round > 1:
        print(f'[CKPT] r{_start_round} ~ r{args.epochs} ', flush=True)
    _is_llm_tm = args.model_name in _LLM_SET
    for iter in range(_start_round, args.epochs + 1):
        t_epoch = time.time()
        is_phase1 = (_warmup_epochs > 0 and iter <= _warmup_epochs)

        apply_lr(args, iter, warmup_epochs=_warmup_epochs, tag='ACC')

        w_locals_c = _new_locals(args)
        w_locals_s = _new_locals(args)
        w_locals_a = []

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

        for idx in idxs_users:
            t_total = time.time()

            if is_phase1:

                local_c = Localupdate_sfl_client_vanilla(
                    args, dataset=dataset_train, idxs=dict_users[idx], wandb=wandb, model_idx=0)

                _nc_upd = copy.deepcopy(w_glob_client); _ns_upd = _srv_copy(args, w_glob_server)
                weight_c, weight_s, _, loss_s, acc_s = local_c.train_client(
                    net_client=_nc_upd, net_server=_ns_upd)
                loss_s, acc_s = train_metric_post(_nc_upd, _ns_upd, local_c.ldr_train, args, _is_llm_tm)
                w_locals_c.append([_sd_keep(args, weight_c), 0])
                w_locals_s.append([_sd_keep(args, weight_s), 0])
                _mn = metric_name()
                print('[Epoch : {}][User {} VANILLA(warmup)] [S_Loss  {:.3f} | S_{} {:.3f}]'
                      .format(iter, idx, loss_s, _mn, acc_s))
                wandb.log({"[Train] Server {} loss".format(args.cut_point): loss_s,
                           "[Train] Server {} {}".format(args.cut_point, _mn): acc_s}, step=iter)
                continue

            t0 = time.time()
            t_extract = time.time() - t0

            aux_client = copy.deepcopy(auxiliary_models[0])

            mask_list = None

            assert not int(getattr(args, 'baseline_fb', 0)), '[acc] baseline_fb has been removed'
            wc_round, ws_round = w_glob_client, w_glob_server

            local_c = LocalUpdate_client(args,  dataset = dataset_train, idxs = dict_users[idx], wandb = wandb, model_idx = 0)

            t0 = time.time()
            _CM.reset()
            if args.model_name in ('RoBerta', 'RoBerta_large', 'DistilRoBerta', 'DistilBert', 'GPT2', 'Qwen'):
                smashed_data, mask_list, label = local_c.get_smashed_data_llm(net=copy.deepcopy(wc_round))
            else :
                smashed_data, label = local_c.get_smashed_data(net=copy.deepcopy(wc_round))

            _CM.up(smashed_data, mask_list, label, msgs=1)

            t_smash = time.time()-t0

            local_s = LocalUpdate_server(args, smashed_data = smashed_data, mask_list=mask_list,
                                         label = label, wandb = wandb, model_idx = 0)

            t0 = time.time()

            _nc_upd = copy.deepcopy(wc_round)
            weight_c, weight_a_c,  args, loss_c, acc_c = local_c.train(
                    net_client = _nc_upd, net_ax = copy.deepcopy(aux_client))

            t_client = time.time() - t0

            t0 = time.time()
            _ns_upd = copy.deepcopy(ws_round)
            weight_s,  args, loss_s, acc_s = local_s.train(
                net = _ns_upd)

            loss_s, acc_s = train_metric_post(_nc_upd, _ns_upd, local_c.ldr_train, args, _is_llm_tm)

            t_server = time.time() - t0

            w_locals_c.append([_sd_keep(args, weight_c), 0])
            w_locals_s.append([_sd_keep(args, weight_s), 0])
            w_locals_a.append(weight_a_c)

            t_user_total = time.time() - t_total
            print(f'[Time] User {idx} | Total {t_user_total:.1f}s | Extract {t_extract:.1f}s | Smash {t_smash:.1f}s | Client {t_client:.1f}s | Server {t_server:.1f}s')

            _mn = metric_name()
            print('[Epoch : {}][User {} with cut_point {}] [C_Loss  {:.3f} | C_{} {:.3f}] [S_Loss  {:.3f} | S_{} {:.3f}]'
                  .format(iter, idx, args.cut_point, loss_c, _mn, acc_c, loss_s, _mn, acc_s))
            wandb.log({"[Train] Client {} loss".format(args.cut_point): loss_c,"[Train] Client {} {}".format(args.cut_point, _mn): acc_c,\
                        "[Train] Server {} loss".format(args.cut_point): loss_s,"[Train] Server {} {}".format(args.cut_point, _mn): acc_s}, step = iter)
            _CM.report(iter, 'ACC', user=idx)

        t0 = time.time()
        w_c_glob = FedAvg(w_locals_c)
        if not is_phase1:
            auxiliary_models = FedAvg_auxnet(w_locals_a, auxiliary_models)

        net_glob_client.load_state_dict(w_c_glob)
        _srv_agg(args, net_glob_server, w_locals_s, w_glob_server)
        t_agg = time.time() - t0

        print(f'[Time] aggregation {t_agg:1f}s |')

        _ev_p = max(1, int(getattr(args, 'hd_eval_every', 10)))
        _ev_w = int(getattr(args, 'fd_warmup_epochs', 0))
        if (iter <= _ev_w) or (iter % _ev_p == 0) or (iter == int(args.epochs) - 1):
            t0_eval_time = time.time()

            t0 = time.time()

            test_acc_list_c = []
            test_acc_list_s = []

            t0_ind_eval = time.time()

            c_keys = set(w_glob_client.state_dict().keys())
            s_keys = set(w_glob_server.state_dict().keys())
            glob_c_keys = set(net_glob_client.state_dict().keys())
            glob_s_keys = set(net_glob_server.state_dict().keys())

            acc_test_c, loss_test_c, acc_test_s, loss_test_s = test_img_acc(net_glob_client, net_glob_server, auxiliary_models[0], dataset_test, args)
            t_eval_ind = time.time() - t0_ind_eval
            print(f'[Time] Epoch {iter} | Split {args.cut_point} | Eval {t_eval_ind:.1f}s | Agg {t_agg:.1f}s | Cumulative Epoch Total {time.time()-t_epoch:.1f}s')

            print("[Epoch {}]Testing accuracy with split point {} : [Client : {:.2f} | Server : {:.2f}] ".format(iter,args.cut_point, acc_test_c, acc_test_s))
            wandb.log({"[Test] Client {} acc".format(args.cut_point): acc_test_c, "[Test] Server {} acc".format(args.cut_point): acc_test_s}, step = iter)
            test_acc_list_c.append(acc_test_c)
            test_acc_list_s.append(acc_test_s)

            acc_test_total_c.append(test_acc_list_c)
            acc_test_total_s.append(test_acc_list_s)

        maybe_save(args, iter, net_glob_client, net_glob_server,
                   aux=auxiliary_models, optimizers=None)

        if  iter % 50 == 0:
            print(program)
    print("finish")


