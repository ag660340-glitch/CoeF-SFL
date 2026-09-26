# Excerpt of train/train_fl.py: vanilla SFL and AccSFL local client/server updates (unmodified code; other utilities omitted).
import torch
import utils.comm_meter as _CM
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from train import task_loss as _TL
from torch.nn.attention import sdpa_kernel, SDPBackend
from data.dataset import DatasetSplit
from utils.utils import calculate_accuracy
import copy
import torch.nn.functional as F
import random
import os


def _dl_kwargs():
    _nw = int(os.environ.get('DL_WORKERS', '4'))
    if _nw <= 0:
        return {}
    return dict(num_workers=_nw, pin_memory=True, persistent_workers=True,
                prefetch_factor=int(os.environ.get('DL_PREFETCH', '2')))


class Localupdate_sfl_client_vanilla(object):
    def __init__(self, args, dataset = None, idxs = None, wandb = None, model_idx = None):
        self.args = args
        collator = getattr(args, "sst2_collator", None) if args.model_name in ('RoBerta', 'RoBerta_large', 'DistilRoBerta', 'DistilBert', 'GPT2', 'Qwen') else None
        DS_alter = DatasetSplit(dataset, idxs)
        random.shuffle(DS_alter.idxs)
        self.ldr_train = DataLoader(DS_alter, batch_size = args.local_bs, shuffle = False, collate_fn=collator, **_dl_kwargs())
        self.wandb = wandb
        self.model_idx = model_idx

    def train_client(self, net_client, net_server, net_client0=None):

        self.round_radius = None; self.round_radius_series = []
        if net_client0 is not None:
            net_client0.to(self.args.device); net_client0.train()
        net_client.to(self.args.device)

        if bool(getattr(self.args, 'van_client_eval', False)) and bool(getattr(self.args, '_van_eval_now', False)):
            net_client.eval()
        else:
            net_client.train()
        net_server.to(self.args.device)
        net_server.train()

        optimizer_client = torch.optim.AdamW(net_client.parameters(), lr=self.args.lr,
            weight_decay=self.args.weight_decay, eps=1e-8)
        optimizer_server = torch.optim.AdamW(net_server.parameters(), lr=self.args.lr,
            weight_decay=self.args.weight_decay, eps=1e-8)

        criterion = _TL.GlobalCriterion()

        epoch_loss_s = []
        epoch_acc_s = []

        is_llm = self.args.model_name in ('RoBerta', 'RoBerta_large', 'DistilRoBerta', 'DistilBert', 'GPT2', 'Qwen')

        for iter in range(self.args.local_ep):

            batch_loss_s = []
            batch_acc_s = []
            len_batch = len(self.ldr_train)

            for batch_idx, batch in enumerate(self.ldr_train):

                if is_llm:
                    ids = batch['input_ids'].to(self.args.device)
                    mask = batch['attention_mask'].to(self.args.device)
                    label = batch['labels'].to(self.args.device)
                    fx, ext_mask = net_client(ids, mask)
                else:
                    images, label = batch[0].to(self.args.device), batch[1].to(self.args.device)
                    if net_client0 is not None:
                        with torch.random.fork_rng(devices=[torch.device(self.args.device)] if str(self.args.device).startswith('cuda') else []):
                            with torch.no_grad():
                                _ref_a0 = net_client0(images)
                    fx = net_client(images)
                    ext_mask = None

                if net_client0 is not None and not is_llm:
                    with torch.no_grad():
                        _B = fx.size(0); _a0 = _ref_a0.detach()
                        self.round_radius_series.append(float((fx.detach() - _a0).reshape(_B, -1).norm(dim=1).mean() / (_a0.reshape(_B, -1).norm(dim=1).mean() + 1e-30)))
                net_client.zero_grad()
                optimizer_client.zero_grad()

                fx_client = fx.clone().detach().requires_grad_(True)

                _CM.up(fx_client, label)

                dfx, net_server, loss_s, acc_s = train_server_vanilla_sfl(
                    fx_client, net_server, label, self.args, self.args.device,
                    optimizer_server=optimizer_server,
                    ext_mask=ext_mask
                )

                _CM.down(dfx)
                fx.backward(dfx)
                optimizer_client.step()

                batch_loss_s.append(loss_s.item())
                batch_acc_s.append(acc_s.item())

            epoch_loss_s.append(sum(batch_loss_s)/len(batch_loss_s))
            epoch_acc_s.append(sum(batch_acc_s)/len(batch_acc_s))

        loss_acc = [sum(epoch_loss_s)/len(epoch_loss_s), epoch_acc_s[-1]]
        if self.round_radius_series:
            self.round_radius = float(max(self.round_radius_series))
        _sja = getattr(self.args, '_sj_van_acc', None)
        if _sja:
            from train.srv_jitter import jitter_log_round as _sj_log
            _sj_log(_sja, self.args, tag=' path=vanilla_perbatch'); self.args._sj_van_acc = None

        return net_client.state_dict(), net_server.state_dict(), self.args, loss_acc[0], loss_acc[1]


def train_server_vanilla_sfl(fx_client, net_s, y, args, device,
                             optimizer_server=None, ext_mask=None):
    is_upper = bool(getattr(args, 'upper_bound', False))

    net_server = net_s.to(device)
    if is_upper:
        net_server.eval()
    else:
        net_server.train()
    criterion = _TL.GlobalCriterion()

    if optimizer_server is None and not is_upper:
        optimizer_server = torch.optim.AdamW(net_server.parameters(), lr=args.lr,
            weight_decay=args.weight_decay, eps=1e-8)

    net_s.zero_grad()
    if optimizer_server is not None:
        optimizer_server.zero_grad()

    fx = fx_client.to(device)
    y = y.to(device)

    if ext_mask is not None:
        fx_server, _ = net_server(fx, ext_mask)
    else:
        fx_server, _ = net_server(fx)

    loss = criterion(fx_server, y)
    acc = calculate_accuracy(fx_server, y)

    _sj_rho = float(getattr(args, 'srv_jitter', 0.0) or 0.0)
    if (_sj_rho > 0) and bool(getattr(args, '_srv_jitter_on', True)) and (not is_upper):
        from train.srv_jitter import jitter as _sj_jitter, make_gen as _sj_gen, next_seed as _sj_seed
        dfx_client = torch.autograd.grad(loss, fx)[0].clone().detach()
        net_server.zero_grad(); optimizer_server.zero_grad()
        _g = _sj_gen(device, _sj_seed(args))
        _fxj, _xr = _sj_jitter(fx.detach(), _sj_rho, _g)
        _oj, _ = (net_server(_fxj, ext_mask) if ext_mask is not None else net_server(_fxj))
        _lj = criterion(_oj, y)
        if str(getattr(args, 'srv_jitter_mode', 'noisy')) == 'both':
            _oc, _ = (net_server(fx.detach(), ext_mask) if ext_mask is not None else net_server(fx.detach()))
            _lj = 0.5 * (_lj + criterion(_oc, y))
        _lj.backward()
        optimizer_server.step()
        _acc = getattr(args, '_sj_van_acc', None)
        if _acc is None:
            _acc = args._sj_van_acc = {'xi_rel': [], 'loss_noisy': [], 'loss_clean': []}
        _acc['xi_rel'] += [float(_xr.min()), float(_xr.max())]; _acc['loss_noisy'].append(float(_lj.item())); _acc['loss_clean'].append(float(loss.item()))
        return dfx_client, net_server, loss, acc

    loss.backward()
    dfx_client = fx.grad.clone().detach()

    if not is_upper:
        optimizer_server.step()

    return dfx_client, net_server, loss, acc


class Localupdate_sfl_server(object):
    def __init__(self, args, dataset = None, idxs = None, wandb = None, model_idx = None):
        self.args = args
        self.wandb = wandb
        self.model_idx = model_idx

    def get_gradient_data(self, net_server, smashed_data, label_list, eval_mode=False):
        gradient_data = []
        if eval_mode:
            net_server.eval()
        else:
            net_server.train()
        criterion = _TL.GlobalCriterion()

        for i in range(len(smashed_data)):

            input_smash = smashed_data[i].to(self.args.device)
            label = label_list[i].to(self.args.device)
            fx_g, _ = net_server(input_smash)

            loss = criterion(fx_g, label)
            loss.backward()
            gradient_s = input_smash.grad.clone().detach()
            gradient_data.append(gradient_s)
            net_server.zero_grad()

        return gradient_data

    def get_gradient_data_llm(self, net_server, smashed_data, mask_list, label_list, eval_mode=False):
        gradient_data = []
        if eval_mode:
            net_server.eval()
        else:
            net_server.train()
        criterion = _TL.GlobalCriterion()

        for i in range(len(smashed_data)):

            input_smash = smashed_data[i].to(self.args.device)
            input_mask = mask_list[i].to(self.args.device)
            label = label_list[i].to(self.args.device)
            fx_g, _ = net_server(input_smash, input_mask)

            loss = criterion(fx_g, label)
            loss.backward()
            gradient_s = input_smash.grad.clone().detach()
            gradient_data.append(gradient_s)
            net_server.zero_grad()

        return gradient_data

    def train(self, net_server, smashed_data, mask_list, label_list, gen=None):
        net_s = net_server.to(self.args.device)
        net_s.train()

        optimizer_server = torch.optim.AdamW(net_s.parameters(), lr = self.args.lr
            , weight_decay=self.args.weight_decay, eps=1e-8)

        criterion = _TL.GlobalCriterion()

        epoch_loss_s = []
        epoch_acc_s = []

        _sj_rho = float(getattr(self.args, 'srv_jitter', 0.0) or 0.0)
        _sj_on = (_sj_rho > 0) and bool(getattr(self.args, '_srv_jitter_on', True))
        if _sj_on:
            from train.srv_jitter import jitter as _sj_jitter, make_gen as _sj_gen, next_seed as _sj_seed, jitter_log_round as _sj_log
            _sj_g = _sj_gen(self.args.device, _sj_seed(self.args))
            _sj_p = float(getattr(self.args, 'srv_jitter_p', 1.0)); _sj_both = (str(getattr(self.args, 'srv_jitter_mode', 'noisy')) == 'both')
            _sj_st = {'xi_rel': [], 'loss_noisy': [], 'loss_clean': []}

        for i in range(self.args.local_ep):
            batch_loss_s, batch_acc_s = [], []

            for j in range(len(smashed_data)):

                net_s.zero_grad()
                optimizer_server.zero_grad()

                fx = smashed_data[j].to(self.args.device)
                y = label_list[j].to(self.args.device)
                _fx_clean = None
                if _sj_on and (_sj_p >= 1.0 or float(torch.rand(1, generator=_sj_g, device=self.args.device)) < _sj_p):
                    _fx_clean = fx
                    fx, _xr = _sj_jitter(fx, _sj_rho, _sj_g)
                    _sj_st['xi_rel'] += [float(_xr.min()), float(_xr.max())]

                if self.args.model_name in ('RoBerta', 'RoBerta_large', 'DistilRoBerta', 'DistilBert', 'GPT2', 'Qwen') :

                    mask = mask_list[j].to(self.args.device)
                    fx_server, probas = net_s(fx, mask)

                else :
                    fx_server, probas = net_s(fx)

                loss = criterion(fx_server, y)
                acc = calculate_accuracy(fx_server, y)
                if _fx_clean is not None:
                    _sj_st['loss_noisy'].append(float(loss.item()))
                    if _sj_both:
                        _oc, _ = (net_s(_fx_clean, mask) if self.args.model_name in ('RoBerta', 'RoBerta_large', 'DistilRoBerta', 'DistilBert', 'GPT2', 'Qwen') else net_s(_fx_clean))
                        _lc = criterion(_oc, y); _sj_st['loss_clean'].append(float(_lc.item()))
                        loss = 0.5 * (loss + _lc)
                loss.backward()

                optimizer_server.step()

                batch_loss_s.append(loss.item())
                batch_acc_s.append(acc.item())

            if gen:
                net_s.zero_grad(); optimizer_server.zero_grad()
                _ntot = sum(int(gy.numel()) for _, gy, _ in gen)
                for _gfx, _gy, _gm in gen:
                    _fx = _gfx.to(self.args.device); _y = _gy.to(self.args.device)
                    if self.args.model_name in ('RoBerta', 'RoBerta_large', 'DistilRoBerta', 'DistilBert', 'GPT2', 'Qwen'):
                        _o, _ = net_s(_fx, _gm.to(self.args.device))
                    else:
                        _o, _ = net_s(_fx)
                    (criterion(_o, _y) * (float(_y.numel()) / max(1, _ntot))).backward()
                optimizer_server.step()
            epoch_loss_s.append(sum(batch_loss_s)/len(batch_loss_s))
            epoch_acc_s.append(sum(batch_acc_s)/len(batch_acc_s))
        server_loss = sum(epoch_loss_s)/len(epoch_loss_s)
        server_acc = epoch_acc_s[-1]
        if _sj_on:
            _sj_log(_sj_st, self.args, tag=' path=server_cache')

        return net_s.state_dict(), self.args, server_loss, server_acc


class LocalUpdate_client(object):
    def __init__(self, args, dataset = None, idxs = None, wandb = None, model_idx = None):
        self.args = args
        collator = getattr(args, "sst2_collator", None) if args.model_name in ('RoBerta', 'RoBerta_large', 'DistilRoBerta', 'DistilBert', 'GPT2', 'Qwen') else None
        DS_alter = DatasetSplit(dataset, idxs)
        random.shuffle(DS_alter.idxs)

        self.ldr_train = DataLoader(
            DS_alter,
            batch_size=args.local_bs,
            shuffle=False,
            collate_fn=collator,
            **_dl_kwargs(),
        )
        self.wandb = wandb
        self.model_idx = model_idx

    def get_smashed_data(self, net):
        smashed_data = []
        label = []
        net.eval()

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                fx_client = net(images)
                smashed_data.append(fx_client.clone().detach().requires_grad_(True))
                label.append(labels)
        return smashed_data, label

    def get_smashed_data_llm(self, net):
        smashed_data = []
        mask_list = []
        label_list = []
        net.eval()

        with torch.no_grad():
            for batch in self.ldr_train:

                data, mask, label = batch['input_ids'].to(self.args.device), batch['attention_mask'].to(self.args.device), batch['labels'].to(self.args.device)

                fx_client, ext_mask = net(data, mask)

                smashed_data.append(fx_client.clone().detach().requires_grad_(True))
                mask_list.append(ext_mask.clone().detach())
                label_list.append(label)

        return smashed_data, mask_list,label_list

    def train(self, net_client, net_ax, kd_logits = None):
        net_client.to(self.args.device)

        net_client.train()

        aux_client = copy.deepcopy(net_ax)

        params = list(net_client.parameters())

        aux_client.to(self.args.device)
        aux_client.train()
        params += list(aux_client.parameters())

        optimizer_client = torch.optim.AdamW(params, lr = self.args.lr,
            weight_decay=self.args.weight_decay, eps=1e-8)

        criterion = _TL.GlobalCriterion()

        epoch_loss_c = []
        epoch_acc_c = []
        epoch_loss_s = []
        epoch_acc_s = []

        for iter in range(self.args.local_ep):
            batch_loss_c = []
            batch_acc_c = []
            batch_loss_s = []
            batch_acc_s = []
            len_batch = len(self.ldr_train)
            for batch in self.ldr_train:

                net_client.zero_grad()
                optimizer_client.zero_grad()

                if self.args.model_name in ('RoBerta', 'RoBerta_large', 'DistilRoBerta', 'DistilBert', 'GPT2', 'Qwen'):
                    ids, mask, label = batch['input_ids'].to(self.args.device), batch['attention_mask'].to(self.args.device), batch['labels'].to(self.args.device)
                    fx, ext_mask = net_client(ids, mask)
                else :
                    images, label = batch[0].to(self.args.device), batch[1].to(self.args.device)
                    fx = net_client(images)

                aux_client.zero_grad()

                client_logits, client_probs = (aux_client(fx, ext_mask)
                    if self.args.model_name in ('RoBerta', 'RoBerta_large', 'DistilRoBerta', 'DistilBert', 'GPT2', 'Qwen')
                    else aux_client(fx))
                loss_client = criterion(client_logits, label)
                acc_client = calculate_accuracy(client_logits, label)

                batch_loss_c.append(loss_client.item())
                batch_acc_c.append(acc_client.item())

                loss_client.backward()
                optimizer_client.step()

            epoch_loss_c.append(sum(batch_loss_c)/len(batch_loss_c))
            epoch_acc_c.append(sum(batch_acc_c)/len(batch_acc_c))

        client_loss_acc = [sum(epoch_loss_c)/len(epoch_loss_c), epoch_acc_c[-1]]

        weight_a_c = copy.deepcopy(aux_client.state_dict())

        return net_client.state_dict(), weight_a_c, self.args, client_loss_acc[0], client_loss_acc[1]


class LocalUpdate_server(object):
    def __init__(self, args, smashed_data = None, mask_list=None, label = None, wandb = None, model_idx = None):
        self.args = args
        self.data = smashed_data
        self.label = label
        self.wandb = wandb
        self.model_idx = model_idx
        self.mask_list = mask_list

    def train(self, net):
        net_server = net
        net_server.to(self.args.device)
        net_server.train()

        params = list(net_server.parameters())

        optimizer_server = torch.optim.AdamW(params, lr = self.args.lr,
            weight_decay=self.args.weight_decay, eps=1e-8)

        criterion = _TL.GlobalCriterion()

        net_server.train()

        epoch_loss_s = []
        epoch_acc_s = []

        for iter in range(self.args.local_ep):
            batch_loss_s = []
            batch_acc_s = []

            for j in range(len(self.data)):
                fx = self.data[j].requires_grad_(True)
                y = self.label[j]

                net_server.zero_grad()
                optimizer_server.zero_grad()

                if self.args.model_name in ('RoBerta', 'RoBerta_large', 'DistilRoBerta', 'DistilBert', 'GPT2', 'Qwen') :
                    mask = self.mask_list[j]
                    fx_server, probas = net_server(fx, mask)

                else :
                    fx_server, probas = net_server(fx)

                loss = criterion(fx_server, y)
                acc = calculate_accuracy(fx_server, y)

                loss.backward()
                optimizer_server.step()

                batch_loss_s.append(loss.item())
                batch_acc_s.append(acc.item())

            epoch_loss_s.append(sum(batch_loss_s)/len(batch_loss_s))
            epoch_acc_s.append(sum(batch_acc_s)/len(batch_acc_s))
        server_loss = sum(epoch_loss_s)/len(epoch_loss_s)
        server_acc = epoch_acc_s[-1]

        return net_server.state_dict(), self.args, server_loss, server_acc
