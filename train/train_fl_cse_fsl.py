import copy
import random
import torch
from torch.utils.data import DataLoader

import utils.comm_meter as _CM
from train import task_loss as _TL
from train.train_fl import _dl_kwargs
from data.dataset import DatasetSplit
from utils.utils import calculate_accuracy

_LLM_SET = ('RoBerta', 'RoBerta_large', 'DistilRoBerta', 'DistilBert', 'GPT2', 'Qwen')


def _unpack_batch(batch, is_llm, device):
    if is_llm:
        return ((batch['input_ids'].to(device), batch['attention_mask'].to(device)),
                batch['labels'].to(device))
    return ((batch[0].to(device),), batch[1].to(device))


def _client_fwd(net_c, inp, is_llm):
    if is_llm:
        fx, ext = net_c(*inp)
        return fx, ext
    fx = net_c(*inp)
    if isinstance(fx, (list, tuple)):
        fx = fx[0]
    return fx, None


def _head_fwd(net, fx, ext_mask, is_llm):
    out = net(fx, ext_mask) if is_llm else net(fx)
    return out[0] if isinstance(out, (list, tuple)) else out


import torch.nn as nn
import torch.nn.functional as F


def _nparams(m, trainable=None):
    return sum(p.numel() for p in m.parameters() if trainable is None or p.requires_grad == trainable)


def cmp_aux_target_pct(args):
    v = float(getattr(args, 'cmp_aux_pct', 0.0) or 0.0)
    if v > 0:
        return v
    m = str(getattr(args, 'method', ''))
    conv = str(args.model_name).lower().startswith('resnet')
    if 'sage' in m:
        return 18.8 if conv else 21.4
    return 7.3


def _pick_n(unit_params, head_params, denom, target):
    best, best_n, acc = None, 1, 0
    for n, u in enumerate(unit_params, 1):
        acc += u
        pct = 100.0 * (acc + head_params) / denom
        if best is None or abs(pct - target) < abs(best - target):
            best, best_n = pct, n
    return best_n, best


class _ResNetPrefixAux(nn.Module):

    def __init__(self, units, ch, num_classes, init='random'):
        super().__init__()
        self.body = nn.Sequential(*[copy.deepcopy(u) for u in units])
        self.fc = nn.Linear(ch, num_classes)
        if init == 'random':
            _reinit_module(self)

    def forward(self, x, ext_mask=None):
        o = F.adaptive_avg_pool2d(self.body(x), 1).flatten(1)
        lg = self.fc(o)
        return lg, F.softmax(lg, dim=1)


def _reinit_module(mod):
    import math
    for m in mod.modules():
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
        elif hasattr(m, 'reset_parameters') and not isinstance(m, nn.Linear):
            m.reset_parameters()
    for m in mod.modules():
        if hasattr(m, 'lora_A') and hasattr(m, 'lora_B'):
            for ad in list(m.lora_A.keys()):
                nn.init.kaiming_uniform_(m.lora_A[ad].weight, a=math.sqrt(5))
                nn.init.zeros_(m.lora_B[ad].weight)
    for p in mod.parameters():
        p.requires_grad_(True)


class _ViTPrefixAux(nn.Module):
    def __init__(self, net_server, keys, num_classes, init='random'):
        super().__init__()
        self.layers = nn.ModuleDict({k: copy.deepcopy(net_server.layers[k]) for k in keys})
        self.layernorm = copy.deepcopy(net_server.layernorm)
        for _p in self.layernorm.parameters():
            _p.requires_grad_(True)
        self.fc = nn.Linear(self.layernorm.normalized_shape[0], num_classes)
        if init == 'random':
            _reinit_module(self)
            self.fc.reset_parameters()

    def forward(self, x, ext_mask=None):
        for k in sorted(self.layers.keys(), key=int):
            x = self.layers[k](x)[0]
        x = self.layernorm(x)
        lg = self.fc(x[:, 0])
        return lg, F.softmax(lg, dim=1)


def build_cmp_aux(args, net_client, net_server, shared_aux, num_classes, tag='CMP'):
    arch = str(getattr(args, 'cmp_aux_arch', 'server_block'))
    name = str(args.model_name)
    dev = args.device

    _cpu_rng = torch.get_rng_state()
    _cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        aux, desc = _build_cmp_aux_impl(args, arch, name, dev, net_client, net_server, shared_aux, num_classes, tag)
    finally:
        torch.set_rng_state(_cpu_rng)
        if _cuda_rng is not None:
            torch.cuda.set_rng_state_all(_cuda_rng)
    aux.train()
    denom = _nparams(net_client) + _nparams(net_server)
    n_all, n_tr = _nparams(aux), _nparams(aux, True)
    print(f'[{tag}-AUX] arch={arch} {desc} | aux={n_all:,} (trainable {n_tr:,}) | client+server={denom:,} '
          f'| pct={100.0 * n_all / denom:.2f}% (target {cmp_aux_target_pct(args):.1f}%)', flush=True)
    return aux, desc


def _build_cmp_aux_impl(args, arch, name, dev, net_client, net_server, shared_aux, num_classes, tag):
    if arch == 'shared':
        return shared_aux, 'shared (acc factory)'
    target = cmp_aux_target_pct(args)
    denom = _nparams(net_client) + _nparams(net_server)

    if name.lower().startswith('resnet'):
        ks = sorted(int(n[5:]) for n, _ in net_server.named_children() if n.startswith('layer') and n[5:].isdigit())
        assert ks, '[CMP-AUX] server has no layerK stage'
        units, unames = [], []
        for k in ks:
            for j, blk in enumerate(getattr(net_server, f'layer{k}').children()):
                units.append(blk); unames.append(f'layer{k}[{j}]')
        with torch.no_grad():
            _w = net_client.training; net_client.eval(); _ws = net_server.training; net_server.eval()
            h = net_client(torch.zeros(2, 3, 32, 32, device=dev))
            h = h[0] if isinstance(h, (list, tuple)) else h
            chs = []
            for u in units:
                h = u(h); chs.append(int(h.shape[1]))
            net_client.train(_w); net_server.train(_ws)
        n, pct = _pick_n([_nparams(u) for u in units], 0, denom, target)
        aux = _ResNetPrefixAux(units[:n], chs[n - 1], num_classes, init=str(getattr(args, 'cmp_aux_init', 'random'))).to(dev)
        return aux, f'server_block({"+".join(unames[:n])}+GAP+fc)'

    if name.startswith('ViT'):
        keys = sorted(net_server.layers.keys(), key=int)
        head = _nparams(net_server.layernorm) + net_server.layernorm.normalized_shape[0] * num_classes + num_classes
        _nb = int(getattr(args, 'cmp_aux_blocks', 0) or 0)
        if _nb > 0:
            n = min(_nb, len(keys)); pct = 100.0 * (sum(_nparams(net_server.layers[k]) for k in keys[:n]) + head) / denom
        else:
            n, pct = _pick_n([_nparams(net_server.layers[k]) for k in keys], head, denom, target)
        _init = str(getattr(args, 'cmp_aux_init', 'random'))
        aux = _ViTPrefixAux(net_server, keys[:n], num_classes, init=_init).to(dev)
        return aux, f'server_block(blocks {keys[:n]} of {len(keys)}+layernorm+fc, init={_init})'

    if name in ('RoBerta', 'DistilRoBerta', 'DistilBert'):
        from model import resnet_hetero as _rh
        fn = {'RoBerta': _rh.RoBerta_aux_server, 'DistilRoBerta': _rh.DistilRoBerta_aux_server,
              'DistilBert': _rh.DistilBert_aux_server}[name]
        layers = getattr(net_server, 'layers', None)
        assert layers is not None, '[CMP-AUX] LLM server has no .layers'
        keys = sorted(layers.keys(), key=int)
        hid = int(getattr(net_server, 'fc').in_features)
        head = (2 * hid if name != 'DistilBert' else hid * hid + hid) + hid * num_classes + num_classes
        if _TL.task_type(args) == 'qa': head = hid * 2 + 2
        n, pct = _pick_n([_nparams(layers[k]) for k in keys], head, denom, target)

        _ga = _rh.args
        _old = getattr(_ga, 'aux_ratio', None)
        _ga.aux_ratio = float(n) / float(len(keys))
        try:
            aux = fn(num_classes, args.cut_point).to(dev)
        finally:
            if _old is None:
                delattr(_ga, 'aux_ratio')
            else:
                _ga.aux_ratio = _old
        assert len(aux.layers) == n, f'[CMP-AUX] tf {len(aux.layers)} != {n}'
        return aux, f'server_block(tf {n} of {len(keys)} blocks+{"LN" if name != "DistilBert" else "pre_classifier"}+fc)'

    if name == 'Qwen':
        from model import resnet_hetero as _rh
        layers = getattr(net_server, 'layers', None); assert layers is not None, '[CMP-AUX] Qwen server has no .layers'
        keys = sorted(layers.keys(), key=int)
        if hasattr(net_server, 'qa_outputs'):
            _head = _nparams(getattr(net_server, 'qa_outputs')) + _nparams(getattr(net_server, 'norm'))
        else:
            _head = _nparams(getattr(net_server, 'lm_head')) + _nparams(getattr(net_server, 'norm'))
        n, pct = _pick_n([_nparams(layers[k]) for k in keys], _head, denom, target)
        _ga = _rh.args; _old = getattr(_ga, 'aux_ratio', None); _ga.aux_ratio = float(n) / float(len(keys))
        try: aux = _rh.Qwen_aux_server(num_classes, args.cut_point).to(dev)
        finally:
            if _old is None: delattr(_ga, 'aux_ratio')
            else: _ga.aux_ratio = _old
        assert len(aux.layers) == n, f'[CMP-AUX] Qwen {len(aux.layers)} != {n}'
        return aux, f'server_block(Qwen {n} of {len(keys)} blocks+norm+{"qa_outputs" if hasattr(aux, "qa_outputs") else "lm_head"})'
    print(f'[{tag}-AUX] {name} server_block → shared ', flush=True)
    return shared_aux, 'shared (fallback)'


class Localupdate_cse_fsl_client(object):

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

    def train(self, net_client, net_server, net_ax):
        args = self.args
        dev = args.device
        Q = max(1, int(getattr(args, 'cse_server_interval', 5)))

        net_c = net_client.to(dev); net_c.train()
        net_s = net_server.to(dev); net_s.train()
        aux = copy.deepcopy(net_ax).to(dev); aux.train()

        opt_c = torch.optim.AdamW(net_c.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=1e-8)
        opt_a = torch.optim.AdamW(aux.parameters(),   lr=args.lr, weight_decay=args.weight_decay, eps=1e-8)
        opt_s = torch.optim.AdamW(net_s.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=1e-8)
        crit = _TL.GlobalCriterion()

        ep_loss_c, ep_acc_c, ep_loss_s, ep_acc_s = [], [], [], []
        n_batches = len(self.ldr_train)
        n_up = 0

        for ep in range(args.local_ep):
            b_loss_c, b_acc_c, b_loss_s, b_acc_s = [], [], [], []
            for k, batch in enumerate(self.ldr_train):
                local_iter = ep * n_batches + k
                inp, y = _unpack_batch(batch, self.is_llm, dev)

                opt_c.zero_grad(); net_c.zero_grad()
                fx, ext = _client_fwd(net_c, inp, self.is_llm)
                a_loc = fx.clone().detach().requires_grad_(True)

                opt_a.zero_grad(); aux.zero_grad()
                logits_c = _head_fwd(aux, a_loc, ext, self.is_llm)
                loss_c = crit(logits_c, y)
                loss_c.backward()
                acc_c = calculate_accuracy(logits_c, y)
                opt_a.step()
                fx.backward(a_loc.grad)
                opt_c.step()
                b_loss_c.append(loss_c.item()); b_acc_c.append(acc_c.item())

                if local_iter % Q == 0:
                    a_srv = fx.clone().detach()
                    ext_srv = ext.clone().detach() if ext is not None else None
                    _CM.up(a_srv, y)
                    opt_s.zero_grad(); net_s.zero_grad()
                    logits_s = _head_fwd(net_s, a_srv, ext_srv, self.is_llm)
                    loss_s = crit(logits_s, y)
                    loss_s.backward()
                    opt_s.step()
                    b_loss_s.append(loss_s.item()); b_acc_s.append(calculate_accuracy(logits_s, y).item())
                    n_up += 1

            ep_loss_c.append(sum(b_loss_c) / len(b_loss_c)); ep_acc_c.append(sum(b_acc_c) / len(b_acc_c))
            ep_loss_s.append(sum(b_loss_s) / max(1, len(b_loss_s)))
            ep_acc_s.append(sum(b_acc_s) / max(1, len(b_acc_s)))

        stats = dict(loss_c=sum(ep_loss_c) / len(ep_loss_c), acc_c=ep_acc_c[-1],
                     loss_s=sum(ep_loss_s) / len(ep_loss_s), acc_s=ep_acc_s[-1],
                     n_up=n_up, n_batches=n_batches * args.local_ep)
        return net_c.state_dict(), copy.deepcopy(aux.state_dict()), net_s.state_dict(), stats
