import math
import random
import torch
from torch.utils.data import DataLoader

import utils.comm_meter as _CM
from train import task_loss as _TL
from train.train_fl import _dl_kwargs
from train.train_fl_cse_fsl import _unpack_batch, _client_fwd, _head_fwd, _LLM_SET
from data.dataset import DatasetSplit
from utils.utils import calculate_accuracy


def _trainable(net):
    return [p for p in net.parameters() if p.requires_grad]


def _dirs(params, seed, dist='gauss'):
    if not params:
        return []
    g = torch.Generator(device=params[0].device).manual_seed(int(seed))
    us = [torch.randn(p.shape, device=p.device, dtype=p.dtype, generator=g) for p in params]
    if dist == 'sphere':
        d = sum(u.numel() for u in us)
        nrm = math.sqrt(sum(float((u * u).sum()) for u in us)) + 1e-12
        us = [u * (math.sqrt(d) / nrm) for u in us]
    return us


def _add(params, us, alpha):
    for p, u in zip(params, us):
        p.add_(u, alpha=float(alpha))


def _seed_of(args, rnd, cid, b, k=0):
    return (int(args.seed) * 1000003 + int(rnd) * 7919 + int(cid) * 104729 + int(b) * 1299709 + int(k) * 15485863) & 0x7fffffff


class Localupdate_mu_splitfed_client(object):

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

    @torch.no_grad()
    def train(self, net_client, net_server, rnd, cid):
        args = self.args
        dev = args.device
        tau = max(1, int(getattr(args, 'mu_tau', 2)))
        lam = float(getattr(args, 'mu_mu', 5e-3))
        P = max(1, int(getattr(args, 'mu_num_pert', 1)))
        dist = str(getattr(args, 'mu_pert_dist', 'gauss'))
        lr_s = float(getattr(args, 'mu_lr_s', 0.0)) or float(args.lr)
        lr_c = float(getattr(args, 'mu_lr_c', 0.0)) or float(args.lr)
        joint = bool(getattr(args, 'mu_joint_final', False))
        eval_fwd = bool(getattr(args, 'mu_eval_forward', False))

        net_c = net_client.to(dev); net_s = net_server.to(dev)
        (net_c.eval() if eval_fwd else net_c.train())
        (net_s.eval() if eval_fwd else net_s.train())
        pc, ps = _trainable(net_c), _trainable(net_s)
        crit = _TL.GlobalCriterion()

        def loss_of(net, a, ext, y):
            logits = _head_fwd(net, a, ext, self.is_llm)
            return float(crit(logits, y).item()), logits

        ep_loss_s, ep_acc_s = [], []
        n_batches = len(self.ldr_train)
        n_srv_steps = 0
        if bool(getattr(args, 'mu_lowfreq', False)):
            return self._train_lowfreq(net_c, net_s, pc, ps, crit, loss_of, rnd, cid, tau, lam, P, dist, lr_s, lr_c, dev)
        for ep in range(args.local_ep):
            b_loss, b_acc = [], []
            for k, batch in enumerate(self.ldr_train):
                b = ep * n_batches + k
                inp, y = _unpack_batch(batch, self.is_llm, dev)

                h, ext = _client_fwd(net_c, inp, self.is_llm)
                uc = [_dirs(pc, _seed_of(args, rnd, cid, b, 1000 + p), dist) for p in range(P)]
                hp, hm = [], []
                for p in range(P):
                    _add(pc, uc[p], +lam); hp.append(_client_fwd(net_c, inp, self.is_llm)[0])
                    _add(pc, uc[p], -2 * lam); hm.append(_client_fwd(net_c, inp, self.is_llm)[0])
                    _add(pc, uc[p], +lam)
                _CM.up(h, *hp, *hm, y)

                for i in range(tau):
                    us = [_dirs(ps, _seed_of(args, rnd, cid, b, 2000 + i * 64 + p), dist) for p in range(P)]
                    sc = []
                    for p in range(P):
                        _add(ps, us[p], +lam); lp, _ = loss_of(net_s, h, ext, y)
                        _add(ps, us[p], -2 * lam); lm, _ = loss_of(net_s, h, ext, y)
                        _add(ps, us[p], +lam)
                        sc.append((lp - lm) / (2 * lam))
                    for p in range(P):
                        _add(ps, us[p], -lr_s * sc[p] / P)
                    n_srv_steps += 1

                dc = []
                if joint:
                    uf = [_dirs(ps, _seed_of(args, rnd, cid, b, 3000 + p), dist) for p in range(P)]
                    for p in range(P):
                        _add(ps, uf[p], +lam); lp, lg = loss_of(net_s, hp[p], ext, y)
                        _add(ps, uf[p], -2 * lam); lm, _ = loss_of(net_s, hm[p], ext, y)
                        _add(ps, uf[p], +lam)
                        dc.append((lp - lm) / (2 * lam))
                    for p in range(P):
                        _add(ps, uf[p], -lr_s * dc[p] / P)
                else:
                    for p in range(P):
                        lp, lg = loss_of(net_s, hp[p], ext, y)
                        lm, _ = loss_of(net_s, hm[p], ext, y)
                        dc.append((lp - lm) / (2 * lam))
                dc_t = torch.tensor(dc, device=dev)
                _CM.down(dc_t)
                for p in range(P):
                    _add(pc, uc[p], -lr_c * dc[p] / P)

                b_loss.append(0.5 * (lp + lm)); b_acc.append(calculate_accuracy(lg, y).item())

            ep_loss_s.append(sum(b_loss) / len(b_loss)); ep_acc_s.append(sum(b_acc) / len(b_acc))

        stats = dict(loss_s=sum(ep_loss_s) / len(ep_loss_s), acc_s=ep_acc_s[-1],
                     n_batches=n_batches * args.local_ep, n_srv_steps=n_srv_steps,
                     tau=tau, P=P, lam=lam, lr_s=lr_s, lr_c=lr_c)
        return net_c.state_dict(), net_s.state_dict(), stats

    @torch.no_grad()
    def _train_lowfreq(self, net_c, net_s, pc, ps, crit, loss_of, rnd, cid, tau, lam, P, dist, lr_s, lr_c, dev):
        args = self.args
        assert int(args.local_ep) == 1, '[MU-LOWFREQ] requires local_ep=1 (one round-initial activation exchange)'
        H, HP, HM, EXT, Y, INP, UC = [], [], [], [], [], [], []
        joint = bool(getattr(args, 'mu_joint_final', False))

        for b, batch in enumerate(self.ldr_train):
            inp, y = _unpack_batch(batch, self.is_llm, dev)
            h, ext = _client_fwd(net_c, inp, self.is_llm)
            uc = [_dirs(pc, _seed_of(args, rnd, cid, b, 1000 + p), dist) for p in range(P)]
            hp, hm = [], []
            for p in range(P):
                _add(pc, uc[p], +lam); hp.append(_client_fwd(net_c, inp, self.is_llm)[0])
                _add(pc, uc[p], -2 * lam); hm.append(_client_fwd(net_c, inp, self.is_llm)[0])
                _add(pc, uc[p], +lam)
            H.append(h); HP.append(hp); HM.append(hm); EXT.append(ext); Y.append(y); INP.append(inp); UC.append(uc)
        B = len(H)
        _CM.up(*H, *[t for hp in HP for t in hp], *[t for hm in HM for t in hm], *Y, msgs=1)
        n_srv_steps = 0
        for i in range(tau):
            for b in range(B):
                us = [_dirs(ps, _seed_of(args, rnd, cid, b, 2000 + i * 64 + p), dist) for p in range(P)]
                sc = []
                for p in range(P):
                    _add(ps, us[p], +lam); lp, _ = loss_of(net_s, H[b], EXT[b], Y[b])
                    _add(ps, us[p], -2 * lam); lm, _ = loss_of(net_s, H[b], EXT[b], Y[b])
                    _add(ps, us[p], +lam)
                    sc.append((lp - lm) / (2 * lam))
                for p in range(P):
                    _add(ps, us[p], -lr_s * sc[p] / P)
                n_srv_steps += 1
        dc = torch.zeros(B, P, device=dev); b_loss, b_acc = [], []

        for b in range(B):
            if joint:
                uf = [_dirs(ps, _seed_of(args, rnd, cid, b, 3000 + p), dist) for p in range(P)]
                for p in range(P):
                    _add(ps, uf[p], +lam); lp, lg = loss_of(net_s, HP[b][p], EXT[b], Y[b])
                    _add(ps, uf[p], -2 * lam); lm, _ = loss_of(net_s, HM[b][p], EXT[b], Y[b])
                    _add(ps, uf[p], +lam)
                    dc[b, p] = (lp - lm) / (2 * lam)
                for p in range(P):
                    _add(ps, uf[p], -lr_s * float(dc[b, p]) / P)
                n_srv_steps += 1
            else:
                for p in range(P):
                    lp, lg = loss_of(net_s, HP[b][p], EXT[b], Y[b]); lm, _ = loss_of(net_s, HM[b][p], EXT[b], Y[b])
                    dc[b, p] = (lp - lm) / (2 * lam)
            b_loss.append(0.5 * (lp + lm)); b_acc.append(calculate_accuracy(lg, Y[b]).item())
        _CM.down(dc, msgs=1)
        for b in range(B):
            for p in range(P):
                _add(pc, UC[b][p], -lr_c * float(dc[b, p]) / P)

        rel = []
        for b in range(B):
            hn, _ = _client_fwd(net_c, INP[b], self.is_llm)
            rel.append(float((hn.float() - H[b].float()).norm() / (H[b].float().norm() + 1e-12)))
        print(f'[MU-DELTA] Epoch {rnd} user {cid} | ||d||/||a0|| mean={sum(rel)/len(rel):.4e} max={max(rel):.4e} n={B} | dc |med|={float(dc.abs().median()):.4e} | srv_zo_steps={n_srv_steps} (tau={tau}, B={B}, P={P})', flush=True)
        stats = dict(loss_s=sum(b_loss) / len(b_loss), acc_s=sum(b_acc) / len(b_acc), n_batches=B, n_srv_steps=n_srv_steps,
                     tau=tau, P=P, lam=lam, lr_s=lr_s, lr_c=lr_c, delta_mean=sum(rel) / len(rel), delta_max=max(rel))
        return net_c.state_dict(), net_s.state_dict(), stats
