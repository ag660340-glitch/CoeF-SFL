import math
import numpy as np
import torch


def client_adjust(labels, num_classes, tro=1.0):
    cnt = np.bincount(np.asarray(labels, dtype=np.int64), minlength=int(num_classes)).astype(np.float64)
    p = cnt / max(1.0, cnt.sum())
    return torch.from_numpy(np.log(p ** float(tro) + 1e-12)).float()


class GasStats(object):

    def __init__(self, num_classes, reg=1e-5):
        self.C = int(num_classes); self.reg = float(reg)
        self.mean = {}; self.var = {}; self.w = {}
        self.T = None; self.d = None

    def has_all(self):
        return all(c in self.mean for c in range(self.C))

    @torch.no_grad()
    def update(self, a, y, mask2d, weight):
        a = a.float(); B, T, d = a.shape
        if self.T is None:
            self.T, self.d = int(T), int(d)
        if T > self.T:
            for c in list(self.mean):
                pad = T - self.T
                self.mean[c] = torch.cat([self.mean[c], torch.zeros(pad, d, device=a.device)], 0)
                self.var[c] = torch.cat([self.var[c], torch.zeros(pad, d, device=a.device)], 0)
                self.w[c] = torch.cat([self.w[c], torch.zeros(pad, device=a.device)], 0)
            self.T = int(T)
        m = (torch.ones(B, T, device=a.device) if mask2d is None else mask2d.to(a.device).float())
        for c in y.unique().tolist():
            sel = (y == c)
            ac = a[sel]; mc = m[sel]
            n_pos = mc.sum(0)
            new_w = n_pos * float(weight)
            valid = n_pos > 0
            mu_new = (ac * mc[..., None]).sum(0) / n_pos.clamp(min=1)[:, None]
            var_new = (((ac - mu_new[None]) ** 2) * mc[..., None]).sum(0) / n_pos.clamp(min=1)[:, None]
            if c not in self.mean:
                mu = torch.zeros(self.T, d, device=a.device); var = torch.zeros(self.T, d, device=a.device); w = torch.zeros(self.T, device=a.device)
                mu[:T][valid] = mu_new[valid]; var[:T][valid] = var_new[valid]; w[:T][valid] = new_w[valid]
                self.mean[c], self.var[c], self.w[c] = mu, var, w
                continue
            mu_old, var_old, w_old = self.mean[c][:T], self.var[c][:T], self.w[c][:T]
            tot = w_old + new_w
            alpha = torch.where(tot > 0, w_old / tot.clamp(min=1e-12), torch.zeros_like(tot))
            mu = alpha[:, None] * mu_old + (1 - alpha[:, None]) * mu_new
            var = alpha[:, None] * (var_old + (mu - mu_old) ** 2) + (1 - alpha[:, None]) * (var_new + (mu - mu_new) ** 2) + self.reg
            upd = valid
            self.mean[c][:T][upd] = mu[upd]; self.var[c][:T][upd] = var[upd]; self.w[c][:T][upd] = tot[upd]

    @torch.no_grad()
    def sample(self, c, n, L, gen, device):
        mu, var, w = self.mean[c][:L], self.var[c][:L], self.w[c][:L]
        std = torch.sqrt(var.clamp(min=0) + 1e-5)
        eps = torch.randn(n, L, mu.size(1), generator=gen, device=device)
        x = mu[None] + std[None] * eps
        return x, (w > 0)


@torch.no_grad()
def make_generated_batches(stats, label_list, mask_list, n_min, bs, gen, device, template_mask, is_llm, lengths=None):
    fx_all, y_all, len_all, info = _generate_pool(stats, label_list, n_min, gen, device, is_llm, lengths)
    if not fx_all:
        return [], [], [], info
    order = sorted(range(len(fx_all)), key=lambda i: -len_all[i])
    out_fx, out_y, out_m = [], [], []
    for s in range(0, len(order), bs):
        ids = order[s:s + bs]
        L = max(len_all[i] for i in ids)
        xb = torch.zeros(len(ids), L, stats.d, device=device); mb = torch.zeros(len(ids), L, device=device)
        for r, i in enumerate(ids):
            xb[r, :len_all[i]] = fx_all[i]; mb[r, :len_all[i]] = 1.0
        out_fx.append(xb); out_y.append(torch.tensor([y_all[i] for i in ids], dtype=torch.long, device=device))
        out_m.append(_ext_mask_like(template_mask, mb) if is_llm else None)
    return out_fx, out_y, out_m, info


@torch.no_grad()
def _generate_pool(stats, label_list, n_min, gen, device, is_llm, lengths):
    ys = torch.cat([y.reshape(-1).cpu() for y in label_list])
    cnt = torch.bincount(ys, minlength=stats.C)
    need = [(c, int(n_min - cnt[c])) for c in range(stats.C) if int(cnt[c]) <= n_min and int(n_min - cnt[c]) > 0]
    absent = sum(1 for c, k in need if int(cnt[c]) == 0)
    info = {'n_gen': 0, 'labels_pad': len(need) - absent, 'labels_absent': absent}
    if not need:
        return [], [], [], info
    fx_all, y_all, len_all = [], [], []
    for c, k in need:
        if is_llm:
            Lmax = int((stats.w[c] > 0).sum())
            idx = torch.randint(0, len(lengths), (k,), generator=gen, device=device)
            for L in [min(int(lengths[i]), max(1, Lmax)) for i in idx.tolist()]:
                x, _ = stats.sample(c, 1, L, gen, device)
                fx_all.append(x[0]); y_all.append(c); len_all.append(L)
        else:
            x, _ = stats.sample(c, k, stats.T, gen, device)
            for r in range(k):
                fx_all.append(x[r]); y_all.append(c); len_all.append(stats.T)
    info['n_gen'] = len(fx_all)
    return fx_all, y_all, len_all, info


@torch.no_grad()
def mix_generated(stats, smashed_data, label_list, mask_list, n_min, gen, device, is_llm, lengths=None):
    fx_all, y_all, len_all, info = _generate_pool(stats, label_list, n_min, gen, device, is_llm, lengths)
    if not fx_all:
        return list(smashed_data), list(label_list), (list(mask_list) if is_llm else None), info
    B = len(smashed_data)
    perm = torch.randperm(len(fx_all), generator=gen, device=device).tolist()
    buckets = [[] for _ in range(B)]
    for r, i in enumerate(perm):
        buckets[r % B].append(i)
    out_fx, out_y, out_m = [], [], []
    for b in range(B):
        fx = smashed_data[b].detach().to(device); y = label_list[b].to(device).reshape(-1)
        ids = buckets[b]
        if not ids:
            out_fx.append(fx); out_y.append(y); out_m.append(mask_list[b] if is_llm else None); continue
        Tb = fx.size(1); Lg = max(len_all[i] for i in ids); L = max(Tb, Lg)
        n = fx.size(0) + len(ids)
        xb = torch.zeros(n, L, fx.size(2), device=device, dtype=fx.dtype); mb = torch.zeros(n, L, device=device)
        xb[:fx.size(0), :Tb] = fx
        if is_llm:
            mb[:fx.size(0), :Tb] = mask2d_from_ext(mask_list[b].to(device))
        else:
            mb[:fx.size(0), :Tb] = 1.0
        for r, i in enumerate(ids):
            xb[fx.size(0) + r, :len_all[i]] = fx_all[i].to(fx.dtype); mb[fx.size(0) + r, :len_all[i]] = 1.0
        out_fx.append(xb)
        out_y.append(torch.cat([y, torch.tensor([y_all[i] for i in ids], dtype=torch.long, device=device)]))
        out_m.append(_ext_mask_like(mask_list[b], mb) if is_llm else None)
    return out_fx, out_y, (out_m if is_llm else None), info


def _ext_mask_like(template, mask2d):
    if template.dim() == 4:
        neg = float(template.min()) if float(template.min()) < 0 else -10000.0
        return ((1.0 - mask2d)[:, None, None, :] * neg).to(template.dtype)
    return mask2d.to(template.dtype)


def mask2d_from_ext(ext):
    if ext is None:
        return None
    if ext.dim() == 4:
        return (ext[:, 0, 0, :] >= -0.5).float()
    return (ext > 0.5).float()


def gen_for(args, rnd, cid):
    g = torch.Generator(device=args.device)
    g.manual_seed((int(getattr(args, 'seed', 0)) * 1000003 + int(rnd) * 1009 + int(cid) * 7919) % (2 ** 31))
    return g
