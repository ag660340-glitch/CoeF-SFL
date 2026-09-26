import math
import torch


def compute_omega(z, a0, mode, m=1, log_both=False):
    out = {}
    if mode == 'omega3b' or log_both:
        w = torch.zeros_like(a0)
        for _ in range(int(m)):
            v = torch.randn_like(z)
            u = torch.autograd.grad((z * v).sum(), a0, retain_graph=True)[0]
            w = w + u.pow(2)
        out['3b'] = (w / float(m)).detach()
    if mode == 'omega3a' or log_both:
        out['3a'] = torch.autograd.grad(0.5 * (z * z).sum(), a0, retain_graph=True)[0].abs().detach()
    return out


def compute_cbar(z, a0, retain_graph=False):
    import torch.nn.functional as F
    with torch.no_grad():
        p = torch.softmax(z.float(), dim=1)
        y_hat = torch.multinomial(p, num_samples=1).squeeze(1)
    loss_hat = F.cross_entropy(z, y_hat, reduction='mean')
    g_hat = torch.autograd.grad(loss_hat, a0, retain_graph=retain_graph)[0]
    return float(g_hat.float().pow(2).sum() / g_hat.reshape(g_hat.size(0), -1).size(1))


def omega_correction_taylor(delta, omega_hat, kappa, cbar, p):
    B = delta.size(0)
    df = delta.reshape(B, -1)
    oh = omega_hat.reshape(omega_hat.size(0), -1).to(df.dtype)
    return (float(kappa) * float(cbar) * oh.pow(float(p)) * df).reshape_as(delta)


def omega_correction_steplock(delta, omega_hat, g0, kappa, p, lam_dc=3000.0):
    B = delta.size(0)
    df = delta.reshape(B, -1); gf = g0.reshape(B, -1).to(df.dtype)
    oh = omega_hat.reshape(omega_hat.size(0), -1).to(df.dtype)
    wd = oh.pow(float(p)) * df
    lam_b = float(kappa) * float(lam_dc) * float((gf * gf * df).norm()) / (float(wd.norm()) + 1e-30)
    return (lam_b * wd).reshape_as(delta), lam_b


def omega_correction_rspring(delta, omega_hat, g0, a0, rstar, p):
    B = delta.size(0)
    df = delta.reshape(B, -1); gf = g0.reshape(B, -1).to(df.dtype); af = a0.reshape(B, -1).to(df.dtype)
    oh = omega_hat.reshape(omega_hat.size(0), -1).to(df.dtype)
    wd = oh.pow(float(p)) * df
    r_b = float(df.norm(dim=1).mean()) / (float(af.norm(dim=1).mean()) + 1e-30)
    mag = float(gf.norm()) * (r_b / float(rstar))
    lam = mag / (float(wd.norm()) + 1e-30)
    return (lam * wd).reshape_as(delta), lam, r_b


def rsloc_audit(corr, delta, omega_hat, g0, a0, rstar, p, r2=None, B=None):
    import torch as _t
    with _t.no_grad():
        N = delta.size(0)
        df = delta.reshape(N, -1).double(); cf = corr.reshape(N, -1).double()
        gf = g0.reshape(N, -1).double(); af = a0.reshape(N, -1).double()
        oh = omega_hat.reshape(omega_hat.size(0), -1).double()
        wd = oh.pow(float(p)) * df
        cn = cf.reshape(-1); wn = wd.reshape(-1)
        cos_dir = float((cn @ wn) / ((cn.norm() * wn.norm()) + 1e-300)) if float(cn.norm()) > 0 else 1.0

        rb = float(_t.stack([df[i].norm() for i in range(N)]).mean()
                   / (_t.stack([af[i].norm() for i in range(N)]).mean() + 1e-300))
        mag_ref = float(gf.norm()) * rb / float(rstar)
        mag = float(cf.norm())
        mag_rel = abs(mag - mag_ref) / (mag_ref + 1e-300)
        d = dict(cos_dir=cos_dir, mag_rel=mag_rel, r_b=rb)
        if r2 is not None and B is not None:
            ref = float(r2) * (float(B) ** 0.5)
            d['rloc_rel'] = abs(float(rstar) - ref) / (ref + 1e-300)
        d['ok'] = bool(abs(cos_dir - 1.0) < 1e-6 and mag_rel < 1e-5 and d.get('rloc_rel', 0.0) < 1e-9)
        return d


def normalize_omega(omega, reduce='none'):
    oh = omega / (omega.mean() + 1e-30)
    if reduce == 'mean':
        oh = oh.mean(dim=0, keepdim=True)
    return oh.detach()


def omega_correction(delta, omega_hat, lam, p):
    B = delta.size(0)
    df = delta.reshape(B, -1)
    oh = omega_hat.reshape(omega_hat.size(0), -1).to(df.dtype)
    return (float(lam) * oh.pow(float(p)) * df).reshape_as(delta)


def calibrate_lambda_kappa(g0, omega_hat, delta_ref, p, kappa, lam_dc=3000.0):
    B = g0.size(0)
    gf = g0.reshape(B, -1).float(); d = delta_ref.reshape(B, -1).float()
    oh = omega_hat.reshape(omega_hat.size(0), -1).float()
    n_om = float((oh.pow(float(p)) * d).norm()); n_g2 = float((gf * gf * d).norm())
    lam = float(kappa) * float(lam_dc) * n_g2 / (n_om + 1e-30)
    return lam, {'g0': float(gf.norm()), 'delta': float(d.norm()), 'om_delta': n_om, 'g2_delta': n_g2}


def calibrate_lambda(g0, omega_hat, delta_ref, p, rho):
    B = g0.size(0)
    oh = omega_hat.reshape(omega_hat.size(0), -1).to(torch.float32)
    d = delta_ref.reshape(B, -1).to(torch.float32)
    den = float((oh.pow(float(p)) * d).norm())
    return float(rho) * float(g0.reshape(B, -1).float().norm()) / (den + 1e-30)


def load_delta_ref(path, key):
    obj = torch.load(path, map_location='cpu')
    if isinstance(obj, dict) and key in obj:
        obj = obj[key]
    if not torch.is_tensor(obj) or obj.dim() < 2:
        raise SystemExit(f'[OMEGA] --hd_delta_ref {path}: δ_ref..')
    return obj


def _spearman(x, y):
    x = x.reshape(-1).float(); y = y.reshape(-1).float()
    rx = x.argsort().argsort().float(); ry = y.argsort().argsort().float()
    rx = rx - rx.mean(); ry = ry - ry.mean()
    return float((rx * ry).sum() / (rx.norm() * ry.norm() + 1e-30))


@torch.no_grad()
def omega_stats(omega_hat, g0, delta, corr, omega_alt=None, omega_end=None):
    B = g0.size(0)
    gf = g0.reshape(B, -1).float(); df = delta.reshape(B, -1).float(); cf = corr.reshape(B, -1).float()
    oh = omega_hat.reshape(omega_hat.size(0), -1).float().expand(B, -1)
    eps_w = 1e-3 * oh.mean(); eps_g = 1e-3 * gf.abs().mean()
    d = {}
    d['cov_omega'] = float((oh > eps_w).float().mean()); d['cov_g'] = float((gf.abs() > eps_g).float().mean())
    mask = (oh > 0) & (gf != 0)
    if int(mask.sum()) > 2:
        lx = torch.log(oh[mask]); ly = torch.log(gf[mask].pow(2))
        lx = lx - lx.mean(); ly = ly - ly.mean()
        d['corr_log'] = float((lx * ly).sum() / (lx.norm() * ly.norm() + 1e-30))
    else:
        d['corr_log'] = float('nan')
    dn = df.norm()
    d['leak_frac'] = float((df * (gf.abs() < eps_g).float()).norm() / (dn + 1e-30)) if float(dn) > 0 else 0.0
    gn = gf.norm(); d['cnr'] = float(cf.norm() / (gn + 1e-30))
    d['omax'] = float(oh.max() / (oh.mean() + 1e-30))
    med = gf.abs().median()
    hi = gf.abs() >= med; lo = ~hi
    d['cnr_hi'] = float((cf * hi.float()).norm() / ((gf * hi.float()).norm() + 1e-30))
    d['cnr_lo'] = float((cf * lo.float()).norm() / ((gf * lo.float()).norm() + 1e-30))

    flip = ((gf + cf) * gf) < 0
    d['flip_hi'] = float(flip[hi].float().mean()) if int(hi.sum()) > 0 else 0.0
    d['flip_lo'] = float(flip[lo].float().mean()) if int(lo.sum()) > 0 else 0.0

    _k1 = max(1, int(0.01 * oh.size(1)))
    _top = torch.topk(oh[0], _k1).indices
    _cn2 = float((cf * cf).sum())
    d['corr_top1_mass'] = float((cf[:, _top] ** 2).sum() / (_cn2 + 1e-30)) if _cn2 > 0 else 0.0
    if omega_alt is not None:
        d['spearman_3a3b'] = _spearman(oh, omega_alt.reshape(omega_alt.size(0), -1).float().expand(B, -1))
    if omega_end is not None:
        d['omega_stab'] = _spearman(oh, omega_end.reshape(omega_end.size(0), -1).float().expand(B, -1))
    return d


def omega_of_batch(net_server, a0, y, mask, args, rnd, cid, bidx):
    from train.train_fl_hess_diag import masked_rng, mask_sync_on
    is_llm = args.model_name in ('RoBerta', 'RoBerta_large', 'DistilRoBerta', 'DistilBert', 'GPT2', 'Qwen')
    _ws = net_server.training
    if mask_sync_on(args):
        net_server.train()
    else:
        (net_server.eval() if getattr(args, 'hd_eval_exchange', False) else net_server.train())
    s0 = a0.to(args.device).detach().float().requires_grad_(True)
    m = mask.to(args.device) if (is_llm and mask is not None) else None
    with masked_rng(args, rnd, cid, bidx, args.device):
        logits, _ = (net_server(s0, m) if is_llm else net_server(s0))
    om = compute_omega(logits.float(), s0, str(args.hd_mode), int(getattr(args, 'hd_omega_m', 1)), False)
    oh = normalize_omega(om['3b' if str(args.hd_mode) == 'omega3b' else '3a'], str(getattr(args, 'hd_omega_reduce', 'none')))
    net_server.zero_grad(); net_server.train(_ws)
    return oh.detach()


def flip_stats(g0, corr):
    import torch as _t
    with _t.no_grad():
        gf = g0.reshape(-1).float(); cf = corr.reshape(-1).float()
        med = gf.abs().median()
        hi = gf.abs() >= med; lo = ~hi
        flip = ((gf + cf) * gf) < 0
        return dict(
            flip_hi=float(flip[hi].float().mean()) if int(hi.sum()) > 0 else 0.0,
            flip_lo=float(flip[lo].float().mean()) if int(lo.sum()) > 0 else 0.0,
            cnr=float(cf.norm() / (gf.norm() + 1e-30)),
            cnr_hi=float((cf * hi.float()).norm() / ((gf * hi.float()).norm() + 1e-30)),
            cnr_lo=float((cf * lo.float()).norm() / ((gf * lo.float()).norm() + 1e-30)))


def omega_line(rnd, b, lam, st):
    keys = ['cov_omega', 'cov_g', 'corr_log', 'leak_frac', 'cnr', 'cnr_hi', 'cnr_lo', 'flip_hi', 'flip_lo', 'omax', 'corr_top1_mass', 'spearman_3a3b', 'omega_stab']
    body = ' '.join(f'{k}={st[k]:.4f}' for k in keys if k in st)
    return f'[OMEGA] r={rnd} b={b} lambda={lam:.6e} {body}'


M_ELL = {"ce": 0.5, "bce": 0.25, "mse": 2.0}


def rslocM_task(args):
    from train.task_loss import task_type
    t = task_type(args)
    if t == 'cls':
        return 'ce', 1
    if t == 'qa':
        return 'ce', 2
    if t == 'lm' and str(getattr(args, 'hd_mode', '')) == 'omegaMpJ':
        return 'ce', 'lm'
    if t == 'reg':
        return 'mse', 1
    raise AssertionError(f'[RSLOCM] /: task_type={t} (ce QA, lm rsloc_MpJ, reg §3)')


def rslocM_weight(args, n_batch):
    _, k = rslocM_task(args)
    return 1.0 / (float(k) * float(n_batch))


def compute_omega_M(z, a0, n_batch, exact_mode='auto', gen=None, return_blocks=False):
    N = int(n_batch); R = int(z.size(0)); C = int(z.size(1))
    assert R % N == 0, f'[RSLOCM] {R} {N} '
    K = R // N
    exact = (exact_mode == 'exact') or (exact_mode == 'auto' and C <= 8)
    blocks = []
    nb = 0
    for k in range(K):
        rows = slice(k * N, (k + 1) * N)
        w = torch.zeros_like(a0)
        if exact:
            for c in range(C):
                v = torch.zeros_like(z); v[rows, c] = 1.0
                u = torch.autograd.grad((z * v).sum(), a0, retain_graph=True)[0]
                w = w + u.pow(2); nb += 1
        else:
            v = torch.zeros_like(z)
            r = (torch.randint(0, 2, (N, C), device=z.device, generator=gen).to(z.dtype) * 2.0 - 1.0)
            v[rows] = r
            u = torch.autograd.grad((z * v).sum(), a0, retain_graph=True)[0]
            w = w + u.pow(2); nb += 1
        blocks.append(w.detach())
    info = {'C': C, 'K': K, 'exact': bool(exact), 'n_backward': nb}
    if return_blocks:
        return torch.stack(blocks, 0), info
    return sum(blocks), info


def rslocM_correction(delta, omega, w_n, m_ell):
    B = delta.size(0)
    df = delta.reshape(B, -1)
    of = omega.reshape(omega.size(0), -1).to(df.dtype)
    return (float(w_n) * float(m_ell) * of * df).reshape_as(delta)


def rslocM_line(rnd, user, b, omega, corr, g_tilde):
    of = omega.reshape(-1).float()
    share = float(corr.reshape(-1).norm() / (g_tilde.reshape(-1).norm() + 1e-30))
    return ('[RSLOCM-OMEGA] Epoch %s user %s b=%d | omega mean/med/max=%.4e/%.4e/%.4e | share=||corr||/||g~||=%.4f'
            % (rnd, user, b, float(of.mean()), float(of.median()), float(of.max()), share))


L_S = 3.0


def rslocMp_m(z, n_batch, m_ell, mode='trace', loss_kind='ce'):
    if loss_kind == 'mse':
        N = int(n_batch); K = int(z.size(0)) // N
        return torch.full((K, N), float(m_ell), device=z.device, dtype=torch.float32)
    with torch.no_grad():
        p = torch.softmax(z.float(), dim=1)
        if mode == 'exact':
            S = torch.diag_embed(p) - p.unsqueeze(2) * p.unsqueeze(1)
            lam = torch.linalg.eigvalsh(S)[:, -1]
        else:
            lam = 1.0 - (p * p).sum(1)
        m = torch.clamp(lam, max=float(m_ell))
    N = int(n_batch); K = int(z.size(0)) // N
    return m.reshape(K, N)


def rslocMp_pack(omega_blocks, m):
    K, N = m.shape
    of = omega_blocks.reshape(K, N, -1)
    return torch.cat([of, m.to(of.dtype).reshape(K, N, 1)], dim=2)


def rslocMp_unpack(packed, delta_shape):
    K, N, D1 = packed.shape
    D = D1 - 1
    return packed[..., :D].reshape(K, N, *delta_shape[1:]), packed[..., D]


def rslocMp_correction(delta, omega_blocks, m, w_n, m_ell, l_s=L_S):
    K = omega_blocks.size(0); N = delta.size(0)
    df = delta.reshape(N, -1)
    corr = torch.zeros_like(df)
    cs, Ds = [], []
    for k in range(K):
        ok = omega_blocks[k].reshape(N, -1).to(df.dtype)
        Dk = torch.sqrt((ok * df * df).sum(1).clamp_min(0.0))
        ck = torch.clamp(m[k].to(df.dtype) + 0.5 * float(l_s) * Dk, max=float(m_ell))
        corr = corr + float(w_n) * ck.unsqueeze(1) * ok * df
        cs.append(ck); Ds.append(Dk)
    c_all = torch.cat(cs); D_all = torch.cat(Ds)
    stats = {'m_med': float(m.float().median()), 'm_max': float(m.float().max()),
             'D_med': float(D_all.float().median()), 'D_max': float(D_all.float().max()),
             'c_med': float(c_all.float().median()), 'c_max': float(c_all.float().max()),
             'frac_cap': float((c_all >= float(m_ell) - 1e-12).float().mean())}
    return corr.reshape_as(delta), stats


def rslocMp_line(rnd, user, b, st):
    return ('[RSLOCMP] Epoch %s user %s b=%d | m_n med/max=%.4f/%.4f | Delta med/max=%.4e/%.4e | c med/max=%.4f/%.4f | frac_cap=%.3f'
            % (rnd, user, b, st['m_med'], st['m_max'], st['D_med'], st['D_max'], st['c_med'], st['c_max'], st['frac_cap']))


G_ELL = {"ce": 1.4142135623730951, "bce": 1.0, "mse": float("inf")}


def _cG_of(m_k, Dk, m_ell, g_ell, l_s):
    c_path = m_k + 0.5 * float(l_s) * Dk
    _g = torch.as_tensor(g_ell, dtype=Dk.dtype, device=Dk.device)
    c_G = torch.where(Dk > 0, _g / Dk.clamp_min(1e-300), torch.full_like(Dk, float('inf')))
    cand = torch.stack([torch.full_like(Dk, float(m_ell)), c_path, c_G], 0)
    which = cand.argmin(0)
    return cand.min(0).values, which


def rslocMpG_correction(delta, omega_blocks, m, w_n, m_ell, g_ell, l_s=L_S):
    K = omega_blocks.size(0); N = delta.size(0)
    df = delta.reshape(N, -1)
    corr = torch.zeros_like(df)
    cs, Ds, ws = [], [], []
    for k in range(K):
        ok = omega_blocks[k].reshape(N, -1).to(df.dtype)
        Dk = torch.sqrt((ok * df * df).sum(1).clamp_min(0.0))
        ck, wk = _cG_of(m[k].to(df.dtype), Dk, m_ell, g_ell, l_s)
        corr = corr + float(w_n) * ck.unsqueeze(1) * ok * df
        cs.append(ck); Ds.append(Dk); ws.append(wk)
    c_all = torch.cat(cs); D_all = torch.cat(Ds); w_all = torch.cat(ws); cD = c_all * D_all
    stats = {'m_med': float(m.float().median()), 'm_max': float(m.float().max()),
             'D_med': float(D_all.float().median()), 'D_max': float(D_all.float().max()),
             'c_med': float(c_all.float().median()), 'c_max': float(c_all.float().max()),
             'cD_med': float(cD.float().median()), 'cD_max': float(cD.float().max()),
             'frac_cap': float((w_all == 0).float().mean()), 'frac_G': float((w_all == 2).float().mean())}
    return corr.reshape_as(delta), stats


def rslocMpG_line(rnd, user, b, st):
    return ('[RSLOCMP] Epoch %s user %s b=%d | m_n med/max=%.4f/%.4f | Delta med/max=%.4e/%.4e | c med/max=%.4f/%.4f | cDelta med/max=%.4e/%.4e | frac_cap=%.3f | frac_G=%.3f'
            % (rnd, user, b, st['m_med'], st['m_max'], st['D_med'], st['D_max'], st['c_med'], st['c_max'], st['cD_med'], st['cD_max'], st['frac_cap'], st['frac_G']))


def compute_J_blocks(z, a0, n_batch, exact_mode='auto', k_probes=1, gen=None):
    N = int(n_batch); Rz = int(z.size(0)); C = int(z.size(1))
    assert Rz % N == 0, f'[RSLOCMPJ] {Rz} {N} '
    K = Rz // N
    exact = (exact_mode == 'exact') or (exact_mode == 'auto' and C <= 8)
    R = C if exact else int(k_probes)
    assert R >= 1
    blocks = []; nb = 0
    for k in range(K):
        rows = slice(k * N, (k + 1) * N)
        us = []
        for r in range(R):
            v = torch.zeros_like(z)
            if exact:
                v[rows, r] = 1.0
            else:
                v[rows] = (torch.randint(0, 2, (N, C), device=z.device, generator=gen).to(z.dtype) * 2.0 - 1.0)
            u = torch.autograd.grad((z * v).sum(), a0, retain_graph=True)[0]
            us.append(u.detach()); nb += 1
        blocks.append(torch.stack(us, 1))
    info = {'C': C, 'K': K, 'mode': 'E' if exact else 'P', 'R': R, 'n_backward': nb}
    return torch.stack(blocks, 0), info


def rslocMpJ_pack(J_blocks, m):
    K, N, R = J_blocks.shape[:3]
    jf = J_blocks.reshape(K, N, -1)
    return torch.cat([jf, m.to(jf.dtype).reshape(K, N, 1)], dim=2)


def rslocMpJ_unpack(packed, delta_shape):
    K, N, D1 = packed.shape
    D = 1
    for s in delta_shape[1:]:
        D *= int(s)
    R = (D1 - 1) // D
    assert R * D + 1 == D1, f'[RSLOCMPJ] D1={D1} D={D}'
    return packed[..., :R * D].reshape(K, N, R, D), packed[..., R * D]


def rslocMpJ_correction(delta, J_blocks, m, w_n, m_ell, g_ell, mode, l_s=L_S, with_diag=True, coef='cG'):
    K, N, R, D = J_blocks.shape
    df = delta.reshape(N, -1).to(J_blocks.dtype)
    assert df.size(1) == D
    scale = 1.0 if mode == 'E' else 1.0 / float(R)
    corr = torch.zeros_like(df); corr_diag = torch.zeros_like(df)
    cs, Ds, ws = [], [], []
    for k in range(K):
        Jk = J_blocks[k]
        s = torch.einsum('nrd,nd->nr', Jk, df)
        Dk = torch.sqrt((scale * (s * s).sum(1)).clamp_min(0.0))
        gk = (g_ell[k].to(df.dtype) if torch.is_tensor(g_ell) else g_ell)
        if coef == 'const':
            ck = torch.full_like(Dk, float(m_ell)); wk = torch.zeros_like(Dk, dtype=torch.long)
        else:
            ck, wk = _cG_of(m[k].to(df.dtype), Dk, m_ell, gk, l_s)
        wgt = (w_n[k].to(df.dtype) if torch.is_tensor(w_n) else float(w_n))
        corr = corr + (wgt * ck).unsqueeze(1) * scale * torch.einsum('nr,nrd->nd', s, Jk)
        cs.append(ck); Ds.append(Dk); ws.append(wk)
        if with_diag:
            ok = scale * (Jk * Jk).sum(1)
            Dd = torch.sqrt((ok * df * df).sum(1).clamp_min(0.0))
            cd, _ = _cG_of(m[k].to(df.dtype), Dd, m_ell, gk, l_s)
            corr_diag = corr_diag + (wgt * cd).unsqueeze(1) * ok * df
    c_all = torch.cat(cs); D_all = torch.cat(Ds); w_all = torch.cat(ws); cD = c_all * D_all

    quad2 = float(torch.stack([((w_n[k].to(df.dtype) if torch.is_tensor(w_n) else float(w_n)) * cs[k] * Ds[k] * Ds[k]).sum() for k in range(K)], 0).sum())
    stats = {'m_med': float(m.float().median()), 'm_max': float(m.float().max()),
             'D_med': float(D_all.float().median()), 'D_max': float(D_all.float().max()),
             'c_med': float(c_all.float().median()), 'c_max': float(c_all.float().max()),
             'cD_med': float(cD.float().median()), 'cD_max': float(cD.float().max()),
             'frac_cap': float((w_all == 0).float().mean()), 'frac_G': float((w_all == 2).float().mean()),
             'ratio_diag': float(corr.norm() / (corr_diag.norm() + 1e-30)) if with_diag else float('nan'),
             'quad2': quad2, 'mode': mode}
    return corr.reshape_as(delta), stats


def rslocMpJ_line(rnd, user, b, mode, R, st, share):
    return ('[RSLOCMPJ] Epoch %s user %s b=%d | mode=%s k=%d | Dhat med/max=%.4e/%.4e | c med/max=%.4f/%.4f | cDhat med/max=%.4e/%.4e | frac_cap=%.3f | frac_G=%.3f | share=||corr||/||g~||=%.4f | ratio_diag=||corr_J||/||corr_omega||=%.4e'
            % (rnd, user, b, mode, R, st['D_med'], st['D_max'], st['c_med'], st['c_max'], st['cD_med'], st['cD_max'], st['frac_cap'], st['frac_G'], share, st['ratio_diag']))


def compute_J_blocks_lm(z, a0, n_batch, y, k_probes=1, gen=None):
    N = int(n_batch); R = int(z.size(0)); V = int(z.size(1)); T1 = R // N
    assert T1 * N == R, f'[MpJ-LM] {R} N={N} '
    yv = y.reshape(N, T1); valid = (yv != -100)
    nvalid = int(valid.sum().item()); assert nvalid > 0, '[MpJ-LM] no valid tokens'
    pos = [t for t in range(T1) if bool(valid[:, t].any())]
    ar = torch.arange(N, device=z.device)
    blocks = []; nb = 0
    for t in pos:
        rows = ar * T1 + t; zt = z.index_select(0, rows)
        us = []
        for r in range(int(k_probes)):
            v = (torch.randint(0, 2, (N, V), device=z.device, generator=gen).to(z.dtype) * 2.0 - 1.0)
            u = torch.autograd.grad((zt * v).sum(), a0, retain_graph=True)[0]
            us.append(u.detach()); nb += 1
        blocks.append(torch.stack(us, 1))
    w = (valid[:, pos].to(a0.dtype).T.contiguous() / float(nvalid))
    info = {'C': V, 'K': len(pos), 'mode': 'L', 'R': int(k_probes), 'n_backward': nb, 'T1': T1, 'n_valid': nvalid}
    return torch.stack(blocks, 0), info, pos, w


def rslocMp_m_lm(z, n_batch, pos, m_ell, chunk=256):
    N = int(n_batch); T1 = int(z.size(0)) // N
    ar = torch.arange(N, device=z.device)
    idx = torch.cat([ar * T1 + t for t in pos])
    out = []
    with torch.no_grad():
        for s0 in range(0, idx.numel(), chunk):
            p = torch.softmax(z.index_select(0, idx[s0:s0 + chunk]).float(), dim=1)
            out.append(torch.clamp(1.0 - (p * p).sum(1), max=float(m_ell)))
    return torch.cat(out).reshape(len(pos), N)


def rslocMpJ_pack_lm(J_blocks, m, w):
    K, N = m.shape; jf = J_blocks.reshape(K, N, -1)
    return torch.cat([jf, m.to(jf.dtype).reshape(K, N, 1), w.to(jf.dtype).reshape(K, N, 1)], dim=2)


def rslocMpJ_unpack_lm(packed, delta_shape):
    K, N, D2 = packed.shape
    D = 1
    for s_ in delta_shape[1:]:
        D *= int(s_)
    R = (D2 - 2) // D
    assert R * D + 2 == D2, f'[MpJ-LM] D2={D2} D={D}'
    return packed[..., :R * D].reshape(K, N, R, D), packed[..., R * D], packed[..., R * D + 1]


def compute_J_sample_lm(z, a0, n_batch, y, k_probes=1, gen=None):
    N = int(n_batch); R = int(z.size(0)); V = int(z.size(1)); T1 = R // N
    assert T1 * N == R, f'[MpJ-LM] {R} N={N} '
    valid = (y.reshape(N, T1) != -100); nt = valid.sum(1)
    nvalid = int(nt.sum().item()); assert nvalid > 0, '[MpJ-LM] no valid tokens'
    vm = valid.reshape(R, 1).to(z.dtype)
    us = []
    for r in range(int(k_probes)):
        v = (torch.randint(0, 2, (R, V), device=z.device, generator=gen).to(z.dtype) * 2.0 - 1.0) * vm
        u = torch.autograd.grad((z * v).sum(), a0, retain_graph=True)[0]
        us.append(u.detach())
    J = torch.stack(us, 1).unsqueeze(0)
    w = ((nt > 0).to(a0.dtype) / float(nvalid)).reshape(1, N)
    G = (float(G_ELL['ce']) * torch.sqrt(nt.to(a0.dtype))).reshape(1, N)
    info = {'C': V, 'K': 1, 'mode': 'S', 'R': int(k_probes), 'n_backward': int(k_probes), 'T1': T1, 'n_valid': nvalid}
    return J, info, w, G


def rslocMp_m_lm_sample(z, n_batch, y, m_ell, chunk=1024):
    N = int(n_batch); R = int(z.size(0)); T1 = R // N
    valid = (y.reshape(N, T1) != -100)
    tr = torch.zeros(R, device=z.device, dtype=torch.float32)
    with torch.no_grad():
        for s0 in range(0, R, chunk):
            p = torch.softmax(z[s0:s0 + chunk].float(), dim=1); tr[s0:s0 + chunk] = 1.0 - (p * p).sum(1)
    tr = (tr.reshape(N, T1) * valid.float()).sum(1)
    return torch.clamp(tr, max=float(m_ell)).reshape(1, N)


def rslocMpJ_pack_lm1(J, m, w, G):
    K, N = m.shape; jf = J.reshape(K, N, -1)
    return torch.cat([jf, m.to(jf.dtype).reshape(K, N, 1), w.to(jf.dtype).reshape(K, N, 1), G.to(jf.dtype).reshape(K, N, 1)], dim=2)


def rslocMpJ_unpack_lm1(packed, delta_shape):
    K, N, D3 = packed.shape
    D = 1
    for s_ in delta_shape[1:]:
        D *= int(s_)
    R = (D3 - 3) // D
    assert R * D + 3 == D3, f'[MpJ-LM] D3={D3} D={D}'
    return packed[..., :R * D].reshape(K, N, R, D), packed[..., R * D], packed[..., R * D + 1], packed[..., R * D + 2]


LOSS_CONST = {"ce": (0.5, 3.0, 1.4142135623730951), "bce": (0.25, 1.0 / (6.0 * 3.0 ** 0.5), 1.0), "mse": (2.0, 0.0, float("inf"))}


def mpjs_loss_kind(args):
    from train.task_loss import task_type
    t = task_type(args)
    k = {'cls': 'ce', 'qa': 'ce', 'lm': 'ce', 'reg': 'mse'}.get(t)
    assert k in LOSS_CONST, f'[MPJS]: task_type={t} LOSS_CONST (M_ℓ, L_S, G_ℓ) '
    return k


def mpjs_hvp_factory(z, y, crit, denom):
    loss_sum = crit(z, y) * float(denom)
    gz = torch.autograd.grad(loss_sum, z, create_graph=True, retain_graph=True)[0]
    def hvp(v):
        return torch.autograd.grad(gz, z, grad_outputs=v, retain_graph=True)[0]
    return gz.detach(), hvp


def mpjs_mu0(hvp, gz, R, C, exact, iters=10, gen=None, device=None):
    if exact:
        cols = []
        for c in range(C):
            e = torch.zeros(R, C, device=gz.device, dtype=gz.dtype); e[:, c] = 1.0
            cols.append(hvp(e).detach())
        S = torch.stack(cols, 2)
        S = 0.5 * (S + S.transpose(1, 2))
        mu = torch.linalg.eigvalsh(S.float())[:, -1].to(gz.dtype)
        return mu, S
    v = torch.randn(R, C, device=gz.device, dtype=gz.dtype, generator=gen)
    v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-30)
    mu = torch.zeros(R, device=gz.device, dtype=gz.dtype)
    for _ in range(int(iters)):
        Sv = hvp(v).detach()
        mu = (v * Sv).sum(1)
        nrm = Sv.norm(dim=1, keepdim=True)
        v = torch.where(nrm > 0, Sv / nrm.clamp_min(1e-30), v)
    return mu.clamp_min(0.0), None


def compute_mpjs_server(z, a0, y, crit, denom, n_batch, exact_mode='auto', k_probes=1, gen=None, lm_major=False):
    N = int(n_batch); R = int(z.size(0)); C = int(z.size(1)); K = R // N
    assert K * N == R, f'[MPJS] {R} N={N} '
    gz, hvp = mpjs_hvp_factory(z, y, crit, denom)
    ones = torch.ones_like(z)
    support = ((gz.abs().sum(1) > 0) | (hvp(ones).detach().abs().sum(1) > 0))
    exact = (exact_mode == 'exact') or (exact_mode == 'auto' and C <= 8 and K == 1)
    D = a0[0].numel()

    if lm_major:
        samp = torch.arange(R, device=z.device) // K
    else:
        samp = torch.arange(R, device=z.device) % N
    nz = torch.zeros(N, device=z.device, dtype=torch.long).index_add_(0, samp, support.long())
    info = {'C': C, 'K': K, 'R': R, 'n_backward_net': 0, 'n_hvp': 0}
    if exact:
        mu_r, S = mpjs_mu0(hvp, gz, R, C, True)
        info['n_hvp'] += C
        Js = []
        for c in range(C):
            e = torch.zeros_like(z); e[:, c] = 1.0
            Js.append(torch.autograd.grad((z * e).sum(), a0, retain_graph=True)[0].detach().reshape(N, -1)); info['n_backward_net'] += 1
        J = torch.stack(Js, 1)
        info['mode'] = 'E'
        return {'J': J, 'S': S.detach(), 'mu': mu_r.detach(), 'nz': nz}, info
    mu_r, _ = mpjs_mu0(hvp, gz, R, C, False, gen=gen); info['n_hvp'] += 10
    mu = torch.zeros(N, device=z.device, dtype=z.dtype).index_reduce_(0, samp, mu_r.detach(), 'amax', include_self=True)
    us, wts = [], []
    for j in range(int(k_probes)):
        v = (torch.randint(0, 2, (R, C), device=z.device, generator=gen).to(z.dtype) * 2.0 - 1.0) * support.to(z.dtype).unsqueeze(1)
        Sv = hvp(v).detach(); info['n_hvp'] += 1
        u = torch.autograd.grad((z * v).sum(), a0, retain_graph=True)[0].detach().reshape(N, -1)
        wt = torch.autograd.grad((z * Sv).sum(), a0, retain_graph=True)[0].detach().reshape(N, -1)
        info['n_backward_net'] += 2
        us.append(u); wts.append(wt)
    info['mode'] = 'P'
    return {'u': torch.stack(us, 1), 'wt': torch.stack(wts, 1), 'mu': mu, 'nz': nz}, info


def mpjs_pack(srv):
    N = srv['mu'].numel()
    if 'J' in srv:
        return torch.cat([srv['J'].reshape(N, -1), srv['S'].reshape(N, -1), srv['mu'].reshape(N, 1), srv['nz'].to(srv['J'].dtype).reshape(N, 1)], 1), 'E'
    return torch.cat([srv['u'].reshape(N, -1), srv['wt'].reshape(N, -1), srv['mu'].reshape(N, 1), srv['nz'].to(srv['u'].dtype).reshape(N, 1)], 1), 'P'


def mpjs_unpack(packed, mode, D, C=None, k=None):
    N = packed.size(0)
    if mode == 'E':
        J = packed[:, :C * D].reshape(N, C, D); S = packed[:, C * D:C * D + C * C].reshape(N, C, C)
        return {'J': J, 'S': S, 'mu': packed[:, -2], 'nz': packed[:, -1]}
    return {'u': packed[:, :k * D].reshape(N, k, D), 'wt': packed[:, k * D:2 * k * D].reshape(N, k, D), 'mu': packed[:, -2], 'nz': packed[:, -1]}


def mpjs_weight(kind_task, n_batch, nz):
    N = float(n_batch)
    if kind_task == 'lm':
        return 1.0 / float(nz.sum().clamp_min(1))
    if kind_task == 'qa':
        return 1.0 / (2.0 * N)
    return 1.0 / N


def mpjs_correction(delta, srv, mode, w, consts, C):
    M, L_S, G = consts
    N = delta.size(0); df = delta.reshape(N, -1); mu = srv['mu'].to(df.dtype)
    if mode == 'E':
        J, S = srv['J'].to(df.dtype), srv['S'].to(df.dtype)
        t = torch.einsum('ncd,nd->nc', J, df)
        Dh = t.norm(dim=1)
        lam_path = mu + 0.5 * L_S * Dh
        use_path = (lam_path <= M)
        q_path = torch.einsum('nij,nj->ni', S, t) + (0.5 * L_S * Dh).unsqueeze(1) * t
        q_ceil = M * t
        q = torch.where(use_path.unsqueeze(1), q_path, q_ceil)
        qn = q.norm(dim=1); sat = torch.clamp(G / qn.clamp_min(1e-30), max=1.0) if G != float('inf') else torch.ones_like(qn)
        q = q * sat.unsqueeze(1)
        corr = torch.einsum('ncd,nc->nd', J, q) * float(w)
        lam_C = torch.where(use_path, lam_path, torch.full_like(lam_path, M))

        c_sc = torch.minimum(torch.full_like(Dh, M), torch.minimum(lam_path, torch.where(Dh > 0, G / Dh.clamp_min(1e-30), torch.full_like(Dh, float('inf')))))
        corr_sc = torch.einsum('ncd,nc->nd', J, c_sc.unsqueeze(1) * t) * float(w)
    else:
        u, wt = srv['u'].to(df.dtype), srv['wt'].to(df.dtype); k = u.size(1)
        su = torch.einsum('nkd,nd->nk', u, df); sw = torch.einsum('nkd,nd->nk', wt, df)
        Dh = torch.sqrt((su * su).mean(1).clamp_min(0.0))
        r_S = 0.5 * (torch.einsum('nk,nkd->nd', su, wt) + torch.einsum('nk,nkd->nd', sw, u)) / k
        r_I = torch.einsum('nk,nkd->nd', su, u) / k
        lam_path = mu + 0.5 * L_S * Dh; use_path = (lam_path <= M)
        r = torch.where(use_path.unsqueeze(1), r_S + (0.5 * L_S * Dh).unsqueeze(1) * r_I, M * r_I)
        Cdim = int(C)
        Jest = torch.sqrt((u * u).sum(2).mean(1) / Cdim)
        rn = r.norm(dim=1); sat = torch.clamp(G * Jest / rn.clamp_min(1e-30), max=1.0) if G != float('inf') else torch.ones_like(rn)
        r = r * sat.unsqueeze(1); corr = r * float(w)
        lam_C = torch.where(use_path, lam_path, torch.full_like(lam_path, M))
        c_sc = torch.minimum(torch.full_like(Dh, M), torch.minimum(lam_path, torch.where(Dh > 0, G / Dh.clamp_min(1e-30), torch.full_like(Dh, float('inf')))))
        corr_sc = c_sc.unsqueeze(1) * r_I * float(w)
    stats = {'D_med': float(Dh.median()), 'D_max': float(Dh.max()), 'mu_med': float(mu.median()), 'mu_max': float(mu.max()),
             'lamC_med': float(lam_C.median()), 'lamC_max': float(lam_C.max()),
             'frac_ceil': float((~use_path).float().mean()), 'frac_G': float((sat < 1.0 - 1e-12).float().mean()),
             'ratio_scalar': float(corr.norm() / (corr_sc.norm() + 1e-30))}
    return corr.reshape_as(delta), stats


def mpjs_line(rnd, user, b, kind, mode, k, K, st, share):
    return ('[MPJS] Epoch %s user %s b=%d | loss=%s | mode=%s k=%d | blocks=%d | Dhat med/max=%.4e/%.4e | mu0 med/max=%.4f/%.4f | lam_max_Chat med/max=%.4f/%.4f | frac_ceil=%.3f | frac_G=%.3f | share=%.4f | ratio_scalar=%.4e'
            % (rnd, user, b, kind, mode, k, K, st['D_med'], st['D_max'], st['mu_med'], st['mu_max'], st['lamC_med'], st['lamC_max'], st['frac_ceil'], st['frac_G'], share, st['ratio_scalar']))
