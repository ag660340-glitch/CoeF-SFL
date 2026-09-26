import math
import torch
import torch.nn.functional as F

from train import task_loss as _TL


def server_kappa_probe(net_server, a0, lab, mask, is_llm, args, return_raw=False):
    assert bool(getattr(args, 'hd_hvp_exact', True)),\
        '[KPROBE] FD-HVP not implemented (only --hd_hvp_exact True)'
    tau = float(getattr(args, 'hd_kprobe_gnorm_tau', 1e-8))
    gate = bool(getattr(args, 'hd_kprobe_gate_neg', True))

    _form = str(getattr(args, 'hd_kprobe_form', 'diag'))
    _hvp = getattr(args, 'hd_kprobe_hvp', None)
    _skip_hvp = (not bool(_hvp)) if _hvp is not None else _form.startswith('const')
    a = a0.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        z = (net_server(a, mask)[0] if is_llm else net_server(a)[0])
        if _skip_hvp:
            g = torch.autograd.grad(_TL.gnll(z, lab), a)[0]
            B = a.size(0)
            gf = g.detach().reshape(B, -1)
            gn = gf.norm(dim=1)
            kappa_raw = torch.zeros(B, device=a.device, dtype=gf.dtype)
        else:
            g = torch.autograd.grad(_TL.gnll(z, lab), a, create_graph=True)[0]
            B = a.size(0)
            gf = g.detach().reshape(B, -1)
            gn = gf.norm(dim=1)
            ghat = gf / (gn.unsqueeze(1) + 1e-12)
            Hg = torch.autograd.grad((g * ghat.reshape_as(g)).sum(), a)[0]
            kappa_raw = (ghat * Hg.detach().reshape(B, -1)).sum(dim=1)
            del Hg
    kappa = kappa_raw.clone()
    kappa[gn < tau] = 0.0
    if gate:
        kappa = kappa.clamp(min=0.0)
    net_server.zero_grad()
    g0 = g.detach()

    with torch.no_grad():
        _zf = z.detach().float().reshape(-1, z.size(-1)); _yy = lab.reshape(-1).long()
        _r0n = None
        if _zf.size(1) == 1 and lab.dtype.is_floating_point:
            _r0n = (2.0 * (_zf.reshape(-1) - lab.reshape(-1).float())).abs(); args._dca_r0vec_last = None; args._dca_p_last = None
        elif _zf.size(0) == _yy.numel():
            _p = torch.softmax(_zf, 1); _v = (_yy != -100); _r = _p.clone()
            _r[torch.arange(_zf.size(0), device=_zf.device)[_v], _yy[_v]] -= 1.0; _r[~_v] = 0.0
            _r0n = _r.norm(dim=1)
            args._dca_r0vec_last = (_r.detach() if _zf.size(0) == B else None)
            args._dca_p_last = (_p.detach() if _zf.size(0) == B else None)
            if _zf.size(0) != B and _zf.size(0) % B == 0:
                _r0n = (_r * _r).sum(1).reshape(-1, B).sum(0).sqrt()
    args._dca_r0_last = _r0n
    del a, z, g
    if return_raw:
        return g0, kappa.detach(), kappa_raw.detach()
    return g0, kappa.detach()


def round_perm(n, rnd, seed, device):
    g = torch.Generator(device='cpu'); g.manual_seed((int(seed) * 1000003 + int(rnd) * 7919) & 0x7FFFFFFF)
    return torch.randperm(int(n), generator=g).to(device)


def correct_kprobe(delta, g0, kappa, form='diag', alpha=1.0, lam_const=None, lam0=None, a_eps=1e-7, perm=None, r0n=None):
    B = g0.size(0)
    df = delta.reshape(B, -1)
    gf = g0.reshape(B, -1)
    if form == 'const':
        assert lam_const is not None, '[KPROBE] const form requires lam_const'
        corr = float(lam_const) * (gf * gf) * df
    elif form == 'const_perm':

        assert lam_const is not None and perm is not None, '[KPROBE] const_perm form requires lam_const and perm'
        corr = float(lam_const) * (gf * gf)[:, perm] * df
    elif form == 'const_iso':

        assert lam_const is not None, '[KPROBE] const_iso form requires lam_const'
        corr = float(lam_const) * (gf * gf).mean(dim=1, keepdim=True) * df
    elif form == 'const_rn':

        assert lam_const is not None and r0n is not None, '[KPROBE] const_rn form requires lam_const and r0n'
        corr = float(lam_const) * (gf * gf) * df / (r0n.reshape(B, 1).to(gf.dtype) ** 2).clamp_min(1e-12)
    elif form == 'const_a':
        assert lam0 is not None, '[KPROBE] const_a form requires lam0'
        corr = float(lam0) * (gf * gf) / (gf.abs() + float(a_eps)) * df
    else:
        k = kappa.reshape(B, 1).to(gf.dtype) * float(alpha)
        n2 = (gf * gf).sum(dim=1, keepdim=True)
        if form == 'diag':
            corr = k * (gf * gf) / (n2 + 1e-24) * df
        elif form == 'rank1':
            ghat = gf / (n2.sqrt() + 1e-12)
            corr = k * ghat * (ghat * df).sum(dim=1, keepdim=True)
        else:
            raise ValueError(f'[KPROBE] form={form!r}')
    return g0 + corr.reshape_as(g0)


def _med(xs):
    return float(torch.as_tensor(xs).float().median()) if len(xs) else float('nan')


@torch.no_grad()
def kprobe_log_perbatch(diag, kappa, kappa_raw, g0, delta, corr, a0, gnorm_tau):
    B = g0.size(0)
    gf = g0.reshape(B, -1); df = delta.reshape(B, -1); cf = corr.reshape(B, -1); af = a0.reshape(B, -1)
    gn = gf.norm(dim=1); dn = df.norm(dim=1); an = af.norm(dim=1)
    k = kappa.float()
    d = diag
    d.setdefault('kp_kappa_min', []).append(float(k.min())); d.setdefault('kp_kappa_med', []).append(float(k.median()))
    d.setdefault('kp_kappa_max', []).append(float(k.max()))
    d.setdefault('kp_neg_frac', []).append(float((kappa_raw < 0).float().mean()) if kappa_raw is not None else float('nan'))
    d.setdefault('kp_tau_frac', []).append(float((gn < gnorm_tau).float().mean()))
    d.setdefault('kp_corr_ratio', []).append(float((cf.norm(dim=1) / (gn + 1e-12)).mean()))
    eff = k / (gn * gn + 1e-24)
    d.setdefault('kp_eff_lam_min', []).append(float(eff.min())); d.setdefault('kp_eff_lam_max', []).append(float(eff.max()))
    d.setdefault('kp_eff_lam_med', []).append(float(eff.median()))
    d.setdefault('kd_norm', []).append(float(dn.mean()))
    d.setdefault('kd_ratio', []).append(float((dn / (an + 1e-12)).mean()))
    d.setdefault('kd_rho_g', []).append(float(F.cosine_similarity(df, gf, dim=1).mean()))
    d.setdefault('kd_rho_a', []).append(float(F.cosine_similarity(df, af, dim=1).mean()))
    prev = d.get('_kp_prev_delta', None)
    if prev is not None and prev.shape == df.shape:
        d.setdefault('kd_pers', []).append(float(F.cosine_similarity(df.reshape(1, -1), prev.reshape(1, -1), dim=1)))
    d['_kp_prev_delta'] = df.detach().clone()


def kprobe_log_round(diag, rnd, user, args):
    d = diag
    if not d.get('kp_kappa_med'):
        return {}
    def _pb(k, f='{:.3e}'):
        return ' '.join(f.format(v) for v in d.get(k, []))
    print(f"[HD-perbatch] kappa min=[{_pb('kp_kappa_min')}]", flush=True)
    print(f"[HD-perbatch] kappa med=[{_pb('kp_kappa_med')}]", flush=True)
    print(f"[HD-perbatch] kappa max=[{_pb('kp_kappa_max')}]", flush=True)
    print(f"[HD-perbatch] kappa neg_frac=[{_pb('kp_neg_frac', '{:.2f}')}] gnorm_tau_frac=[{_pb('kp_tau_frac', '{:.2f}')}]", flush=True)
    print(f"[HD-perbatch] corr_norm_ratio(|c|/|g0|)=[{_pb('kp_corr_ratio', '{:.4f}')}]", flush=True)
    print(f"[HD-perbatch] eff_lambda(k/|g0|^2) min=[{_pb('kp_eff_lam_min')}] max=[{_pb('kp_eff_lam_max')}]", flush=True)
    print(f"[DELTA] ||d_b||=[{_pb('kd_norm')}]", flush=True)
    print(f"[DELTA] ||d_b||/||a0||=[{_pb('kd_ratio')}]", flush=True)
    print(f"[DELTA] rho_g cos(d,g0)=[{_pb('kd_rho_g', '{:+.4f}')}]", flush=True)
    print(f"[DELTA] rho_a cos(d,a0)=[{_pb('kd_rho_a', '{:+.4f}')}]", flush=True)
    print(f"[DELTA] persistence cos(d_b,d_b-1)=[{_pb('kd_pers', '{:+.4f}')}]", flush=True)
    maxd = max(d.get('kd_norm', [float('nan')]))
    form = str(getattr(args, 'hd_kprobe_form', 'diag')); alpha = float(getattr(args, 'hd_kprobe_alpha', 1.0))
    kmed = _med(d['kp_kappa_med']); lmed = _med(d['kp_eff_lam_med'])
    print(f"[HD-KPROBE] r={rnd} user={user} form={form} alpha={alpha:g} "
          f"{'lambda=%g ' % float(getattr(args, 'hd_kprobe_lambda', 3000)) if form == 'const' else ''}"
          f"{'lambda0=%g eps=%g ' % (float(getattr(args, 'hd_kprobe_lambda0', 0.0)), float(getattr(args, 'hd_kprobe_a_eps', 1e-7))) if form == 'const_a' else ''}"
          f"| kappa med={kmed:.4e} | eff_lambda med={lmed:.4e} | maxdelta={maxd:.4e} "
          f"| neg_frac mean={float(torch.as_tensor(d['kp_neg_frac']).float().mean()):.3f} "
          f"| corr_ratio med={_med(d['kp_corr_ratio']):.4f} | n={len(d['kp_kappa_med'])}", flush=True)
    d.pop('_kp_prev_delta', None)
    return {'[KP] kappa med': kmed, '[KP] eff_lambda med': lmed, '[KP] maxdelta': float(maxd),
            '[KP] neg_frac mean': float(torch.as_tensor(d['kp_neg_frac']).float().mean()),
            '[KP] corr_ratio med': _med(d['kp_corr_ratio']),
            '[KP] rho_g med': _med(d['kd_rho_g']), '[KP] rho_a med': _med(d['kd_rho_a']),
            '[KP] delta_ratio max': float(max(d.get('kd_ratio', [float('nan')])))}
