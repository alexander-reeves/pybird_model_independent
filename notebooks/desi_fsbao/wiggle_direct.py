"""Reference direct chains for the recovery test of the wiggle models, in output/desi_fsbao/<kind>_<variant>/:

  kind = rdfix   model.MIModel's direct model with the sound horizon corrected (rd_h_fixed: omega_b exponent -0.13;
                 pybird's symbolic.rs_drag has +0.13)
  kind = exact   the TRUE direct likelihood: the engine's spectrum (CosmoPower at z_early grown to z_N, or ede-v2) evaluated
                 at the emulator's own knots, kk = knots_h, P = h^3 P(knots_h h), with r_d corrected; growth, AP and BAO
                 as model.MIModel. The direct model of model.MIModel instead samples P/T on the 60 ln a nodes, interpolates
                 to the knots in 1/Mpc and lets the emulator re-interpolate after the h_conv relabelling.

Before sampling, the representation test (wiggle_envelope.py W4) is repeated against this reference on draws of the
existing direct chain. Everything else is settings.VARIANTS[variant] and sample.direct_stages.

    SCRIPT=wiggle_direct.py ARGS="lcdm5 exact" sbatch exec_wiggle.sbatch
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import jax, jax.numpy as jnp
import settings as S
import wiggle_settings as W
from wiggle_model import rd_h_fixed
from model import COSMO_MODELS, Omega_m, rd_h, sym_H, sym_DM, C_LIGHT
from sample import direct_stages, Tee
from sampling import log


def fix_rd(M):
    """The exact direct model with r_d from rd_h_fixed in the BAO alphas (instance override of MIModel.bao_alphas)."""
    def bao_alphas(c):
        Om, rdh, out = c.get('Omega_m', Omega_m(c)), c.get('rd_h', rd_h_fixed(c)), []
        for b, iso in zip(M.bao, M.bao_iso):
            a_par = C_LIGHT / (100. * sym_H(Om, b['zeff'], c['w0'], c['wa'])) / rdh / b['DH_over_rd_fid']
            a_per = sym_DM(Om, b['zeff'], c['w0'], c['wa']) / rdh / b['DM_over_rd_fid']
            out += [a_par**(1 / 3.) * a_per**(2 / 3.)] if iso else [a_par, a_per]
        return jnp.array(out)
    M.bao_alphas = bao_alphas
    return M


def true_loglkl(M, eft, c):
    """ln L with the engine's spectrum at the emulator's knots; growth, AP, BAO from M.phi_ln_full(c)."""
    kh = jnp.array(M.knots_h)
    phi = M.phi_ln_full(c); d = M.cosmo_dicts(phi)
    P = (M.ede_emulator().plin(kh * c['h'], c, M.z_N) if c.get('fEDE') is not None else M.plin_zN(kh * c['h'], c)) * c['h']**3
    D = jnp.concatenate([jnp.exp(phi[M.pidx['D']]), jnp.ones(1)])
    d = [dict(di, kk=kh, pk_lin=P * D[M.sky_to_z[i]]**2) for i, di in enumerate(d)]
    return M.L.loglkl(eft, M.eft_names_flat, need_cosmo_update=True, cosmo_dict=d, cosmo_module=None, cosmo_engine=None)


def make_true(M):
    """M's direct likelihood -> true_loglkl (instance override; logpost_direct picks it up)."""
    M.loglkl_direct = lambda x: true_loglkl(M, x[:M.n_eft], M.theta_to_full(x[M.n_eft:]))
    return M


def rdfix_files(variant, kind='rdfix'):
    s = S.resolve(variant); s['out'] = os.path.join(S.OUT_ROOT, f'{kind}_{variant}'); f = S.files(s)
    return s, {'bestfit_direct': f['bestfit_direct'], 'chain_direct': f['chain_direct'], 'meta': f['meta']}


if __name__ == '__main__':
    variant = sys.argv[1] if len(sys.argv) > 1 else 'lcdm5'
    kind = sys.argv[2] if len(sys.argv) > 2 else 'rdfix'
    s, F = rdfix_files(variant, kind); os.makedirs(s['out'], exist_ok=True)
    sys.stdout = Tee(os.path.join(s['out'], 'sample.log')); sys.stderr = sys.stdout
    log(f"wiggle_direct.py {variant}: exact direct chain with r_d from rd_h_fixed -> {s['out']}; JAX devices {jax.devices()}")
    M, _ = S.build(s); fix_rd(M)
    if kind == 'exact': make_true(M); log("direct likelihood: the engine's spectrum at the emulator knots (true_loglkl)")
    c_fid = dict(M.cosmo_fixed)
    c_lo = dict(c_fid, omega_b=c_fid['omega_b'] - 0.00055)
    log(f"h r_d at the fiducial: model.rd_h {float(rd_h(c_fid)):.4f}, rd_h_fixed {float(rd_h_fixed(c_fid)):.4f} Mpc/h; at omega_b - 1 BBN sigma "
        f"{float(rd_h(c_lo)):.4f} vs {float(rd_h_fixed(c_lo)):.4f}")

    # ---- representation test against the corrected exact likelihood (draws of the existing chain)
    x = np.load(S.files(S.resolve(variant))['chain_direct'])['x']; x = x.reshape(-1, x.shape[-1])
    x = x[np.random.default_rng(0).choice(len(x), 300, replace=False)]; keys = COSMO_MODELS[s['cosmo_model']]
    def batched(f, *args, B=50):
        fb = jax.jit(jax.vmap(f)); out = []
        for i in range(0, len(args[0]), B):
            a = [np.asarray(v[i:i + B]) for v in args]; m = len(a[0])
            if m < B: a = [np.concatenate([v, np.repeat(v[-1:], B - m, 0)]) for v in a]
            out.append(np.asarray(fb(*[jnp.array(v) for v in a]))[:m])
        return np.concatenate(out)
    fx = dict(M.cosmo_fixed); f0 = M.phi_ln_keys(keys)
    L0 = batched((lambda e, t: true_loglkl(M, e, M.theta_to_full(t, keys, fx))) if kind == 'exact' else (lambda e, t: M.loglkl_phi(e, f0(t))),
                 x[:, :M.n_eft], x[:, M.n_eft:])
    for name in ('B', 'Be1', 'Be2'):
        Mw = W.build(W.resolve(name, variant), verbose=False)[0]; fw = Mw.phi_ln_keys(keys)
        d = batched(lambda e, t: Mw.loglkl_phi(e, fw(t)), x[:, :M.n_eft], x[:, M.n_eft:]) - L0
        w = np.exp(d - d.max()); w /= w.sum(); th = x[:, M.n_eft:]; sh = (w @ th - th.mean(0)) / th.std(0)
        log(f"W4-{kind} [{name}] {variant}: Delta ln L mean {d.mean():+.3f}, std {d.std():.3f}; ESS/N {1 / np.sum(w**2) / len(w):.2f}; "
            f"implied shift " + ", ".join(f"{k} {a:+.2f}" for k, a in zip(keys, sh)) + " sigma")
        np.save(os.path.join(s['out'], f'W4_{kind}_{name}.npy'), d)

    # ---- the direct chain
    with open(F['meta'], 'w') as fh: json.dump(S.describe(s, M, S.load_data()) | {'rd': 'wiggle_model.rd_h_fixed', 'kind': kind}, fh, indent=1)
    lp_d = jax.jit(M.logpost_direct)
    nuts = dict(max_num_doublings=s['max_doublings'], target_accept=s['target_accept'], initial_step_size=s['initial_step_size'])
    t0 = time.time(); direct_stages(M, s, F, False, lp_d, nuts)
    log(f"DONE in {(time.time() - t0) / 60:.1f} min")
