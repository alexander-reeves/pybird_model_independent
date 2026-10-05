"""Direct chains of the TRUE likelihood with the BAO amplitude and damping free, as in a BAO fit: the reference for the
M projections (wiggle_recovery.py --wiggle ...M: A, dSigma^2 marginalized). The linear spectrum of each cosmology is

    P(k) = P_nw(k) [1 + (1 + A) exp(-k^2 dSigma^2 / 2) O(k)],      k in 1/Mpc,

with P_nw, O the cosmology's OWN wiggle/no-wiggle split (wiggle_model.WiggleModel._split, here in JAX and per cosmology)
and A ~ N(0, s_env), dSigma^2 ~ N(0, s_dsig2) Mpc^2 (wiggle_settings.WIDE_PRIOR, the priors of the Ag245W chain). Everything
else is wiggle_direct.py's exact likelihood (CosmoPower / ede-v2 spectrum at the emulator knots, corrected r_d). No MI
model and no least-squares map are involved. Output: output/desi_fsbao/exactM_<variant>/.

    SCRIPT=wiggle_direct_marg.py ARGS="lcdm5" sbatch exec_wiggle.sbatch            (ARGS="lcdm5 gates": gates only)
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import jax, jax.numpy as jnp
import model
import settings as S
import wiggle_settings as W
from model import COSMO_MODELS
from wiggle_direct import fix_rd, true_loglkl, rdfix_files
from sample import direct_stages, Tee
from sampling import log

S_A, S_D = W.WIDE_PRIOR['s_env'], W.WIDE_PRIOR['s_dsig2']
EXTRA = {'A_bao': ((-0.95, 5.0), r'A', 0.05), 'dsig2': ((-50.0, 300.0), r'\Delta\Sigma^2', 2.0)}
for _k, (_box, _tex, _sc) in EXTRA.items():          # register the two nuisances as direct-model keys (this process only)
    if _k not in model.COSMO_KEYS: model.COSMO_KEYS.append(_k)
    model.COSMO_BOX[_k] = _box; model.COSMO_TEX[_k] = _tex; model.COSMO_SCALES[_k] = _sc


def eh_nowiggle_T(k, omega_m, omega_b, h, T_cmb=2.7255):
    """wiggle_model.eh_nowiggle_T in jax.numpy (Eisenstein & Hu 1998 eqs. 26-31, k in 1/Mpc)."""
    th2, fb = (T_cmb / 2.7)**2, omega_b / omega_m
    s = 44.5 * jnp.log(9.83 / omega_m) / jnp.sqrt(1 + 10 * omega_b**0.75)
    ag = 1 - 0.328 * jnp.log(431 * omega_m) * fb + 0.38 * jnp.log(22.3 * omega_m) * fb**2
    gam = omega_m / h * (ag + (1 - ag) / (1 + (0.43 * k * s)**4))
    q = k / h * th2 / gam
    L0, C0 = jnp.log(2 * jnp.e + 1.8 * q), 14.2 + 731 / (1 + 62.5 * q)
    return L0 / (L0 + C0 * q**2)


class Split:
    """WiggleModel._split for any cosmology, differentiable: O = P / (P_EH,nw G[P / P_EH,nw]) - 1 on a uniform ln k grid,
    G = scipy.ndimage.gaussian_filter1d(mode='nearest', truncate=4) of sigma_dex in log10 k as a matrix, tapered to 0
    below k_taper."""

    def __init__(self, n=1500, k_grid=(1e-5, 9.0), sigma_dex=0.25, k_taper=(1e-3, 3e-3)):
        lnk = np.linspace(np.log(k_grid[0]), np.log(k_grid[1]), n)
        sd = sigma_dex * np.log(10.) / (lnk[1] - lnk[0]); r = int(4 * sd + 0.5)
        off = np.arange(-r, r + 1); w = np.exp(-0.5 * (off / sd)**2); w /= w.sum()
        G = np.zeros((n, n))
        for i in range(n): np.add.at(G[i], np.clip(i + off, 0, n - 1), w)
        t = np.clip((lnk - np.log(k_taper[0])) / np.log(k_taper[1] / k_taper[0]), 0, 1)
        self.lnk, self.k, self.G, self.taper = jnp.array(lnk), jnp.array(np.exp(lnk)), jnp.array(G), jnp.array(t * t * (3 - 2 * t))

    def O(self, P, c):
        """O(k) on the grid for the spectrum P (on self.k, Mpc^3) of cosmology dict c."""
        Peh = self.k**c['n_s'] * eh_nowiggle_T(self.k, c['omega_cdm'] + c['omega_b'], c['omega_b'], c['h'])**2
        return (P / (Peh * jnp.exp(self.G @ jnp.log(P / Peh))) - 1.0) * self.taper


def marg_loglkl(M, SP, eft, c):
    """true_loglkl with the cosmology's wiggles rescaled by (1 + A) exp(-k^2 dSigma^2 / 2); equal to it at A = dSigma^2 = 0."""
    kh = jnp.array(M.knots_h); kq = kh * c['h']
    eng = (lambda k: M.ede_emulator().plin(k, c, M.z_N)) if c.get('fEDE') is not None else (lambda k: M.plin_zN(k, c))
    Ok = jnp.interp(jnp.log(kq), SP.lnk, SP.O(eng(SP.k), c))
    E = (1.0 + c['A_bao']) * jnp.exp(-0.5 * kq**2 * c['dsig2'])
    P = eng(kq) * (1.0 + E * Ok) / (1.0 + Ok) * c['h']**3
    phi = M.phi_ln_full(c); d = M.cosmo_dicts(phi)
    D = jnp.concatenate([jnp.exp(phi[M.pidx['D']]), jnp.ones(1)])
    d = [dict(di, kk=kh, pk_lin=P * D[M.sky_to_z[i]]**2) for i, di in enumerate(d)]
    return M.L.loglkl(eft, M.eft_names_flat, need_cosmo_update=True, cosmo_dict=d, cosmo_module=None, cosmo_engine=None)


if __name__ == '__main__':
    variant = sys.argv[1] if len(sys.argv) > 1 else 'lcdm5'
    gates_only = len(sys.argv) > 2 and sys.argv[2] == 'gates'
    s, F = rdfix_files(variant, os.environ.get('KIND', 'exactM')); os.makedirs(s['out'], exist_ok=True)     # KIND: another output dir
    s['box_transform'] = True        # sample in unbounded box coordinates: the wider n_s posterior reaches its box edge, and
                                     # hard walls there made 87% of lcdm5 transitions divergent (R-hat 1.10 in dSigma^2)
    sys.stdout = Tee(os.path.join(s['out'], 'sample.log')); sys.stderr = sys.stdout
    log(f"wiggle_direct_marg.py {variant}: true likelihood + free A ~ N(0, {S_A}), dSigma^2 ~ N(0, {S_D}) Mpc^2 -> {s['out']}; "
        f"JAX devices {jax.devices()}")
    M, _ = S.build(s); fix_rd(M)
    M.cosmo_fid = dict(M.cosmo_fid, A_bao=0.0, dsig2=0.0)
    keys0 = COSMO_MODELS[s['cosmo_model']]; keys = keys0 + ['A_bao', 'dsig2']
    M.set_cosmo_model(keys, gauss=dict(s['cosmo_gauss'] or {}, A_bao=(0.0, S_A), dsig2=(0.0, S_D)))
    log(f"direct model {M.cosmo_keys}, box {M.cosmo_box}, Gaussian {M.cosmo_gauss}")
    SP = Split()

    # ---- gates: (1) the JAX split reproduces the wiggle model's fiducial O at the knots; (2) at A = dSigma^2 = 0 the
    # likelihood is the exact one (wiggle_direct.true_loglkl) at the exact chain's best fit; (3) the response to A, dSigma^2
    c0 = dict(M.cosmo_fixed)
    Mw = W.build(W.resolve('Ag245', variant), verbose=False)[0]
    Pg = M.ede_emulator().plin(SP.k, c0, M.z_N) if c0.get('fEDE') is not None else M.plin_zN(SP.k, c0)
    O_j = np.interp(np.log(np.asarray(Mw.knots_mpc)), np.asarray(SP.lnk), np.asarray(SP.O(Pg, c0)))
    O_w = np.asarray(Mw.O_of_k(Mw.knots_mpc))
    log(f"G1 split at the fiducial: max |O_jax - O_wiggle_model| on the knots {np.abs(O_j - O_w).max():.2e} (max |O| {np.abs(O_w).max():.3f})")
    b = np.load(rdfix_files(variant, 'exact')[1]['bestfit_direct'])['x_bf']; eft_b, th_b = b[:M.n_eft], b[M.n_eft:]
    c_b = M.theta_to_full(np.concatenate([th_b, [0.0, 0.0]]))
    L_true, L_m = float(true_loglkl(M, jnp.array(eft_b), c_b)), float(marg_loglkl(M, SP, jnp.array(eft_b), c_b))
    log(f"G2 at the exact best fit, A = dSigma^2 = 0: ln L true {L_true:.6f}, with the envelope {L_m:.6f}, difference {L_m - L_true:+.2e}")
    for A_, D_ in ((0.1, 0.0), (-0.1, 0.0), (0.0, 10.0), (0.0, -10.0)):
        Lx = float(marg_loglkl(M, SP, jnp.array(eft_b), dict(c_b, A_bao=A_, dsig2=D_)))
        log(f"G3 A = {A_:+.1f}, dSigma^2 = {D_:+.0f} Mpc^2: Delta chi2 = {-2 * (Lx - L_m):+.3f}")
    if gates_only: log("gates only: done"); sys.exit(0)

    # ---- the direct chain (settings.VARIANTS[variant] sampler settings, sample.direct_stages)
    M.loglkl_direct = lambda x: marg_loglkl(M, SP, x[:M.n_eft], M.theta_to_full(x[M.n_eft:]))
    with open(F['meta'], 'w') as fh:
        json.dump({'variant': variant, 'kind': 'exactM', 'keys': keys, 'box': {k: list(v) for k, v in M.cosmo_box.items()},
                   'gauss': {k: list(v) for k, v in M.cosmo_gauss.items()}, 'rd': 'wiggle_model.rd_h_fixed',
                   'time': time.strftime('%Y-%m-%d %H:%M:%S')}, fh, indent=1)
    lp_d = jax.jit(M.logpost_direct)
    nuts = dict(max_num_doublings=s['max_doublings'], target_accept=s['target_accept'], initial_step_size=s['initial_step_size'])
    t0 = time.time(); direct_stages(M, s, F, False, lp_d, nuts)
    log(f"DONE in {(time.time() - t0) / 60:.1f} min")
