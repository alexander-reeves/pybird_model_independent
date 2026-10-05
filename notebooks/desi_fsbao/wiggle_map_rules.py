"""How much does the rule that translates a cosmology into MI parameters matter? For draws of each true-likelihood direct
chain (exact_<model>), the MI likelihood at phi(theta) is compared with the true likelihood at theta,
Delta ln L = ln L_MI(phi(theta)) - ln L_true(theta), for three rules (model Ag245, BAO-template envelope, 21 nodes):

  F  data-weighted least squares on the 80 knots (W = J^T C^-1 J): the KL-optimal rule, used by Ag245F
  U  unweighted least squares (uniform in ln k)
  I  no fit: ln a at each node read off the cosmology's target ln[P / (T_nw (1 + O))] at that k, A = dSigma^2 = 0

A rule is good when Delta ln L is nearly constant over the posterior (std << 1); the importance weights exp(Delta ln L)
turn it into the shift it would cause in the cosmological posterior.

    SCRIPT=wiggle_map_rules.py sbatch exec_wiggle.sbatch      -> output/desi_fsbao/wiggle_gates/map_rules.npz
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import jax, jax.numpy as jnp
import settings as S
import wiggle_settings as W
from model import COSMO_MODELS
from wiggle_direct import fix_rd, true_loglkl, rdfix_files
from sampling import log


def batched(f, *args, B=50):
    fb = jax.jit(jax.vmap(f)); out = []
    for i in range(0, len(args[0]), B):
        a = [np.asarray(v[i:i + B]) for v in args]; m = len(a[0])
        if m < B: a = [np.concatenate([v, np.repeat(v[-1:], B - m, 0)]) for v in a]
        out.append(np.asarray(fb(*[jnp.array(v) for v in a]))[:m])
    return np.concatenate(out)


M0 = fix_rd(S.build(S.resolve('baseline'), verbose=False)[0])
res = {}
for cv in W.COSMO_VARIANTS:
    sv = S.resolve(cv); keys = COSMO_MODELS[sv['cosmo_model']]; M0.set_cosmo_model(sv['cosmo_model'], gauss=sv['cosmo_gauss'])
    fx = dict(M0.cosmo_fixed)
    x = np.load(rdfix_files(cv, 'exact')[1]['chain_direct'])['x']; x = x.reshape(-1, x.shape[-1])
    x = x[np.random.default_rng(3).choice(len(x), 1000, replace=False)]; e, th = x[:, :M0.n_eft], x[:, M0.n_eft:]
    L0 = batched(lambda e_, t: true_loglkl(M0, e_, M0.theta_to_full(t, keys, fx)), e, th)
    MF = W.build(W.resolve('Ag245F', cv), verbose=False)[0]; MU = W.build(W.resolve('Ag245', cv), verbose=False)[0]
    for M_ in (MF, MU): M_.set_cosmo_model(sv['cosmo_model'], gauss=sv['cosmo_gauss'])      # builds the EDE emulator outside jit

    def phi_I(t, M=MF):
        c = M.theta_to_full(t, keys, fx); phi = M.phi_ln_full(c)
        c2, plin = M.engine(c); r = M.lna_target_knots(c2, plin)
        return phi.at[M.pidx['lna']].set(jnp.interp(M.nodes_lnk, M.lnk_knots, r)).at[M.pidx['env']].set(0.0)
    rules = {'F': (MF, lambda t: MF.phi_ln_full(MF.theta_to_full(t, keys, fx))),
             'U': (MU, lambda t: MU.phi_ln_full(MU.theta_to_full(t, keys, fx))), 'I': (MF, phi_I)}
    for name, (M, fphi) in rules.items():
        d = batched(lambda e_, t: M.loglkl_phi(e_, fphi(t)), e, th) - L0
        w = np.exp(d - d.max()); w /= w.sum()
        sh = (w @ th - th.mean(0)) / th.std(0); ess = 1 / np.sum(w**2) / len(w)
        res[f'{cv}_{name}'] = d
        log(f"[{name}] {cv:9s} Delta ln L mean {d.mean():+8.3f} std {d.std():7.3f}  ESS/N {ess:.2f}  implied shifts "
            + " ".join(f"{k} {a:+.2f}" for k, a in zip(keys, sh)) + f"  -> max {np.abs(sh).max():.2f} sigma")
np.savez(os.path.join(S.OUT_ROOT, 'wiggle_gates', 'map_rules.npz'), **res)
log("done")
