#!/usr/bin/env python
"""Direct and model-independent NUTS chains for the DESI DR1 FS+BAO analysis.

    python sample.py --variant baseline [--force] [--dry-run]
    VARIANT=baseline sbatch exec_sample.sbatch

Stages, each skipped when its output exists (unless --force), in output/desi_fsbao/<variant>/:
    bestfit_direct.npz   L-BFGS-B best fit of the direct model + its Hessian (preconditioner)
    chain_direct.npz     direct NUTS chain
    bestfit_mi.npz       L-BFGS-B best fit of the MI posterior, started from the direct EFT best fit
                         at the fiducial phi, + the Gauss-Newton Fisher with the prior (preconditioner)
    chain_mi.npz         MI NUTS chain
Variants with `mi_from` (lcdm5, w0wa7) sample only the direct model and share the MI chain.
Settings: settings.py. The log is also appended to sample.log in the output directory."""
import os, sys, argparse, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import jax, jax.numpy as jnp
import settings as S
from model import COSMO_MODELS
from sampling import log, WhitenedNUTS, minimize_lbfgs, fisher_at, precond_cov, chain_diagnostics, BoxTransform


class Tee:
    def __init__(self, path): self.f = open(path, 'a'); self.out = sys.stdout
    def write(self, s): self.out.write(s); self.f.write(s)
    def flush(self): self.out.flush(); self.f.flush()


def todo(path, force, n=None):
    """True if the stage must run: missing, --force, or saved with a different dimension."""
    if force or not os.path.exists(path): return True
    if n is not None:
        z = np.load(path); k = 'x_bf' if 'x_bf' in z.files else 'x'
        if z[k].shape[-1] != n: log(f"{path}: dimension {z[k].shape[-1]} != {n} -> recomputing"); return True
    log(f"{path} exists -> skipped (--force to redo)"); return False


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--variant', default='baseline', choices=list(S.VARIANTS))
    p.add_argument('--force', action='store_true'); p.add_argument('--dry-run', action='store_true')
    p.add_argument('--n-samples', type=int); p.add_argument('--n-chains', type=int); p.add_argument('--n-warmup', type=int)
    p.add_argument('--out-dir', help='write here instead of output/desi_fsbao/<variant> (smoke tests)')
    p.add_argument('--seed-mi', type=int, help='MI chain seed (extra independent chains, e.g. into another --out-dir)')
    p.add_argument('--stage', choices=['all', 'direct', 'mi'], default='all', help='run only the direct or only the MI stages (fits the 1 h 30 debug limit)')
    a = p.parse_args(argv)
    s = S.resolve(a.variant, {'n_samples': a.n_samples, 'n_chains': a.n_chains, 'n_warmup': a.n_warmup, 'seed_mi': a.seed_mi})
    if a.out_dir: s['out'] = s['mi_out'] = os.path.abspath(a.out_dir)
    os.makedirs(s['out'], exist_ok=True)
    sys.stdout = Tee(os.path.join(s['out'], 'sample.log')); sys.stderr = sys.stdout
    log(f"sample.py --variant {a.variant} ({' '.join(sys.argv[1:])}); JAX devices {jax.devices()}")
    M, data = S.build(s)
    F = S.files(s)
    with open(F['meta'], 'w') as fh: json.dump(S.describe(s, M, data), fh, indent=1)
    if a.dry_run: log("dry run: no sampling"); return 0
    t0 = time.time()
    lp_d, lp_mi = jax.jit(M.logpost_direct), jax.jit(M.logpost_mi)
    nuts = dict(max_num_doublings=s['max_doublings'], target_accept=s['target_accept'], initial_step_size=s['initial_step_size'])

    if a.stage == 'mi':
        if not os.path.exists(F['bestfit_direct']): raise FileNotFoundError(f"{F['bestfit_direct']}: run --stage direct first")
        xd_bf = np.load(F['bestfit_direct'])['x_bf']
    else:
        xd_bf = direct_stages(M, s, F, a.force, lp_d, nuts)
    if s['mi_from'] or a.stage == 'direct':
        log(f"DONE (direct stages) in {(time.time()-t0)/60:.1f} min"); return 0
    mi_stages(M, s, F, a.force, lp_mi, nuts, xd_bf)
    log(f"DONE in {(time.time()-t0)/60:.1f} min; outputs in {s['out']}")
    return 0


def direct_stages(M, s, F, force, lp_d, nuts):
    n_d = M.n_eft + M.n_cosmo
    # ---- direct best fit + Hessian
    x0 = M.x_direct_fid(S.eft_start(M))
    if s['mi_from']:            # enlarged model: start from the baseline lcdm3 best fit, extra parameters at the fiducial
        b0 = np.load(os.path.join(S.OUT_ROOT, s['mi_from'], 'bestfit_direct.npz'))['x_bf']
        c = dict(M.cosmo_fixed); c.update(zip(COSMO_MODELS['lcdm3'], b0[M.n_eft:])); c.update(s.get('start') or {})
        x0 = np.concatenate([b0[:M.n_eft], [c[k] for k in M.cosmo_keys]])
    if s.get('box_transform'):
        return direct_stages_boxed(M, s, F, force, lp_d, nuts, x0)
    if todo(F['bestfit_direct'], force, n_d):
        x_bf, _ = minimize_lbfgs(lp_d, x0, name='direct', scales=M.min_scales('direct'), bounds=M.min_bounds('direct'))
        np.savez(F['bestfit_direct'], x_bf=x_bf, F=fisher_at(lp_d, x_bf, name='direct'))
    b = np.load(F['bestfit_direct']); xd_bf, Fd = b['x_bf'], b['F']
    log(f"direct best fit ({M.cosmo_model}): chi2 = {-2 * float(M.loglkl_direct(jnp.array(xd_bf))):.2f}; "
        f"{dict(zip(M.cosmo_names, np.round(xd_bf[M.n_eft:], 5)))}")

    # ---- direct chain
    if todo(F['chain_direct'], force, n_d):
        r = WhitenedNUTS(lp_d, xd_bf, precond_cov(Fd), name='direct').run(
            jax.random.key(s['seed_direct']), s['n_warmup'], s['n_samples'], n_chains=s['n_chains'], jitter=s['jitter_direct'], **nuts)
        np.savez(F['chain_direct'], **r)
        chain_diagnostics(r['x'], M.eft_labels + M.cosmo_names)
    return xd_bf


def direct_stages_boxed(M, s, F, force, lp_d, nuts, x0):
    """The direct stages in the unbounded coordinates of BoxTransform (cosmological parameters only)."""
    n_d = M.n_eft + M.n_cosmo
    bt = BoxTransform([M.cosmo_box[k][0] for k in M.cosmo_keys], [M.cosmo_box[k][1] for k in M.cosmo_keys], n_free=M.n_eft)
    lp_y = jax.jit(bt.wrap(lp_d))
    if todo(F['bestfit_direct'], force, n_d):
        y_bf, _ = minimize_lbfgs(lp_y, np.asarray(bt.to_y(x0)), name='direct (box coordinates)', scales=np.concatenate([np.full(M.n_eft, 0.5), np.full(M.n_cosmo, 0.3)]))
        x_bf = np.asarray(bt.to_x(jnp.array(y_bf)))
        np.savez(F['bestfit_direct'], x_bf=x_bf, y_bf=y_bf, F_y=fisher_at(lp_y, y_bf, name='direct (box coordinates)'))
    b = np.load(F['bestfit_direct']); x_bf, y_bf, F_y = b['x_bf'], b['y_bf'], b['F_y']
    log(f"direct MAP ({M.cosmo_model}): chi2 = {-2 * float(M.loglkl_direct(jnp.array(x_bf))):.2f}; "
        f"{dict(zip(M.cosmo_names, np.round(x_bf[M.n_eft:], 5)))}")
    if todo(F['chain_direct'], force, n_d):
        r = WhitenedNUTS(lp_y, y_bf, precond_cov(F_y, floor_prec=0.25), name='direct').run(
            jax.random.key(s['seed_direct']), s['n_warmup'], s['n_samples'], n_chains=s['n_chains'], jitter=s['jitter_direct'], **nuts)
        r['y'] = r['x']; r['x'] = np.asarray(jax.vmap(jax.vmap(bt.to_x))(jnp.array(r['y'])))
        np.savez(F['chain_direct'], **r)
        chain_diagnostics(r['x'], M.eft_labels + M.cosmo_names)
    return x_bf


def mi_stages(M, s, F, force, lp_mi, nuts, xd_bf):
    # ---- MI best fit + Gauss-Newton preconditioner
    if todo(F['bestfit_mi'], force, M.n_mi):
        xm_bf, _ = minimize_lbfgs(lp_mi, M.x_mi_fid(xd_bf[:M.n_eft]), maxiter=3000, name='MI', scales=M.min_scales('mi'),
                                  bounds=M.min_bounds('mi'))
        t1 = time.time(); Fgn = M.gn_fisher('mi', xm_bf)
        log(f"[GN] Fisher ({M.n_mi}x{M.n_mi}) incl. prior in {time.time()-t1:.1f}s")
        np.savez(F['bestfit_mi'], x_bf=xm_bf, F_gn=Fgn)
    b = np.load(F['bestfit_mi']); xm_bf, Fgn = b['x_bf'], b['F_gn']
    ph = M.unpack(jnp.array(xm_bf[M.n_eft:]))
    log(f"MI best fit: chi2 = {-2 * float(M.loglkl_mi(jnp.array(xm_bf))):.2f}; h_conv = {float(ph['hconv'][0]):.4f}; "
        f"D(z_i)/D(z_N) = {np.round(np.asarray(ph['D']), 3)}")

    # ---- MI chain
    if todo(F['chain_mi'], force, M.n_mi):
        # floor = the smallest prior precision eigenvalue: no direction is whitened wider than the prior allows
        C = precond_cov(Fgn, floor_prec=float(np.linalg.eigvalsh(M.prior_prec).min()))
        r = WhitenedNUTS(lp_mi, xm_bf, C, name='MI').run(
            jax.random.key(s['seed_mi']), s['n_warmup'], s['n_samples'], n_chains=s['n_chains'], jitter=s['jitter_mi'], **nuts)
        np.savez(F['chain_mi'], **r)
        chain_diagnostics(r['x'], M.names, max_print=8)


if __name__ == '__main__':
    sys.exit(main())
