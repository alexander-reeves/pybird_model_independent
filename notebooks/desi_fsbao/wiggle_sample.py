#!/usr/bin/env python
"""MI best fit and NUTS chain of a wiggle/no-wiggle model (wiggle_model.py, wiggle_settings.py).

    python wiggle_sample.py --wiggle B [--force] [--n-samples N] [--out-dir DIR]
    SCRIPT=wiggle_sample.py ARGS="--wiggle B" sbatch exec_wiggle.sbatch

Stages in output/desi_fsbao/wiggle_<wiggle>/, each skipped when its output exists (unless --force):
    bestfit_mi.npz   L-BFGS-B best fit of the MI posterior from the fiducial phi, EFT from the baseline direct
                     best fit; Gauss-Newton Fisher with the prior (the preconditioner)
    chain_mi.npz     NUTS, same sampler settings and seed as the baseline MI chain (settings.BASELINE)
The direct chains are not resampled: the recovery test (wiggle_recovery.py) compares with the exact ones."""
import os, sys, argparse, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import jax, jax.numpy as jnp
import wiggle_settings as W
from sample import Tee, todo
from sampling import log, WhitenedNUTS, minimize_lbfgs, precond_cov, chain_diagnostics


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--wiggle', default='B'); p.add_argument('--force', action='store_true')
    p.add_argument('--n-samples', type=int); p.add_argument('--n-chains', type=int); p.add_argument('--n-warmup', type=int)
    p.add_argument('--out-dir'); p.add_argument('--seed-mi', type=int)
    p.add_argument('--precond-chain', help='an earlier chain_mi.npz of the same model: whiten NUTS with its sample covariance '
                                           'instead of the Gauss-Newton Fisher at the MAP (better for non-Gaussian posteriors)')
    a = p.parse_args(argv)
    s = W.resolve(a.wiggle)
    for k in ('n_samples', 'n_chains', 'n_warmup', 'seed_mi'):
        if getattr(a, k) is not None: s[k] = getattr(a, k)
    if a.out_dir: s['out'] = s['mi_out'] = os.path.abspath(a.out_dir)
    os.makedirs(s['out'], exist_ok=True)
    sys.stdout = Tee(os.path.join(s['out'], 'sample.log')); sys.stderr = sys.stdout
    log(f"wiggle_sample.py --wiggle {a.wiggle} ({' '.join(sys.argv[1:])}); JAX devices {jax.devices()}")
    M, _ = W.build(s); F = W.files(s)
    with open(F['meta'], 'w') as fh: json.dump(W.describe(s, M), fh, indent=1)
    t0 = time.time(); lp = jax.jit(M.logpost_mi)
    eft0 = np.load(F['baseline_bestfit_direct'])['x_bf'][:M.n_eft]

    if todo(F['bestfit_mi'], a.force, M.n_mi):
        x_bf, _ = minimize_lbfgs(lp, M.x_mi_fid(eft0), maxiter=3000, name='MI', scales=M.min_scales('mi'), bounds=M.min_bounds('mi'))
        t1 = time.time(); Fgn = M.gn_fisher('mi', x_bf)
        log(f"[GN] Fisher ({M.n_mi}x{M.n_mi}) incl. prior in {time.time()-t1:.1f}s")
        np.savez(F['bestfit_mi'], x_bf=x_bf, F_gn=Fgn)
    b = np.load(F['bestfit_mi']); x_bf, Fgn = b['x_bf'], b['F_gn']
    log(f"MI best fit: chi2 = {-2 * float(M.loglkl_mi(jnp.array(x_bf))):.2f}; "
        + "; ".join(f"{n} {v:+.4f}" for n, v in zip(M.names, x_bf) if 'alpha' in n))

    if todo(F['chain_mi'], a.force, M.n_mi):
        C = precond_cov(Fgn, floor_prec=float(np.linalg.eigvalsh(M.prior_prec).min()))
        if a.precond_chain:
            xp = np.load(a.precond_chain)['x']; C = np.cov(xp.reshape(-1, xp.shape[-1]), rowvar=False)
            log(f"preconditioner: sample covariance of {a.precond_chain} ({xp.shape[0]} x {xp.shape[1]} draws)")
        r = WhitenedNUTS(lp, x_bf, C, name='MI').run(jax.random.key(s['seed_mi']), s['n_warmup'], s['n_samples'], n_chains=s['n_chains'],
                                                    jitter=s['jitter_mi'], max_num_doublings=s['max_doublings'],
                                                    target_accept=s['target_accept'], initial_step_size=s['initial_step_size'])
        np.savez(F['chain_mi'], **r)
        chain_diagnostics(r['x'], M.names, max_print=8)
    log(f"DONE in {(time.time()-t0)/60:.1f} min; outputs in {s['out']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
