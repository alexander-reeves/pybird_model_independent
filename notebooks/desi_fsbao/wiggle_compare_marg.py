"""Is marginalizing A, dSigma^2 a bias of the MI projection, or what marginalizing costs any analysis? (numpy only; runs on
the login node.) For each cosmological model:

  (1) exactM vs exact      the TRUE-likelihood direct chain with free A, dSigma^2 (wiggle_direct_marg.py) against the
                           standard true-likelihood chain: the cost of treating A, dSigma^2 as nuisances, in a direct fit;
  (2) MI-M vs exactM       the MI projection with A, dSigma^2 marginalized (Ag245WFM) against (1): the projection's error;
  (3) MI-M vs exact        what the recovery test reported.

Shifts in units of the reference sigma; widths as ratios.     python3 wiggle_compare_marg.py [MI variant, default Ag245WFM]
"""
import os, sys
import numpy as np

OUT = '/capstor/store/cscs/swissai/a0158/areeves/pybird_model_independent/output/desi_fsbao'
CHAIN = {'baseline': 'chain_direct.npz', 'lcdm5': 'chain_direct_lcdm5.npz', 'w0wa7': 'chain_direct_w0wa7.npz', 'ede7': 'chain_direct_ede7.npz'}
N_EFT = 12


def chain(kind, cv):
    f = os.path.join(OUT, f'{kind}_{cv}', CHAIN[cv])
    if not os.path.exists(f): return None
    x = np.load(f)['x']; return x.reshape(-1, x.shape[-1])[:, N_EFT:]


def cmp(ref, x):
    return (x.mean(0) - ref.mean(0)) / ref.std(0), x.std(0) / ref.std(0)


if __name__ == '__main__':
    mi = sys.argv[1] if len(sys.argv) > 1 else 'Ag245WFM'
    R = np.load(os.path.join(OUT, f'wiggle_{mi}', 'recovery_c5.npz'))
    for cv in CHAIN:
        d, dm = chain('exact', cv), chain('exactM', cv)
        if dm is None: print(f"== {cv}: exactM chain not there yet"); continue
        th_m, env = dm[:, :-2], dm[:, -2:]; p = R[f'{cv}_samples']; names = list(R[f'{cv}_names'])
        s1, r1 = cmp(d, th_m); s2, r2 = cmp(th_m, p); s3, r3 = cmp(d, p)
        print(f"== {cv}: {len(dm)} exactM draws; A = {env[:, 0].mean():+.3f} +- {env[:, 0].std():.3f}, "
              f"dSigma^2 = {env[:, 1].mean():+.1f} +- {env[:, 1].std():.1f} Mpc^2")
        print(f"   {'':11s} {'(1) exactM vs exact':>22s} {'(2) ' + mi + ' vs exactM':>26s} {'(3) ' + mi + ' vs exact':>25s}")
        for i, n in enumerate(names):
            print(f"   {n:11s} {s1[i]:+7.2f} sigma  x{r1[i]:.2f}    {s2[i]:+9.2f} sigma  x{r2[i]:.2f}      {s3[i]:+7.2f} sigma  x{r3[i]:.2f}")
        print(f"   max |shift|: (1) {np.abs(s1).max():.2f}, (2) {np.abs(s2).max():.2f}, (3) {np.abs(s3).max():.2f}; "
              f"widths (2) {r2.min():.2f}-{r2.max():.2f}")
