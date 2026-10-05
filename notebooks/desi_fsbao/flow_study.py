"""B0: flow-ensemble settings judged by what they are for, the recovery of the direct posteriors:
LCDM (lcdm3) and EDE (ede7), each MI -> model projection against its direct NUTS chain. The ensemble
of 2026-10-01 (8 layers x 128, lr 5e-4) stopped every member at step 300 and missed EDE by 0.4-0.6 sigma.
Writes output/desi_fsbao/meeting/B0_flow_study.{log,npz} and baseline/flow_<config>.pkl for each setting.

    SCRIPT=flow_study.py ARGS="c0 c1" sbatch exec_meeting.sbatch
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import settings as S
from model import BBN_OMEGA_B
from sampling import log
from compress import load_or_fit, sample_projected, gaussian_estimate

from settings import FLOW_CONFIGS as CONFIGS
todo = sys.argv[1:] or list(CONFIGS)
OUT = os.path.join(S.OUT_ROOT, 'meeting'); os.makedirs(OUT, exist_ok=True)
s = S.resolve('baseline'); M, _ = S.build(s, verbose=False)
xm = S.load_mi_samples(s)
TAG = ''.join('_' + d for d in os.environ.get('MI_EXTRA', '').split(',') if d)   # which MI samples the flows saw
th_l = np.load(S.files(s)['chain_direct'])['x'].reshape(-1, M.n_eft + 3)[:, M.n_eft:]
se = S.resolve('ede7'); th_e = np.load(S.files(se)['chain_direct'])['x']; th_e = th_e.reshape(-1, th_e.shape[-1])[:, M.n_eft:]
res = {}
for c in todo:
    kw = CONFIGS.get(c, {}); t0 = time.time()
    if c == 'gauss':
        fl = gaussian_estimate(M, xm); gain, best = [0.0], [0]
    else:
        fl = load_or_fit(M, xm, os.path.join(s['out'], f'flow_{c}{TAG}.pkl'), n_seeds=10, steps=8000, **kw)
        gain = [m['ll_ho'] - m['ll_gauss_ho'] for m in fl['members']]; best = [m['best_step'] for m in fl['members']]
    out = {}
    for model, gauss, th_d in (('lcdm3', None, th_l), ('ede7', {'omega_b': BBN_OMEGA_B}, th_e)):
        M.set_cosmo_model(model, gauss=gauss)
        t = sample_projected(M, fl, np.median(th_d, 0)).reshape(-1, M.n_cosmo)
        sh, ra = (t.mean(0) - th_d.mean(0)) / th_d.std(0), t.std(0) / th_d.std(0)
        out[model] = (sh, ra, t)
        log(f"[{c} {kw}] {model}: shifts {np.round(sh, 2)} sigma, width ratios {np.round(ra, 2)}; max |shift| {np.abs(sh).max():.2f}")
    M.set_cosmo_model('lcdm3')
    log(f"[{c}] held-out gain {np.round(np.mean(gain), 3)} nats (members {np.round(gain, 2)}), best steps {best}; {(time.time()-t0)/60:.1f} min")
    res[c] = out
    np.savez(os.path.join(OUT, f'B0_flow_study_{c}{TAG}.npz'), gain=gain, best=best,
             **{f'{m}_{k}': v for m, (sh, ra, t) in out.items() for k, v in (('shift', sh), ('ratio', ra), ('samples', t))})
log("=== FLOW STUDY ===")
for c, out in res.items():
    log(f"{c} {CONFIGS.get(c, {})}: " + "; ".join(f"{m} max|shift| {np.abs(sh).max():.2f} ratios [{ra.min():.2f}, {ra.max():.2f}]" for m, (sh, ra, t) in out.items()))
