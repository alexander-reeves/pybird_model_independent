"""Settings of the wiggle/no-wiggle MI models (wiggle_model.py), on top of settings.py: same data, fiducial,
priors and sampler; only the MI parametrization changes. Outputs in output/desi_fsbao/wiggle_<variant>/.

    A   alpha_rs free (the proposal: AP alphas per redshift + one sound-horizon rescaling of the wiggles)
    B   alpha_rs = 1  (the alphas are the BAO alphas; the template is the spectrum in sound-horizon units)
    Be2 ...   B with a free wiggle envelope [1 + (1 + e0 + e1 x) O], x = ln(k / 0.1 Mpc^-1): the BAO amplitude and its
              damping follow omega_b / omega_m, which alpha_rs alone cannot (wiggle_gates.py W4)
    A120, B30, Be2120 ...  the same with node spacing 1.20, 0.30, ... in ln k (default 0.6 = 16 nodes)
    ...N      full shape only: the post-reconstruction BAO points left out of the likelihood (its own MI chain)
    ...W      wide priors (WIDE_PRIOR): its own MI chain
    ...F      the direct map weighted by the data (projection only; same MI chain)
    ...M      the projection marginalizes the wiggle envelope (A, dSigma^2) instead of predicting it (same MI chain)

A direct model is attached with `cosmo` = a settings.VARIANTS key (baseline = lcdm3, lcdm5, w0wa7, ede7):
its Gaussian priors, box and direct chain (the exact likelihood, settings.files) are the reference of
the recovery test.
"""
import os, copy, json, time
import numpy as np

import settings as S
from wiggle_model import WiggleModel
from sampling import log

KNOT_FISHER = os.path.join(S.OUT_ROOT, 'wiggle_gates', 'knot_fisher.npz')     # wiggle_fisher_map.py
# 'W': every prior the 2026-10-04 chain hit widened (posterior/prior sd 0.53-0.83): the amplitudes s_lnP, the shape s_lna,
# ln f, the BAO amplitude A and damping dSigma^2. The alphas (0.13-0.23), alpha_rs and the smoothness are kept.
WIDE_PRIOR = {'s_lnP': 2.0, 's_lna': 2.0, 's_lnf': 1.0, 's_env': 1.0, 's_dsig2': 50.0}


def wiggle_spec(name):
    """'A' / 'B', optional 'e<n>' (n wiggle-envelope terms), optional node spacing in hundredths of ln k:
    'B90' -> B at 0.9, 'Be2' -> B + 2 envelope terms at 0.6, 'Be2120' -> B + 2 envelope terms at 1.2; a final 'F' makes
    the direct map data-weighted (KNOT_FISHER) instead of uniform in ln k: 'Be2F'. The MI model itself is the same.
    'g<n>' in place of 'e<n>' is the BAO-fit envelope: (1 + A) exp(-k^2 dSigma^2 / 2) (g2) or the damping alone (g1)."""
    import re
    m = re.fullmatch(r'([AB])(?:([eg])(\d))?(\d+)?(N)?(W)?(F)?(M)?', name)
    assert m, name
    return dict(rs_free=m.group(1) == 'A', env_type='gauss' if m.group(2) == 'g' else 'log', n_env=int(m.group(3) or 0),
                nodes={'spacing': int(m.group(4)) / 100 if m.group(4) else 0.6}, ls_weight=KNOT_FISHER if m.group(7) else None,
                nobao=bool(m.group(5)), prior=WIDE_PRIOR if m.group(6) else {}, marg_env=bool(m.group(8)),
                mi_name=''.join(m.group(i) or '' for i in range(1, 7)))

LABEL = {'A': r'A: $\alpha_\parallel,\alpha_\perp$ per $z$ + $\alpha_{r_s}$',
         'B': r'B: $\alpha_\parallel,\alpha_\perp$ per $z$ = BAO $\alpha$'}
COSMO_VARIANTS = ['baseline', 'lcdm5', 'w0wa7', 'ede7']          # the recovery tests (direct chains exist)


def resolve(wiggle='B', cosmo='baseline'):
    s = S.resolve(cosmo)
    w = wiggle_spec(wiggle)
    s['prior'] = dict(s['prior'], **w.get('prior', {})); s['nodes'] = dict(s['nodes'], **w['nodes'])
    s.update(wiggle=wiggle, rs_free=w['rs_free'], n_env=w['n_env'], env_type=w['env_type'], ls_weight=w['ls_weight'], cosmo_variant=cosmo,
             marg_env=w['marg_env'], nobao=w['nobao'])
    s['direct_out'] = s['out']                                    # the exact direct chain of this cosmo variant
    tag = os.environ.get('MI_TAG', '')                                          # e.g. x3: a combined / extended chain
    s['out'] = os.path.join(S.OUT_ROOT, f'wiggle_{wiggle}{tag}')                  # projections, logs
    s['mi_out'] = os.path.join(S.OUT_ROOT, f"wiggle_{w['mi_name']}{tag}")         # the MI chain ('Be2F', 'Be2FM' share Be2's)
    return s


def files(s):
    d = S.files(S.resolve(s['cosmo_variant']))
    return {'chain_direct': d['chain_direct'], 'bestfit_direct': d['bestfit_direct'],
            'bestfit_mi': os.path.join(s['mi_out'], 'bestfit_mi.npz'), 'chain_mi': os.path.join(s['mi_out'], 'chain_mi.npz'),
            'meta': os.path.join(s['out'], 'meta.json'), 'baseline_bestfit_direct': S.files(S.resolve('baseline'))['bestfit_direct']}


def build(s, verbose=True):
    data = S.load_data()
    if s.get('nobao'): data['cfg']['with_bao_rec'] = False        # N: the full-shape multipoles only
    M = WiggleModel(data, S.COSMO_FID, prior=s['prior'], nodes=s['nodes'], rs_free=s['rs_free'], n_env=s.get('n_env', 0), env_type=s.get('env_type', 'log'), ls_weight=s.get('ls_weight'), z_early=s['z_early'],
                    get_maxlkl=s['get_maxlkl'], verbose=verbose)
    if s['cmb']:
        import cmb
        M.extra_loglike = cmb.loglike
    if s['cosmo_model'] != 'lcdm3' or s['cosmo_gauss']:
        M.set_cosmo_model(s['cosmo_model'], gauss=s['cosmo_gauss'])
    return M, data


def describe(s, M):
    js = lambda v: json.loads(json.dumps(v, default=lambda o: np.asarray(o).tolist()))
    return js({'wiggle': s['wiggle'], 'settings': s, 'n_mi': M.n_mi, 'parameter_names_mi': M.names, 'nodes_h': M.nodes_h,
               'node_spacing': M.node_spacing, 'lambda_eff': M.lambda_eff, 'rdh_fid': M.rdh_fid, 'phi_fid': M.phi_fid,
               'prior_widths': M.prior_widths(), 'bao_dm_factor': M.bao_dm_factor, 'time': time.strftime('%Y-%m-%d %H:%M:%S')})
