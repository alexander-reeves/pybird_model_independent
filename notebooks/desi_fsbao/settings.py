"""Settings of the DESI DR1 FS+BAO model-independent analysis: data, fiducial cosmology, priors,
node basis and sampler, in one place, shared by the sampler (sample.py), the gates (gates.py)
and the analysis (analysis.py), so the analysis always rebuilds the model that was sampled.

Variants differ from the baseline only in what they override; each has its own output
directory output/desi_fsbao/<variant>/.
"""
import os, copy, json, subprocess, time
import numpy as np
import yaml

from model import MIModel, ROOT, BBN_OMEGA_B
from sampling import log

DATA_DIR = os.path.join(ROOT, 'data', 'desi_dr1_fs_bao')
CONFIG = os.path.join(DATA_DIR, 'likelihood_config', 'desi_dr1_fs_bao.yaml')
OUT_ROOT = os.path.join(ROOT, 'output', 'desi_fsbao')

# DESI fiducial cosmology: the template and the centre of every MI prior
COSMO_FID = {'omega_b': 0.02237, 'omega_cdm': 0.1200, 'h': 0.6736, 'ln10^{10}A_s': 3.044, 'n_s': 0.9649}

BASELINE = dict(
    # MI prior (model.MIModel docstring): widths in ln
    prior={'s_lna': 0.5, 'smooth_lambda': 200.0, 's_lnP': 0.6, 's_lng': 0.3, 's_lnh': 0.05, 's_lnbao': 0.3},
    # ln a nodes: log-spaced over k_range (h/Mpc at h_fid), `spacing` in ln k -> 60 nodes
    nodes={'k_range': (1e-4, 0.7), 'spacing': 0.15},
    z_early=5.0,             # CosmoPower redshift of the direct model (dark energy negligible there)
    get_maxlkl=True,         # profile the analytically marginalized EFT parameters
    cosmo_model='lcdm3', cosmo_gauss=None, cmb=False,
    # NUTS
    n_warmup=500, n_samples=2000, n_chains=32, max_doublings=8, target_accept=0.8, initial_step_size=0.1,
    jitter_direct=0.5, jitter_mi=0.1, seed_direct=21, seed_mi=22,
)

VARIANTS = {
    'baseline': {},
    # the prior audit of the P+B Fisher (../prior_decomposition_pb.py): lambda is the block that
    # informs LCDM, s_lna sets most of the reported P(k) band
    'wide': {'prior': {'s_lna': 2.0, 'smooth_lambda': 50.0}},
    # enlarged direct models (the MI chain is the baseline's; only the direct chain is new)
    'lcdm5': {'cosmo_model': 'lcdm5', 'cosmo_gauss': {'omega_b': BBN_OMEGA_B}, 'seed_direct': 31, 'mi_from': 'baseline'},
    'w0wa7': {'cosmo_model': 'w0wa7', 'cosmo_gauss': {'omega_b': BBN_OMEGA_B}, 'seed_direct': 32, 'mi_from': 'baseline', 'box_transform': True},
    # + the CMB marginalized over late-time physics (cmb.py; Planck PR4, Lemos & Lewis 2023); no BBN prior then
    'lcdm5_cmb': {'cosmo_model': 'lcdm5', 'cmb': True, 'seed_direct': 35, 'mi_from': 'baseline'},
    'w0wa7_cmb': {'cosmo_model': 'w0wa7', 'cmb': True, 'seed_direct': 36, 'mi_from': 'baseline', 'box_transform': True},
    # early dark energy (CosmoPower ede-v2 emulators, ede.py), n_s fixed; sampled in box coordinates.
    # The compression test: this direct chain against the baseline MI chain projected onto ede7.
    'ede7': {'cosmo_model': 'ede7', 'cosmo_gauss': {'omega_b': BBN_OMEGA_B}, 'seed_direct': 33, 'mi_from': 'baseline',
             'box_transform': True, 'start': {'fEDE': 0.05}},
    # the EDE engine at its LCDM limit (fEDE = 0.001): separates emulator differences from EDE
    'lcdm4e': {'cosmo_model': 'lcdm4e', 'cosmo_gauss': {'omega_b': BBN_OMEGA_B}, 'seed_direct': 34, 'mi_from': 'baseline'},
}

# EFT starting values for L-BFGS-B (b1 per sample BGS, LRG1, LRG2, LRG3, ELG2, QSO)
EFT_START = {'b1': [1.5, 2.0, 2.1, 2.3, 1.3, 2.3], 'c2': 0.5}


def resolve(variant='baseline', overrides=None):
    """BASELINE <- VARIANTS[variant] <- overrides (dicts are merged one level deep)."""
    if variant not in VARIANTS: raise ValueError(f"unknown variant '{variant}'; choose from {list(VARIANTS)}")
    s = copy.deepcopy(BASELINE); s.update(mi_from=None, box_transform=False, start=None)
    for src in (VARIANTS[variant], overrides or {}):
        for k, v in copy.deepcopy(src).items():
            if v is None: continue
            if isinstance(v, dict) and isinstance(s.get(k), dict): s[k].update(v)
            else: s[k] = v
    s['variant'] = variant
    s['out'] = os.path.join(OUT_ROOT, variant)
    s['mi_out'] = os.path.join(OUT_ROOT, s['mi_from']) if s['mi_from'] else s['out']
    return s


def files(s):
    tag = '' if s['cosmo_model'] == 'lcdm3' else f"_{s['cosmo_model']}"
    return {'bestfit_direct': os.path.join(s['out'], f'bestfit_direct{tag}.npz'), 'chain_direct': os.path.join(s['out'], f'chain_direct{tag}.npz'),
            'bestfit_mi': os.path.join(s['mi_out'], 'bestfit_mi.npz'), 'chain_mi': os.path.join(s['mi_out'], 'chain_mi.npz'),
            'meta': os.path.join(s['out'], 'meta.json')}


def read_bao_fid(g):
    """BAO fiducial of one sky from the data file; `iso` is stored as a string or a bool."""
    iso = g['iso'][()]
    if isinstance(iso, bytes): iso = iso.decode()
    if isinstance(iso, str): iso = iso.strip().lower() in ('true', '1', 'yes')
    return {'zeff': float(g['zeff'][()]), 'iso': bool(iso),
            'DH_over_rd_fid': float(g['DH_over_rd_fid'][()]), 'DM_over_rd_fid': float(g['DM_over_rd_fid'][()])}


def load_data(config=CONFIG):
    """The DESI DR1 FS+BAO likelihood config with resolved paths, the effective redshift of each
    sky, the AP fiducial and the BAO fiducials (one per unique redshift, in increasing z)."""
    import h5py
    cfg = yaml.full_load(open(config)); cfg['data_path'] = DATA_DIR
    cfg['write'] = {'save': False, 'fake': False, 'plot': False, 'show': False}
    skies = list(cfg['sky'])
    with h5py.File(os.path.join(DATA_DIR, cfg['data_file']), 'r') as hf:
        zeff = [float(hf[sky]['z']['eff'][()]) for sky in skies]
        ap_Om = float(hf[skies[0]]['fid']['Omega_m'][()])
        bao_sky = {sky: read_bao_fid(hf[sky]['bao_rec']['fid']) for sky in skies} if cfg.get('with_bao_rec') else None
    bao = None
    if bao_sky:
        bao = [next(bao_sky[sky] for sky, z in zip(skies, zeff) if z == zu) for zu in sorted(set(zeff))]
    return {'cfg': cfg, 'config_file': config, 'skies': skies, 'zeff': zeff, 'ap_Omega_m_fid': ap_Om, 'bao': bao}


def build(s, verbose=True):
    """Likelihood + MIModel for resolved settings s. Returns (M, data)."""
    data = load_data()
    if verbose:
        log(f"DESI DR1 FS+BAO, variant '{s['variant']}': {data['config_file']}")
        log(f"  skies {data['skies']}, z_eff {data['zeff']}, AP fiducial Omega_m {data['ap_Omega_m_fid']:.4f}")
    M = MIModel(data, COSMO_FID, prior=s['prior'], nodes=s['nodes'], z_early=s['z_early'], get_maxlkl=s['get_maxlkl'], verbose=verbose)
    if s['cmb']:
        import cmb
        M.extra_loglike = cmb.loglike
    if s['cosmo_model'] != 'lcdm3' or s['cosmo_gauss']:
        M.set_cosmo_model(s['cosmo_model'], gauss=s['cosmo_gauss'])
    if verbose:
        log(f"  direct model {M.cosmo_model} {M.cosmo_keys}, box {M.cosmo_box}, Gaussian {M.cosmo_gauss}"
            + ("; + CMB marginalized over late-time physics (cmb.py)" if s['cmb'] else ""))
        log(f"  data points {sum(len(y) for y in M.L.y_sky)}")
    return M, data


def eft_start(M):
    return np.array([EFT_START['b1'][i] if n == 'b1' else EFT_START.get(n, 0.0) for i in range(M.num_skies) for n in M.eft_free])


def describe(s, M, data):
    """JSON record of the resolved settings and the constructed model (-> meta.json)."""
    try: git = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception: git = None
    import jax
    js = lambda v: json.loads(json.dumps(v, default=lambda o: np.asarray(o).tolist()))
    return js({'variant': s['variant'], 'settings': {k: v for k, v in s.items()}, 'config_file': data['config_file'],
               'skies': data['skies'], 'zeff': data['zeff'], 'z_N': M.z_N, 'n_mi': M.n_mi, 'n_direct': M.n_eft + M.n_cosmo,
               'parameter_names_mi': M.names, 'parameter_names_direct': M.eft_labels + M.cosmo_names,
               'nodes_h': M.nodes_h, 'phi_fid': M.phi_fid, 'prior_widths': M.prior_widths(), 'bao': M.bao,
               'jax_devices': [str(d) for d in jax.devices()], 'git_head': git, 'time': time.strftime('%Y-%m-%d %H:%M:%S')})

# flow-ensemble settings for the compression (compared in flow_study.py, meeting B0)
FLOW_CONFIGS = {'c0': dict(n_layers=8, hidden=128, lr=5e-4),          # 2026-10-01 production
                'c1': dict(n_layers=12, hidden=256, lr=5e-4),         # the larger flow of the BOSS study
                'c2': dict(n_layers=8, hidden=128, lr=1e-4),
                'c3': dict(n_layers=12, hidden=256, lr=2e-4),
                # the 2026-10-01 runs stopped every member at step 200-300 (patience 20 x 100 steps), i.e. at the
                # peak learning rate: train the whole cosine schedule and keep the best held-out checkpoint
                'c4': dict(n_layers=8, hidden=128, lr=5e-4, patience=1000),
                'c5': dict(n_layers=8, hidden=128, lr=1e-4, patience=1000)}


def load_mi_samples(s, extra=None):
    """The MI chain of a variant, flattened, plus independent extra chains of the same model (directories
    under output/desi_fsbao/, e.g. 'baseline_ext'; default from the MI_EXTRA environment variable)."""
    extra = os.environ.get('MI_EXTRA', '') if extra is None else extra
    dirs = [s['mi_out']] + [os.path.join(OUT_ROOT, d) for d in str(extra).split(',') if d]
    x = [np.load(os.path.join(d, 'chain_mi.npz'))['x'] for d in dirs]
    x = np.concatenate([a.reshape(-1, a.shape[-1]) for a in x])
    log(f"MI samples: {len(x)} from {[os.path.basename(d) for d in dirs]}")
    return x
