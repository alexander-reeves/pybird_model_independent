"""What the pipeline fits, in one place: data, fiducial cosmology, the templates for the linear power spectrum, the
cosmological models and their priors, the sampler, and where everything is written. Every step of run.py builds its
model from here.

TEMPLATES: the model-independent (MI) parametrizations of the linear spectrum at z_N = 1.491 (templates.TemplateModel)
    agnostic       a(k) T(k): the fiducial spectrum times a free a(k) on 60 nodes, fine enough to reshape the BAO wiggles,
                   so nothing is assumed about them
    bao_template   a(k) T_nw(k) [1 + (1 + A) exp(-k^2 dSigma^2/2) O(alpha_rs k)]: a(k) on 21 nodes (too coarse to move a
                   wiggle) times the no-wiggle spectrum plus the fiducial BAO wiggles O, moved by alpha_rs = h r_d/(h r_d)_fid,
                   with a BAO amplitude A and damping dSigma^2 as in a BAO fit
    bao_template_no_rs   the same with alpha_rs = 1: the AP is parametrized by the BAO alphas (alpha_par, alpha_per = (D/r_d) /
                   (D/r_d)_fid) and P_lin is in sound-horizon units; alpha_rs x lam with every alpha x lam and the broadband
                   dilated is a symmetry of bao_template (6_alpha_rs_degeneracy.ipynb), so this drops its prior-only direction
  All three also have, per redshift: the growth rate f, the growth ratio D(z_i)/D(z_N) and two dilations alpha_par, alpha_perp,
  shared by the full-shape AP and the post-reconstruction BAO points. 12 EFT parameters are sampled, 42 profiled.

MODELS: the cosmologies fitted directly with the full likelihood (the references) and through the MI posterior
    lcdm   omega_cdm, omega_b, n_s, ln10^10 A_s, h
    w0wa   + w0, wa
    ede    omega_cdm, omega_b, ln10^10 A_s, h, f_EDE, log10 z_c, theta_i   (n_s fixed; CosmoPower ede-v2 emulators)
    mnu    lcdm + m_ncdm, the mass of each of 3 degenerate neutrinos (the ede-v2 emulators without EDE; cdm + baryon spectrum)
  Priors: flat boxes (model.COSMO_BOX) and the DESI-alone priors (desi_priors): BBN omega_b, n_s with 10 x Planck's error.
  The CMB acoustic angle theta_s (THETA_S) is added only where asked (run.py project, run.py reference --theta-s).

MOCKS: with the environment variable MI_MOCK=<name> every step runs on a noiseless mock instead of the data (same covariance,
  priors, sampler), in its own output tree; `checks.py truth` makes it first.

MI_DATA=y6_p | y6_pb: the noiseless DESI-Y6-like mock of 5_fisher_pk_bk.ipynb (7 samples, P0, P2, P4 for 0.01 < k < 0.2 h/Mpc,
  Gaussian covariance, no window, no BAO points) instead of DESI DR1: the power spectrum alone, or with the tree-level
  bispectrum monopole (triangles 0.02 <= k <= 0.10 h/Mpc). The same P(k) data, EFT priors, template and sampler in both. The
  bispectrum code is in pybird-dev (PYBIRD_DEV, default: a checkout next to this repository), which then replaces this
  repository's pybird for the whole run.
"""
import os, sys, glob, json, time
import numpy as np
import yaml

DATASET = os.environ.get('MI_DATA', 'dr1')
if DATASET != 'dr1':             # before anything imports pybird
    # pybird-dev's spline solves through equinox, which raises on a non-finite input; NUTS needs the NaN back (a rejected step)
    os.environ.setdefault('EQX_ON_ERROR', 'nan')
    sys.path.insert(0, os.environ.get('PYBIRD_DEV', os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'pybird-dev'))))

from model import MIModel, COSMO_MODELS, BBN_OMEGA_B
from templates import TemplateModel, sound_horizon_loglkl
from cosmology import ROOT, THETA_S, theta_s
from inference import log

# ------------------------------------------------------------------------------------------------- data and outputs
if DATASET == 'dr1':
    DATA_DIR = os.path.join(ROOT, 'data', 'desi_dr1_fs_bao')                 # DESI DR1 P0, P2 + post-reconstruction BAO
    LIKELIHOOD_CONFIG = os.path.join(DATA_DIR, 'likelihood_config', 'desi_dr1_fs_bao.yaml')
    FIDUCIAL = {'omega_b': 0.02237, 'omega_cdm': 0.1200, 'h': 0.6736, 'ln10^{10}A_s': 3.044, 'n_s': 0.9649}   # DESI fiducial
else:                                                                        # the Y6 mock (pk_bk/rebuild.py jacobians writes it)
    DATA_DIR = os.path.join(ROOT, 'output'); LIKELIHOOD_CONFIG = os.path.join(DATA_DIR, 'fake_desi_y6_fisher_pb_config.yaml')
    FIDUCIAL = {'omega_b': 0.02235, 'omega_cdm': 0.120, 'h': 0.675, 'ln10^{10}A_s': 3.044, 'n_s': 0.965}      # the mock's truth
LOOPS = os.environ.get('MI_LOOPS', 'emulator')   # 'exact': pybird's FFTLog one-loop instead of the 80-knot loop emulator (~7x
                                                 # slower in HMC); a cross-check with its own output tree, references included
MOCK = os.environ.get('MI_MOCK')                 # a noiseless mock (MOCKS) in place of the data, with its own output tree
_TREE = ('_exact_loops' if LOOPS == 'exact' else '') + (f'_mock_{MOCK}' if MOCK else '')
DATA_OUT = os.path.join(ROOT, 'output', 'desi_dr1')                          # the outputs of the real data (the mocks start from them)
OUT = os.path.join(ROOT, 'output', 'desi_' + DATASET + _TREE)                # everything the pipeline writes (not in git)
PRODUCTS = os.path.join(ROOT, 'data', 'mi_chains' + ('' if DATASET == 'dr1' else '_' + DATASET) + _TREE)   # what the notebooks read (run.py export)
# data weights of the map theta -> MI (run.py knots); both Y6 runs use the P(k)-alone weights, so that they share the map
KNOT_FISHER = os.path.join(ROOT, 'data', 'mi_chains' + ('' if DATASET == 'dr1' else '_y6_p'), 'knot_fisher.npz')
Z_EARLY = 5.0                    # CosmoPower redshift of the direct model, grown to z_N (dark energy negligible there)

if MOCK and os.path.exists(os.path.join(OUT, 'truth.npz')):                 # the theta_s prior of a mock: centred on its truth, Planck's width
    THETA_S = (float(np.load(os.path.join(OUT, 'truth.npz'))['theta_s']), THETA_S[1])

# ------------------------------------------------------------------------------------------------- the templates
# draws are the number of draws per chain, per template
# envelope is the envelope of the BAO template, i.e. the BAO amplitude A and damping dSigma^2 
# alpha_rs: the BAO position alpha_rs is free (False: alpha_rs = 1); node_spacing: of the ln a(k) nodes, in ln k
TEMPLATES = {
    'agnostic':     dict(node_spacing=0.15, bao_template=False, alpha_rs=False, envelope=False, draws=500),   # 60 nodes; 500 draws x 32 chains ~ 55 min on 4 GH200
    'bao_template': dict(node_spacing=0.45, bao_template=True, alpha_rs=True, envelope=True, draws=1200),     # 21 nodes; 1200 x 32 ~ 85 min
    'bao_template_no_rs': dict(node_spacing=0.45, bao_template=True, alpha_rs=False, envelope=True, draws=1000),   # alpha_rs = 1
    # the same with the growth derived from the expansion: a spline of H(z) r_d and Omega_m instead of f, D, alphas per redshift
    'growth_from_h': dict(node_spacing=0.45, bao_template=True, alpha_rs=False, envelope=True, growth_from_h=True, draws=1000),
}
# The MI prior: Gaussian in the log of each quantity, centred on the fiducial (A and dSigma^2: linear); the same for all templates
MI_PRIOR = {
    's_lna': 2.0,            # shape of ln a(k): per-node width around the node mean
    'smooth_lambda': 200.0,  # smoothness: lambda x sum (second difference of ln a)^2, quoted at node spacing 0.15
    's_lnP': 2.0,            # growth factor: width of (mean of ln a over the nodes) + 2 ln D(z_i)/D(z_N) at each redshift
    's_lnf': 1.0,            # the growth rate f(z_i)
    's_lng': 0.3,            # the dilations alpha_par(z_i), alpha_perp(z_i)
    's_lnrs': 0.05,          # bao_template: alpha_rs
    's_env': 1.0,            # bao_template: the BAO amplitude A
    's_dsig2': 50.0,         # bao_template: the BAO damping dSigma^2 [Mpc^2]
    's_lnm': 1.0,            # growth_from_h: Omega_m (H0 r_d)^2; its H(z) r_d nodes take s_lng
}

# ------------------------------------------------------------------------------------------------- cosmological models
MODELS = {                       # seed: of the reference chain; box_coordinates: sample in unbounded coordinates
    'lcdm': dict(seed=31),
    'w0wa': dict(seed=32, box_coordinates=True),
    'ede':  dict(seed=33, box_coordinates=True, start={'fEDE': 0.05}),
    'mnu':  dict(seed=34, box_coordinates=True, start={'m_ncdm': 0.05}),
}
NS_PRIOR = (0.9649, 0.042)       # n_s: Planck 2018's mean and 10 x its error (the DESI full-shape choice)
# Noiseless mocks: (model, parameters changed). The data vector is the true likelihood's prediction with the EFT parameters and
# the cosmology of the real data's lcdm best fit, these parameters changed, and what the model fixes at its fixed value
MOCKS = {'lcdm': ('lcdm', {}),
         'w0wa': ('w0wa', {'w0': -0.75, 'wa': -0.9}),                               # close to DESI + CMB + DES-SN5YR
         'ede':  ('ede', {'fEDE': 0.1, 'log10z_c': 3.56, 'thetai_scf': 2.8}),
         'mnu':  ('mnu', {'m_ncdm': 0.02})}                                         # sum 0.06 eV


def desi_priors(model):
    """The DESI-alone Gaussian priors of a model: BBN omega_b ~ N(0.02218, 0.00055), and n_s ~ NS_PRIOR where n_s is free."""
    return {'omega_b': BBN_OMEGA_B, **({'n_s': NS_PRIOR} if 'n_s' in COSMO_MODELS[model] else {})}


# ------------------------------------------------------------------------------------------------- sampler, flows
SAMPLER = dict(n_warmup=500, n_chains=32, max_doublings=8, target_accept=0.8, initial_step_size=0.1,
               jitter_mi=0.1, jitter_reference=0.5, reference_draws=2000 if LOOPS != 'exact' else 1000)
EFT_START = {'b1': [1.5, 2.0, 2.1, 2.3, 1.3, 2.3], 'c2': 0.5}         # L-BFGS-B start (b1 of BGS, LRG1-3, ELG2, QSO)
# The projection density. Default: the normalizing flow. It estimates the MI posterior p(phi | d) from the chain; divided by the
# MI prior that is the likelihood of the MI parameters, which is evaluated at phi(theta) for a cosmology theta.
# Back-up and cross-check: score regression (inference.fit_score), which fits the same likelihood from its gradients at the
# chain draws. `run.py recovery` runs both against the direct fits; `run.py project --density score` projects with the back-up.
FLOW = dict(n_layers=8, hidden=128, lr=1e-4, patience=1000)          # 10 RealNVP flows of the MI posterior
SCORE = dict(n_members=20, hidden=256, steps=60000, lr=3e-4)         # back-up: the mean of 20 fits (single fits scatter 0.1-0.3 sigma along w0-wa)
DENSITY = 'flow'                 # what run.py project uses: 'flow' (default) or 'score' (the back-up)


# ------------------------------------------------------------------------------------------------- where things are
def template_dir(template, full_shape_only=False):
    return os.path.join(OUT, template + ('_full_shape_only' if full_shape_only else ''))


def reference_dir(model, with_theta_s=False, sound_horizon=False):
    return os.path.join(OUT, 'reference', model + ('_rd' if sound_horizon else '') + ('_theta_s' if with_theta_s else ''))


def chain_files(template, full_shape_only=False):
    """All MI chains of a template, one per seed: <template>/chain_seed<k>.npz. Every step downstream uses all of them."""
    return sorted(glob.glob(os.path.join(template_dir(template, full_shape_only), 'chain_seed*.npz')))


def load_chain(template, full_shape_only=False):
    """The MI chains of a template concatenated along the chain axis: x (chains, draws, n_mi) and the per-draw arrays."""
    fs = chain_files(template, full_shape_only)
    if not fs: raise FileNotFoundError(f"no chain_seed*.npz in {template_dir(template, full_shape_only)}: run.py chain first")
    cs = [np.load(f) for f in fs]; n = min(c['x'].shape[1] for c in cs)
    out = {k: np.concatenate([c[k][:, :n] for c in cs]) for k in ('x', 'logp', 'divergent')}
    out['files'] = fs
    return out


# ------------------------------------------------------------------------------------------------- building the models
def read_bao_fid(g):
    """BAO fiducial of one sky from the data file; `iso` is stored as a string or a bool."""
    iso = g['iso'][()]
    if isinstance(iso, bytes): iso = iso.decode()
    if isinstance(iso, str): iso = iso.strip().lower() in ('true', '1', 'yes')
    return {'zeff': float(g['zeff'][()]), 'iso': bool(iso),
            'DH_over_rd_fid': float(g['DH_over_rd_fid'][()]), 'DM_over_rd_fid': float(g['DM_over_rd_fid'][()])}


def fiducial_distances(zs):
    """D_H / r_d and D_M / r_d of FIDUCIAL at the redshifts zs, as the BAO fiducials of a data file: the reference of the alphas
    for data without post-reconstruction BAO."""
    from model import Omega_m, rd_h, sym_H, sym_DA, C_LIGHT
    Om, rdh = float(Omega_m(FIDUCIAL)), float(rd_h(FIDUCIAL))
    return [{'zeff': z, 'iso': False, 'DH_over_rd_fid': C_LIGHT / (100. * float(sym_H(Om, z))) / rdh,
             'DM_over_rd_fid': C_LIGHT / 100. * (1. + z) * float(sym_DA(Om, z)) / rdh} for z in zs]


def with_bispectrum_times(M):
    """Y6 mock, the two things pybird-dev wants differently: the EdS time coefficients in the cosmology dicts (the mock was made
    with them), and the EFT parameters named per sample (<name>_<sky>; bare names are silently read as 0 there)."""
    if DATASET != 'dr1':
        f = M.cosmo_dicts; M.cosmo_dicts = lambda phi: [dict(d, G1=1., Y1=0., G1t=3. / 7., V12t=1. / 7.) for d in f(phi)]
        M.eft_names_flat = list(M.eft_labels)
    return M


def load_data(full_shape_only=False):
    """The DESI DR1 FS+BAO likelihood config with resolved paths, the effective redshift of each sky and the BAO fiducials
    (one per unique redshift, in increasing z). full_shape_only: the BAO points are left out of the likelihood."""
    import h5py
    cfg = yaml.full_load(open(LIKELIHOOD_CONFIG)); cfg['data_path'] = DATA_DIR
    cfg['write'] = {'save': False, 'fake': False, 'plot': False, 'show': False}
    cfg['with_emu'] = LOOPS != 'exact'
    if DATASET == 'y6_p':                                # the power spectrum alone: no bispectrum, nor its two EFT parameters
        cfg['output'] = 'bPk'; [cfg['eft_prior'].pop(n) for n in ('Bd1', 'Be5')]
    if LOOPS == 'exact' and not getattr(load_data, 'matrices', False):
        # pybird's loop matrices, built with numpy (under JAX: hours) and cached in ~/.cache, which the container does not keep
        import subprocess, sys; load_data.matrices = True
        subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checks.py'), 'loop_matrices'], check=True)
    skies = list(cfg['sky'])
    with h5py.File(os.path.join(DATA_DIR, cfg['data_file']), 'r') as hf:
        zeff = [float(hf[sky]['z']['eff'][()]) for sky in skies]
        bao_sky = {sky: read_bao_fid(hf[sky]['bao_rec']['fid']) for sky in skies} if cfg.get('with_bao_rec') else None
    bao = ([next(bao_sky[sky] for sky, z in zip(skies, zeff) if z == zu) for zu in sorted(set(zeff))] if bao_sky
           else fiducial_distances(sorted(set(zeff))))
    if full_shape_only: cfg['with_bao_rec'] = False
    return {'cfg': cfg, 'skies': skies, 'zeff': zeff, 'bao': bao}


def truth():
    """The truth of the mock MI_MOCK (checks.py truth): y the data vector, x = [EFT, cosmological parameters], keys, theta_s."""
    return np.load(os.path.join(OUT, 'truth.npz'))


def mock_data(M):
    """M with the mock's data vector in place of the data when MI_MOCK is set (the covariance stays)."""
    if MOCK:
        y = truth()['y']; n = np.cumsum([len(v) for v in M.L.y_sky])[:-1]
        M.L.y_sky = [np.array(v) for v in np.split(y, n)]; M.L.y_all = np.array(y)
    return M


def build_template(template, model='lcdm', full_shape_only=False, verbose=True):
    """The MI model of a template, with `model` (and its DESI-alone priors) as the cosmology it projects onto."""
    t = TEMPLATES[template]
    M = TemplateModel(load_data(full_shape_only), FIDUCIAL, prior=MI_PRIOR, nodes={'spacing': t['node_spacing']},
                      bao_template=t['bao_template'], alpha_rs=t.get('alpha_rs'), envelope=t.get('envelope'),
                      ls_weight=KNOT_FISHER if os.path.exists(KNOT_FISHER) else None,      # absent only before `run.py knots` on a new data set
                      growth_from_h=t.get('growth_from_h', False), z_early=Z_EARLY, verbose=verbose)
    return mock_data(with_bispectrum_times(M).set_cosmo_model(model, gauss=desi_priors(model)))


def build_reference(model, verbose=True, mock=True, with_theta_s=False, sound_horizon=False):
    """The true likelihood of a cosmological model: the engine's spectrum at the emulator knots, r_d from rd_h_fixed.
    mock=False: on the real data whatever MI_MOCK. with_theta_s: 100 theta_s ~ N(THETA_S) as part of the model's prior.
    sound_horizon: the same likelihood in the alpha_rs = 1 frame (templates.sound_horizon_loglkl), i.e. with the EFT priors in
    units of the sound horizon: what bao_template_no_rs represents (7_alpha_rs_comparison.ipynb)."""
    from model import fix_rd, make_true
    if sound_horizon:
        M = TemplateModel(load_data(), FIDUCIAL, prior=MI_PRIOR, nodes={'spacing': TEMPLATES['bao_template_no_rs']['node_spacing']},
                          bao_template=True, alpha_rs=False, envelope=True, ls_weight=KNOT_FISHER, z_early=Z_EARLY, verbose=verbose)
        M.set_cosmo_model(model, gauss=desi_priors(model))
        M.loglkl_direct = lambda x: sound_horizon_loglkl(M, x[:M.n_eft], M.theta_to_full(x[M.n_eft:]))
    else:
        M = MIModel(load_data(), FIDUCIAL, z_early=Z_EARLY, verbose=verbose)
        M = make_true(fix_rd(M.set_cosmo_model(model, gauss=desi_priors(model))))
    if with_theta_s: M.extra_loglike = lambda c: -0.5 * ((theta_s(c) - THETA_S[0]) / THETA_S[1])**2
    with_bispectrum_times(M)
    return mock_data(M) if mock else M


def eft_start(M):
    """EFT start of a fit: the mock's truth; on the data the lcdm reference best fit if it exists, else EFT_START."""
    if MOCK: return truth()['x'][:M.n_eft]
    f = os.path.join(reference_dir('lcdm'), 'bestfit.npz')
    if os.path.exists(f): return np.load(f)['x_bf'][:M.n_eft]
    pr = M.L.c['eft_prior']                              # parameters without an entry in EFT_START start at their prior mean
    return np.array([EFT_START['b1'][i] if n == 'b1' else EFT_START.get(n, pr[n]['mean'][i]) for i in range(M.num_skies) for n in M.eft_free])


def describe(M, **extra):
    """JSON record of a built model (-> meta.json next to its outputs)."""
    js = lambda v: json.loads(json.dumps(v, default=lambda o: np.asarray(o).tolist()))
    d = {'n_eft': M.n_eft, 'parameter_names': M.names, 'model': M.cosmo_model, 'priors': M.cosmo_gauss, 'time': time.strftime('%Y-%m-%d %H:%M:%S')}
    if isinstance(M, TemplateModel):
        d.update(prior=M.prior, nodes_h=M.nodes_h, node_spacing=M.node_spacing, lambda_eff=M.lambda_eff, phi_fid=M.phi_fid,
                 prior_widths=M.prior_widths(), bao_template=M.bao_template)
    else:
        d.update(parameter_names=M.eft_labels + M.cosmo_names)
    return js(dict(d, **extra))
