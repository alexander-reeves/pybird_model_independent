"""Rebuilds the cached inputs of ../../notebooks/5_fisher_pk_bk.ipynb (output/fisher_pb/) and the notebook.
On Alps: ARGS="<subcommand> [options]" sbatch run.sbatch (CPU). Each cache subcommand takes
--out PATH (default: the cache itself).

    jacobians            jacobians_pb.npz (+ jacobians_pb.log, which carries the mock nbar): the
                         model Jacobians, data precision, EFT priors, symmetry tests   5-10 min
    cosmo_map            cosmo_map_pb.npz: d(MI parameters)/d(omega_cdm, lnAs, h), and the gate
                         J_direct = J_MI M                                             ~3 min
    robust               robust_pb_results.npz: BAO stage A, the 80-knot model without a
                         template prior, by SVD                                        ~1 min
    se07                 se07_forecast.npz: standard BAO (Seo & Eisenstein 2007) on the same mock
    notebook [--execute] writes ../../notebooks/5_fisher_pk_bk.ipynb with cleared outputs [and runs it in place]

From scratch: jacobians, then cosmo_map, robust, se07 (any order), then notebook --execute.
jacobians and cosmo_map need pybird-dev (the bispectrum JAX code; PYBIRD_DEV overrides the path)
and write the mock (fake_desi_y6_fisher_pb.h5 + _config.yaml) next to --out.
"""
import os, re, sys, time, inspect, argparse
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.normpath(os.path.join(HERE, '..', '..', 'output', 'fisher_pb'))
PYBIRD_DEV = os.environ.get('PYBIRD_DEV', os.path.normpath(os.path.join(HERE, '..', '..', '..', 'pybird-dev')))   # a checkout next to this repo
NB = os.path.normpath(os.path.join(HERE, '..', '..', 'notebooks', '5_fisher_pk_bk.ipynb'))


# ============================================================================================
# the mock and the two likelihoods (pybird-dev)
# ============================================================================================
def model(fake_dir, log):
    """The DESI-Y6-like mock, the P and P+B likelihoods and the MI model vector on them.

    MI parametrization: the P_lin template in Mpc^3 at z_ref = 5 on the 80 emulator knots (fixed
    1/Mpc grid) times free amplitudes a_j, and [f, H/H0, D_A H0 per z, D(z)/D(z_ref) per z,
    h_conv]; the direct LCDM model is the exact composition with the cosmology maps. The data are
    the direct model at the fiducial: P0+P2+P4 (0.01 < k < 0.2 h/Mpc, Gaussian covariance) and the
    tree-level B0 (closed triangles 0.02 <= k_i <= 0.10 h/Mpc, PPP covariance, no P-B cross term),
    written by a re-implementation of Fake.set() (pybird-dev's own would run its Boltzmann path,
    not this model). Both likelihoods carry the SAME P(k) multipoles, so P is a strict subset of
    P+B. EFT basis 'eth' (the only one the bispectrum accepts; for P(k) it is eftoflss relabelled),
    one-loop P(k) from the 80-knot emulator with the cubic input interpolation of v3.1 and the EdS
    time coefficients the MI tree hard-codes. Set with_bk_tree_level False + a bk_loop_matrix_path
    per sky once the one-loop bispectrum matrices exist."""
    from types import SimpleNamespace
    from copy import deepcopy
    from collections import defaultdict
    sys.path.insert(0, PYBIRD_DEV)             # must shadow the editable pybird_model_independent
    import yaml
    import h5py
    import jax
    import jax.numpy as jnp
    from scipy.interpolate import interp1d as scipy_interp1d
    jax.config.update("jax_enable_x64", True)
    from pybird import config
    assert os.path.realpath(config.__file__).startswith(os.path.realpath(PYBIRD_DEV)), config.__file__
    config.set_jax_enabled(True)
    from pybird.likelihood import Likelihood
    from pybird.fake import Fake
    from pybird.utils_cov import get_cov_gauss, get_cov_b_PPP
    from pybird.symbolic import D as symD, f as symf, Hubble as symHubble, DA as symDA
    from cosmopower_jax.cosmopower_jax import CosmoPowerJAX as CPJ
    from pybird import emulator as _pb_emu
    from pybird.jax_special import interp1d as _pb_interp1d

    # emulator input interpolation cubic, not piecewise-linear (v3.1 fix): the template sits ON
    # the knots at h_conv = h_fid, so the curvature along h_conv would jump by ~20% there
    def _make_params_smooth(self, kk, pk, f=1.0, time=False, ir=False, pca=False):
        pk_max = jnp.max(pk)
        self.pk_max = jnp.array(pk_max)
        ilogpk = _pb_interp1d(jnp.log(kk), jnp.log(pk / self.pk_max.reshape(-1, 1)),
                              axis=-1, kind='cubic')
        logpk = ilogpk(self.logknots)
        if ir:
            self.params = jnp.concatenate([logpk, jnp.array([[pk_max]]), jnp.array([[f]])], axis=1)
        else:
            self.params = logpk
        return logpk

    _pb_emu.Emulator.make_params = _make_params_smooth
    os.makedirs(fake_dir, exist_ok=True)
    log(f"pybird from {os.path.dirname(config.__file__)}; jax devices: {jax.devices()}")

    # ---- fiducial cosmology, survey, CosmoPower-JAX
    cosmo_fid = {'omega_b': 0.02235, 'omega_cdm': 0.120, 'h': 0.675, 'ln10^{10}A_s': 3.044,
                 'n_s': 0.965}
    h_true = cosmo_fid['h']
    Om_fid = (cosmo_fid['omega_cdm'] + cosmo_fid['omega_b']) / h_true**2
    z_ref = 5.0  # P_lin(k [1/Mpc], z_ref) in Mpc^3 is ~h-independent here
    DESI_Y6 = {
        'n_sky': 7,
        'zmin': [.1, .4, .6, .8, .8, 1.1, .8],
        'zmax': [.4, .6, .8, 1.1, 1.1, 1.6, 2.1],
        'zeff': [0.295, 0.51, 0.706, 0.930, 0.930, 1.317, 1.491],
        'Veff': np.array([4., 8., 12., 15., 8., 12., 4.]) * 1.e9,
        'degsq': [14000] * 7,
        'P0': np.array([9.2, 8.9, 8.9, 8.4, 8.4, 2.9, 5.]) * 1.e3,
        'nbar_prior': [3.e-4, 3.e-4, 3.e-4, 3.e-4, 3.e-4, 2.e-3, 1.e-4],
    }
    num_skies = DESI_Y6['n_sky']
    zeff_list = list(DESI_Y6['zeff'])
    zeff_unique = sorted(set(zeff_list))
    n_z_unique = len(zeff_unique)
    sky_to_z_idx = [zeff_unique.index(z) for z in zeff_list]
    cpj_pklin = CPJ(probe='mpk_lin')
    cpj_k_modes = jnp.array(cpj_pklin.modes)  # 1/Mpc

    def pklin_cpj(k_mpc, omega_b, omega_cdm, h, lnAs, n_s, z):
        """P_lin in Mpc^3 at k in 1/Mpc from CosmoPower-JAX (log-log interp onto k_mpc)."""
        input_dict = {
            'omega_b': jnp.atleast_1d(omega_b), 'omega_cdm': jnp.atleast_1d(omega_cdm),
            'h': jnp.atleast_1d(h), 'ln10^{10}A_s': jnp.atleast_1d(lnAs),
            'n_s': jnp.atleast_1d(n_s), 'z': jnp.atleast_1d(z),
        }
        pk = cpj_pklin.predict(input_dict)
        return jnp.exp(jnp.interp(jnp.log(k_mpc), jnp.log(cpj_k_modes), jnp.log(pk)))

    # ---- knots = the emulator's 80 native knots; the physical template
    knots_h = np.load(os.path.join(PYBIRD_DEV, 'pybird', 'emu_data', 'knots.npy'))  # h/Mpc
    n_knots = len(knots_h)
    knots_mpc = jnp.array(knots_h * h_true)  # FIXED physical grid in 1/Mpc
    template_mpc = jnp.array(pklin_cpj(knots_mpc, cosmo_fid['omega_b'], cosmo_fid['omega_cdm'],
                                       h_true, cosmo_fid['ln10^{10}A_s'], cosmo_fid['n_s'], z_ref))
    log(f"{n_knots} emulator knots, k=[{knots_h.min():.1e}, {knots_h.max():.2f}] h/Mpc")

    # ---- cosmology -> MI parameters
    def cosmo_to_amps(cosmo_params_3d):
        """[omega_cdm, lnAs, h] -> P(k) amplitude ratios at the FIXED physical knots (1/Mpc)."""
        omega_cdm, lnAs, h = cosmo_params_3d
        pk = pklin_cpj(knots_mpc, cosmo_fid['omega_b'], omega_cdm, h, lnAs, cosmo_fid['n_s'], z_ref)
        return pk / template_mpc

    def cosmo_to_growth(cosmo_params_2d):
        """[omega_cdm, h] -> [f, H/H0, DA*H0 per z, D(z)/D(z_ref) per z, h_conv]."""
        omega_cdm, h = cosmo_params_2d
        Om = (omega_cdm + cosmo_fid['omega_b']) / h**2
        out = []
        for z in zeff_unique:
            out.extend([symf(Om, z, -1., 0.), symHubble(Om, z, -1., 0.), symDA(Om, z, -1., 0.)])
        D_ref = symD(Om, z_ref, -1., 0.)
        for z in zeff_unique:
            out.append(symD(Om, z, -1., 0.) / D_ref)
        out.append(h)  # h_conv
        return jnp.array(out)

    def cosmo_to_observables(cosmo_params_3d):
        return jnp.concatenate([cosmo_to_amps(cosmo_params_3d),
                                cosmo_to_growth(jnp.array([cosmo_params_3d[0], cosmo_params_3d[2]]))])

    cosmo_fid_vec = jnp.array([cosmo_fid['omega_cdm'], cosmo_fid['ln10^{10}A_s'], cosmo_fid['h']])
    cosmo_fid_2d = jnp.array([cosmo_fid['omega_cdm'], cosmo_fid['h']])
    growth_fid = np.array(cosmo_to_growth(cosmo_fid_2d))
    n_growth = len(growth_fid)  # 3*n_z + n_z + 1
    idx_f = [3 * i for i in range(n_z_unique)]
    idx_H = [3 * i + 1 for i in range(n_z_unique)]
    idx_DA = [3 * i + 2 for i in range(n_z_unique)]

    # ---- MI parameters -> per-sky pybird inputs
    def build_cosmo_dicts(amps, growth):
        """In the data's h-units via h_conv: kk = knots_mpc / h_conv [h/Mpc], pk = P_mpc h_conv^3
        (same physical points, no interpolation); EdS time coefficients, as the MI tree."""
        f_z = [growth[i * 3] for i in range(n_z_unique)]
        H_z = [growth[i * 3 + 1] for i in range(n_z_unique)]
        DA_z = [growth[i * 3 + 2] for i in range(n_z_unique)]
        D_ratios = [growth[3 * n_z_unique + i] for i in range(n_z_unique)]
        h_conv = growth[-1]
        kk_h = knots_mpc / h_conv
        pk_h = amps * template_mpc * h_conv**3
        dicts = []
        for i_sky in range(num_skies):
            i_z = sky_to_z_idx[i_sky]
            dicts.append({"H": H_z[i_z], "DA": DA_z[i_z], "f": f_z[i_z],
                          "kk": kk_h, "pk_lin": pk_h * D_ratios[i_z]**2,
                          "G1": 1.0, "Y1": 0.0, "G1t": 3.0 / 7.0, "V12t": 1.0 / 7.0})
        return dicts

    def build_direct_cosmo_dicts(cosmo_params_3d):
        """Direct model = MI model composed with the cosmology mapping (exactly)."""
        amps = cosmo_to_amps(cosmo_params_3d)
        growth = cosmo_to_growth(jnp.array([cosmo_params_3d[0], cosmo_params_3d[2]]))
        return build_cosmo_dicts(amps, growth)

    # ---- the mock: pybird-dev's Abacus 'ELG_pk_bktree' config ('eth' basis), every projection
    # effect off; P0+P2+P4 (multipole 3) for BOTH likelihoods so P stays a subset of P+B
    template_configfile = os.path.join(PYBIRD_DEV, 'data', 'abacus', 'likelihood_config',
                                       'ELG_pk_bktree.yaml')
    lkl_cfg = yaml.safe_load(open(template_configfile))
    for key in ['data_path', 'data_file', 'nd', 'sky']:
        lkl_cfg.pop(key, None)
    lkl_cfg.update({
        'output': 'bPk,bBk', 'multipole': 3,
        'with_emu': True, 'with_resum': True, 'with_ap': True,
        'with_binning': False, 'with_survey_mask': False,
        'bk_multipole': [{'l': 0, 'mu_i': 1}],
        'with_bk_tree_level': True,
        'with_bk_resum': True, 'with_bk_ap': True,
        'with_bk_binning': False, 'with_bk_survey_mask': False,
    })
    lkl_cfg['write'].update({'save': False, 'fake': False, 'plot': False, 'show': False})
    # fiducial EFT parameters (pybird-dev demo/fake_bispectrum.ipynb): Bb1, Bb2, Bb5 non-zero
    eft_all = ("Bb1 Bb2 Bb3 Bb5 Bb8 Bc1 Bc2 Bc3 Bc4 Bd1 Be1 Be2 Be5 ce2 cr4 cr6").split()
    fiducial_nuisance = {name: 0.0 for name in eft_all}
    fiducial_nuisance.update({"Bb1": 1.8, "Bb2": 0.4, "Bb5": 0.2})
    k_bk = np.arange(0.02, 0.101, 0.01)
    k123 = np.array([(k1, k2, k3) for i, k1 in enumerate(k_bk) for j, k2 in enumerate(k_bk[i:], start=i)
                     for k3 in k_bk[j:] if k1 + k2 >= k3 - 1e-12]).T
    fake_data_name = 'fake_desi_y6_fisher_pb'
    fake_config_name = fake_data_name + '_config'
    s = DESI_Y6
    fake = Fake(s['n_sky'], s['zmin'], s['zmax'], s['zeff'], s['Veff'], s['degsq'], s['P0'],
                cosmo_fid, likelihood_config=lkl_cfg, fiducial_nuisance=fiducial_nuisance,
                boltzmann='Symbolic', Omega_m_fid=Om_fid, kmin=0.01, kmax=0.2, dk=0.01,
                nbar_prior=s['nbar_prior'], k123=k123, k123_eff=k123, bk_binsize=0.01,
                fake_data_filename=fake_data_name, path_to_data=fake_dir,
                fake_likelihood_config_filename=fake_config_name, path_to_config=fake_dir)
    log(f"Fake initialized: {len(fake.kd)} P(k) bins per multipole, {k123.shape[1]} triangles per sky, "
        f"nbar = {np.array(fake.nbar).round(6).tolist()}")

    def fake_set_from_cosmo_dicts(F, cosmo_dicts, prior_center_on_truth=True):
        """`Fake.set()` with the model evaluated from explicit pybird cosmo_dicts."""
        d = defaultdict(dict)
        for i in range(F.n_sky):
            F.e[i].compute(cosmo_dict=cosmo_dicts[i])
            bpk, bbk = F.e[i].get(deepcopy(F.fiducial_nuisance[i]))
            bpk, bbk = np.asarray(bpk), np.asarray(bbk)
            b1, f1 = F.fiducial_nuisance[i]['Bb1'], float(F.e[i].bird.f)
            ipk_lin = scipy_interp1d(np.asarray(F.e[i].bird.kin), np.asarray(F.e[i].bird.Pin))
            F.io.write_common(d[f'sky_{i+1}'], F.zmin[i], F.zmax[i], F.zeff[i], F.Omega_m_fid, F.H_fid[i], F.D_fid[i])
            cov_pk = get_cov_gauss(F.kd, F.dk, ipk_lin, b1, f1, Vs=F.Veff[i], nbar=F.nbar[i], n_mult=F.c['multipole'])
            F.io.write_pk(d[f'sky_{i+1}'], F.c['multipole'], F.kd, bpk, cov_pk, nsims=-1,
                          survey_mask_arr_p=None, survey_mask_mat_kp=None, binsize=F.dk)
            cov_bk = get_cov_b_PPP(F.k123_eff[i], F.bk_binsize, ipk_lin, b1, f1, Vs=F.Veff[i], nbar=F.nbar[i])
            cov_pkbk = np.zeros((bpk.size, bbk.size))
            F.io.write_bk(d[f'sky_{i+1}'], bbk, k123=F.k123[i], k123_eff=F.k123_eff[i], cov_bk=cov_bk,
                          cov_pkbk=cov_pkbk, bk_multipole=F.c.get('bk_multipole'), nsims=-1)
        with h5py.File(F.path_to_file, 'w') as hf:
            F.save_dict_to_hdf5(hf, d)
        F.c['data_path'], F.c['data_file'] = F.path, F.fake_data_filename
        if prior_center_on_truth:
            for name in F.c['eft_prior']:
                if name not in F.fiducial_nuisance[0]:
                    continue
                F.c['eft_prior'][name]['mean'] = [F.fiducial_nuisance[i][name] for i in range(F.n_sky)]
                if 'range' in F.c['eft_prior'][name]:
                    F.c['eft_prior'][name]['range'] = [F.c['eft_prior'][name]['range'][0]] * F.n_sky
        with open(F.path_to_config, 'w') as file:
            yaml.dump(F.c, file)

    fake_set_from_cosmo_dicts(fake, build_direct_cosmo_dicts(cosmo_fid_vec))
    log(f"Fake P0+P2+B0 data (direct model at fiducial) written to {fake.path_to_file}")

    # ---- the two likelihoods, P ('bPk') and P+B ('bPk,bBk'), on the same file
    cfg_pb = yaml.full_load(open(os.path.join(fake_dir, f'{fake_config_name}.yaml')))
    cfg_pb['get_maxlkl'] = True
    cfg_p = deepcopy(cfg_pb)
    cfg_p['output'] = 'bPk'
    for name in ['Bd1', 'Be5']:   # bispectrum-only EFT parameters (tree level)
        cfg_p['eft_prior'].pop(name)
    L = {'P': Likelihood(cfg_p, verbose=False), 'P+B': Likelihood(cfg_pb, verbose=False)}
    for tag, Li in L.items():
        n_data = [len(Li.y_sky[i]) for i in range(Li.nsky)]
        eft_list = Li.correlator_sky[0].eft_parameters_list
        assert set(Li.c['eft_prior']) == set(eft_list), (tag, set(Li.c['eft_prior']) ^ set(eft_list))
        log(f"[{tag}] data points per sky {n_data} (total {sum(n_data)}); EFT: "
            f"{len(Li.bg_name)} marginalized {Li.bg_name}, non-marginalized {Li.bng_name + ['Bb1']}")

    def make_model_vector(Li):
        """Concatenated, masked theory vector for all skies, as a function of
        [all EFT (n_eft_sky per sky), 80 amplitudes, 25 growth]. No marginalization inside."""
        # pybird-dev lists 'Be1' TWICE for output 'bPk,bBk' (correlator.py appends the stochastic
        # block without a duplicate check): a phantom parameter with an identically zero column
        # and an ambiguous name->index map, so the names are de-duplicated here
        raw = list(Li.correlator_sky[0].eft_parameters_list)
        eft_names_sky = list(dict.fromkeys(raw))
        if len(eft_names_sky) != len(raw):
            dup = [n for n in eft_names_sky if raw.count(n) > 1]
            log(f"  note: de-duplicated pybird's eft_parameters_list ({len(raw)} -> "
                f"{len(eft_names_sky)}); repeated: {dup}")
        n_e = len(eft_names_sky)

        def model_vector(params):
            eft = params[:n_e * num_skies].reshape(num_skies, n_e)
            amps = params[n_e * num_skies:n_e * num_skies + n_knots]
            growth = params[n_e * num_skies + n_knots:]
            dicts = build_cosmo_dicts(amps, growth)
            out = []
            for i in range(num_skies):
                Li.correlator_sky[i].compute(cosmo_dict=dicts[i], cosmo_module=None)
                bias = {nm: eft[i, j] for j, nm in enumerate(eft_names_sky)}
                out.append(Li.correlator_sky[i].get(bias, concatenated=True)[Li.m_sky[i]])
            return jnp.concatenate(out)

        return model_vector, eft_names_sky

    def eft_prior_precision(Li, eft_names_sky):
        """Diagonal EFT prior precision per sky from the likelihood config: 1/range^2 for every
        'gauss' or 'marg_gauss' coefficient, nothing for a 'flat' one (Bb1); the amplitudes and
        the growth parameters carry no prior here."""
        n_e = len(eft_names_sky)
        prec = np.zeros(n_e * num_skies + n_knots + n_growth)
        for j, nm in enumerate(eft_names_sky):
            pr = Li.c['eft_prior'].get(nm, {})
            if pr.get('type') in ('gauss', 'marg_gauss'):
                rng = np.atleast_1d(pr['range']).astype(float)
                for i in range(num_skies):
                    prec[i * n_e + j] = 1.0 / (rng[i] if len(rng) > 1 else rng[0])**2
        return prec

    return SimpleNamespace(**locals())


def _logger(path=None):
    """Timestamped log to stdout and, if given, to `path`."""
    t0 = time.time()
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    fh = open(path, 'w') if path else None

    def log(msg):
        line = f"[{time.time()-t0:7.1f}s] {msg}"
        print(line, flush=True)
        if fh:
            fh.write(line + '\n'); fh.flush()
    return log


# ============================================================================================
# subcommands
# ============================================================================================
def jacobians(out):
    """J = d(model vector)/d(theta), theta = [EFT per sky, 80 amplitudes, 25 growth], for P and
    P+B (forward-mode JVPs in chunks of 16), with the data precision and the EFT prior precision.
    The SVD/whitened analyses work on J itself: J^T P J squares the condition number.

    Plus finite-difference tests of the three exact symmetries (fisher.py): Delta chi^2 along
    each measures how far the DISCRETIZED model (80 knots, emulator re-sampling, AP
    interpolation, fixed-scale IR resummation) is from it. g1 is exact because pybird-dev's AP
    carries the full volume factor, 1/(q_perp^2 q_par) for P(k) and (q_par q_perp^2)^-2 for B(k)."""
    log = _logger(os.path.splitext(out)[0] + '.log')
    m = model(os.path.dirname(out), log)
    jax, jnp = m.jax, m.jnp
    num_skies, n_knots, n_growth, n_z_unique = m.num_skies, m.n_knots, m.n_growth, m.n_z_unique
    growth_fid, template_mpc = np.asarray(m.growth_fid), np.asarray(m.template_mpc)
    cosmo_fid, h_true, z_ref, knots_mpc = m.cosmo_fid, m.h_true, m.z_ref, m.knots_mpc

    def jacobian_chunked(f, x, chunk=16):
        """Forward-mode Jacobian, `chunk` tangent directions at a time (memory-bounded)."""
        y, f_lin = jax.linearize(f, x)
        n = x.shape[0]
        eye = jnp.eye(n, dtype=x.dtype)
        blocks = [np.asarray(jax.vmap(f_lin)(eye[s:s + chunk])).T for s in range(0, n, chunk)]
        return np.asarray(y), np.concatenate(blocks, axis=1)

    def template_at(k_mpc):
        return np.asarray(m.pklin_cpj(k_mpc, cosmo_fid['omega_b'], cosmo_fid['omega_cdm'], h_true,
                                      cosmo_fid['ln10^{10}A_s'], cosmo_fid['n_s'], z_ref))

    def _lnT(lnk):          # exact log-slope of the template at the knots (the g3 direction)
        return jnp.log(m.pklin_cpj(jnp.exp(lnk), cosmo_fid['omega_b'], cosmo_fid['omega_cdm'],
                                   h_true, cosmo_fid['ln10^{10}A_s'], cosmo_fid['n_s'], z_ref))
    dlnT_dlnk = np.asarray(jax.jvp(_lnT, (jnp.log(knots_mpc),), (jnp.ones(n_knots),))[1])

    knots_h = np.asarray(m.knots_h)
    res = dict(knots_h=knots_h, template_mpc=template_mpc, dlnT_dlnk=dlnT_dlnk, h_true=h_true,
               zeff_unique=np.asarray(m.zeff_unique), growth_fid=growth_fid,
               idx_f=np.asarray(m.idx_f), idx_H=np.asarray(m.idx_H), idx_DA=np.asarray(m.idx_DA),
               n_knots=n_knots, n_growth=n_growth, n_z_unique=n_z_unique, num_skies=num_skies,
               sky_to_z_idx=np.asarray(m.sky_to_z_idx))
    # the no-wiggle template (pybird's gaussian log-k filter), for the BAO phase direction
    from pybird.cosmo import get_smooth_wiggle
    _k, _pk, _pnw, _pw = get_smooth_wiggle(knots_h, template_mpc * h_true**3, h=None,
                                           method='gaussian')
    res['template_nw_mpc'] = np.interp(np.log(knots_h), np.log(np.asarray(_k)),
                                       np.asarray(_pnw)) / h_true**3

    models = {}
    for tag, key in [('P', 'p'), ('P+B', 'pb')]:
        Li = m.L[tag]
        t1 = time.time()
        mv, names = m.make_model_vector(Li)
        n_e = len(names)
        x0 = np.concatenate([np.tile([m.fiducial_nuisance[nm] for nm in names], num_skies),
                             np.ones(n_knots), growth_fid])
        y0, J = jacobian_chunked(mv, jnp.array(x0))
        y = np.asarray(Li.y_all)
        rel = np.max(np.abs(y0 - y) / np.maximum(np.abs(y), 1e-30))
        log(f"[{tag}] J: {J.shape[0]} data x {J.shape[1]} params ({n_e}x{num_skies} EFT + "
            f"{n_knots} amps + {n_growth} growth) in {time.time()-t1:.0f}s; "
            f"max relative residual at fiducial {rel:.1e}")
        res.update({f'J_{key}': J, f'P_{key}': np.asarray(Li.p_all),
                    f'prior_{key}': m.eft_prior_precision(Li, names), f'x0_{key}': x0,
                    f'y_{key}': y, f'eft_names_{key}': np.array(names), f'n_e_{key}': n_e,
                    f'nsky_data_{key}': np.array([len(Li.y_sky[i]) for i in range(Li.nsky)])})
        models[tag] = (mv, names, x0)

    # which entries of the P+B vector are P(k): per sky the vector is [P multipoles ; B0]
    n_p, n_pb = res['nsky_data_p'], res['nsky_data_pb']
    isB = np.concatenate([np.r_[np.zeros(n_p[i], bool), np.ones(n_pb[i] - n_p[i], bool)]
                          for i in range(num_skies)])
    res['isB_pb'] = isB
    same = np.allclose(res['y_pb'][~isB], res['y_p'], rtol=0, atol=0)
    log(f"P entries of the P+B data vector identical to the P-only vector: {same}")

    log("--- exact-symmetry tests: Delta chi^2 of the DISCRETIZED model along directions the "
        "continuum model leaves exactly invariant ---")
    for tag, key in [('P', 'p'), ('P+B', 'pb')]:
        mv, names, x0 = models[tag]
        P = res[f'P_{key}']
        n_e = len(names)
        o_a, o_g = n_e * num_skies, n_e * num_skies + n_knots
        ih = o_g + n_growth - 1
        iH = np.array([o_g + i for i in m.idx_H]); iD = np.array([o_g + i for i in m.idx_DA])
        iDr = np.array([o_g + 3 * n_z_unique + i for i in range(n_z_unique)])
        m0 = np.asarray(mv(jnp.array(x0)))

        def dchi2(x):
            d = np.asarray(mv(jnp.array(x))) - m0
            return float(d @ P @ d)

        rows = []
        for eps in (1e-3, 1e-2):
            q = np.exp(eps)
            x = x0.copy(); x[ih] *= q                                   # h_conv alone (reference)
            c_h = dchi2(x)
            x = x0.copy(); x[ih] *= q; x[iD] *= q; x[iH] /= q           # g1 ruler
            c_g1 = dchi2(x)
            x = x0.copy(); x[o_a:o_g] *= q; x[iDr] /= np.sqrt(q)        # g2 amplitude
            c_g2 = dchi2(x)
            x = x0.copy()                                               # g3 template dilation
            x[o_a:o_g] = q**-3 * template_at(knots_mpc / q) / template_mpc
            x[ih] *= q
            c_g3 = dchi2(x)
            x = x0.copy(); x[iD] *= q                                   # D_A alone at every z
            c_da = dchi2(x)
            rows.append((eps, c_h, c_g1, c_g2, c_g3, c_da))
            log(f"  {tag:4s} eps={eps:.0e}:  h_conv alone {c_h:.3e} | g1 ruler {c_g1:.3e} "
                f"(= {c_g1/c_h:.1e} of h_conv alone) | g2 amplitude {c_g2:.1e} | g3 template "
                f"dilation {c_g3:.3e} (= {c_g3/c_h:.1e}) | D_A alone {c_da:.3e}")
        res[f'sym_tests_{key}'] = np.array(rows)
    np.savez(out, **res)
    log(f"saved {out}")


def cosmo_map(out):
    """d(MI parameters)/d(omega_cdm, lnAs, h) in the scaled units of the whitened Jacobian (d ln:
    amplitudes are 1 at the fiducial, growth divided by its fiducial). THE GATE: the direct LCDM
    model vector differentiated through the SAME likelihood call equals J_MI @ M, i.e. fitting
    LCDM directly and imposing LCDM on the MI model are the same operation."""
    m = model(os.path.dirname(out), _logger())
    jax, jnp = m.jax, m.jnp
    theta0, n_knots = m.cosmo_fid_vec, int(m.n_knots)
    growth_fid = np.asarray(m.growth_fid)
    obs0 = np.asarray(m.cosmo_to_observables(theta0))
    Jm = np.asarray(jax.jacfwd(m.cosmo_to_observables)(theta0))        # (n_knots + n_growth, 3)
    M = Jm / np.concatenate([np.ones(n_knots), growth_fid])[:, None]
    print(f"amps at the fiducial: max|a-1| = {np.abs(obs0[:n_knots] - 1).max():.2e}")
    print(f"growth at the fiducial matches growth_fid: max rel dev "
          f"{np.abs(obs0[n_knots:] / growth_fid - 1).max():.2e}")

    R = np.load(os.path.join(OUT, 'jacobians_pb.npz'), allow_pickle=True)
    gate = {}
    for tag, key in (('P', 'p'), ('P+B', 'pb')):
        mv, names = m.make_model_vector(m.L[tag])
        n_e = len(names)
        eft0 = jnp.array(np.tile([m.fiducial_nuisance[nm] for nm in names], m.num_skies))

        def m_direct(th, mv=mv, eft0=eft0):
            amps = m.cosmo_to_amps(th)
            growth = m.cosmo_to_growth(jnp.array([th[0], th[2]]))
            return mv(jnp.concatenate([eft0, amps, growth]))

        J_dir = np.asarray(jax.jacfwd(m_direct)(theta0))                 # (n_data, 3)
        J_comp = R[f'J_{key}'][:, n_e * m.num_skies:] @ Jm               # chain rule
        gate[key] = np.linalg.norm(J_dir - J_comp) / np.linalg.norm(J_dir)
        print(f"[GATE] {tag:4s} ||J_direct - J_MI M|| / ||J_direct|| = {gate[key]:.2e}   "
              f"({'PASS' if gate[key] < 1e-4 else 'CHECK'})")
    np.savez(out, M=M, M_raw=Jm, theta_fid=np.asarray(theta0), obs_fid=obs0,
             names=np.array(['omega_cdm', 'lnAs', 'h']), gate_p=gate['p'], gate_pb=gate['pb'])
    print(f"saved {out}")


def robust(out):
    """BAO stage A: square-root (SVD) marginals of the 80-knot model, symmetries projected.

    Each case whitens W = [L^T J; sqrt(EFT prior)] (+ the smoothness rows), projects out the exact
    symmetries that survive in its reduced parameter space (W -> (I - Q Q^T) W: the Schur
    complement for a free nuisance along g, it can only remove information) and reads
    var(c) = sum_i (v_i . c)^2 / s_i^2 over singular values > 1e-12 s_0. The SVD carries the square
    root of the condition number of J^T P J, so it resolves what the normal matrix cannot. A
    functional with a component > 1e-7 along the null space moves along an exact symmetry: nan.
      template  fixed (at the fiducial) | free (80 knots, no prior) | smooth (lambda 200 on second
                differences + 0.5 per knot; locks the wiggles, so g3 is not projected)
      ruler     free (h_conv free) | known (h_conv fixed, i.e. calibrated BAO)
    D_M/r_d, D_H/r_d use the template's own ruler rho = w . d ln a, w the wiggle derivative dO/dlnk
    with cubics in ln k projected out over 0.02-0.30 h/Mpc, normalized to w . gamma = 1.
    The notebook reads the free-template, free-ruler distances and the wiggle S/N; the rest of
    the file is kept so that a rebuild reproduces it."""
    import fisher as F
    nk, ng, nz, zz, knots_h, gamma = F.nk, F.ng, F.nz, F.zz, F.knots_h, F.gamma
    iF, iH, iD, iDr, ihc = F.iF, F.iH, F.iD, F.iDr, F.ihc
    Z_REF = int(np.argmin(np.abs(zz - 0.93)))       # reference bin for relative distances
    TOL, NULL_TOL = 1e-12, 1e-13

    lk = np.log(knots_h)
    dO = np.gradient(F.wiggle_O, lk)
    sel = (knots_h >= 0.02) & (knots_h <= 0.30)
    Xp = np.vander(lk[sel] - lk[sel].mean(), 4)
    w0 = dO[sel] - Xp @ np.linalg.lstsq(Xp, dO[sel], rcond=None)[0]
    w = np.zeros(nk); w[sel] = w0
    w_ruler = w / (w @ gamma)

    def system(key):
        wk = F.whiten(key)
        Wd, o_a, o_g = wk['Wd'], wk['o_a'], wk['o_gk']
        npar = o_g + ng
        pr = np.where(wk['has'])[0]
        Wp = np.zeros((len(pr), npar)); Wp[np.arange(len(pr)), pr] = 1.0   # sqrt(prior)*s = 1
        G = np.zeros((npar, 3))
        G[o_g + ihc, 0] = 1.0; G[o_g + iD, 0] = 1.0; G[o_g + iH, 0] = -1.0     # g1 ruler
        G[o_a:o_g, 1] = 1.0; G[o_g + iDr, 1] = -0.5                            # g2 amplitude
        G[o_g + ihc, 2] = 1.0; G[o_a:o_g, 2] = -gamma                          # g3 dilation
        D2 = np.zeros((nk - 2, nk))
        for i in range(nk - 2):
            D2[i, i:i + 3] = (1.0, -2.0, 1.0)
        Ra = np.vstack([np.sqrt(200.0) * D2, np.eye(nk) / 0.5])
        Rm = np.zeros((Ra.shape[0], npar)); Rm[:, o_a:o_a + nk] = Ra          # smoothness rows
        return dict(W=np.vstack([Wd, Wp]), Rm=Rm, o_a=o_a, o_g=o_g, npar=npar, G=G)

    def case(B, template, ruler_known, growth_known=False):
        npar, o_a, o_g = B['npar'], B['o_a'], B['o_g']
        keep = np.ones(npar, bool)
        if template == 'fixed':
            keep[o_a:o_g] = False
        if ruler_known:
            keep[o_g + ihc] = False
        if growth_known:
            keep[o_g:] = False
        W = B['W'] if template != 'smooth' else np.vstack([B['W'], B['Rm']])
        Gs = B['G'][:, [0, 1, 2] if template != 'smooth' else [0, 1]]
        drop = ~keep
        if drop.any():               # symmetries that survive with the dropped parameters fixed
            _, sv, vt = np.linalg.svd(Gs[drop], full_matrices=True)
            rank = int((sv > 1e-10 * max(sv.max(), 1.0)).sum()) if sv.size else 0
            Bc = vt[rank:].T
        else:
            Bc = np.eye(Gs.shape[1])
        Gk = (Gs @ Bc)[keep] if Bc.shape[1] else np.zeros((int(keep.sum()), 0))
        Wk = W[:, keep]
        if Gk.shape[1]:
            Q, Rq = np.linalg.qr(Wk @ Gk)
            ok = np.abs(np.diag(Rq)) > 1e-14 * np.linalg.norm(Wk)
            Q = Q[:, ok]
            Wk = Wk - Q @ (Q.T @ Wk)
        _, S, Vt = np.linalg.svd(Wk, full_matrices=False)
        return dict(keep=keep, S=S, Vt=Vt)

    def marginal(sy, c_full):
        """(sigma of the functional c, its component along the null space)."""
        c = c_full[sy['keep']]
        S, Vt = sy['S'], sy['Vt']
        nrm = max(np.linalg.norm(c), 1e-300)
        null = S <= NULL_TOL * S[0]
        ninv = float(np.linalg.norm(Vt[null] @ c) / nrm) if null.any() else 0.0
        a = Vt @ c
        g = S > TOL * S[0]
        return float(np.sqrt(np.sum((a[g] / S[g])**2))), ninv

    def geo_c(B, name, iz):
        c = np.zeros(B['npar']); o = B['o_g']
        if name == 'f':
            c[o + iF[iz]] = 1.0
        elif name == 'F_AP':                      # ln(H D_A)
            c[o + iH[iz]] = 1.0; c[o + iD[iz]] = 1.0
        elif name == 'alpha_iso':                 # ln D_V up to a constant: (2 ln D_A - ln H)/3
            c[o + iD[iz]] = 2 / 3; c[o + iH[iz]] = -1 / 3
        elif name == 'alpha_iso_rel':             # relative to the reference bin
            c[o + iD[iz]] += 2 / 3; c[o + iH[iz]] += -1 / 3
            c[o + iD[Z_REF]] -= 2 / 3; c[o + iH[Z_REF]] -= -1 / 3
        elif name in ('alpha_perp', 'alpha_par'):  # D_M/r_d, D_H/r_d with the template's own ruler
            if name == 'alpha_perp':
                c[o + iD[iz]] = 1.0
            else:
                c[o + iH[iz]] = -1.0
            c[o + ihc] = -1.0
            c[B['o_a']:B['o_a'] + nk] = -w_ruler
        elif name == 'lnDA_rel':                  # D_A(z) / D_A(z_ref)
            c[o + iD[iz]] += 1.0; c[o + iD[Z_REF]] -= 1.0
        elif name == 'lnH_rel':                   # H(z) / H(z_ref)
            c[o + iH[iz]] += 1.0; c[o + iH[Z_REF]] -= 1.0
        elif name == 'lnDA':
            c[o + iD[iz]] = 1.0
        elif name == 'lnH':
            c[o + iH[iz]] = 1.0
        return c

    def band_c(B, j, conditional):
        """Band power at knot j: ln a_j if growth, geometry and h_conv are known; otherwise the
        gauge-invariant linear power at z_ref in the observed coordinates of that bin,
        ln a_j + 2 ln D(z_ref) + gamma_j [ln h_conv - alpha_iso(z_ref)]."""
        c = np.zeros(B['npar']); o_a, o_g = B['o_a'], B['o_g']
        c[o_a + j] = 1.0
        if not conditional:
            g = gamma[j]
            c[o_g + iDr[Z_REF]] += 2.0
            c[o_g + ihc] += g
            c[o_g + iD[Z_REF]] += -g * 2 / 3
            c[o_g + iH[Z_REF]] += g / 3
        return c

    BUILT = {key: system(key) for key in ('p', 'pb')}
    wig = F.wiggle_O
    isel = np.where(sel)[0]
    bins = [list(isel[i:i + 3]) for i in range(0, len(isel) - 2, 3)]
    res = dict(zeff=zz, z_ref=zz[Z_REF], knots_h=knots_h, wiggle=wig,
               kc=np.array([np.exp(np.mean(np.log(knots_h[bj]))) for bj in bins]),
               wig_bin=np.array([np.mean(wig[bj]) for bj in bins]))
    CASES = [('fixed', False), ('fixed', True), ('free', False), ('free', True),
             ('smooth', False), ('smooth', True)]
    QTY = ['alpha_perp', 'alpha_par', 'F_AP', 'alpha_iso_rel', 'lnDA_rel', 'lnH_rel', 'f',
           'alpha_iso', 'lnDA', 'lnH']
    for template, ruler_known in CASES:
        for key in ('p', 'pb'):
            B = BUILT[key]
            sy = case(B, template, ruler_known)
            for q in QTY:
                vals = []
                for iz in range(nz):
                    if q.endswith('_rel') and iz == Z_REF:
                        vals.append(0.0); continue
                    s_, ninv = marginal(sy, geo_c(B, q, iz))
                    vals.append(np.nan if ninv > 1e-7 else s_)
                res[f"geo_{template}_{'known' if ruler_known else 'free'}_{key}_{q}"] = np.array(vals)

    # band powers per knot, the 3-knot bins, and the BAO detection S/N = sqrt(w^T C^+ w): C the
    # marginal covariance of the band powers over 0.02-0.30 h/Mpc, w the fiducial wiggles there
    bands, binned, snr = {}, {}, {}
    for template in ('free', 'smooth'):
        for key in ('p', 'pb'):
            B = BUILT[key]
            for cond in (True, False):
                tag = f"{template}_{key}_{'cond' if cond else 'marg'}"
                sy = case(B, template, ruler_known=False, growth_known=cond)
                bands[f'band_{tag}'] = np.array([marginal(sy, band_c(B, j, cond))[0]
                                                 for j in range(nk)])
                if template == 'free' and not cond:
                    vals = []
                    for bj in bins:
                        s_, n_ = marginal(sy, np.mean([band_c(B, j, False) for j in bj], axis=0))
                        vals.append(s_ if n_ < 1e-7 else np.nan)
                    binned[f'binned_free_{key}'] = np.array(vals)
                Phi = np.array([band_c(B, j, cond)[sy['keep']] for j in isel])
                g = sy['S'] > TOL * sy['S'][0]
                Mx = (Phi @ sy['Vt'][g].T) / sy['S'][g]
                Um, sm, _ = np.linalg.svd(Mx, full_matrices=False)
                vals = []
                for t in (1e-6, 1e-8, 1e-10):
                    gg = sm > t * sm[0]
                    vals.append(np.sqrt(np.sum(((Um[:, gg].T @ wig[isel]) / sm[gg])**2)))
                snr[f'snr_{tag}'] = np.array(vals)
    res.update(bands); res.update(binned); res.update(snr)
    iz = 2
    print(f"free template, ruler free, z = {zz[iz]:.3f}:  sigma(F_AP) "
          f"{res['geo_free_free_p_F_AP'][iz]:.4f} (P)  {res['geo_free_free_pb_F_AP'][iz]:.4f} (P+B);"
          f"  wiggle S/N {snr['snr_free_p_marg'][0]:.1f} (P)  {snr['snr_free_pb_marg'][0]:.1f} (P+B)")
    np.savez(out, **res)
    print(f"saved {out}: {len(res)} arrays")


def se07(out):
    """Standard BAO-only forecast (Seo & Eisenstein 2007, eq. 26, the DESI design formula) for
    the SAME mock bins: volumes, nbar (from jacobians_pb.log), b1 = 1.8, f, linear P(k).
      F_ij = V A0^2 int dk k^2 exp[-2 (k Sigma_s)^1.4] int_0^1 dmu f_i f_j
             exp[-k^2 Sigma_perp^2 (1-mu^2) - k^2 Sigma_par^2 mu^2] / [P/P_0.2 + 1/(n P_0.2(mu))]^2
      f_1 = mu^2 - 1 (ln D_A/r_d), f_2 = mu^2 (ln H r_d), A0 = 0.4529, Sigma_s = 7.76 Mpc/h,
      Sigma_perp = 9.4 (sigma_8(z)/0.9) Mpc/h, Sigma_par = (1+f) Sigma_perp; post-recon x 0.5."""
    import fisher as F
    trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    kh, T, h = F.knots_h, F.T_, F.h_fid
    zz, gf, iF, iDr, sky_z = F.zz, F.gfid, F.iF, F.iDr, F.R['sky_to_z_idx']
    Veff = np.array([4., 8., 12., 15., 8., 12., 4.]) * 1e9                  # (Mpc/h)^3, as the mock
    nbar = np.array([float(x) for x in re.search(r"nbar = \[([^\]]+)\]",
                     open(os.path.join(OUT, 'jacobians_pb.log')).read()).group(1).split(',')])
    b1 = 1.8
    A0, SIGS = 0.4529, 7.76
    Plin0 = lambda k: np.exp(np.interp(np.log(k * h), np.log(kh * h), np.log(T))) * h**3  # z_ref

    def sigma8(Dr):
        k = np.geomspace(kh[0], kh[-1], 4000); x = 8.0 * k
        W = 3 * (np.sin(x) - x * np.cos(x)) / x**3
        return np.sqrt(trapz(k**2 * Plin0(k) * Dr**2 * W**2, k) / (2 * np.pi**2))

    def fisher_bao(V, n, Dr, f, recon, kmax):
        s8 = sigma8(Dr); Sp = 9.4 * s8 / 0.9 * (0.5 if recon else 1.0); Sl = (1 + f) * Sp
        k = np.linspace(0.01, kmax, 600); mu = np.linspace(0, 1, 201)
        K, M = np.meshgrid(k, mu, indexing='ij')
        P02 = Plin0(0.2) * Dr**2
        denom = (Plin0(K) / Plin0(0.2) + 1.0 / (n * (b1 + f * M**2)**2 * P02))**2
        damp = np.exp(-2 * (K * SIGS)**1.4 - K**2 * Sp**2 * (1 - M**2) - K**2 * Sl**2 * M**2)
        fs = [M**2 - 1, M**2]
        Fm = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                Fm[i, j] = V * A0**2 * trapz(trapz(K**2 * damp * fs[i] * fs[j] / denom, mu, axis=1), k)
        return Fm, s8

    print(f"mock: b1={b1}, nbar per sky = {np.round(nbar, 6).tolist()}")
    print(f"{'z':>6s} {'sig8':>6s} | {'pre-recon D_M/r_d  D_H/r_d':>27s} | {'post-recon D_M/r_d  D_H/r_d':>28s}")
    rows = {'pre_perp': [], 'pre_par': [], 'post_perp': [], 'post_par': []}
    for iz, z in enumerate(zz):
        Dr, f = gf[iDr[iz]], gf[iF[iz]]
        sg = {}
        for recon in (False, True):
            Fm = np.zeros((2, 2))
            for s in np.where(sky_z == iz)[0]:
                Fi, s8 = fisher_bao(Veff[s], nbar[s], Dr, f, recon, 0.5); Fm += Fi
            sg[recon] = np.sqrt(np.diag(np.linalg.inv(Fm)))
        for recon, tag in ((False, 'pre'), (True, 'post')):
            rows[f'{tag}_perp'].append(sg[recon][0]); rows[f'{tag}_par'].append(sg[recon][1])
        print(f"{z:6.3f} {s8:6.3f} | {100*sg[False][0]:17.2f}% {100*sg[False][1]:8.2f}% | "
              f"{100*sg[True][0]:18.2f}% {100*sg[True][1]:8.2f}%")
    np.savez(out, zeff=zz, **{k: np.array(v) for k, v in rows.items()})
    print(f"saved {out}")


# ============================================================================================
# the notebook
# ============================================================================================
def _md(t):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': t.strip('\n')}


def _code(t):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
            'source': t.strip('\n')}


SETUP = r"""
import os, sys
sys.path.insert(0, os.path.abspath('../mi/pk_bk'))
if not os.path.exists('../output/fisher_pb/jacobians_pb.npz'):
    %run ../mi/pk_bk/rebuild.py jacobians     # ~5-10 min, pybird-dev
import fisher
fisher.data_summary()
"""

CELLS = [
    _md(r"""
# P(k) vs P(k)+B(k): model-independent Fisher forecast

DESI-Y6-like noiseless mock; legacy MI template ($\ln a$ on 60 nodes over the 80 emulator knots); tree-level B0 from pybird-dev.
"""),
    _md(r"""
## Data

7 skies, 6 redshifts; P0, P2, P4 at $0.01<k<0.2\,h$/Mpc, B0 on triangles with $0.02\le k_i\le0.10\,h$/Mpc; Gaussian covariance, no P–B cross term.
"""),
    _code(SETUP),
    _md(r"""
## Likelihood

pybird-dev, `eth` basis: one-loop P(k) from the emulator, tree-level B0; per-sky EFT parameters (sky 1 shown).
"""),
    _code("fisher.likelihood_summary()"),
    _md(r"""
## Fisher

Whitened Jacobian on the 60 nodes with the HMC priors ($\sigma_{\ln a}$ 0.5, $\lambda$ 200, $\sigma_{\ln g}$ 0.3, $\sigma_{\ln h}$ 0.05); data response along $g_1$ (ruler) and $g_2$ (amplitude) projected out.
"""),
    _code("fisher.growth()"),
    _md(r"""
## Linear P(k)

Everything else marginalized; solid: wider priors ($\sigma_{\ln a}$ 2, $\lambda$ 50), dotted: HMC priors; grey bands: P and B data ranges.
"""),
    _code("fisher.show('fig_bands_pb.png', 880)"),
    _md(r"""
## Growth

$(f, H, D_A)$ at $z=0.71$, and $\sigma(f)$, $\sigma(F_{\rm AP})$ vs $z$.
"""),
    _code("fisher.show('fig_growth_corner.png', 720)\nfisher.show('fig_growth_vs_z.png', 980)"),
    _md(r"""
## ΛCDM recovery

Dashed: direct ΛCDM fit; solid: the same data through the MI parametrization, MI priors off.
"""),
    _code("fisher.show('fig_cosmo_pb.png', 760)"),
    _md(r"""
## Priors

Each MI prior block on and off, then a scan over $\sigma_{\ln a}$ and $\lambda$.
"""),
    _code("fisher.prior_audit()"),
    _code("fisher.show('fig_prior_decomposition.png', 1150)"),
    _md(r"""
## BAO A: free template

80 free knots, no template prior.
"""),
    _code("fisher.stage_a()"),
    _md(r"""
## BAO B: smoothness prior

The sampled model (60 nodes, $\lambda=200$); $v=d\ln(1+O)/d\ln k$ is the pattern that slides the wiggles.
"""),
    _code("fisher.bao()"),
    _code("fisher.show('fig_phase_prior_explained.png', 860)"),
    _md(r"""
## BAO C: phase prior

Gaussian prior $\sigma_\delta$ on the slide $\delta(k)\,v(k)$, $\delta=\sum_{j=0}^{5}c_j P_j(\ln k)$; broadband and wiggle amplitude free.
"""),
    _code("import inspect\n"
          "print(inspect.getsource(fisher.phase_modes))\n"
          "print(inspect.getsource(fisher.phase_precision))"),
    _code("fisher.show('fig_phase_scan.png', 1150)"),
    _md(r"""
## P vs P+B per redshift

Solid: no phase prior; dashed: $\sigma_\delta=0.2\%$; grey: standard BAO (Seo & Eisenstein 2007), pre/post-reconstruction.
"""),
    _code("fisher.show('fig_bao_per_z.png', 1150)"),
    _md(r"""
## All redshifts

$D_M/r_d$ and $D_H/r_d$ with $\sigma_\delta=0.2\%$, everything else marginalized.
"""),
    _code("fisher.show('fig_bao_triangle_all_z.png', 1250)"),
    _md(r"""
## z = 0.71

Dashed: no phase prior; solid: $\sigma_\delta=0.2\%$.
"""),
    _code("fisher.show('fig_bao_corner.png', 760)"),
]


def notebook(execute):
    import json
    for i, c in enumerate(CELLS):
        c['id'] = f'c{i:02d}'
    nb = {'cells': CELLS,
          'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python',
                                      'name': 'python3'},
                       'language_info': {'name': 'python'}},
          'nbformat': 4, 'nbformat_minor': 5}
    with open(NB, 'w') as fh:
        json.dump(nb, fh, indent=1)
    print(f"wrote {NB}: {len(CELLS)} cells")
    if not execute:
        return
    import nbformat
    from nbclient import NotebookClient
    nb = nbformat.read(NB, as_version=4)
    ok = True
    try:                             # cwd notebooks/, as when the notebook is opened there
        NotebookClient(nb, timeout=1800, kernel_name='python3',
                       resources={'metadata': {'path': os.path.dirname(NB)}}).execute()
    except Exception as e:                                          # keep the partial record
        ok = False
        print(f'FAILED: {type(e).__name__}: {str(e)[:2000]}')
    finally:
        nbformat.write(nb, NB)
    nerr = sum(1 for c in nb.cells for o in c.get('outputs', []) if o.get('output_type') == 'error')
    print(f'EXECUTED {NB}: {len(nb.cells)} cells, {nerr} errors')
    if not ok or nerr:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split('\n', 1)[1])
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name, fn, cache, hlp in (
            ('jacobians', jacobians, 'jacobians_pb.npz', 'model Jacobians + symmetry tests, 5-10 min'),
            ('cosmo_map', cosmo_map, 'cosmo_map_pb.npz', 'LCDM map + gate, ~3 min'),
            ('robust', robust, 'robust_pb_results.npz', 'BAO stage A (free template), ~1 min'),
            ('se07', se07, 'se07_forecast.npz', 'standard-BAO reference, seconds')):
        p = sub.add_parser(name, help=hlp, description=inspect.cleandoc(fn.__doc__),
                           formatter_class=argparse.RawDescriptionHelpFormatter)
        p.add_argument('--out', default=os.path.join(OUT, cache), help='default: %(default)s')
        p.set_defaults(fn=lambda a, fn=fn: fn(os.path.abspath(a.out)))
    p = sub.add_parser('notebook', help='write ../../notebooks/5_fisher_pk_bk.ipynb [and execute it in place]')
    p.add_argument('--execute', action='store_true')
    p.set_defaults(fn=lambda a: notebook(a.execute))
    a = ap.parse_args()
    a.fn(a)


if __name__ == '__main__':
    main()
