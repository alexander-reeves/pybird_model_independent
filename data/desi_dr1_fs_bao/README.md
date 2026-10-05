# DESI DR1 full shape + post-reconstruction BAO (pybird format)

`desi_dr1_kp_fs_bao.h5` holds P0 and P2 of BGS, LRG1, LRG2, LRG3, ELG2 and QSO, over 0.02 < k < 0.20 h/Mpc. It also
holds the window matrices and the 10 post-reconstruction BAO dilations, with one joint FS+BAO covariance.

- **Source:** the setup of DESI's own FS+BAO cosmology analysis (DESI 2024 V / VII). The file was built by
  `pybird-milan/demo/run_desi_fs_bao.ipynb` from DESI's public DR1 likelihood files,
  `likelihood_spectrum-poles-rotated+bao-recon_syst-hod_<tracer>_…_thetacut0.05.h5` (full-shape-bao-clustering
  release v1.0).
- **BAO block:** the post-reconstruction correlation-function fits of DESI 2024 III, with systematics added. Its
  values therefore differ slightly from the BAO paper's tables of statistical-only posterior means.

`likelihood_config/desi_dr1_fs_bao.yaml` is DESI's key-project configuration (`pybird-milan/data/desi_dr1/
likelihood_config_v1/ALL.yaml`) with two edits: `with_emu: True` and an absolute `data_path`. `notebooks/desi_fsbao/
settings.py` overrides that path with this folder.
