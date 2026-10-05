"""Standard BAO-only projection (Seo & Eisenstein 2007, eq. 26) for the SAME mock bins as the
P+B Fisher: same volumes, number densities, bias, growth rate and linear spectrum. This is the
formula DESI's design forecasts used (with 50% reconstruction). Compared against the full-shape
MI numbers in bao_summary_pb.npz.

  F_ij = V A0^2 int dk k^2 exp[-2 (k Sigma_s)^1.4] int_0^1 dmu f_i f_j
         exp[-k^2 Sigma_perp^2 (1-mu^2) - k^2 Sigma_par^2 mu^2] / [P(k)/P_0.2 + 1/(n P_0.2(mu))]^2
  f_1 = mu^2 - 1 (ln D_A/r_d), f_2 = mu^2 (ln H r_d), A0 = 0.4529, Sigma_s = 7.76 Mpc/h,
  Sigma_perp = 9.4 (sigma_8(z)/0.9) Mpc/h, Sigma_par = (1+f) Sigma_perp; post-recon x 0.5.
"""
import os, re, numpy as np
OUT = '../../output/fisher_pb'
R = np.load('../../output/fisher_pb/jacobians_pb.npz', allow_pickle=True)
S = np.load('../../output/fisher_pb/bao_summary_pb.npz')
kh, T, h = R['knots_h'], R['template_mpc'], float(R['h_true'])
zz, gf = R['zeff_unique'], R['growth_fid']; nz = int(R['n_z_unique'])
iF = R['idx_f']; iDr = 3 * nz + np.arange(nz); sky_z = R['sky_to_z_idx']
Veff = np.array([4., 8., 12., 15., 8., 12., 4.]) * 1e9                  # (Mpc/h)^3, as the mock
nbar = np.array([float(x) for x in re.search(r"nbar = \[([^\]]+)\]",
                 open('../../output/fisher_pb/jacobians_pb.log').read()).group(1).split(',')])
b1 = 1.8
A0, SIGS = 0.4529, 7.76
Plin0 = lambda k: np.exp(np.interp(np.log(k * h), np.log(kh * h), np.log(T))) * h**3  # z_ref

def sigma8(Dr):
    k = np.geomspace(kh[0], kh[-1], 4000); x = 8.0 * k
    W = 3 * (np.sin(x) - x * np.cos(x)) / x**3
    return np.sqrt(np.trapz(k**2 * Plin0(k) * Dr**2 * W**2, k) / (2 * np.pi**2))

def fisher(V, n, Dr, f, recon, kmax):
    s8 = sigma8(Dr); Sp = 9.4 * s8 / 0.9 * (0.5 if recon else 1.0); Sl = (1 + f) * Sp
    k = np.linspace(0.01, kmax, 600); mu = np.linspace(0, 1, 201)
    K, M = np.meshgrid(k, mu, indexing='ij')
    P02 = Plin0(0.2) * Dr**2
    denom = (Plin0(K) / Plin0(0.2) + 1.0 / (n * (b1 + f * M**2)**2 * P02))**2
    damp = np.exp(-2 * (K * SIGS)**1.4 - K**2 * Sp**2 * (1 - M**2) - K**2 * Sl**2 * M**2)
    fs = [M**2 - 1, M**2]
    F = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            F[i, j] = V * A0**2 * np.trapz(np.trapz(K**2 * damp * fs[i] * fs[j] / denom, mu, axis=1), k)
    return F, s8

print(f"mock: b1={b1}, nbar per sky = {np.round(nbar, 6).tolist()}")
print(f"{'z':>6s} {'sig8':>6s} | {'BAO-only pre-recon':>21s} | {'BAO-only post-recon':>21s} | "
      f"{'MI full-shape P (60 nodes)':>27s} | {'MI P, rigid (16 nodes)':>23s} | {'MI P+B, rigid':>15s}")
print(f"{'':6s} {'':6s} | {'D_M/r_d   D_H/r_d':>21s} | {'D_M/r_d   D_H/r_d':>21s} | {'D_M/r_d   D_H/r_d':>27s} | "
      f"{'D_M/r_d  D_H/r_d':>23s} | {'D_M    D_H':>15s}")
rows = {'pre_perp': [], 'pre_par': [], 'post_perp': [], 'post_par': []}
for iz, z in enumerate(zz):
    Dr, f = gf[iDr[iz]], gf[iF[iz]]
    out = {}
    for recon in (False, True):
        F = np.zeros((2, 2))
        for s in np.where(sky_z == iz)[0]:
            Fi, s8 = fisher(Veff[s], nbar[s], Dr, f, recon, 0.5); F += Fi
        C = np.linalg.inv(F); out[recon] = np.sqrt(np.diag(C))
    for pre, tag in ((False, 'pre'), (True, 'post')):
        rows[f'{tag}_perp'].append(out[pre][0]); rows[f'{tag}_par'].append(out[pre][1])
    mi = lambda sp, k, q: S[f"s{sp:.2f}_{k}_{q}"][iz]
    print(f"{z:6.3f} {s8:6.3f} | {100*out[False][0]:8.2f}% {100*out[False][1]:8.2f}% | "
          f"{100*out[True][0]:8.2f}% {100*out[True][1]:8.2f}% | "
          f"{100*mi(0.15,'p','perp'):10.2f}% {100*mi(0.15,'p','par'):10.2f}% | "
          f"{100*mi(0.60,'p','perp'):9.2f}% {100*mi(0.60,'p','par'):8.2f}% | "
          f"{100*mi(0.60,'pb','perp'):5.2f}% {100*mi(0.60,'pb','par'):5.2f}%")

# saved so the notebooks can draw the standard-BAO reference without re-running this file
np.savez(os.path.join(OUT, 'se07_forecast.npz'), zeff=zz,
         **{k: np.array(v) for k, v in rows.items()})
print(f"\nsaved {os.path.join(OUT, 'se07_forecast.npz')}")
