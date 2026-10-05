"""Which prior does the work? A block-by-block decomposition, for LCDM and for the MI results.

The model-independent model carries four prior blocks (setups.COMMON, and mi_model.prior_prec_mat):

  s_lna  = 0.5    a Gaussian of this width on ln a at EACH of the 60 nodes
  lambda = 200    a penalty lambda * ||D2 ln a||^2 on second differences of ln a between nodes
  s_lng  = 0.3    a Gaussian of this width on ln of every growth quantity (f, H, D_A, D-ratios)
  s_lnh  = 0.05   a Gaussian of this width on ln h_conv

None of them is meant to carry information: s_lna and lambda exist to keep the sampler (and the
loop emulator, which misbehaves on jagged ln a) in a sane region, s_lng to keep the growth
parameters positive and O(1), s_lnh because h_conv is exactly unconstrained by the data.

This script measures what they actually do:
  1. the induced prior on LCDM, block by block, with no data at all;
  2. sigma(omega_cdm, lnAs, h) with each block ON alone and with each block OFF alone;
  3. the same for the quantities the MI analysis reports (the P(k) band, f, D_M/r_d);
  4. a scan over s_lna and lambda, with the roughness each setting still allows, so a wider
     prior can be chosen without giving up the smoothness the emulator needs.

Runs in ~1 min from the cached jacobians_pb.npz.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import mi_prior_fisher as M

OUT = M.OUT
FIG = os.path.join(OUT, 'meeting')
os.makedirs(FIG, exist_ok=True)
C_P, C_PB, C_INK, C_MUTED = '#E8710A', '#1E88E5', '#1A1A1A', '#6E6E6E'
zz, nz, IZ = M.zz, M.nz, 2
PAR = ['omega_cdm', 'ln(1e10 As)', 'h']
BIG = 1e6                     # "prior off": a width so wide it cannot constrain anything

import cosmo_fisher_pb as CF
Sb, Db, n_amp, Mnode = CF.Sb, CF.Db, CF.n_amp, CF.Mnode


def prior_blocks(s_lna=0.5, lam=200.0, s_lng=0.3, s_lnh=0.05):
    """The four prior precision blocks, separately, on [ln a nodes, ln growth]."""
    n = n_amp + M.ng
    B = {}
    for name in ('s_lna', 'lambda', 's_lng', 's_lnh'):
        P = np.zeros((n, n))
        if name == 's_lna':
            P[:n_amp, :n_amp] = np.eye(n_amp) / s_lna**2
        elif name == 'lambda':
            P[:n_amp, :n_amp] = lam * (Db.T @ Db)
        elif name == 's_lng':
            for j in range(4 * nz):
                P[n_amp + j, n_amp + j] = 1.0 / s_lng**2
        else:
            P[n_amp + M.ihc, n_amp + M.ihc] = 1.0 / s_lnh**2
        B[name] = P
    return B


def lcdm_data_fisher(key='p'):
    """The 3x3 LCDM Fisher from the DATA alone (EFT marginalized, MI priors off)."""
    return CF.cosmo_fisher(key, route='through_mi_flat')


def sig3(F):
    return np.sqrt(np.diag(np.linalg.inv(F)))


def row(label, s):
    return f"  {label:34s}" + "".join(f"{v:13.4f}" for v in s)


print(__doc__.split('This script measures')[0])
print("=" * 86)
print("1. THE INDUCED PRIOR ON LCDM, NO DATA AT ALL")
print(f"{'':36s}" + "".join(f"{p:>13s}" for p in PAR))
B = prior_blocks()
tot = sum(B.values())
for name, P in list(B.items()) + [('all four together', tot)]:
    Pi = Mnode.T @ P @ Mnode
    ev = np.linalg.eigvalsh(Pi)
    s = sig3(Pi) if ev.min() > 1e-12 * max(ev.max(), 1) else np.full(3, np.inf)
    print(row(name, s))

F_data = {k: lcdm_data_fisher(k) for k in ('p', 'pb')}
print("\n  for scale, the DATA alone:")
for k, lab in (('p', 'P(k)'), ('pb', 'P(k)+B(k)')):
    print(row(lab, sig3(F_data[k])))

print("\n" + "=" * 86)
print("2. LCDM WITH EACH BLOCK ON ALONE, AND WITH EACH BLOCK OFF ALONE   [P(k)+B(k)]")
print(f"{'':36s}" + "".join(f"{p:>13s}" for p in PAR))
base = sig3(F_data['pb'])
print(row('data only (no MI priors)', base))
for name in B:
    print(row(f'+ {name} alone', sig3(F_data['pb'] + Mnode.T @ B[name] @ Mnode)))
print(row('+ all four (the model as sampled)', sig3(F_data['pb'] + Mnode.T @ tot @ Mnode)))
for name in B:
    off = sum(P for n_, P in B.items() if n_ != name)
    print(row(f'all but {name}', sig3(F_data['pb'] + Mnode.T @ off @ Mnode)))
print("\n  ratio to data-only, all four on: " +
      "  ".join(f"{p}={v:.2f}" for p, v in
               zip(PAR, sig3(F_data['pb'] + Mnode.T @ tot @ Mnode) / base)))

print("\n" + "=" * 86)
print("3. THE SAME BLOCKS, ON THE QUANTITIES THE MI ANALYSIS REPORTS   [P(k)+B(k)]")
inb = (M.knots_h >= 0.01) & (M.knots_h <= 0.2)


def mi_numbers(priors, phase_sigma=None):
    sy = M.build('pb', priors=priors, project_data=True, project=(0, 1), phase_sigma=phase_sigma)
    Ca = sy['C'][sy['o_a']:sy['o_a'] + sy['n_amp'], sy['o_a']:sy['o_a'] + sy['n_amp']]
    band = np.sqrt(np.clip(np.diag(sy['S'] @ Ca @ sy['S'].T), 0, None))[inb].mean()
    c = np.zeros(sy['npar']); c[sy['o_g'] + M.ihc] = -1.0; c[sy['o_g'] + M.iD[IZ]] = 1.0
    return band, M.sig(sy, M.growth_c(sy, 'f', IZ)), M.sig(sy, c)


print(f"{'':36s}{'band sigma(lnP)':>16s}{'sigma(f)':>11s}{'D_M/r_d':>10s}{'D_M/r_d + phase':>17s}")
SETS = [('as sampled: 0.5 / 200', dict(s_lna=0.5, lam=200.0, s_lng=0.3, s_lnh=0.05)),
        ('s_lna off (1e6)', dict(s_lna=BIG, lam=200.0, s_lng=0.3, s_lnh=0.05)),
        ('lambda off (0)', dict(s_lna=0.5, lam=0.0, s_lng=0.3, s_lnh=0.05)),
        ('s_lng off (1e6)', dict(s_lna=0.5, lam=200.0, s_lng=BIG, s_lnh=0.05)),
        ('template priors off', dict(s_lna=BIG, lam=0.0, s_lng=0.3, s_lnh=0.05))]
for lab, pr in SETS:
    b, f_, d = mi_numbers(pr)
    _, _, dp = mi_numbers(pr, phase_sigma=0.002)
    print(f"  {lab:34s}{b:16.4f}{100*f_:10.1f}%{100*d:9.2f}%{100*dp:16.2f}%")
print("  (s_lnh is not varied here: h_conv is exactly unconstrained by the data, so with that\n"
      "   prior off the absolute distances are simply undefined -- by construction, not by bad luck)")

print("\n" + "=" * 86)
print("4. HOW FAR CAN THE TEMPLATE PRIOR BE WIDENED?")
print("   target: the induced prior on LCDM should be well WEAKER than the data, say 5x, so it"
      "\n   cannot contribute; and the allowed node-to-node roughness should stay small enough"
      "\n   for the loop emulator, which is why lambda was there in the first place.\n")
S_LNA = [0.5, 1.0, 2.0, 5.0, 10.0]
LAM = [200.0, 50.0, 20.0, 5.0, 0.0]
print(f"{'s_lna':>7s}{'lam':>6s} | {'LCDM sigma ratio to data-only':>31s} | {'rough':>7s} | "
      f"{'band':>7s}{'sig(f)':>9s}{'D_M/r_d':>9s}{'+phase':>9s}")
print(f"{'':7s}{'':6s} | {'omega_cdm     lnAs        h':>31s} | {'':7s} | {'':7s}{'':9s}"
      f"{'free':>9s}{'0.2%':>9s}")
scan = {}
for s_lna in S_LNA:
    for lam in LAM:
        pr = dict(s_lna=s_lna, lam=lam, s_lng=0.3, s_lnh=0.05)
        Pi_full = M.prior_precision(n_amp, s_lna, 0.3, 0.05, lam, Db)
        sw = sig3(F_data['pb'] + Mnode.T @ Pi_full @ Mnode)         # LCDM with the MI prior on
        ratio = sw / base
        Pa = np.eye(n_amp) / s_lna**2 + lam * (Db.T @ Db)
        rough = np.sqrt(np.median(np.diag(Db @ np.linalg.inv(Pa) @ Db.T)))
        b, f_, d = mi_numbers(pr)
        _, _, dp = mi_numbers(pr, phase_sigma=0.002)
        scan[(s_lna, lam)] = (ratio, rough, b, f_, d, dp)
        print(f"{s_lna:7.1f}{lam:6.0f} | " + "".join(f"{v:10.2f}" for v in ratio) +
              f" | {rough:7.3f} | {b:7.3f}{100*f_:8.1f}%{100*d:8.2f}%{100*dp:8.2f}%")
print("\n  'LCDM sigma ratio to data-only' is the contamination: 1.00 = the prior adds nothing"
      "\n  to LCDM, 0.42 = it halves the error bar out of nowhere. 'rough' is the node-to-node"
      "\n  roughness the prior still allows (1 sigma on one second difference of ln a), which is"
      "\n  what the smoothness penalty was for. 'band' is the fractional width of the recovered"
      "\n  P_lin inside the data window: part of the current 0.10 is prior, not data.")

np.savez(os.path.join(OUT, 'prior_decomposition_pb.npz'),
         s_lna=np.array(S_LNA), lam=np.array(LAM), par=np.array(PAR),
         data_sigma_p=sig3(F_data['p']), data_sigma_pb=base,
         **{f'scan_{s:g}_{l:g}': np.array([*scan[(s, l)][0], scan[(s, l)][1], scan[(s, l)][2],
                                           scan[(s, l)][3], scan[(s, l)][4], scan[(s, l)][5]])
            for s in S_LNA for l in LAM})

print("\n" + "=" * 86)
print("5. A WARNING ABOUT lambda AND THE NODE SPACING")
print("   lambda multiplies UNNORMALIZED second differences, D2 = a[i+1] - 2a[i] + a[i-1]. For a"
      "\n   smooth function those scale as (node spacing)^2, so the same lambda is a much STRONGER"
      "\n   prior on a coarse basis. Anything that changes node_spacing changes the prior too:\n")
print(f"{'spacing':>9s}{'nodes':>7s}{'roughness':>12s}{'equivalent lambda at 0.15':>27s}")
for spacing in (0.15, 0.25, 0.40, 0.60):
    nl_s, S_s, D_s, _ = M.node_basis(spacing)
    Pa = np.eye(len(nl_s)) / 0.5**2 + 200.0 * (D_s.T @ D_s)
    rough = np.sqrt(np.median(np.diag(D_s @ np.linalg.inv(Pa) @ D_s.T)))
    print(f"{spacing:9.2f}{len(nl_s):7d}{rough:12.3f}{200 * (spacing / 0.15)**4:27.0f}")
print("   So the 16-node 'BAO-rigid' basis of the earlier notebook carries a smoothness prior"
      "\n   ~250x stronger in physical terms. The phase prior does not have this problem: it is"
      "\n   stated directly as a fraction, and it is applied on the 60-node basis throughout.")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), layout='constrained')
cols = plt.cm.viridis(np.linspace(0.05, 0.85, len(LAM)))
for i, lam in enumerate(LAM):
    axes[0].plot(S_LNA, [scan[(s, lam)][0][0] for s in S_LNA], color=cols[i], lw=2.2,
                 marker='o', ms=5, label=rf'$\lambda$ = {lam:g}')
    axes[1].plot(S_LNA, [scan[(s, lam)][1] for s in S_LNA], color=cols[i], lw=2.2, marker='o', ms=5)
    axes[2].plot(S_LNA, [100 * scan[(s, lam)][2] for s in S_LNA], color=cols[i], lw=2.2,
                 marker='o', ms=5)
axes[0].axhline(1, color=C_INK, lw=1.4, ls='--')
axes[0].set_ylabel(r'$\sigma(\omega_{\rm cdm})$ with the prior / data alone', fontsize=12)
axes[1].set_ylabel('node-to-node roughness allowed', fontsize=12)
axes[2].set_ylabel(r'$1\sigma$ on $P_{\rm lin}$ in the data window [%]', fontsize=12)
axes[0].set_ylim(0.35, 1.08)
for ax in axes:
    ax.set_xscale('log'); ax.grid(alpha=0.2, which='both')
for ax in axes[1:]:
    ax.set_yscale('log')
    ax.set_xlabel(r'$\sigma_{\ln a}$ per node', fontsize=12)
axes[0].legend(frameon=False, fontsize=10)
fig.suptitle('How much the template prior constrains, and what widening it costs', fontsize=14)
p = os.path.join(FIG, 'fig_prior_decomposition.png')
fig.savefig(p, dpi=170); plt.close(fig); print(f"\nsaved {p}")

print("\n" + "=" * 86)
print("RECOMMENDATION")
print("""  * lambda is the block that matters: with s_lna anywhere from 0.5 to 10 the LCDM errors are
    identical, while lambda alone accounts for the whole factor 2 on omega_cdm and 2.4 on A_s.
    s_lna is not doing harm to LCDM -- but it IS setting a large part of the reported P(k) band,
    so it should still be widened to report an honest band.
  * Recommended for the next chain: s_lna = 2, lambda = 50 (from 0.5 / 200). The node-to-node
    roughness the prior allows goes 0.066 -> 0.135, still smooth enough for the loop emulator,
    which is the job lambda was introduced for. The MI results barely move: sigma(f) 9.7 -> 10.0%,
    D_M/r_d with the phase prior 0.85 -> 1.03%. The reported P(k) band widens 0.10 -> 0.23, which
    is the point: that width was previously prior, not data.
  * No practical lambda makes the induced LCDM prior negligible (lambda = 5 still leaves A_s at
    0.76 of its data-only width, and lambda = 0 gives a roughness of 5). So widening is not by
    itself a fix for the compression: an MI posterior compressed onto LCDM must divide out the
    induced prior pi(M theta). Done that way the answer matches the direct fit exactly, at any
    prior width -- which is what the LCDM figure of 08 shows.""")
