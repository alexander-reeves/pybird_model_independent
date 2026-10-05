"""The LCDM Fisher as a projection of the model-independent one. Shared by
meeting_growth_pb.py (the recovery figure) and prior_decomposition_pb.py (the prior audit),
so the two cannot drift apart.

The direct LCDM model IS the MI model composed with the cosmology map (cosmo_map_pb.npz);
`cosmo_map_pb.py` checks that by differentiating the direct model through the same likelihood
call and comparing with J_MI @ M (they agree to 1.4e-6). What differs between the routes is
only which priors are carried, which is what `route` selects.
"""
import os
import numpy as np
import mi_prior_fisher as M

CM = np.load(os.path.join(M.OUT, 'cosmo_map_pb.npz'))
Mmap, theta_fid = CM['M'], CM['theta_fid']
GATE = {k: float(CM[f'gate_{k}']) for k in ('p', 'pb') if f'gate_{k}' in CM.files}

nl, Sb, Db, _ = M.node_basis()
n_amp = len(nl)
# the LCDM template response expressed on the MI nodes (least squares through the spline)
M_amp_nodes = np.linalg.lstsq(Sb, Mmap[:M.nk], rcond=None)[0]
REP = [np.linalg.norm(Sb @ M_amp_nodes[:, j] - Mmap[:M.nk, j]) / np.linalg.norm(Mmap[:M.nk, j])
       for j in range(3)]
Mnode = np.vstack([M_amp_nodes, Mmap[M.nk:]])


def whiten(key, project_data=True):
    """Whitened Jacobian in the [EFT, 80 knots, 25 growth] basis, exact symmetries projected."""
    R = M.R
    J, P, eftpr, x0 = R[f'J_{key}'], R[f'P_{key}'], R[f'prior_{key}'], R[f'x0_{key}']
    n_e = int(R[f'n_e_{key}']); o_a = n_e * M.nsky; o_gk = o_a + M.nk
    Lc = np.linalg.cholesky(0.5 * (P + P.T))
    s = np.ones(len(x0)); e = np.arange(o_a); has = eftpr[e] > 0
    s[e] = np.where(has, 1 / np.sqrt(np.where(has, eftpr[e], 1.0)), np.maximum(np.abs(x0[e]), 1.0))
    s[o_a:o_gk] = x0[o_a:o_gk]; s[o_gk:] = M.gfid
    Wd = (Lc.T @ J) * s
    if project_data:
        G = np.zeros((Wd.shape[1], 3))
        G[o_gk + M.ihc, 0] = 1.0; G[o_gk + M.iD, 0] = 1.0; G[o_gk + M.iH, 0] = -1.0
        G[o_a:o_gk, 1] = 1.0; G[o_gk + M.iDr, 1] = -0.5
        G[o_gk + M.ihc, 2] = 1.0; G[o_a:o_gk, 2] = -M.gamma
        Q, Rq = np.linalg.qr(Wd @ G)
        Q = Q[:, np.abs(np.diag(Rq)) > 1e-14 * np.linalg.norm(Wd)]
        Wd = Wd - Q @ (Q.T @ Wd)
    return Wd, o_a, np.where(has)[0]


def cosmo_fisher(key, route='direct', project_data=True, priors=None, prior_prec=None):
    """3x3 LCDM Fisher, EFT marginalized.

    route 'direct'           LCDM fitted to the data at the 80 knots, no MI priors: an
                             ordinary full-shape LCDM fit.
    route 'through_mi_flat'  the same data and likelihood routed through the MI
                             parametrization (60 spline nodes), no MI priors. Must reproduce
                             'direct' -- that is the no-information-inserted check.
    route 'through_mi'       ... plus the MI priors evaluated on the LCDM submanifold, i.e.
                             what compressing an MI posterior onto LCDM gives if the induced
                             prior is not divided out.
    prior_prec               an explicit (n_amp + n_growth) precision to use instead of the
                             named MI priors, for auditing one block at a time.
    """
    Wd, o_a, idx = whiten(key, project_data)
    if route == 'direct':
        W = np.hstack([Wd[:, :o_a], Wd[:, o_a:] @ Mmap])
        Pi = np.zeros((3, 3))
    else:
        Wn = np.hstack([Wd[:, :o_a], Wd[:, o_a:o_a + M.nk] @ Sb, Wd[:, o_a + M.nk:]])
        W = np.hstack([Wn[:, :o_a], Wn[:, o_a:] @ Mnode])
        if route == 'through_mi_flat' and prior_prec is None:
            Pi = np.zeros((3, 3))
        else:
            if prior_prec is None:
                pr = dict(M.PRIORS_HMC if priors is None else priors)
                prior_prec = M.prior_precision(n_amp, pr['s_lna'], pr['s_lng'], pr['s_lnh'],
                                               pr['lam'], Db)
            Pi = Mnode.T @ prior_prec @ Mnode
    Wp = np.zeros((len(idx), o_a + 3)); Wp[np.arange(len(idx)), idx] = 1.0
    F = W.T @ W + Wp.T @ Wp
    F = 0.5 * (F + F.T)
    Fe, Fx, Fxx = F[:o_a, :o_a], F[:o_a, o_a:], F[o_a:, o_a:]
    return Fxx + Pi - Fx.T @ np.linalg.solve(Fe, Fx)
