"""Compression of the MI posterior onto any cosmological model.

The likelihood depends on cosmology only through phi, so for any model theta -> phi(theta)

    p(theta | d)  proportional to  p_MI(phi(theta) | d) / pi_MI(phi(theta)) * pi(theta).

p_MI is estimated from the MI chain with an ensemble of affine-coupling normalizing flows
(RealNVP, pure JAX + optax) fitted to the physical block in sample-whitened coordinates; the
members differ in seed (initialization, masks, held-out split) and are combined as the mean of
their log-densities, which is robust to a single member with a heavy tail. The Gaussian MI
prior is divided out exactly.
"""
import numpy as np
import jax, jax.numpy as jnp
import optax

from sampling import log


# ---- the flow ------------------------------------------------------------------------------
def _init_mlp(key, d_in, d_out, hidden):
    ks = jax.random.split(key, 3)
    lin = lambda k, a, b, sc: {'w': sc * jax.random.normal(k, (a, b)) / jnp.sqrt(a), 'b': jnp.zeros(b)}
    return [lin(ks[0], d_in, hidden, 1.0), lin(ks[1], hidden, hidden, 1.0), lin(ks[2], hidden, d_out, 1e-3)]

def _mlp(p, x):
    h = jax.nn.gelu(x @ p[0]['w'] + p[0]['b']); h = jax.nn.gelu(h @ p[1]['w'] + p[1]['b'])
    return h @ p[2]['w'] + p[2]['b']

def _init_flow(key, dim, n_layers, hidden, seed):
    rng = np.random.default_rng(seed); masks = []
    for l in range(n_layers):
        perm = rng.permutation(dim) if l > 0 else np.arange(dim)
        m = np.zeros(dim); m[perm[: dim // 2]] = 1.0; masks.append(m)       # 1 = conditioning half
    keys = jax.random.split(key, n_layers)
    return [_init_mlp(keys[l], dim, 2 * dim, hidden) for l in range(n_layers)], jnp.array(np.stack(masks))

def log_prob(params, masks, u):
    z, logdet = u, 0.0
    for p, m in zip(params, masks):
        out = _mlp(p, z * m); d = z.shape[0]
        log_s = 2.0 * jnp.tanh(out[d:] / 2.0) * (1 - m)                    # bounded log-scale
        z = z * jnp.exp(log_s) + out[:d] * (1 - m); logdet = logdet + jnp.sum(log_s)
    return -0.5 * jnp.sum(z**2) - 0.5 * z.shape[0] * jnp.log(2 * jnp.pi) + logdet

def train_flow(u, n_layers=8, hidden=128, steps=8000, batch=2048, lr=5e-4, holdout=0.2, seed=0, eval_every=100, patience=20):
    """Maximum likelihood (Adam, cosine schedule) with held-out early stopping; u: (n, dim)."""
    u = np.asarray(u, float); n, dim = u.shape
    perm = np.random.default_rng(seed).permutation(n); n_ho = int(holdout * n)
    u_ho, u_tr = jnp.array(u[perm[:n_ho]]), jnp.array(u[perm[n_ho:]])
    params, masks = _init_flow(jax.random.key(seed), dim, n_layers, hidden, seed)
    opt = optax.chain(optax.clip_by_global_norm(5.0), optax.adam(optax.warmup_cosine_decay_schedule(0.0, lr, min(200, steps // 10), steps, lr * 0.02)))
    state = opt.init(params); lp = jax.vmap(log_prob, in_axes=(None, None, 0))
    @jax.jit
    def step(params, state, ub):
        l, g = jax.value_and_grad(lambda p: -jnp.mean(lp(p, masks, ub)))(params)
        upd, state = opt.update(g, state, params); return optax.apply_updates(params, upd), state, l
    ll_ho = jax.jit(lambda p: jnp.mean(lp(p, masks, u_ho)))
    best, bad, key = (-np.inf, params, 0), 0, jax.random.key(seed + 1)
    for it in range(1, steps + 1):
        key, k = jax.random.split(key)
        params, state, _ = step(params, state, u_tr[jax.random.choice(k, u_tr.shape[0], (min(batch, u_tr.shape[0]),))])
        if it % eval_every == 0:
            v = float(ll_ho(params))
            if v > best[0] + 1e-4: best, bad = (v, params, it), 0
            else:
                bad += 1
                if bad >= patience: break
    gauss = float(np.mean(-0.5 * np.sum(np.asarray(u_ho)**2, 1) - 0.5 * dim * np.log(2 * np.pi)))
    return {'params': best[1], 'masks': masks, 'll_ho': best[0], 'll_gauss_ho': gauss, 'best_step': best[2]}


# ---- ensemble fit and projected posterior --------------------------------------------------
def complement_basis(vecs, n):
    """Orthonormal basis (n x (n - m)) of the complement of the m given directions in R^n."""
    if not len(vecs): return np.eye(n)
    V = np.stack([np.asarray(v, float) / np.linalg.norm(v) for v in vecs], 1)
    Q, _ = np.linalg.qr(np.concatenate([V, np.eye(n)], 1))
    return Q[:, V.shape[1]:n]


def fit_ensemble(M, x_samples, n_seeds=10, seed=0, block=None, project=(), **kw):
    """Flow ensemble on the physical block phi = x[n_eft:] of the MI chain (flattened), or on a SECTOR of
    it: `block` indexes phi, and the directions `project` (vectors in block coordinates, e.g. prior-only
    directions the sector cannot see without the others) are projected out, i.e. the flow models the
    marginal over their complement y = phi[block] Q. The Gaussian MI prior of the same y is stored, so
    the projection divides the marginal posterior by the marginal prior."""
    phi = np.asarray(x_samples)[:, M.n_eft:]
    sector = block is not None
    block = np.arange(M.n_phys) if block is None else np.asarray(block, int)
    Q = complement_basis(list(project), len(block))
    y = phi[:, block] @ Q
    mu = y.mean(0); w, V = np.linalg.eigh(np.linalg.inv(np.cov(y, rowvar=False)))
    Linv = (V * np.sqrt(np.clip(w, 1e-8 * w.max(), None))).T                   # whitening
    u = (y - mu) @ Linv.T
    members = [train_flow(u, seed=seed + k, **kw) for k in range(n_seeds)]
    log(f"[flow] {n_seeds} members on {y.shape[1]} dims: held-out gain over the whitened Gaussian "
        f"{np.round([m['ll_ho'] - m['ll_gauss_ho'] for m in members], 3)} nats/sample; best steps {[m['best_step'] for m in members]}")
    fl = {'members': members, 'mu': mu, 'Linv': Linv}
    if sector:
        C = np.linalg.inv(M.prior_prec)[np.ix_(block, block)]
        fl.update(block=block, Q=Q, prior_prec_y=np.linalg.inv(Q.T @ C @ Q), prior_mean_y=Q.T @ M.prior_mean[block])
    return fl


def gaussian_estimate(M, x_samples, block=None, project=()):
    """The sample mean and covariance of the MI physical block (or of a sector, as fit_ensemble) as the
    density estimate: what the flows add beyond a Gaussian."""
    phi = np.asarray(x_samples)[:, M.n_eft:]
    sector = block is not None
    block = np.arange(M.n_phys) if block is None else np.asarray(block, int)
    Q = complement_basis(list(project), len(block)); y = phi[:, block] @ Q
    w, V = np.linalg.eigh(np.linalg.inv(np.cov(y, rowvar=False)))
    fl = {'gauss': True, 'members': [], 'mu': y.mean(0), 'Linv': (V * np.sqrt(np.clip(w, 1e-8 * w.max(), None))).T}
    if sector:
        C = np.linalg.inv(M.prior_prec)[np.ix_(block, block)]
        fl.update(block=block, Q=Q, prior_prec_y=np.linalg.inv(Q.T @ C @ Q), prior_mean_y=Q.T @ M.prior_mean[block])
    return fl


def density_logpost(M, fl, phi_map, lo, hi, logprior=None):
    """theta -> log p_MI(y(theta)) - log pi_MI(y(theta)) + log pi(theta), with y the (sector) coordinates of
    phi_map(theta) (ln phi, full physical block); flat in the box (lo, hi) times exp(logprior)."""
    mu, Linv, members = jnp.array(fl['mu']), jnp.array(fl['Linv']), fl['members']
    if 'block' in fl:
        blk, Q = jnp.array(np.asarray(fl['block'], int)), jnp.array(fl['Q'])
        Pi, pm = jnp.array(fl['prior_prec_y']), jnp.array(fl['prior_mean_y'])
        to_y = lambda phi: phi[blk] @ Q
    else:
        Pi, pm, to_y = jnp.array(M.prior_prec), jnp.array(M.prior_mean), (lambda phi: phi)
    lo, hi = jnp.array(lo), jnp.array(hi)
    def _lp(theta):
        y = to_y(phi_map(jnp.clip(theta, lo, hi))); u = Linv @ (y - mu)
        lp_mi = (-0.5 * u @ u) if fl.get('gauss') else jnp.mean(jnp.array([log_prob(m['params'], m['masks'], u) for m in members]))
        lpt = (0.0 if logprior is None else logprior(theta))
        ok = jnp.all((theta > lo) & (theta < hi)) & jnp.isfinite(lpt)
        return jnp.where(ok, lp_mi + 0.5 * (y - pm) @ Pi @ (y - pm) + jnp.where(ok, lpt, 0.0), -jnp.inf)
    return _lp


def projected_logpost(M, fl, keys=None):
    """theta -> log p_MI(phi(theta)) - log pi_MI(phi(theta)) + log pi(theta) for the direct model
    `keys` (default: the current cosmo model; the closure is a snapshot)."""
    keys = list(M.cosmo_keys if keys is None else keys)
    return density_logpost(M, fl, M.phi_ln_keys(keys), [M.cosmo_box[k][0] for k in keys], [M.cosmo_box[k][1] for k in keys],
                           M.logprior_theta_keys(keys))


def load_or_fit(M, x_samples, path, force=False, **kw):  # kw -> fit_ensemble (incl. block, project)
    """The flow ensemble of an MI chain, cached as a pickle next to it (refit when the chain is newer
    or with force), so every model is projected from the same density estimate."""
    import os, pickle
    chain = os.path.join(os.path.dirname(path), 'chain_mi.npz')
    if os.path.exists(path) and not force and (not os.path.exists(chain) or os.path.getmtime(path) > os.path.getmtime(chain)):
        with open(path, 'rb') as fh: fl = pickle.load(fh)
        log(f"[flow] loaded {path} ({len(fl['members'])} members)")
        return jax.tree_util.tree_map(lambda a: jnp.asarray(a) if isinstance(a, np.ndarray) else a, fl)
    fl = fit_ensemble(M, x_samples, **kw)
    with open(path, 'wb') as fh: pickle.dump(jax.tree_util.tree_map(lambda a: np.asarray(a) if hasattr(a, 'shape') else a, fl), fh)
    log(f"[flow] saved {path}")
    return fl


def sample_logpost(lp, lo, hi, th_start, seed=41, n_warmup=500, n_samples=2000, n_chains=4):
    """NUTS on a log-posterior with a flat box (lo, hi), in the unbounded coordinates of BoxTransform,
    started from th_start. Returns theta samples (chains, draws, n)."""
    from sampling import BoxTransform, minimize_lbfgs, fisher_at, precond_cov, WhitenedNUTS
    bt = BoxTransform(lo, hi)
    lp_y = jax.jit(bt.wrap(lp))
    y_bf, _ = minimize_lbfgs(lp_y, np.asarray(bt.to_y(jnp.array(th_start))), name='projected', scales=np.full(len(lo), 0.3), verbose=False)
    C = precond_cov(fisher_at(lp_y, y_bf, name='projected'), floor_prec=0.25)
    r = WhitenedNUTS(lp_y, y_bf, C, name='projected').run(jax.random.key(seed), n_warmup, n_samples, n_chains=n_chains, jitter=0.5,
                                                         max_num_doublings=7, verbose=False)
    return np.asarray(jax.vmap(jax.vmap(bt.to_x))(jnp.array(r['x'])))


def sample_projected(M, fl, th_start, seed=41, n_warmup=500, n_samples=2000, n_chains=4, keys=None):
    """sample_logpost on the projected posterior of the current cosmo model (or `keys`)."""
    keys = list(M.cosmo_keys if keys is None else keys)
    return sample_logpost(projected_logpost(M, fl, keys), [M.cosmo_box[k][0] for k in keys], [M.cosmo_box[k][1] for k in keys],
                          th_start, seed, n_warmup, n_samples, n_chains)
