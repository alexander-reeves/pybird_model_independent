"""Inference: L-BFGS-B, Hessian, whitened NUTS (BlackJAX), box coordinates, and the normalizing-flow density of an MI chain
that the projection onto a cosmological model uses,

    p(theta | d)  proportional to  p_MI(phi(theta) | d) / pi_MI(phi(theta)) * pi(theta),

p_MI an ensemble of affine-coupling flows (RealNVP, pure JAX + optax) on the physical block in sample-whitened coordinates,
members differing in seed and combined as the mean of their log-densities; the Gaussian MI prior is divided out exactly.

The alternative, more accurate, density (score regression, fit_score): the EFT-marginal MI likelihood
l(phi) = log int L(e, phi) de learned from its gradient at the chain draws, log p(theta | d) = l(phi(theta)) + log pi(theta).
"""
import os, sys, time, pickle
import numpy as np
import jax
import jax.numpy as jnp
import optax

jax.config.update("jax_enable_x64", True)

jax.config.update("jax_enable_x64", True)

_T0 = time.time()
def log(msg):
    print(f"[{time.time()-_T0:7.1f}s] {msg}", flush=True)


def precond_cov(F, floor_prec=None, rtol=1e-8):
    """Whitening covariance from a (possibly noisy) Fisher: eigenvalues floored at
    max(floor_prec, rtol * max) so no direction gets a spuriously large scale."""
    F = 0.5 * (F + F.T); w, V = np.linalg.eigh(F)
    fl = rtol * float(w.max())
    if floor_prec is not None: fl = max(fl, float(floor_prec))
    return (V / np.clip(w, fl, None)) @ V.T


# ------------------------------------------------------------------------------------------
# optimizer and exact Hessian
# ------------------------------------------------------------------------------------------
def minimize_lbfgs(logpost, x0, maxiter=500, name='min', scales=None, bounds=None, verbose=True):
    """Maximize logpost with scipy L-BFGS-B in scaled coordinates u = (x - x0)/scales, JAX
    gradient, optional bounds [(lo, hi) or (None, None)] (a -inf inside a line search stalls
    L-BFGS-B). Returns (x_best, -logpost(x_best))."""
    from scipy.optimize import minimize
    x0 = np.asarray(x0, float); sc = np.ones_like(x0) if scales is None else np.asarray(scales, float)
    f = jax.jit(lambda u: -logpost(jnp.array(x0) + jnp.array(sc) * u))
    g = jax.jit(jax.grad(lambda u: -logpost(jnp.array(x0) + jnp.array(sc) * u)))
    def fun(u):
        v = float(f(jnp.array(u))); return v if np.isfinite(v) else 1e30
    def jac(u):
        gr = np.asarray(g(jnp.array(u)), float); return np.where(np.isfinite(gr), gr, 0.0)
    ub = None if bounds is None else [((None if lo is None else (lo - x0[i]) / sc[i]), (None if hi is None else (hi - x0[i]) / sc[i]))
                                      for i, (lo, hi) in enumerate(bounds)]
    t0 = time.time()
    r = minimize(fun, np.zeros_like(x0), jac=jac, method='L-BFGS-B', bounds=ub,
                 options={'maxiter': maxiter, 'maxcor': 30, 'ftol': 1e-12, 'gtol': 1e-6})
    if verbose:
        log(f"[{name}] L-BFGS-B: -logpost {fun(np.zeros_like(x0)):.3f} -> {r.fun:.3f} in {time.time()-t0:.1f}s "
            f"({r.nit} it, {r.nfev} fev: {r.message})")
    return x0 + sc * r.x, float(r.fun)


def fisher_at(logpost, x, name='F'):
    """Exact Hessian of -logpost at x (jax.hessian), symmetrized."""
    t0 = time.time()
    F = -np.array(jax.hessian(logpost)(jnp.array(x))); F = 0.5 * (F + F.T)
    w = np.linalg.eigvalsh(F)
    log(f"[{name}] Hessian ({len(x)}x{len(x)}) in {time.time()-t0:.1f}s; min/max eigenvalue {w.min()/w.max():+.2e}")
    return F


# ------------------------------------------------------------------------------------------
# NUTS in whitened coordinates
# ------------------------------------------------------------------------------------------
class WhitenedNUTS:
    """BlackJAX NUTS on u with x = center + L u, C = L L^T a preconditioning covariance
    (the inverse Gauss-Newton Fisher for the MI posterior). Window adaptation then only fixes the
    residual anisotropy. Chains are vmapped and sharded over the GPUs of the node with pmap."""

    def __init__(self, logpost, center, cov, name='chain'):
        import blackjax
        self.bj = blackjax
        C = 0.5 * (np.asarray(cov) + np.asarray(cov).T); w, V = np.linalg.eigh(C); w = np.clip(w, 1e-12 * w.max(), None)
        self.center = jnp.array(center); self.Lmat = jnp.array(V * np.sqrt(w)); self.n = len(center)
        self.logpost, self.name = logpost, name

    def u_to_x(self, u): return self.center + self.Lmat @ u
    def logdensity_u(self, u): return self.logpost(self.u_to_x(u))

    def run(self, key, num_warmup, num_samples, n_chains=32, jitter=0.1, target_accept=0.8, max_num_doublings=8,
            initial_step_size=0.1, verbose=True, resume=None):
        """resume: the dict a previous run returned. Sampling continues from its last positions with its adapted step sizes
        and mass matrices (no warmup), so a chain can be split over several short jobs."""
        bj = self.bj
        keys = jax.random.split(key, n_chains + 1)
        init_u = jitter * jax.random.normal(keys[-1], (n_chains, self.n))
        if resume is not None:
            init_u, num_warmup = jnp.asarray(resume['u_last']), 0; n_chains = len(init_u)
            keys = jax.random.split(jax.random.fold_in(key, resume['x'].shape[1]), n_chains + 1)
        lp_fn = jax.vmap(jax.jit(self.logdensity_u))
        lp0, kr = np.asarray(lp_fn(init_u)), keys[-1]
        for _ in range(8):                        # a start outside the support is redrawn closer to the centre
            bad = ~np.isfinite(lp0)
            if not bad.any(): break
            kr, k2 = jax.random.split(kr); jitter *= 0.5
            init_u = np.array(init_u); init_u[bad] = jitter * np.asarray(jax.random.normal(k2, (int(bad.sum()), self.n)))
            init_u = jnp.asarray(init_u); lp0 = np.asarray(lp_fn(init_u))
        if verbose:
            log(f"[{self.name}] start: logdensity(center) = {float(self.logdensity_u(jnp.zeros(self.n))):.3f}; "
                f"chains in [{lp0.min():.2f}, {lp0.max():.2f}]")
        if not np.isfinite(lp0).all():
            raise RuntimeError(f"[{self.name}] non-finite log-density at the start positions")
        warmup = bj.window_adaptation(bj.nuts, self.logdensity_u, is_mass_matrix_diagonal=True, target_acceptance_rate=target_accept,
                                      initial_step_size=initial_step_size, max_num_doublings=max_num_doublings)

        def run_one(k, u0, eps, imm):
            kw, ks = jax.random.split(k)
            if resume is None:
                (state, params), _ = warmup.run(kw, u0, num_steps=num_warmup); eps, imm = params['step_size'], params['inverse_mass_matrix']
            else:
                state = bj.nuts.init(u0, self.logdensity_u)
            kernel = bj.nuts(self.logdensity_u, step_size=eps, inverse_mass_matrix=imm, max_num_doublings=max_num_doublings).step
            def step(st, kk):
                st, info = kernel(kk, st)
                return st, (st.position, st.logdensity, info.acceptance_rate, info.is_divergent, info.num_integration_steps)
            _, out = jax.lax.scan(step, state, jax.random.split(ks, num_samples))
            return out, eps, imm

        eps0 = jnp.asarray(resume['step_size']) if resume is not None else jnp.zeros(n_chains)             # unused without resume
        imm0 = jnp.asarray(resume['inverse_mass_matrix']) if resume is not None else jnp.zeros((n_chains, self.n))
        t0 = time.time(); n_dev = jax.local_device_count()
        if n_dev > 1 and n_chains % n_dev == 0 and n_chains >= 2 * n_dev:
            per = n_chains // n_dev
            if verbose: log(f"[{self.name}] {n_chains} chains on {n_dev} devices ({per} per device, pmap x vmap)")
            sh = lambda a: a.reshape((n_dev, per) + a.shape[1:])
            out, step, imm = jax.pmap(jax.vmap(run_one))(sh(keys[:n_chains]), sh(init_u), sh(eps0), sh(imm0))
            out, step, imm = [[np.asarray(a).reshape((n_chains,) + np.asarray(a).shape[2:]) for a in v] for v in (out, [step], [imm])]
            step, imm = step[0], imm[0]
        else:
            out, step, imm = jax.jit(jax.vmap(run_one))(keys[:n_chains], init_u, eps0, imm0)
            out, step, imm = [np.asarray(a) for a in out], np.asarray(step), np.asarray(imm)
        pos, logp, acc, div, nsteps = out
        wall = time.time() - t0
        x = np.asarray(jax.vmap(jax.vmap(self.u_to_x))(jnp.array(pos)))
        if verbose:
            log(f"[{self.name}] {n_chains} chains x {num_samples} draws (+{num_warmup} warmup) in {wall/60:.1f} min; "
                f"accept {acc.mean():.3f}, divergent {div.mean():.4f}, leapfrog/iter {nsteps.mean():.1f}")
        r = {'x': x, 'logp': logp, 'accept': acc, 'divergent': div, 'n_leapfrog': nsteps, 'step_size': step,
             'walltime': wall, 'num_warmup': num_warmup, 'num_samples': num_samples, 'inverse_mass_matrix': imm, 'u_last': pos[:, -1]}
        if resume is not None:                                # the draws of the previous jobs come first
            r.update({k: np.concatenate([resume[k], r[k]], 1) for k in ('x', 'logp', 'accept', 'divergent', 'n_leapfrog')},
                     walltime=wall + float(resume['walltime']), num_warmup=int(resume['num_warmup']), num_samples=num_samples + int(resume['num_samples']))
        return r


def chain_diagnostics(x, names=None, max_print=5):
    """Split-Rhat and ESS per parameter (numpyro) for x of shape (chains, draws, n)."""
    from numpyro.diagnostics import split_gelman_rubin, effective_sample_size
    x = np.asarray(x)
    rhat = np.array([split_gelman_rubin(x[:, :, i]) for i in range(x.shape[-1])])
    ess = np.array([effective_sample_size(x[:, :, i]) for i in range(x.shape[-1])])
    if names is not None and max_print:
        for i in np.argsort(-rhat)[:max_print]:
            log(f"   {names[i]:>26s}: Rhat {rhat[i]:.3f}  ESS {ess[i]:.0f}")
    log(f"   max Rhat {rhat.max():.3f}, min ESS {ess.min():.0f}, median ESS {np.median(ess):.0f}")
    return rhat, ess


class BoxTransform:
    """Unbounded coordinates for a flat box on the last len(lo) entries of x (the first n_free are
    left alone): x_box = lo + (hi - lo) sigmoid(y). `wrap(logpost)` adds the log-Jacobian, so the
    sampled density is exactly the boxed one while NUTS never meets the -inf walls, which along a
    parameter the data do not bound (fEDE at 0, theta_i, z_c) otherwise makes most transitions diverge."""

    def __init__(self, lo, hi, n_free=0):
        self.lo, self.hi = jnp.array(np.asarray(lo, float)), jnp.array(np.asarray(hi, float))
        self.w, self.n = self.hi - self.lo, int(n_free)

    def to_x(self, y): return jnp.concatenate([y[:self.n], self.lo + self.w * jax.nn.sigmoid(y[self.n:])])
    def to_y(self, x):
        x = jnp.asarray(x); p = jnp.clip((x[self.n:] - self.lo) / self.w, 1e-6, 1 - 1e-6)
        return jnp.concatenate([x[:self.n], jnp.log(p) - jnp.log1p(-p)])
    def logjac(self, y): return jnp.sum(jnp.log(self.w) + jax.nn.log_sigmoid(y[self.n:]) + jax.nn.log_sigmoid(-y[self.n:]))
    def wrap(self, logpost): return lambda y: logpost(self.to_x(y)) + self.logjac(y)


class Tee:
    def __init__(self, path): self.f = open(path, 'a'); self.out = sys.stdout
    def write(self, s): self.out.write(s); self.f.write(s)
    def flush(self): self.out.flush(); self.f.flush()


def todo(path, force, n=None):
    """True if the stage must run: missing, --force, or saved with a different dimension."""
    if force or not os.path.exists(path): return True
    if n is not None:
        z = np.load(path); k = 'x_bf' if 'x_bf' in z.files else 'x'
        if z[k].shape[-1] != n: log(f"{path}: dimension {z[k].shape[-1]} != {n} -> recomputing"); return True
    log(f"{path} exists -> skipped (--force to redo)"); return False


def vbatch(f, *args, B=50):
    """jit(vmap(f)) over the leading axis of args in fixed-size batches (the last one padded), concatenated."""
    fb = jax.jit(jax.vmap(f)); out = []
    for i in range(0, len(args[0]), B):
        a = [np.asarray(v[i:i + B]) for v in args]; m = len(a[0])
        if m < B: a = [np.concatenate([v, np.repeat(v[-1:], B - m, 0)]) for v in a]
        out.append(np.asarray(fb(*[jnp.array(v) for v in a]))[:m])
    return np.concatenate(out)


def run_nuts(lp, lo, hi, th0, name, seed=41, warmup=1000, draws=5000):
    bt = BoxTransform(lo, hi); lp_y = jax.jit(bt.wrap(lp))
    y_bf, _ = minimize_lbfgs(lp_y, np.asarray(bt.to_y(jnp.array(th0))), name=name, scales=np.full(len(lo), 0.3))
    r = WhitenedNUTS(lp_y, y_bf, precond_cov(fisher_at(lp_y, y_bf, name=name), floor_prec=0.25), name=name).run(
        jax.random.key(seed), warmup, draws, n_chains=8, jitter=0.5, max_num_doublings=7)
    r['y'] = r['x']; r['x'] = np.asarray(jax.vmap(jax.vmap(bt.to_x))(jnp.array(r['y']))); return r


# ------------------------------------------------------------------------------------------ flows
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

def train_flow(u, n_layers=8, hidden=128, steps=8000, batch=2048, lr=5e-4, holdout=0.2, seed=0, eval_every=100, patience=20, groups=None):
    """Maximum likelihood (Adam, cosine schedule) with held-out early stopping; u: (n, dim). groups (n,): the chain of each
    draw. Then whole chains are held out: draws of one chain are correlated, so held-out draws of a training chain stop the
    training too late (on DESI DR1 the projected LCDM mean then moved by up to 0.40 sigma between fits, 0.24 with whole chains)."""
    u = np.asarray(u, float); n, dim = u.shape; rng = np.random.default_rng(seed)
    if groups is None:
        perm = rng.permutation(n); i_ho, i_tr = perm[:int(holdout * n)], perm[int(holdout * n):]
    else:
        g = np.unique(groups); ho = np.isin(groups, rng.permutation(g)[:max(1, int(holdout * len(g)))]); i_ho, i_tr = np.where(ho)[0], np.where(~ho)[0]
    u_ho, u_tr = jnp.array(u[i_ho]), jnp.array(u[i_tr])
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
    if fl.get('score'):                                    # score regression: l(phi) itself, no MI prior to divide out
        P, lo, hi = jax.tree_util.tree_map(jnp.asarray, members), jnp.array(lo), jnp.array(hi)
        def _lps(theta):
            u = Linv @ (phi_map(jnp.clip(theta, lo, hi)) - mu); lpt = (0.0 if logprior is None else logprior(theta))
            ok = jnp.all((theta > lo) & (theta < hi)) & jnp.isfinite(lpt)
            return jnp.where(ok, jnp.mean(jax.vmap(score_g, in_axes=(0, None))(P, u)) + jnp.where(ok, lpt, 0.0), -jnp.inf)
        return _lps
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


def load_or_fit(M, x_samples, path, chains=(), force=False, **kw):  # kw -> fit_ensemble
    """The flow ensemble of an MI chain, cached as a pickle (refit when any of `chains` is newer, or with force), so every
    model is projected from the same density estimate."""
    if os.path.exists(path) and not force and all(os.path.getmtime(path) > os.path.getmtime(c) for c in chains):
        with open(path, 'rb') as fh: fl = pickle.load(fh)
        log(f"[flow] loaded {path} ({len(fl['members'])} members)")
        return jax.tree_util.tree_map(lambda a: jnp.asarray(a) if isinstance(a, np.ndarray) else a, fl)
    fl = fit_ensemble(M, x_samples, **kw)
    with open(path, 'wb') as fh: pickle.dump(jax.tree_util.tree_map(lambda a: np.asarray(a) if hasattr(a, 'shape') else a, fl), fh)
    log(f"[flow] saved {path}")
    return fl


# ------------------------------------------------------------------------------------------ score regression
# l(phi) = log int L(e, phi) de, the MI likelihood with the 12 sampled EFT parameters marginalized, from its gradient.
# Fisher's identity: grad l(phi) = E_{e|phi}[grad_phi log L(e, phi)], so the gradient at each chain draw is an unbiased, noisy
# sample of grad l at that phi. Functions with zero mean given phi are subtracted as control variates (Stein's identity:
# grad_e log L times (1, u), and e_k d_j log L + delta_jk). g(u) = b.u + u.A.u/2 + MLP(u) in the sample-whitened u is fitted
# to the corrected gradients: the quadratic by least squares, then all of it by weighted gradient matching with held-out
# chains for early stopping; an ensemble of fits (each with its own held-out chains) is averaged. Unlike the flow, nothing is
# a density estimate of the chain and no prior is divided out: on DESI DR1 it recovers w0wa to <= 0.24 sigma (flow 0.59).
SCORE = dict(n_members=5, hidden=256, steps=60000, lr=3e-4, batch=4096, holdout=0.2)


def chain_gradients(M, x, batch=32):
    """log L and its gradient (EFT block, phi block) at every draw of x (n, n_mi): pmap over the local devices x vmap."""
    nd = jax.local_device_count(); vg = jax.pmap(jax.vmap(jax.value_and_grad(M.loglkl_mi))); step = nd * batch; n = len(x)
    ll, g = np.zeros(n), np.zeros(x.shape); t0 = time.time()
    for i in range(0, n, step):
        idx = np.arange(i, i + step) % n; m = min(step, n - i)
        v, gr = vg(jnp.array(x[idx].reshape(nd, batch, -1)))
        ll[i:i + m] = np.asarray(v).reshape(-1)[:m]; g[i:i + m] = np.asarray(gr).reshape(step, -1)[:m]
    log(f"[score] {n} gradients in {time.time() - t0:.0f}s on {nd} device(s)")
    return ll, g


def score_g(p, u):
    """One member of the score fit at whitened u: quadratic + two-layer GELU MLP."""
    h = jax.nn.gelu(u @ p['w0'] + p['b0']); h = jax.nn.gelu(h @ p['w1'] + p['b1'])
    return p['b'] @ u + 0.5 * u @ p['A'] @ u + (h @ p['w2'])[0]


def fit_score(M, x, grads, n_members=5, hidden=256, steps=60000, lr=3e-4, batch=4096, holdout=0.2, seed=0):
    """Score regression of l(phi) on an MI chain: x and grads = d log L / dx at the draws, both (chains, draws, n_mi)."""
    nc, nt, _ = x.shape; ne = M.n_eft; x, grads = x.reshape(nc * nt, -1), grads.reshape(nc * nt, -1)
    ok = np.isfinite(grads).all(1); cid = np.repeat(np.arange(nc), nt)
    e, phi, ge, gp = x[:, :ne], x[:, ne:], grads[:, :ne], grads[:, ne:]
    mu = phi[ok].mean(0); w, V = np.linalg.eigh(np.cov(phi[ok], rowvar=False)); L, Linv = V * np.sqrt(w), (V / np.sqrt(w)).T
    u = (phi - mu) @ Linv.T; s = gp @ L; D = u.shape[1]                                     # s = L^T grad_phi = grad_u
    sg, se = ge[ok].std(0), e[ok].std(0); ges, ec = ge / sg, (e - e[ok].mean(0)) / se
    c2 = (np.eye(ne) / (se[:, None] * sg[None, :])).ravel()                                # E[e_k d_j log L | phi] = -delta_jk
    cv = lambda m: np.concatenate([ges[m], (ges[m][:, :, None] * u[m][:, None, :]).reshape(m.sum(), -1),
                                   (ec[m][:, :, None] * ges[m][:, None, :]).reshape(m.sum(), -1) + c2], 1)
    def split(k):                                                                            # (train, held-out) draws: whole chains
        h = np.random.default_rng(seed + k).permutation(nc)[:max(1, int(holdout * nc))]
        return ~np.isin(cid, h) & ok, np.isin(cid, h) & ok
    tr, ho = split(0); F = lambda m: np.concatenate([np.ones((m.sum(), 1)), u[m], cv(m)], 1)
    Th = np.linalg.lstsq(F(tr), s[tr], rcond=None)[0]; r = s[ho] - F(ho) @ Th
    A = Th[1:1 + D].T; A = 0.5 * (A + A.T); b = Th[0]
    s_c = s - cv(np.ones(len(u), bool)) @ Th[1 + D:]                                      # control-variate-corrected gradients
    wdir = jnp.array(1.0 / r.var(0))
    log(f"[score] {ok.sum()} draws, {D} dims: held-out gradient residual of the quadratic, per whitened direction: median "
        f"{np.median(r.std(0)):.3f}, max {r.std(0).max():.3f} (raw gradient sd median {np.median(s[ok].std(0)):.3f})")
    grad_g = jax.vmap(jax.grad(score_g, argnums=1), in_axes=(None, 0))
    loss = lambda p, uu, ss: jnp.mean(jnp.sum(wdir * (grad_g(p, uu) - ss)**2, 1))
    members, info = [], []
    for k in range(n_members):
        kk = jax.random.split(jax.random.key(seed + k), 3)
        p = {'b': jnp.array(b), 'A': jnp.array(A), 'w0': jax.random.normal(kk[0], (D, hidden)) / np.sqrt(D), 'b0': jnp.zeros(hidden),
             'w1': jax.random.normal(kk[1], (hidden, hidden)) / np.sqrt(hidden), 'b1': jnp.zeros(hidden),
             'w2': 1e-3 * jax.random.normal(kk[2], (hidden, 1)) / np.sqrt(hidden)}
        tr, ho = split(k); utr, str_, uho, sho = (jnp.array(a_) for a_ in (u[tr], s_c[tr], u[ho], s_c[ho]))
        ho_loss = jax.jit(lambda p: loss(p, uho, sho))
        opt = optax.chain(optax.clip_by_global_norm(10.0), optax.adam(optax.warmup_cosine_decay_schedule(0.0, lr, 500, steps, lr / 100)))
        st = opt.init(p)
        @jax.jit
        def step(p, st, key):
            i = jax.random.choice(key, utr.shape[0], (batch,))
            l, gr = jax.value_and_grad(loss)(p, utr[i], str_[i])
            up, st = opt.update(gr, st, p); return optax.apply_updates(p, up), st, l
        p0 = dict(p, w2=jnp.zeros_like(p['w2'])); l_quad = float(ho_loss(p0)); best = (l_quad, p0, 0); key = jax.random.key(1000 + seed + k)
        for it in range(1, steps + 1):
            key, kk_ = jax.random.split(key); p, st, _ = step(p, st, kk_)
            if it % 250 == 0:
                v = float(ho_loss(p))
                if v < best[0]: best = (v, p, it)
        members.append(best[1]); info.append((l_quad, best[0], best[2]))
        log(f"[score] member {k}: held-out weighted loss {best[0]:.2f} (quadratic alone {l_quad:.2f}; noise alone ~{D}), best step {best[2]}")
    return {'score': True, 'members': jax.tree_util.tree_map(lambda *a_: np.stack([np.asarray(v) for v in a_]), *members),
            'mu': mu, 'Linv': Linv, 'info': np.array(info)}


def load_or_fit_score(M, x, path, chains=(), force=False, **kw):  # kw -> fit_score
    """The score fit of an MI chain x (chains, draws, n_mi), cached as a pickle like the flows (refit when any of `chains` is
    newer, or with force); the gradients at the draws are cached next to it (<path minus .pkl>_gradients.npz)."""
    fresh = lambda f: os.path.exists(f) and not force and all(os.path.getmtime(f) > os.path.getmtime(c) for c in chains)
    if fresh(path):
        with open(path, 'rb') as fh: sc = pickle.load(fh)
        log(f"[score] loaded {path} ({len(sc['info'])} members)"); return sc
    fg = path[:-4] + '_gradients.npz'
    if fresh(fg): g = np.load(fg)['g']
    else: _, g = chain_gradients(M, x.reshape(-1, x.shape[-1])); np.savez(fg, g=g)
    sc = fit_score(M, x, g.reshape(x.shape), **{**SCORE, **kw})
    with open(path, 'wb') as fh: pickle.dump(sc, fh)
    log(f"[score] saved {path}"); return sc
