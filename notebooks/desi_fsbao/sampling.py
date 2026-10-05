"""Optimizer, Hessian, preconditioner and NUTS (BlackJAX) used by the sampler and the analysis."""
import time

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

_T0 = time.time()
def log(msg):
    print(f"[{time.time()-_T0:7.1f}s] {msg}", flush=True)


# ------------------------------------------------------------------------------------------
# Fisher / precision linear algebra. pos_pinv gives a null direction ZERO weight (inside a
# marginalization); fisher_to_cov gives it a LARGE variance (Fisher -> covariance). Mixing the
# two up makes unconstrained directions look measured.
# ------------------------------------------------------------------------------------------
def pos_pinv(F, rtol=1e-12):
    F = 0.5 * (F + F.T); w, V = np.linalg.eigh(F)
    good = w > rtol * (float(np.abs(w).max()) or 1.0)
    return (V * np.where(good, 1.0 / np.where(good, w, 1.0), 0.0)) @ V.T

def fisher_to_cov(F, rtol=1e-10, big_var=1e10):
    F = 0.5 * (F + F.T); w, V = np.linalg.eigh(F)
    good = w > rtol * (float(np.abs(w).max()) or 1.0)
    return (V * np.where(good, 1.0 / np.where(good, w, 1.0), big_var)) @ V.T

def schur_marg(F, keep, marg):
    """Marginalize the indices `marg` out of F (Schur complement onto `keep`)."""
    return F[np.ix_(keep, keep)] - F[np.ix_(keep, marg)] @ pos_pinv(F[np.ix_(marg, marg)]) @ F[np.ix_(marg, keep)]

def psd_clip(M):
    M = 0.5 * (M + M.T); w, V = np.linalg.eigh(M)
    return (V * np.clip(w, 0.0, None)) @ V.T

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
            initial_step_size=0.1, verbose=True):
        bj = self.bj
        keys = jax.random.split(key, n_chains + 1)
        init_u = jitter * jax.random.normal(keys[-1], (n_chains, self.n))
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

        def run_one(k, u0):
            kw, ks = jax.random.split(k)
            (state, params), _ = warmup.run(kw, u0, num_steps=num_warmup)
            kernel = bj.nuts(self.logdensity_u, step_size=params['step_size'], inverse_mass_matrix=params['inverse_mass_matrix'],
                             max_num_doublings=max_num_doublings).step
            def step(st, kk):
                st, info = kernel(kk, st)
                return st, (st.position, st.logdensity, info.acceptance_rate, info.is_divergent, info.num_integration_steps)
            _, out = jax.lax.scan(step, state, jax.random.split(ks, num_samples))
            return out, params['step_size']

        t0 = time.time(); n_dev = jax.local_device_count()
        if n_dev > 1 and n_chains % n_dev == 0 and n_chains >= 2 * n_dev:
            per = n_chains // n_dev
            if verbose: log(f"[{self.name}] {n_chains} chains on {n_dev} devices ({per} per device, pmap x vmap)")
            out, step = jax.pmap(jax.vmap(run_one))(keys[:n_chains].reshape(n_dev, per), init_u.reshape(n_dev, per, self.n))
            out, step = [np.asarray(a).reshape((n_chains,) + np.asarray(a).shape[2:]) for a in out], np.asarray(step).reshape(n_chains)
        else:
            out, step = jax.jit(jax.vmap(run_one))(keys[:n_chains], init_u)
            out, step = [np.asarray(a) for a in out], np.asarray(step)
        pos, logp, acc, div, nsteps = out
        wall = time.time() - t0
        x = np.asarray(jax.vmap(jax.vmap(self.u_to_x))(jnp.array(pos)))
        if verbose:
            log(f"[{self.name}] {n_chains} chains x {num_samples} draws (+{num_warmup} warmup) in {wall/60:.1f} min; "
                f"accept {acc.mean():.3f}, divergent {div.mean():.4f}, leapfrog/iter {nsteps.mean():.1f}")
        return {'x': x, 'logp': logp, 'accept': acc, 'divergent': div, 'n_leapfrog': nsteps, 'step_size': step,
                'walltime': wall, 'num_warmup': num_warmup, 'num_samples': num_samples}


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
