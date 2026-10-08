"""
Intrinsic noise on a fully deterministic system: two-scale Lorenz-96.

The truth is deterministic. 8 resolved variables X_k (the "WRF grid") each drive 32 fast
variables Y_{j,k} (the sub-grid eddies the model cannot see):

    dX_k/dt = -X_{k-1}(X_{k-2} - X_{k+1}) - X_k + F + U_k,     U_k = -(h c / b) sum_j Y_{j,k}
    dY_j/dt = -c b Y_{j+1}(Y_{j+2} - Y_{j-1}) - c Y_j + (h c / b) X_{k(j)}
    F = 20, h = 1, b = c = 10, K = 8, J = 32     (Wilks 2005; Arnold, Moroz & Palmer 2013)

Nothing random is added anywhere. The "noise" of a coarse model of X is U, the unresolved part of
the system itself. Its conditional law given the resolved state, and its memory, are MEASURED
from the truth (the role of a met mast). Nothing is tuned.

Coarse models of X (time step 0.005), all with the same resolved dynamics:
  C0  no sub-grid term                       (deterministic PDE without S-)
  C1  U = E[U | X], a cubic fitted to truth  (best deterministic closure: unbiased pointwise)
  C2  C1 + Gaussian AR(1), same variance and lag-1 memory as the residual (classic additive)
  C3  C1 + intrinsic residual: drawn from the measured CONDITIONAL distribution of
      e = U - E[U|X] given X (quantile tables), with its memory carried in rank space
      (Gaussian copula AR(1)), so the law of e at every X is the measured one, non-Gaussian.

Questions:
  1. Does a coarse model whose sub-grid term is exact in the mean (C1) still drift in climate?
     That is "the bias comes from the missing S-": a nonlinear model rectifies the fluctuations.
  2. Does putting the unresolved part back (C3) remove that bias and the forecast error?
  3. Exact MSE split (bias^2, std, correlation) of each model against the truth.
"""
import time
import numpy as np
from scipy import stats

K, J, F, h, b, c = 8, 32, 20.0, 1.0, 10.0, 10.0
HCB = h * c / b
DT_TRUTH, SAMPLE_EVERY = 0.001, 5
DT = DT_TRUTH * SAMPLE_EVERY                    # coarse step = sampling interval = 0.005
rng = np.random.default_rng(96)


# ---------------------------------------------------------------- truth
def tend_truth(X, Y):
    Ysum = Y.reshape(Y.shape[0], K, J).sum(-1)
    dX = -np.roll(X, 1, 1) * (np.roll(X, 2, 1) - np.roll(X, -1, 1)) - X + F - HCB * Ysum
    dY = (-c * b * np.roll(Y, -1, 1) * (np.roll(Y, -2, 1) - np.roll(Y, 1, 1)) - c * Y
          + HCB * np.repeat(X, J, axis=1))
    return dX, dY


def rk4_truth(X, Y, dt):
    a1, b1 = tend_truth(X, Y)
    a2, b2 = tend_truth(X + 0.5 * dt * a1, Y + 0.5 * dt * b1)
    a3, b3 = tend_truth(X + 0.5 * dt * a2, Y + 0.5 * dt * b2)
    a4, b4 = tend_truth(X + dt * a3, Y + dt * b3)
    return X + dt / 6 * (a1 + 2 * a2 + 2 * a3 + a4), Y + dt / 6 * (b1 + 2 * b2 + 2 * b3 + b4)


def run_truth(n_traj, spin_mtu, rec_mtu):
    X = rng.normal(0, 1, (n_traj, K)) + 5
    Y = rng.normal(0, 0.1, (n_traj, K * J))
    for _ in range(int(round(spin_mtu / DT_TRUTH))):
        X, Y = rk4_truth(X, Y, DT_TRUTH)
    n_rec = int(round(rec_mtu / DT))
    Xs = np.empty((n_rec, n_traj, K))
    Us = np.empty((n_rec, n_traj, K))
    for t in range(n_rec):
        Xs[t] = X
        Us[t] = -HCB * Y.reshape(n_traj, K, J).sum(-1)
        for _ in range(SAMPLE_EVERY):
            X, Y = rk4_truth(X, Y, DT_TRUTH)
    return Xs, Us                                   # (time, trajectory, k)


# ---------------------------------------------------------------- closures, measured from truth
class Measured:
    """Everything the coarse models know about the unresolved part, measured on training data."""

    def __init__(self, Xs, Us, n_bins=20, n_q=401):
        x, u = Xs.ravel(), Us.ravel()
        self.poly = np.polyfit(x, u, 3)                         # E[U | X], cubic
        e = Us - np.polyval(self.poly, Xs)
        # conditional quantile tables of e given X
        edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
        self.centres = 0.5 * (edges[1:] + edges[:-1])
        self.probs = (np.arange(n_q) + 0.5) / n_q
        ib = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, n_bins - 1)
        er = e.ravel()
        self.qtab = np.array([np.quantile(er[ib == i], self.probs) for i in range(n_bins)])
        self.sd_bin = np.array([er[ib == i].std() for i in range(n_bins)])
        self.skew_bin = np.array([stats.skew(er[ib == i]) for i in range(n_bins)])
        # Gaussianised ranks within each bin -> memory of the residual in rank space
        z = np.empty_like(er)
        for i in range(n_bins):
            m = ib == i
            z[m] = stats.norm.ppf((stats.rankdata(er[m]) - 0.5) / m.sum())
        z = z.reshape(e.shape)
        self.rho_z = float(np.mean(z[1:] * z[:-1]) / np.mean(z * z))
        self.sigma_e = float(e.std())
        ec = e - e.mean()
        self.rho_e = float(np.mean(ec[1:] * ec[:-1]) / np.mean(ec * ec))
        # spatial correlation of the residual between neighbouring k (reported, not used)
        self.rho_k = float(np.mean(z * np.roll(z, 1, axis=2)))
        lags = np.arange(0, 201, 10)
        self.acf_z = [float(np.mean(z[l:] * z[:z.shape[0] - l]) / np.mean(z * z)) for l in lags]
        self.acf_lags = lags * DT

    def det(self, X):
        return np.polyval(self.poly, X)

    def cond_quantile(self, X, p):
        """Quantile p of e given X: linear interpolation between bin centres, in X and in p."""
        xb = np.clip(X, self.centres[0], self.centres[-1])
        i = np.clip(np.searchsorted(self.centres, xb) - 1, 0, len(self.centres) - 2)
        w = (xb - self.centres[i]) / (self.centres[i + 1] - self.centres[i])
        pi = np.interp(p, self.probs, np.arange(self.probs.size))
        j0 = np.clip(np.floor(pi).astype(int), 0, self.probs.size - 2)
        f = pi - j0
        q_lo = self.qtab[i, j0] * (1 - f) + self.qtab[i, j0 + 1] * f
        q_hi = self.qtab[i + 1, j0] * (1 - f) + self.qtab[i + 1, j0 + 1] * f
        return q_lo * (1 - w) + q_hi * w


def tend_coarse(X, U):
    return -np.roll(X, 1, -1) * (np.roll(X, 2, -1) - np.roll(X, -1, -1)) - X + F + U


def run_coarse(X0, n_steps, model, meas, record_every=1, z0=None):
    """Integrate a coarse model; the sub-grid term is held fixed over each RK4 step."""
    X = X0.copy()
    z = rng.standard_normal(X.shape) if z0 is None else z0
    eps = np.zeros_like(X)
    out = []
    for n in range(n_steps):
        if n % record_every == 0:
            out.append(X.copy())
        if model == "C0":
            U = 0.0
        else:
            U = meas.det(X)
            if model == "C2":
                eps = meas.rho_e * eps + np.sqrt(1 - meas.rho_e ** 2) * meas.sigma_e * rng.standard_normal(X.shape)
                U = U + eps
            elif model == "C3":
                z = meas.rho_z * z + np.sqrt(1 - meas.rho_z ** 2) * rng.standard_normal(X.shape)
                U = U + meas.cond_quantile(X, stats.norm.cdf(z))
        k1 = tend_coarse(X, U)
        k2 = tend_coarse(X + 0.5 * DT * k1, U)
        k3 = tend_coarse(X + 0.5 * DT * k2, U)
        k4 = tend_coarse(X + DT * k3, U)
        X = X + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        X = np.where(np.isfinite(X), X, np.nan)
    out.append(X.copy())
    return np.array(out)


def mse_split(obs, mod):
    o, m = np.ravel(obs), np.ravel(mod)
    ok = np.isfinite(o) & np.isfinite(m)
    o, m = o[ok], m[ok]
    bias, so, sm, r = m.mean() - o.mean(), o.std(), m.std(), np.corrcoef(o, m)[0, 1]
    mse = np.mean((m - o) ** 2)
    return dict(rmse=np.sqrt(mse), bias2=bias ** 2 / mse, std=(sm - so) ** 2 / mse,
                corr=2 * sm * so * (1 - r) / mse, bias=bias)


# ================================================================= run
if __name__ == "__main__":
    MODELS = ("C0", "C1", "C2", "C3")
    NAMES = {"C0": "C0 no sub-grid term", "C1": "C1 E[U|X] (deterministic)",
             "C2": "C2 C1 + Gaussian AR(1)", "C3": "C3 C1 + intrinsic residual"}

    t0 = time.perf_counter()
    Xtr, Utr = run_truth(n_traj=48, spin_mtu=5, rec_mtu=60)       # training ("mast record")
    Xte, Ute = run_truth(n_traj=32, spin_mtu=5, rec_mtu=60)       # independent test truth
    print(f"truth: {Xtr.size + Xte.size:,} resolved samples in {time.perf_counter() - t0:.0f} s")

    meas = Measured(Xtr, Utr)
    e_all = Utr - meas.det(Xtr)
    print("\n=== What the unresolved part looks like (measured on the training truth) ===")
    print(f"E[U|X] cubic: {np.array2string(meas.poly, precision=4)}   (U = sub-grid tendency)")
    print(f"residual e = U - E[U|X]: sd {meas.sigma_e:.3f}, skew {stats.skew(e_all.ravel()):+.3f}, "
          f"excess kurtosis {stats.kurtosis(e_all.ravel()):+.3f}")
    print(f"conditional sd of e across X bins: {meas.sd_bin.min():.3f} .. {meas.sd_bin.max():.3f} "
          f"(state-dependent: ratio {meas.sd_bin.max() / meas.sd_bin.min():.1f})")
    print(f"conditional skewness of e across X bins: {meas.skew_bin.min():+.2f} .. {meas.skew_bin.max():+.2f}")
    print(f"memory: lag-{DT} autocorrelation of e {meas.rho_e:.3f}, of its Gaussianised rank {meas.rho_z:.3f}"
          f" (e-folding time {-DT / np.log(meas.rho_z):.3f} MTU)")
    print("rank autocorrelation vs lag (MTU): " + "  ".join(
        f"{l:.2f}:{a:.2f}" for l, a in zip(meas.acf_lags[::4], meas.acf_z[::4])))
    print(f"neighbour (k, k-1) correlation of the residual ranks: {meas.rho_k:+.3f}")

    # ------------------------------------------------ climate
    print("\n=== Climate: 300 coarse runs x 50 MTU from truth states, vs test truth ===")
    t0 = time.perf_counter()
    starts = Xte[-1][np.arange(300) % Xte.shape[1]] + rng.normal(0, 0.01, (300, K))
    xt = Xte.ravel()
    lo, hi = np.quantile(xt, [0.0005, 0.9995])
    bins = np.linspace(lo - 2, hi + 2, 81)
    pt, _ = np.histogram(xt, bins, density=True)
    clim = {}
    for mname in MODELS:
        traj = run_coarse(starts, int(55 / DT), mname, meas, record_every=10)
        xs = traj[int(5 / DT / 10):].ravel()
        xs = xs[np.isfinite(xs)]
        pm, _ = np.histogram(xs, bins, density=True)
        hell = np.sqrt(0.5 * np.sum((np.sqrt(pm) - np.sqrt(pt)) ** 2 * np.diff(bins)))
        lag = int(0.5 / (DT * 10))
        xa = traj[int(5 / DT / 10):]
        xa = xa - np.nanmean(xa)
        ac = np.nanmean(xa[lag:] * xa[:-lag]) / np.nanmean(xa * xa)
        clim[mname] = dict(mean=xs.mean(), sd=xs.std(), skew=stats.skew(xs), q99=np.quantile(xs, 0.99),
                           hell=hell, ac=ac)
    xa = Xte - Xte.mean()
    lag = int(0.5 / DT)
    ac_t = np.mean(xa[lag:] * xa[:-lag]) / np.mean(xa * xa)
    print(f"{'model':<30}{'mean':>8}{'sd':>8}{'skew':>8}{'99th pct':>10}{'acf(0.5)':>10}{'Hellinger':>11}")
    print(f"{'truth':<30}{xt.mean():>8.3f}{xt.std():>8.3f}{stats.skew(xt):>8.3f}{np.quantile(xt, 0.99):>10.3f}"
          f"{ac_t:>10.3f}{0:>11.3f}")
    for mname in MODELS:
        r = clim[mname]
        print(f"{NAMES[mname]:<30}{r['mean']:>8.3f}{r['sd']:>8.3f}{r['skew']:>8.3f}{r['q99']:>10.3f}"
              f"{r['ac']:>10.3f}{r['hell']:>11.3f}")
    print(f"({time.perf_counter() - t0:.0f} s)")

    # ------------------------------------------------ forecasts
    print("\n=== Forecasts from perfect resolved initial states (sub-grid state unknown) ===")
    t0 = time.perf_counter()
    leads = (0.2, 0.5, 1.0, 1.5, 2.0)
    n_lead = int(round(max(leads) / DT))
    step0 = np.arange(0, Xte.shape[0] - n_lead - 1, int(1.0 / DT))
    ics = Xte[step0]                                          # (n_start, traj, K)
    truth_fc = np.stack([Xte[step0 + int(round(L / DT))] for L in leads])   # (lead, n_start, traj, K)
    ics = ics.reshape(-1, K)
    truth_fc = truth_fc.reshape(len(leads), -1, K)
    n_ic, n_ens = ics.shape[0], 20
    print(f"{n_ic} forecasts, {n_ens} members each for C2/C3")
    res = {}
    for mname in MODELS:
        ens = n_ens if mname in ("C2", "C3") else 1
        X0 = np.repeat(ics, ens, axis=0)
        traj = run_coarse(X0, n_lead, mname, meas)
        fc = np.stack([traj[int(round(L / DT))] for L in leads]).reshape(len(leads), n_ic, ens, K)
        em = np.nanmean(fc, axis=2)
        err = em - truth_fc
        rmse = np.sqrt(np.nanmean(err ** 2, axis=(1, 2)))
        spread = np.sqrt(np.nanmean(np.nanvar(fc, axis=2, ddof=1), axis=(1, 2))) if ens > 1 else np.full(len(leads), np.nan)
        # spread-error consistency: rmse of the mean vs spread corrected for finite ensemble
        ratio = spread / rmse * np.sqrt((ens + 1) / ens) if ens > 1 else np.full(len(leads), np.nan)
        # CRPS of the ensemble (fair estimator not needed for comparison at fixed size)
        if ens > 1:
            o = truth_fc[:, :, None, :]
            t1 = np.nanmean(np.abs(fc - o), axis=2)
            t2 = np.nanmean(np.abs(fc[:, :, :, None, :] - fc[:, :, None, :, :]), axis=(2, 3))
            crps = np.nanmean(t1 - 0.5 * t2, axis=(1, 2))
        else:
            crps = np.nanmean(np.abs(fc[:, :, 0, :] - truth_fc), axis=(1, 2))
        split = mse_split(truth_fc[2], em[2])
        res[mname] = dict(rmse=rmse, ratio=ratio, crps=crps, split=split)
    print(f"{'model':<30}" + "".join(f"{'RMSE ' + str(L):>11}" for L in leads))
    for mname in MODELS:
        print(f"{NAMES[mname]:<30}" + "".join(f"{v:>11.3f}" for v in res[mname]["rmse"]))
    print(f"\n{'spread / error (1 = reliable)':<30}" + "".join(f"{'lead ' + str(L):>11}" for L in leads))
    for mname in ("C2", "C3"):
        print(f"{NAMES[mname]:<30}" + "".join(f"{v:>11.2f}" for v in res[mname]["ratio"]))
    print(f"\n{'CRPS (lower = better)':<30}" + "".join(f"{'lead ' + str(L):>11}" for L in leads))
    for mname in MODELS:
        print(f"{NAMES[mname]:<30}" + "".join(f"{v:>11.3f}" for v in res[mname]["crps"]))
    print(f"\nExact MSE split at lead {leads[2]} MTU (ensemble mean vs truth), shares of MSE:")
    print(f"{'model':<30}{'RMSE':>8}{'bias':>9}{'bias^2':>9}{'std':>9}{'corr':>9}")
    for mname in MODELS:
        s = res[mname]["split"]
        print(f"{NAMES[mname]:<30}{s['rmse']:>8.3f}{s['bias']:>+9.3f}{s['bias2']:>9.3f}{s['std']:>9.3f}{s['corr']:>9.3f}")
    print(f"({time.perf_counter() - t0:.0f} s)")
