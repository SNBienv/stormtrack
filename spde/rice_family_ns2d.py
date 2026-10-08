"""
Rice vector family solve on 2-D Navier-Stokes: the unresolved wind is derived, not injected.

Truth: 2-D Navier-Stokes (vorticity form), pseudo-spectral, 256^2, steady Kolmogorov forcing
F = (F0 sin(n_f y), 0), linear drag. Fully deterministic: nothing random anywhere.

The "model" (the role of WRF) holds the wind averaged over B x B cells. The question is the law of
the point wind speed a mast would measure inside a cell. The family solve carries, per cell,

    V_res  = the resolved (cell-mean) wind vector         [from the coarse solver]
    sigma^2 = unresolved variance per component          [derived from V_res, no tuning]

and the speed law is Rice(|V_res|, sigma): the exact law of |V_res + v'| for v' ~ N(0, sigma^2 I).

sigma^2 has two parts, both derived from the system itself:
  1. sub-cell variance, the exact leading term of what a cell average hides: a field that varies
     linearly across a square cell of side D has within-cell variance D^2/12 |grad u|^2, so
         sigma_sub^2 = c_dyn (D^2 / 24) (|grad u_res|^2 + |grad v_res|^2)    (isotropic share)
     c_dyn is not tuned: it comes from the resolved field itself (Germano's dynamic procedure).
     At the test scale 2D the true answer IS resolved (the spread of the four cell means inside
     each 2x2 block); its ratio to the same gradient formula at 2D gives c_dyn, assuming scale
     similarity. c_dyn = 1 recovers the pure Taylor term.
  2. error of the resolved state at lead t. Short leads: the unresolved scales act on the
     resolved ones and the error grows like Green-Kubo, MSE(t) = 2 int_0^t (t - s) C_r(s) ds
     (team/quantum). Longer leads: chaos of the resolved scales themselves, which is not
     isotropic noise around the forecast: the truth regresses towards climatology. Both are
     MEASURED on a separate TRAINING period as a regression of the true cell wind on the
     forecast, V ~ a(t) V_fc + (1 - a(t)) V_clim, plus the residual variance sigma_err^2(t).

Scored on an independent TEST period against the true point speeds inside every cell:
  deterministic        speed = |V_res|                         (a point mass, no S-)
  injected, uniform    Rice(|V_res|, sigma_c(t)), sigma_c(t) the domain mean of the same total
                       variance, measured on the training period: right amount, not state-dependent
  derived, Taylor      Rice(|V_res|, sqrt(sigma_sub^2 (c_dyn = 1) + raw error variance(t)))
  derived (family)     Rice(|a V_res + (1-a) V_clim|, sqrt(c_dyn sigma_sub^2 + sigma_err^2(t)))
  derived + floor      derived family plus a uniform floor for what the derivation misses,
                       measured on the training period (true minus derived sub-cell variance):
                       the role of a mast when energy enters below the grid
  oracle sub-cell      Rice with the TRUE sub-cell variance of each cell (lead 0 only)

Regimes (--regime):
  smooth   forcing n_f = 4 (resolved); enstrophy-cascade spectrum, steep: sub-cell = smooth
  rough    forcing n_f = 40, inside the cells: energy is injected at unresolved scales, which no
           closure built only from the resolved field can see (the honest stress test)

    python rice_family_ns2d.py --regime smooth
    python rice_family_ns2d.py --regime rough
"""
import argparse
import time
import numpy as np
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("--regime", default="smooth", choices=("smooth", "rough"))
ap.add_argument("--block", type=int, default=8)
args = ap.parse_args()

N, L = 256, 2 * np.pi
B = args.block
NC = N // B
D = L / NC
if args.regime == "smooth":
    NF, F0, NU, MU, DT = 4, 1.0, 1e-3, 0.05, 0.005
else:
    NF, F0, NU, MU, DT = 40, 8.0, 2e-4, 0.05, 0.002
rng = np.random.default_rng(7)
LEADS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)


class Spectral:
    def __init__(self, n, nu, hyper=0.0):
        self.n = n
        k = np.fft.fftfreq(n, 1 / n)
        kr = np.fft.rfftfreq(n, 1 / n)
        self.KX, self.KY = np.meshgrid(kr, k)
        self.K2 = self.KX ** 2 + self.KY ** 2
        self.K2i = np.where(self.K2 > 0, 1 / np.maximum(self.K2, 1e-12), 0)
        self.mask = (np.abs(self.KX) < n / 3) & (np.abs(self.KY) < n / 3)
        y = np.arange(n) * L / n
        Y = np.meshgrid(y, y)[1]
        # the forcing exists on this grid only if it is resolved; sampling cos(NF y) on a coarser
        # grid would alias it onto a resolved wavenumber (40 on 32 points -> 8)
        resolved = NF < n / 3
        self.fw = np.fft.rfft2(-NF * F0 * np.cos(NF * Y)) * self.mask if resolved else np.zeros_like(self.K2, complex)
        self.lin = nu * self.K2 + hyper * self.K2 ** 4

    def vel(self, wh):
        ph = wh * self.K2i
        n = self.n
        return np.fft.irfft2(1j * self.KY * ph, s=(n, n)), np.fft.irfft2(-1j * self.KX * ph, s=(n, n))

    def rhs(self, wh):
        n = self.n
        u, v = self.vel(wh)
        wx = np.fft.irfft2(1j * self.KX * wh, s=(n, n))
        wy = np.fft.irfft2(1j * self.KY * wh, s=(n, n))
        return -np.fft.rfft2(u * wx + v * wy) * self.mask - MU * wh + self.fw

    def stepper(self, dt):
        E, E2 = np.exp(-self.lin * dt), np.exp(-self.lin * dt / 2)

        def step(wh):
            a = self.rhs(wh)
            b = self.rhs(E2 * (wh + dt / 2 * a))
            c = self.rhs(E2 * wh + dt / 2 * b)
            d = self.rhs(E * wh + dt * E2 * c)
            return E * wh + dt / 6 * (E * a + 2 * E2 * (b + c) + d)
        return step

    def from_velocity(self, u, v):
        uh, vh = np.fft.rfft2(u), np.fft.rfft2(v)
        return (1j * self.KX * vh - 1j * self.KY * uh) * self.mask


fine = Spectral(N, NU)
# coarse model: same equations on the cell grid; hyperviscosity damps the grid scale at rate ~1
kmax_c = NC / 3
coarse = Spectral(NC, NU, hyper=1.0 / kmax_c ** 8)
step_f = fine.stepper(DT)
step_c = coarse.stepper(DT * 2)


def block(a):
    return a.reshape(NC, B, NC, B).mean((1, 3))


def cells_points(a):
    """(NC*NC, B*B) array: the fine points inside every cell."""
    return a.reshape(NC, B, NC, B).transpose(0, 2, 1, 3).reshape(NC * NC, B * B)


def grad2(f):
    """|grad f|^2 on the coarse periodic grid, centred differences."""
    fx = (np.roll(f, -1, 1) - np.roll(f, 1, 1)) / (2 * D)
    fy = (np.roll(f, -1, 0) - np.roll(f, 1, 0)) / (2 * D)
    return fx ** 2 + fy ** 2


def sigma_sub2(ub, vb):
    return D ** 2 / 24 * (grad2(ub) + grad2(vb))


def c_dynamic(ub, vb):
    """Germano-style coefficient from the resolved field: at the test scale 2D the spread of the
    four cell means in each 2x2 block is known; compare it with the gradient formula there.
    For four cell centres at +-D/2 sampling a linear field, the spread is D^2/4 |grad|^2."""
    def t2(a):
        return a.reshape(NC // 2, 2, NC // 2, 2).mean((1, 3))
    ub2, vb2 = t2(ub), t2(vb)
    actual = (0.5 * ((ub - np.repeat(np.repeat(ub2, 2, 0), 2, 1)) ** 2
                     + (vb - np.repeat(np.repeat(vb2, 2, 0), 2, 1)) ** 2)).mean()

    def g2(f):
        d2 = 2 * D
        fx = (np.roll(f, -1, 1) - np.roll(f, 1, 1)) / (2 * d2)
        fy = (np.roll(f, -1, 0) - np.roll(f, 1, 0)) / (2 * d2)
        return fx ** 2 + fy ** 2
    pred = (0.5 * D ** 2 / 4 * (g2(ub2) + g2(vb2))).mean()
    return actual / pred


def rice_mean(nu, sig):
    """Stable Rice mean (scaled Bessel functions), also for nu / sigma >> 1."""
    from scipy import special
    x = -nu ** 2 / (2 * sig ** 2)
    return sig * np.sqrt(np.pi / 2) * ((1 - x) * special.i0e(-x / 2) - x * special.i1e(-x / 2))


def run_fine(wh, T, every):
    out = []
    n_every = int(round(every / DT))
    for n in range(int(round(T / DT)) + 1):
        if n % n_every == 0:
            out.append(wh.copy())
        wh = step_f(wh)
    return out, wh


def coarse_forecast(ub, vb, leads):
    wc = coarse.from_velocity(ub, vb)
    out, t, dtc = {}, 0.0, DT * 2
    for L_ in leads:
        while t < L_ - 1e-9:
            wc = step_c(wc)
            t += dtc
        out[L_] = coarse.vel(wc)
    return out


# ----------------------------------------------------------------------------- truth
t0 = time.perf_counter()
INTERVAL = 6.0                                                       # between forecast starts
N_TRAIN, N_TEST = 12, 12
import os
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", f"ns2d_truth_{args.regime}.npy")
if os.path.exists(CACHE):
    snaps = list(np.load(CACHE).astype(np.complex128))
else:
    wh = np.fft.rfft2(0.1 * rng.standard_normal((N, N))) * fine.mask
    _, wh = run_fine(wh, 40.0, 40.0)                                # spin-up
    snaps, wh = run_fine(wh, INTERVAL * (N_TRAIN + N_TEST) + max(LEADS), 0.25)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.save(CACHE, np.array(snaps).astype(np.complex64))
print(f"[{args.regime}] truth: {N}^2, n_f = {NF}, cells {NC}^2 of {B}x{B} points "
      f"({time.perf_counter() - t0:.0f} s)", flush=True)


def truth_at(t):
    return fine.vel(snaps[int(round(t / 0.25))])


u0, v0 = truth_at(0.0)
k = np.sqrt(fine.K2).round().astype(int)
Ek = 0.5 * (np.abs(fine.KY * snaps[0] * fine.K2i) ** 2 + np.abs(fine.KX * snaps[0] * fine.K2i) ** 2)
spec = np.bincount(k.ravel(), (Ek * np.where(fine.KX == 0, 1, 2)).ravel())
kc = NC / 2
slope = np.polyfit(np.log(np.arange(4, int(kc))), np.log(spec[4:int(kc)]), 1)[0]
print(f"rms speed {np.sqrt((u0 ** 2 + v0 ** 2).mean()):.2f}; resolved-band spectral slope {slope:.2f}")

# ----------------------------------------------------------------------------- training: error growth
# climatological cell wind (time mean of the training period; the Kolmogorov mean flow)
clim_u = np.mean([block(truth_at(t)[0]) for t in np.arange(0, N_TRAIN * INTERVAL, 1.0)], axis=0)
clim_v = np.mean([block(truth_at(t)[1]) for t in np.arange(0, N_TRAIN * INTERVAL, 1.0)], axis=0)
pairs = {L_: [] for L_ in LEADS}
for s in range(N_TRAIN):
    ts = s * INTERVAL
    u, v = truth_at(ts)
    fc = coarse_forecast(block(u), block(v), LEADS)
    for L_ in LEADS:
        ut, vt = truth_at(ts + L_)
        pairs[L_].append((fc[L_][0] - clim_u, fc[L_][1] - clim_v, block(ut) - clim_u, block(vt) - clim_v))
reg_a, sig_err2, sig_err2_raw = {}, {}, {}
for L_ in LEADS:
    f = np.concatenate([np.r_[p[0].ravel(), p[1].ravel()] for p in pairs[L_]])
    o = np.concatenate([np.r_[p[2].ravel(), p[3].ravel()] for p in pairs[L_]])
    reg_a[L_] = float(np.dot(f, o) / np.dot(f, f))
    sig_err2[L_] = float(np.mean((o - reg_a[L_] * f) ** 2))
    sig_err2_raw[L_] = float(np.mean((o - f) ** 2))
# sub-cell part: how well do the derived formulas match the truth (training)?
true_sub, pred_sub, cdyn = [], [], []
for s in range(N_TRAIN):
    u, v = truth_at(s * INTERVAL)
    ub, vb = block(u), block(v)
    du = cells_points(u) - ub.reshape(-1, 1)
    dv = cells_points(v) - vb.reshape(-1, 1)
    true_sub.append(0.5 * (du ** 2 + dv ** 2).mean(1))
    pred_sub.append(sigma_sub2(ub, vb).ravel())
    cdyn.append(c_dynamic(ub, vb))
true_sub, pred_sub = np.concatenate(true_sub), np.concatenate(pred_sub)
print("\n=== Derived sub-cell variance vs truth (training period, per cell) ===")
print(f"domain mean: true {true_sub.mean():.4f}, Taylor D^2/24|grad V|^2 {pred_sub.mean():.4f} "
      f"(ratio {pred_sub.mean() / true_sub.mean():.2f}); cell-by-cell correlation "
      f"{np.corrcoef(true_sub, pred_sub)[0, 1]:.3f}; log-log slope "
      f"{np.polyfit(np.log(pred_sub + 1e-12), np.log(true_sub + 1e-12), 1)[0]:.2f}")
print(f"dynamic coefficient from the resolved field: c_dyn = {np.mean(cdyn):.2f} (range {min(cdyn):.2f}-{max(cdyn):.2f})"
      f" -> derived/true ratio {np.mean(cdyn) * pred_sub.mean() / true_sub.mean():.2f} (needed: "
      f"{true_sub.mean() / pred_sub.mean():.2f})")
# what the derivation misses, measured on the training period (the role of a mast): uniform floor
floor_sub = float(max(true_sub.mean() - np.mean(cdyn) * pred_sub.mean(), 0.0))
print(f"measured floor (true minus derived sub-cell variance, training mean): {floor_sub:.4f}")
print("resolved-wind error per component (training), by lead: regression slope a, residual variance"
      " (raw variance without the regression)")
print("   " + "  ".join(f"{L_:g}: a={reg_a[L_]:.2f} {sig_err2[L_]:.4f} ({sig_err2_raw[L_]:.4f})" for L_ in LEADS))


# ----------------------------------------------------------------------------- test
def score(speeds, nu, sig, n_crps=16, m_draws=48):
    """PIT stats and CRPS of Rice(nu, sig) per cell against the true point speeds in the cell."""
    sig = np.maximum(sig, 1e-4)[:, None]
    pit = stats.rice.cdf(speeds, nu[:, None] / sig, scale=sig)
    idx = rng.choice(speeds.shape[1], n_crps, replace=False)
    obs = speeds[:, idx]
    z = rng.standard_normal((2, m_draws))
    draws = np.hypot(nu[:, None] + sig * z[0], sig * z[1])               # (cells, draws)
    t1 = np.abs(draws[:, None, :] - obs[:, :, None]).mean(-1).mean()
    t2 = np.abs(draws[:, :, None] - draws[:, None, :]).mean()
    return pit.ravel(), t1 - 0.5 * t2


rows = {}
models = ("deterministic", "injected, uniform", "derived, Taylor", "derived (family)", "derived + measured floor",
          "oracle sub-cell")
for L_ in LEADS:
    acc = {m: dict(pit=[], crps=[], bias=[], gap=[]) for m in models}
    for s in range(N_TRAIN, N_TRAIN + N_TEST):
        ts = s * INTERVAL
        u, v = truth_at(ts)
        fc = coarse_forecast(block(u), block(v), (L_,))[L_]
        ub, vb = fc
        ut, vt = truth_at(ts + L_)
        speeds = cells_points(np.hypot(ut, vt))
        nu = np.hypot(ub, vb).ravel()
        s_sub = sigma_sub2(ub, vb).ravel()
        cd = c_dynamic(ub, vb)
        a_ = reg_a[L_]
        nu_reg = np.hypot(a_ * ub + (1 - a_) * clim_u, a_ * vb + (1 - a_) * clim_v).ravel()
        s_tay = np.sqrt(s_sub + sig_err2_raw[L_])
        s_der = np.sqrt(cd * s_sub + sig_err2[L_])
        s_uni = np.full_like(nu, np.sqrt(true_sub.mean() + sig_err2_raw[L_]))
        s_hyb = np.sqrt(cd * s_sub + floor_sub + sig_err2[L_])
        cand = {"injected, uniform": (nu, s_uni), "derived, Taylor": (nu, s_tay),
                "derived (family)": (nu_reg, s_der), "derived + measured floor": (nu_reg, s_hyb)}
        if L_ == 0.0:
            dtu = cells_points(ut) - block(ut).reshape(-1, 1)
            dtv = cells_points(vt) - block(vt).reshape(-1, 1)
            cand["oracle sub-cell"] = (nu, np.sqrt(0.5 * (dtu ** 2 + dtv ** 2).mean(1)))
        acc["deterministic"]["crps"].append(np.abs(speeds - nu[:, None]).mean())
        acc["deterministic"]["bias"].append((speeds.mean(1) - nu).mean())
        for m, (nu_m, sg) in cand.items():
            pit, cr = score(speeds, nu_m, sg)
            acc[m]["pit"].append(pit)
            acc[m]["crps"].append(cr)
            acc[m]["bias"].append((speeds.mean(1) - rice_mean(nu_m, np.maximum(sg, 1e-4))).mean())
    rows[L_] = acc

print("\n=== Test period: law of the point speed inside each cell, given the model's resolved wind ===")
print("PIT calibration (ideal: KS 0, central-90% coverage 0.90, >q99 0.010); CRPS lower = better;")
print("bias = mean point speed minus the law's mean speed (deterministic: minus |V_res|).")
print(f"{'lead':>5} {'model':<20}{'KS':>7}{'cov90':>8}{'>q99':>8}{'CRPS':>8}{'bias':>9}")
for L_ in LEADS:
    for m in models:
        a = rows[L_][m]
        if not a["crps"]:
            continue
        if a["pit"]:
            p = np.concatenate(a["pit"])
            ks = stats.kstest(p[::7], "uniform").statistic
            cov, up = np.mean((p > 0.05) & (p < 0.95)), np.mean(p > 0.99)
            print(f"{L_:>5g} {m:<20}{ks:>7.3f}{cov:>8.3f}{up:>8.3f}{np.mean(a['crps']):>8.4f}{np.mean(a['bias']):>+9.4f}")
        else:
            print(f"{L_:>5g} {m:<20}{'':>7}{'':>8}{'':>8}{np.mean(a['crps']):>8.4f}{np.mean(a['bias']):>+9.4f}")
    print()
print(f"total {time.perf_counter() - t0:.0f} s")
