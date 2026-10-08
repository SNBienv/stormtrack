"""Task 4: tail asymptotics of S = sum_j w_j X_j, X_j ~ Weibull(k, lam_j) independent, k > 1.

  -log P(S>s) = min{ sum_j (x_j/lam_j)^k : sum_j w_j x_j = s } + O(log s) = (s/Lam)^k + O(log s),
  Lam = ( sum_j (w_j lam_j)^q )^{1/q},  q = k/(k-1)   (Hoelder conjugate)
  sharp:  P(S>s) ~ C s^{(n-1)k/2} exp(-(s/Lam)^k)   with C from Laplace's method (computed below).
"""
import numpy as np, mpmath as mp
import saddlepoint_closure as S
mp.mp.dps = 40

def sharp_tail(s, k, lam, w):
    lam, w = np.asarray(lam, float), np.asarray(w, float)
    a = w * lam; q = k / (k - 1); D = np.sum(a ** q)
    Lam = D ** (1 / q)
    c = lam * a ** (1 / (k - 1)) / D                      # x_j* = c_j s
    x = c * s
    I = (s / Lam) ** k
    g = k * x ** (k - 1) / lam ** k
    H = k * (k - 1) * x ** (k - 2) / lam ** k
    n = len(w)
    fS = np.prod(g) * np.exp(-I) * (2 * np.pi) ** ((n - 1) / 2) / np.sqrt(np.prod(H) * np.sum(w ** 2 / H))
    return fS / (k * s ** (k - 1) / Lam ** k), Lam

def exact_tail_2(s, k, lam, w):
    """P(w1 X1 + w2 X2 > s) by one-dimensional quadrature in high precision."""
    k = mp.mpf(k); l1, l2 = map(mp.mpf, lam); w1, w2 = map(mp.mpf, w); s = mp.mpf(s)
    f1 = lambda x: k / l1 * (x / l1) ** (k - 1) * mp.exp(-(x / l1) ** k)
    Fb2 = lambda y: mp.exp(-(y / l2) ** k) if y > 0 else mp.mpf(1)
    xm = s / w1
    # integrand peaks near the large-deviation optimum: split there
    a = np.array([w1 * l1, w2 * l2], float); qq = float(k / (k - 1))
    xs = float(l1 * a[0] ** (1 / (float(k) - 1)) / np.sum(a ** qq)) * float(s)
    pts = sorted(set([0, xs * 0.5, xs, min(xs * 1.5, float(xm)), float(xm)]))
    return mp.quad(lambda x: f1(x) * Fb2((s - w1 * x) / w2), pts) + mp.exp(-(xm / l1) ** k)

if __name__ == "__main__":
    k, lam, w = 2.0, [7.0, 9.0], [0.6, 0.4]
    print("Two Weibulls, k=2, lam=(7,9), w=(.6,.4):  Lam =", sharp_tail(1, k, lam, w)[1])
    print("   s     exact P(S>s)      sharp asymptotic    ratio    LR (saddlepoint) ratio   -logP/(s/Lam)^k")
    mc = S.MixedCGF(np.array([w]), np.array([k, k]), np.array(lam))
    for s in [8, 12, 16, 24, 32, 48]:
        ex = exact_tail_2(s, k, lam, w)
        asy, Lam = sharp_tail(s, k, lam, w)
        # LR at q = s: solve K'(t) = s
        from scipy.optimize import brentq
        t = brentq(lambda t: mc.cgf(np.array([[t]]), nder=1)[1][0, 0] - s, -5, 50)
        K, K1, K2, K3 = [v[0, 0] for v in mc.cgf(np.array([[t]]), nder=3)]
        lr, _ = S.lr_tail(np.array(t), np.array(s), K, K2, K3)
        print(f"  {s:3d}  {mp.nstr(ex,10):>16}  {asy:16.9e}  {float(asy/ex):.5f}   {float(lr/ex):.6f}        {float(-mp.log(ex))/(s/Lam)**k:.4f}")
    k, lam, w = 1.7, [7.0, 9.0], [0.6, 0.4]
    print("Two Weibulls, k=1.7:")
    for s in [16, 32, 64]:
        ex = exact_tail_2(s, k, lam, w); asy, Lam = sharp_tail(s, k, lam, w)
        print(f"  {s:3d}  exact {mp.nstr(ex,8):>14}  asympt {asy:.8e}  ratio {float(asy/ex):.5f}")

    # mixed grid point after 40 steps: true tail exponent vs the exp-Weibull fit
    print("\nGrid points x=0.25 (i=50) and x=0.75 (i=150) after 40 steps: far-tail quantiles")
    P1 = S.step(np.eye(S.N)); W = np.linalg.matrix_power(P1, 40)
    for i0 in [50, 150]:
        mcs = S.MixedCGF(W[[i0]], S.K0, S.L0)
        probs = np.array([0.999, 1 - 1e-5, 1 - 1e-7, 1 - 1e-9])
        q_lr2 = S.lr_quantiles(mcs, probs, order=2)[0][0]
        q_ex = S.exact_quantiles(mcs, probs[:2])[0]
        q_ew, par = S.ew_fit_quantiles(mcs.kappa, probs)
        print(f"  point {i0}: sources' k in [{mcs.k.min():.2f},{mcs.k.max():.2f}], exp-Weibull fitted k={par[0,0]:.2f}, a={par[0,2]:.2f}")
        print("      p          LR2       exact(GP)    expWeibull  (EW-LR2)/LR2")
        for j, p in enumerate(probs):
            ex = f"{q_ex[j]:10.4f}" if j < 2 else "      --  "
            print(f"   1-{1-p:.0e}  {q_lr2[j]:10.4f}  {ex}  {q_ew[0,j]:10.4f}   {100*(q_ew[0,j]/q_lr2[j]-1):+6.2f}%")
