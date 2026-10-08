import numpy as np
c1, c2 = 9.1e-4, 3.0e-10       # Lindborg 1999 mesoscale fit, E(k)=c1 k^-5/3 + c2 k^-3 (k in rad/m), per component
k = np.logspace(-7, 0, 400001)
E = c1*k**(-5/3) + c2*k**(-3)
def sinc(x): return np.sinc(x/np.pi)
U = 8.0
print("Residual variance per velocity component, point (time-averaged over T, Taylor length U*T) minus WRF cell")
print("WRF filter = sharp cutoff at 2pi/(ceff*Delta)")
for ceff in [1, 4, 7]:
  for T in [0, 600, 3600]:
    row=[]
    for D in [9e3, 3e3, 1e3]:
        HD = (k < 2*np.pi/(ceff*D)).astype(float)
        HT = sinc(k*U*T/2) if T>0 else np.ones_like(k)
        v = np.trapezoid(E*(HT-HD)**2, k)
        row.append(v)
    row=np.array(row)
    print(f"ceff={ceff} T={T:5d}s: var(9,3,1 km)= {np.round(row,3)} m2/s2 ; rms={np.round(np.sqrt(row),2)} ; ratios 9:3:1 = {np.round(row/row[-1],2)}")
# pure inertial check: point minus sharp cutoff: (3/2) c1 kc^(-2/3)
D=np.array([9e3,3e3,1e3]); kc=2*np.pi/(7*D)
print("analytic 1.5*c1*kc^-2/3 (ceff=7):", np.round(1.5*c1*kc**(-2/3),3), " Delta^(2/3) ratios:", np.round((D/1e3)**(2/3),2))
# van Kampen micro-turbulence term: sigma_u^2 * 2 tau / T
su, tau = 1.2, 30.
for T in [600, 3600]: print(f"microturb sampling var T={T}: {su**2*2*tau/T:.3f} m2/s2 (rms {np.sqrt(su**2*2*tau/T):.2f})")
