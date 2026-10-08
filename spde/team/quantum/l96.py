import numpy as np
K, J = 8, 32
F, h, b, c = 20.0, 1.0, 10.0, 10.0
hcb = h * c / b
DT = 0.001

def tend(X, Y, Xfrozen=False):
    # X: (M,K), Y: (M,K*J) single ring (Wilks 2005 / Arnold et al 2013)
    S = Y.reshape(Y.shape[0], K, J).sum(-1)
    if Xfrozen:
        dX = np.zeros_like(X)
    else:
        dX = (-np.roll(X, 1, -1) * (np.roll(X, 2, -1) - np.roll(X, -1, -1))
              - X + F - hcb * S)
    dY = (-c * b * np.roll(Y, -1, -1) * (np.roll(Y, -2, -1) - np.roll(Y, 1, -1))
          - c * Y + hcb * np.repeat(X, J, axis=-1))
    return dX, dY

def rk4(X, Y, dt=DT, Xfrozen=False):
    k1x, k1y = tend(X, Y, Xfrozen)
    k2x, k2y = tend(X + .5*dt*k1x, Y + .5*dt*k1y, Xfrozen)
    k3x, k3y = tend(X + .5*dt*k2x, Y + .5*dt*k2y, Xfrozen)
    k4x, k4y = tend(X + dt*k3x, Y + dt*k3y, Xfrozen)
    return (X + dt/6*(k1x+2*k2x+2*k3x+k4x), Y + dt/6*(k1y+2*k2y+2*k3y+k4y))

def U_of(Y):
    return -hcb * Y.reshape(Y.shape[0], K, J).sum(-1)

def edge_flux(Y):
    # B_k = sum over sector k of advective term  -cb*Y_{n+1}(Y_{n+2}-Y_{n-1})
    adv = -c * b * np.roll(Y, -1, -1) * (np.roll(Y, -2, -1) - np.roll(Y, 1, -1))
    return adv.reshape(Y.shape[0], K, J).sum(-1)
