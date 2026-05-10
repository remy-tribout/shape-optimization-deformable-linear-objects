import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# PARAMETERS
L = 0.29
N = 12
l0 = L / (N - 1)

k_s   = 1000
k_b   = 1
k_enc = 100

p0 = np.array([0.0, 0.0, 0.0])
t0 = np.array([0.0, 0.0, 1.0])
p_end = np.array([0.20, 0.10, 0.104])

# ANGLE
def bending_angle(a, b, c):
    v1 = b - a
    v2 = c - b

    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)

    return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))

# ENERGY
def energy(x):

    p = np.zeros((N,3))

    p[0]  = p0
    p[-1] = p_end
    p[1:-1] = x.reshape((N-2,3))

    Es = 0
    for i in range(N-1):
        d = np.linalg.norm(p[i+1] - p[i])
        Es += 0.5 * k_s * (d - l0)**2

    Eb = 0
    for i in range(1, N-1):
        Eb += 0.5 * k_b * bending_angle(p[i-1], p[i], p[i+1])**2

    seg0 = p[1] - p[0]
    t = seg0 / (np.linalg.norm(seg0) + 1e-12)
    Eenc = 0.5 * k_enc * np.linalg.norm(t - t0)**2

    return Es + Eb + Eenc

# INITIALIZATION
p_init = np.zeros((N,3))
for d in range(3):
    p_init[:,d] = np.linspace(p0[d], p_end[d], N)

# OPTIMIZATION
res = minimize(energy, p_init[1:-1].flatten(), method='L-BFGS-B')

# RESULT
p = np.zeros((N,3))
p[0] = p0
p[-1] = p_end
p[1:-1] = res.x.reshape((N-2,3))

# PLOT
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.plot(p[:,0], p[:,1], p[:,2], '-o', lw=2, label="Mass-spring")

ax.scatter(p_end[0], p_end[1], p_end[2],
           color='red', s=120, label='End target')

targets = np.array([
    [0.02, 0.02, 0.068],
    [0.07, 0.05, 0.120],
    [0.20, 0.10, 0.104],
])

ax.scatter(targets[:,0], targets[:,1], targets[:,2],
           color='green', s=80, label='Targets')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax.legend()
plt.show()