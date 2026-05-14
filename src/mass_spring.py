import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import time

# Parameters
L = 0.29
N = 12
l_s = L / (N - 1)

E = 5e5
h, b = 0.02, 0.02
A = b * h
Ix = b * h**3 / 12
k_s = E * A / l_s
k_b = E * Ix / l_s

p_encas = np.array([0.0, 0.0, 0.0])
t0_enc = np.array([0.0, 0.0, 1.0])

targets_final = np.array([
            [0.02, 0.02, 0.068],
            [0.07, 0.05, 0.120],
            [0.20, 0.10, 0.104],
        ])

# Weights
W1 = 1 
W2 = 100

# Resample
def resample_targets(points, N):
    from scipy.interpolate import CubicSpline
    t = np.linspace(0, 1, len(points))
    t_new = np.linspace(0, 1, N)
    cs_x = CubicSpline(t, points[:, 0])
    cs_y = CubicSpline(t, points[:, 1])
    cs_z = CubicSpline(t, points[:, 2])
    return np.vstack([cs_x(t_new), cs_y(t_new), cs_z(t_new)]).T

# Bending angle
def bending_angle(p_prev, p_n, p_next):
    v1 = p_n - p_prev
    v2 = p_next - p_n
    return np.arctan2(np.linalg.norm(np.cross(v1, v2)), np.dot(v1, v2))

# Mass spring energy
def equilibrium(p_inner_flat):
    p = np.zeros((N, 3))
    p[0] = p_encas
    p[1] = p_encas + t0_enc * l_s   # encastrement : direction imposée
    p[2:N] = p_inner_flat.reshape((N - 2, 3))

    Es = sum(0.5 * k_s * (np.linalg.norm(p[n] - p[n-1]) - l_s)**2
             for n in range(1, N))
    Eb = sum(0.5 * k_b * bending_angle(p[n-1], p[n], p[n+1])**2
             for n in range(1, N - 1))
    return Es + Eb

def solve_equilibrium(p_init_free):
    res = minimize(equilibrium, p_init_free, method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-9})
    p = np.zeros((N, 3))
    p[0] = p_encas
    p[1] = p_encas + t0_enc * l_s
    p[2:N] = res.x.reshape((N - 2, 3))
    return p

# Distance
def dist_targets(p, targets):
    return sum(np.min(np.sum((p - t)**2, axis=1)) for t in targets)


# Cost function
def cost(p_inner_flat, targets):
    p = np.zeros((N, 3))
    p[0] = p_encas
    p[1] = p_encas + t0_enc * l_s
    p[2:N] = p_inner_flat.reshape((N - 2, 3))

    E_elastic = equilibrium(p_inner_flat)

    E_targets = dist_targets(p, targets)

    return W1*E_elastic + W2*E_targets  

# ISG
def isg_3d(c0, s_des, lam):
    delta = np.linalg.norm(s_des - c0, axis=1)
    sigma = int(np.argmax(delta[1:N-1])) + 1
    delta_sigma = delta[sigma]
    K = max(2, int(np.floor(delta_sigma / lam)) + 1)
    print(f"\nISG : sigma={sigma}, delta_sigma={delta_sigma*1000:.1f} mm, K={K} shapes")

    def seg_angles(p_a, p_b):
        v = (p_b - p_a) / (np.linalg.norm(p_b - p_a) + 1e-12)
        return np.arctan2(v[1], v[0]), np.arcsin(np.clip(v[2], -1, 1))

    theta_c = np.zeros(N); phi_c = np.zeros(N)
    theta_s = np.zeros(N); phi_s = np.zeros(N)
    for n in range(1, N):
        theta_c[n], phi_c[n] = seg_angles(c0[n-1],    c0[n])
        theta_s[n], phi_s[n] = seg_angles(s_des[n-1], s_des[n])
    for n in range(N-2, -1, -1):
        theta_c[n], phi_c[n] = seg_angles(c0[n+1],    c0[n])
        theta_s[n], phi_s[n] = seg_angles(s_des[n+1], s_des[n])

    Theta_theta = (theta_s - theta_c + np.pi) % (2*np.pi) - np.pi
    Theta_phi = (phi_s   - phi_c   + np.pi) % (2*np.pi) - np.pi
    vartheta_theta = Theta_theta / (K - 1) if K > 1 else np.zeros(N)
    vartheta_phi = Theta_phi   / (K - 1) if K > 1 else np.zeros(N)

    shapes = []
    c_prev = c0.copy()
    for kappa in range(1, K):
        c_new = np.zeros_like(c_prev)
        direction = s_des[sigma] - c0[sigma]
        dist = np.linalg.norm(direction)
        c_new[sigma] = (c_prev[sigma] + lam * direction / dist
                        if dist > 1e-12 else c_prev[sigma])
        for n in range(sigma + 1, N):
            th = theta_c[n] + kappa * vartheta_theta[n]
            ph = phi_c[n]   + kappa * vartheta_phi[n]
            v = np.array([np.cos(ph)*np.cos(th), np.cos(ph)*np.sin(th), np.sin(ph)])
            c_new[n] = c_new[n-1] + l_s * v
        for n in range(sigma - 1, -1, -1):
            th = theta_c[n] + kappa * vartheta_theta[n]
            ph = phi_c[n]   + kappa * vartheta_phi[n]
            v = np.array([np.cos(ph)*np.cos(th), np.cos(ph)*np.sin(th), np.sin(ph)])
            c_new[n] = c_new[n+1] - l_s * v
        shapes.append(c_new.copy())
        c_prev = c_new.copy()

    shapes.append(s_des.copy())
    return shapes, sigma, K

# Tip pose
def tip_tangent(p):
    t = p[-1] - p[-2]
    return t / (np.linalg.norm(t) + 1e-12)

def tip_frame(p):
    t = tip_tangent(p)
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(t, ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    x = np.cross(ref, t);  x /= np.linalg.norm(x) + 1e-12
    y = np.cross(t, x)
    return np.column_stack((x, y, t))

def print_tip_pose(p, p_base):
    pos_tip, R_tip = p[-1] - p_base, tip_frame(p)
    T     = np.eye(4); T[:3,:3] = R_tip; T[:3,3] = pos_tip
    print("\n=== Tip pose ===")
    print(f"Position : {pos_tip.round(4)}")
    print(f"T :\n{T.round(4)}")
    return pos_tip, R_tip

# Main loop
s_desired = resample_targets(targets_final, N)
c0 = np.zeros((N, 3))
c0[:, 2] = np.linspace(0, L, N)

lam = 0.030
intermediary_shapes, sigma, K = isg_3d(c0, s_desired, lam)

targets_init = np.column_stack([
    np.zeros(len(targets_final)),
    np.zeros(len(targets_final)),
    targets_final[:, 2]
])

N_steps = len(intermediary_shapes)

# Initial condition
x = c0[2:].flatten()
t_start = time.time()

for k in range(N_steps):
    alpha = (k + 1) / N_steps
    targets_k = (1 - alpha) * targets_init + alpha * targets_final

    result = minimize(cost, x, args=(targets_k,), method='BFGS',
                      options={'maxiter': 300, 'gtol': 1e-6})
    x = result.x

    if k % max(1, N_steps//5) == 0 or k == N_steps - 1:
        p_k = np.zeros((N, 3))
        p_k[0] = p_encas
        p_k[1] = p_encas + t0_enc * l_s
        p_k[2:] = x.reshape((N-2, 3))
        e_mean = np.mean([np.sqrt(np.min(np.sum((p_k - t)**2, axis=1)))
                          for t in targets_final])
        print(f"step {k+1:2d}/{N_steps} (alpha={alpha:.2f}) : "
              f"e_mean={e_mean*1000:.2f} mm, p(L)={p_k[-1].round(3)}")

print(f"\nComputation time : {time.time() - t_start:.3f} s")

p_final = np.zeros((N, 3))
p_final[0] = p_encas
p_final[1] = p_encas + t0_enc * l_s
p_final[2:] = x.reshape((N-2, 3))

print_tip_pose(p_final, p_encas)


fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.plot(s_desired[:,0], s_desired[:,1], s_desired[:,2],
        'g--', lw=1.5, alpha=0.7, label='Goal (ISG)')
ax.plot(p_final[:,0], p_final[:,1], p_final[:,2],
        color='steelblue', lw=2, label='Rod (mass-spring)')
ax.scatter(p_final[:,0], p_final[:,1], p_final[:,2],
           color='steelblue', s=40, zorder=4)
ax.scatter(targets_final[:,0], targets_final[:,1], targets_final[:,2],
           color='red', s=80, zorder=5, label='Targets')

ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
ax.legend(fontsize=7)
ax.set_xlim(-0.05, 0.25)
ax.set_ylim(-0.15, 0.15)
ax.set_zlim(0.0, 0.30)
plt.tight_layout()
plt.savefig("mass_spring_result.png", dpi=150)
plt.show()

# Save
np.savez('resultats_mass_spring.npz',
         p       = p_final,
         targets = targets_final)