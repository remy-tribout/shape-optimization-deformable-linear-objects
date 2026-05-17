import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time

# Parameters
L = 0.29
E = 5e5
nu = 0.5
G = E/(2*(1+nu))
h, b = 0.02, 0.02
A = b * h
rho = 1392.0
g = 9.81
Ix = b * h**3 / 12
Iy = h * b**3 / 12
J = Ix + Iy

K_se = np.diag([G*A, G*A, E*A])
K_bt = np.diag([E*Ix, E*Iy, G*J])
K_se_inv = np.linalg.inv(K_se)
K_bt_inv = np.linalg.inv(K_bt)

v_ref = np.array([0.0, 0.0, 1.0])
u_ref = np.array([0.0, 0.0, 0.0])
f_ext = np.array([0.0, 0.0, -rho*A*g])

# Targets
TARGETS = {
    1: {
        "name": "Base shape",
        "targets": np.array([
            [0.02, 0.02, 0.068],
            [0.07, 0.05, 0.120],
            [0.20, 0.10, 0.104],
        ])
    },
    2: {
        "name": "C-shape",
        "targets": np.array([
            [0.04, 0.00, 0.08],
            [0.10, 0.00, 0.14],
            [0.08, 0.00, 0.22],
        ])
    },
    3: {
        "name": "S-shape",
        "targets": np.array([
            [0.05,  0.03, 0.08],
            [0.10, -0.04, 0.15],
            [0.15,  0.05, 0.22],
        ])
    },
    4: {
        "name": "J-shape",
        "targets": np.array([
            [0.02, 0.01, 0.10],
            [0.06, 0.02, 0.18],
            [0.12, 0.03, 0.24],
        ])
    },
}

# Select target
SCENARIO = 1 

targets_final = TARGETS[SCENARIO]["targets"]
targets_init  = np.column_stack([
    np.zeros(len(targets_final)),   
    np.zeros(len(targets_final)),   
    targets_final[:, 2]             
])

print(f"Scénario {SCENARIO} : {TARGETS[SCENARIO]['name']}")
print(f"targets_final :\n{targets_final}")

# Weights
W1 = 1
W2 = 100

def hat(v):
    return np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])

# Cosserat equations
def cosserat_ode(s, y):
    R = y[3:12].reshape((3,3))
    n, m = y[12:15], y[15:18]
    v = v_ref + K_se_inv @ (R.T @ n)
    u = u_ref + K_bt_inv @ (R.T @ m)
    dpds = R @ v
    dRds = R @ hat(u)
    dnds = -f_ext
    dmds = -np.cross(dpds, n)
    return np.concatenate([dpds, dRds.flatten(), dnds, dmds])

# Integration
def integrate_rod(n0, m0):
    y0  = np.concatenate([np.zeros(3), np.eye(3).flatten(), n0, m0])
    return solve_ivp(cosserat_ode, [0, L], y0, t_eval=np.linspace(0,L,200),
                     method='RK45', rtol=1e-8, atol=1e-10)

# Elastic energy
def elastic_energy(sol):
    s, y = sol.t, sol.y
    integrand = np.zeros(len(s))
    for i in range(len(s)):
        R  = y[3:12,i].reshape((3,3))
        v  = v_ref + K_se_inv @ (R.T @ y[12:15,i])
        u  = u_ref + K_bt_inv @ (R.T @ y[15:18,i])
        dv, du = v - v_ref, u - u_ref
        integrand[i] = dv @ K_se @ dv + du @ K_bt @ du
    return (W1 / (2*L)) * np.trapz(integrand, s)

# Distance
def dist_targets(sol, targets):
    p = sol.y[0:3,:].T
    return sum(np.min(np.sum((p - t)**2, axis=1)) for t in targets)

# Cost function
def cost(x, targets):
    sol = integrate_rod(x[0:3], x[3:6])
    return elastic_energy(sol) + W2 * dist_targets(sol, targets)

# Homotopy
N_steps = 20
x = np.zeros(6)   
shapes_to_plot = {}

t_start = time.time()
for k in range(N_steps + 1):
    alpha = k / N_steps
    targets_k = (1 - alpha) * targets_init + alpha * targets_final
    result = minimize(cost, x, args=(targets_k,), method='BFGS',
                      options={'maxiter': 200, 'gtol': 1e-6})
    x = result.x
    if k % 5 == 0:
        sol_k = integrate_rod(x[:3], x[3:])
        shapes_to_plot[alpha] = sol_k.y[0:3, :]
        print(f"step {k:2d}/{N_steps} (alpha={alpha:.2f}): cost={result.fun:.5f}")

print(f"\nComputation time: {time.time() - t_start:.3f} s")
print("\nn(0) =", result.x[0:3])
print("m(0) =", result.x[3:6])

sol = integrate_rod(result.x[0:3], result.x[3:6])
p = sol.y[0:3,:]

# Visualization
fig = plt.figure()
ax  = fig.add_subplot(projection='3d')

cmap   = plt.cm.Blues
alphas = sorted(shapes_to_plot.keys())
for i, alpha in enumerate(alphas[:-1]):
    p_k = shapes_to_plot[alpha]
    ax.plot(p_k[0], p_k[1], p_k[2],
            color=cmap(0.3 + 0.6 * i / len(alphas)),
            linewidth=1.8, linestyle='--', alpha=0.85,
            label=f'α={alpha:.2f}')

p_final = shapes_to_plot[alphas[-1]]
ax.plot(p_final[0], p_final[1], p_final[2],
        color='royalblue', linewidth=3, label='Final shape')

ax.scatter(targets_final[:,0], targets_final[:,1], targets_final[:,2],
           color='red', s=60, zorder=5, label='Targets')

ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
ax.set_xlim(-0.05, 0.25)
ax.set_ylim(-0.15, 0.15)
ax.set_zlim(0.0, 0.30)
ax.legend(fontsize=7)
plt.savefig("cosserat_result.png", dpi=150)
plt.show()

# Save
n_L = sol.y[12:15, -1]
m_L = sol.y[15:18, -1]
R_L = sol.y[3:12, -1].reshape((3, 3))

F_global = R_L @ n_L     
M_global = R_L @ m_L 

np.savez('cosserat_results.npz',
         p        = sol.y[0:3, :],
         s        = sol.t,
         n_L      = n_L,
         m_L      = m_L,
         R_L      = R_L,
         M_global = M_global,
         targets  = targets_final)