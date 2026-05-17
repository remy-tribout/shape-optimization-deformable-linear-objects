import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Load
data    = np.load('cosserat_results.npz')
p_rod   = data['p']        
s       = data['s']        
n_L     = data['n_L']      
m_L     = data['m_L']      
R_L     = data['R_L']      
targets = data['targets'] 
p_L = p_rod[:, -1]


# Robot parameters (Franka Panda Emika)
L = 0.29

DH = [
    (0,       0.333,   0,         0),
    (0,       0,      -np.pi/2,   0),
    (0,       0.316,   np.pi/2,   0),
    (0.0825,  0,       np.pi/2,   0),
    (-0.0825, 0.384,  -np.pi/2,   0),
    (0,       0,       np.pi/2,   0),
    (0.088,   0.107,   np.pi/2,   0),
]

Q_LIM = np.array([
    [-2.8973,  2.8973],
    [-1.7628,  1.7628],
    [-2.8973,  2.8973],
    [-3.0718, -0.0698],
    [-2.8973,  2.8973],
    [-0.0175,  3.7525],
    [-2.8973,  2.8973],
])

# Rotation rod frame -> robot frame (to be justified according to the physical setup)
R_O_to_R = np.array([
    [ 0, -1,  0],
    [ 1,  0,  0],
    [ 0,  0,  1],
])


# Robot functions
def dh_transform(a, d, alpha, theta):
    sa, ca = np.sin(alpha), np.cos(alpha)
    st, ct = np.sin(theta), np.cos(theta)
    return np.array([
        [ct,    -st,     0,    a    ],
        [st*ca,  ct*ca, -sa,  -d*sa ],
        [st*sa,  ct*sa,  ca,   d*ca ],
        [0,      0,      0,    1    ]
    ])

def forward_kinematics(q):
    T  = np.eye(4)
    Ts = [T.copy()]
    for i in range(7):
        a, d, alpha, offset = DH[i]
        T = T @ dh_transform(a, d, alpha, q[i] + offset)
        Ts.append(T.copy())
    positions = np.array([Ti[:3, 3] for Ti in Ts])
    return T, Ts, positions

def jacobian(Ts):
    p_end = Ts[7][:3, 3]
    J = np.zeros((6, 7))
    for i in range(7):
        z        = Ts[i][:3, 2]
        p        = Ts[i][:3, 3]
        J[:3, i] = np.cross(z, p_end - p)
        J[3:, i] = z
    return J

def rotation_error(R_fk, R_target):
    return 0.5 * np.sum((R_fk.T @ R_target - np.eye(3))**2)

# Target in the robot frame
q_init = np.clip([0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4],
                        Q_LIM[:, 0], Q_LIM[:, 1])
T_init, _, _ = forward_kinematics(q_init)
p_TCP_init = T_init[:3, 3]

# Calibration : at q_init, the TCP is located at the tip of the straight rod
t_O_to_R = p_TCP_init - R_O_to_R @ np.array([0.0, 0.0, L])

# Target TCP position and orientation
p_TCP_target = R_O_to_R @ p_L + t_O_to_R
R_TCP_target = R_O_to_R @ R_L 

print(f"TCP target position  : {p_TCP_target.round(4)}")
print(f"TCP target orientation :\n{R_TCP_target.round(3)}")


# IK
lambda_rot = 1.0    # orientation vs position weighting
lambda_reg = 1e-3   # regularization (stay close to q_init)

def cost_ik(q):
    T, _, _ = forward_kinematics(q)
    pos_err = np.sum((T[:3, 3] - p_TCP_target)**2)
    rot_err = rotation_error(T[:3, :3], R_TCP_target)
    reg = np.sum((q - q_init)**2)
    return pos_err + lambda_rot * rot_err + lambda_reg * reg

res_ik = minimize(cost_ik, q_init, method='L-BFGS-B',
                  bounds=[(Q_LIM[i, 0], Q_LIM[i, 1]) for i in range(7)],
                  options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-9})

q = res_ik.x
T_final, Ts_final, positions_final = forward_kinematics(q)

# Error
err_pos   = np.linalg.norm(T_final[:3, 3] - p_TCP_target)
R_diff    = T_final[:3, :3].T @ R_TCP_target
angle_err = np.degrees(np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1)))

print(f"\nPosition error      : {err_pos*1000:.3f} mm")
print(f"Orientation error     : {angle_err:.3f} °")
print(f"Joint parameter (rad) : {np.round(q, 4)}")

# n_L, m_L are expressed in the material frame -> first transform them into the global rod frame
n_L_global = R_L @ n_L
m_L_global = R_L @ m_L

# Then into the robot frame (negative sign: reaction force/torque on the robot)
F_ext_robot = R_O_to_R @ (-n_L_global)
M_ext_robot = R_O_to_R @ (-m_L_global)

W_TCP = np.concatenate([F_ext_robot, M_ext_robot])
tau = jacobian(Ts_final).T @ W_TCP

print(f"\nForce TCP (robot)  : {F_ext_robot.round(5)}")
print(f"Moment TCP (robot) : {M_ext_robot.round(5)}")
print(f"Torques τ (N·m)    : {np.round(tau, 4)}")


# Visualization
p_rod_robot = (R_O_to_R @ p_rod).T + t_O_to_R

def draw_frame(ax, origin, R=np.eye(3), scale=0.05, labels=('x','y','z')):
    for i, (col, lbl) in enumerate(zip(['red','green','blue'], labels)):
        v = origin + scale * R[:, i]
        ax.quiver(*origin, *(v - origin), color=col, linewidth=2, arrow_length_ratio=0.25)
        ax.text(*v, f' {lbl}', color=col, fontsize=7)

fig = plt.figure(figsize=(11, 4.5))
fig.patch.set_facecolor('white')

# Rod frame
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot(p_rod[0], p_rod[1], p_rod[2], color='royalblue', linewidth=2.5, label='Rod')
ax1.scatter(targets[:,0], targets[:,1], targets[:,2], color='red', s=70, label='Targets')
ax1.scatter(*p_L, color='limegreen', s=90, label='Tip p(L)')
ax1.scatter(0, 0, 0, color='black', s=60, label='Clamp')
draw_frame(ax1, np.zeros(3), scale=0.04, labels=('x_rod','y_rod','z_rod'))
draw_frame(ax1, p_L, R_L, scale=0.04, labels=('x_tip','y_tip','z_tip'))  # orientation tip
ax1.set_xlabel('x'); ax1.set_ylabel('y'); ax1.set_zlabel('z')
ax1.set_title('Rod — rod frame', fontsize=10)
ax1.set_xlim(-0.05, 0.25)
ax1.set_ylim(-0.15, 0.15)  
ax1.set_zlim(0.0, 0.30)
ax1.legend(fontsize=7)

# Robot frame
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot(positions_final[:,0], positions_final[:,1], positions_final[:,2],
         '-o', color='royalblue', linewidth=2.5, markersize=5, label='Franka Panda')
ax2.plot(p_rod_robot[:,0], p_rod_robot[:,1], p_rod_robot[:,2],
         color='darkorange', linewidth=2, label='Rod')
ax2.scatter(*T_final[:3, 3], color='red', s=90, label='TCP')
ax2.scatter(*t_O_to_R, color='limegreen', s=80, label='Clamp')
ax2.scatter(0, 0, 0, color='black', s=60, label='Robot base')
draw_frame(ax2, np.zeros(3), scale=0.08, labels=('X_r','Y_r','Z_r'))
draw_frame(ax2, t_O_to_R, R_O_to_R, scale=0.06, labels=('x_rod','y_rod','z_rod'))
draw_frame(ax2, T_final[:3, 3], T_final[:3, :3], scale=0.05, labels=('x_tcp','y_tcp','z_tcp'))
ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
ax2.set_title('Panda + rod — robot frame', fontsize=10)
ax2.legend(fontsize=7)
ax2.set_box_aspect([1,1,1])
ax2.view_init(elev=25, azim=45)

plt.tight_layout()
plt.savefig('robot_arm_results.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()