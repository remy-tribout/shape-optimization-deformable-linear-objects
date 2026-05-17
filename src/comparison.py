import numpy as np
import matplotlib.pyplot as plt

# Load
cos = np.load('cosserat_results.npz',        allow_pickle=True)
ms = np.load('mass_spring_results.npz', allow_pickle=True)

p_cos = cos['p'].T
n_L = cos['n_L']
m_L = cos['m_L']
targets = cos['targets']
p_ms = ms['p']

# Resample
def resample_to(p_src, n_out):
    diffs = np.diff(p_src, axis=0)
    seg_len = np.linalg.norm(diffs, axis=1)
    s_cum = np.concatenate([[0], np.cumsum(seg_len)])
    s_uni = np.linspace(0, s_cum[-1], n_out)
    out = np.zeros((n_out, 3))
    for d in range(3):
        out[:, d] = np.interp(s_uni, s_cum, p_src[:, d])
    return out

N = len(p_ms)
p_cos_r = resample_to(p_cos, N)

diff = np.linalg.norm(p_cos_r - p_ms, axis=1)
e_mean = diff.mean()
e_max = diff.max()
e_tip = np.linalg.norm(p_cos[-1] - p_ms[-1])

L_cos = np.sum(np.linalg.norm(np.diff(p_cos, axis=0), axis=1))
L_ms = np.sum(np.linalg.norm(np.diff(p_ms,  axis=0), axis=1))

def dist_to_curve(target, p_curve):
    return np.sqrt(np.min(np.sum((p_curve - target)**2, axis=1)))

err_cos_tgt = [dist_to_curve(t, p_cos) for t in targets]
err_ms_tgt = [dist_to_curve(t, p_ms)  for t in targets]

print(f"\n{'Metric':<35} {'Cosserat':>8}  {'M-S':>8}")
print("─" * 55)
print(f"{'Arc length [mm]':<35} {L_cos*1000:>8.2f}  {L_ms*1000:>8.2f}")
print(f"{'Tip position x [mm]':<35} {p_cos[-1,0]*1000:>8.2f}  {p_ms[-1,0]*1000:>8.2f}")
print(f"{'Tip position y [mm]':<35} {p_cos[-1,1]*1000:>8.2f}  {p_ms[-1,1]*1000:>8.2f}")
print(f"{'Tip position z [mm]':<35} {p_cos[-1,2]*1000:>8.2f}  {p_ms[-1,2]*1000:>8.2f}")

print(f"\n{'─'*55}")
print(f"Shape error (resampled to N={N} pts) :")
print(f"  Mean pointwise error : {e_mean*1000:.2f} mm")
print(f"  Max pointwise error  : {e_max*1000:.2f} mm")
print(f"  Free-end deviation   : {e_tip*1000:.2f} mm")

print(f"\nTarget errors [mm] :")
print(f"  {'Target':<10} {'Cosserat':>10}  {'M-S':>10}")
for i, (ec, em) in enumerate(zip(err_cos_tgt, err_ms_tgt)):
    print(f"  target {i+1:<4} {ec*1000:>10.2f}  {em*1000:>10.2f}")
print("=" * 55)

# Visualization
fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')

ax1.plot(p_cos[:,0], p_cos[:,1], p_cos[:,2],
         color='steelblue', lw=2.5, label='Cosserat')
ax1.scatter(p_cos[0,0],  p_cos[0,1],  p_cos[0,2],
            color='steelblue', s=60, marker='s', zorder=5)
ax1.scatter(p_cos[-1,0], p_cos[-1,1], p_cos[-1,2],
            color='steelblue', s=60, marker='D', zorder=5)

ax1.plot(p_ms[:,0], p_ms[:,1], p_ms[:,2],
         color='coral', lw=2, linestyle='--', label='Mass-spring')
ax1.scatter(p_ms[:,0], p_ms[:,1], p_ms[:,2],
            color='coral', s=25, zorder=4)
ax1.scatter(p_ms[-1,0], p_ms[-1,1], p_ms[-1,2],
            color='coral', s=60, marker='D', zorder=5)

ax1.scatter(targets[:,0], targets[:,1], targets[:,2],
            color='red', s=80, zorder=6, marker='^', label='Targets')

ax1.set_xlabel('x [m]'); ax1.set_ylabel('y [m]'); ax1.set_zlabel('z [m]')
ax1.set_title('Comparison')
ax1.legend(fontsize=8, loc='upper left')

all_pts = np.vstack([p_cos, p_ms, targets])
margin  = 0.03
for setter, idx in [(ax1.set_xlim, 0), (ax1.set_ylim, 1), (ax1.set_zlim, 2)]:
    setter(all_pts[:,idx].min() - margin, all_pts[:,idx].max() + margin)

# ── Écart point-à-point ───────────────────────────────────────────────────────
ax2 = fig.add_subplot(1, 2, 2)

s_norm = np.linspace(0, 1, N)
ax2.plot(s_norm, diff * 1000,
         color='darkorange', lw=2, marker='o', ms=4, label='Pointwise error')
ax2.axhline(e_mean * 1000, color='gray', lw=1, linestyle='--',
            label=f'Mean {e_mean*1000:.2f} mm')
ax2.fill_between(s_norm, 0, diff * 1000, alpha=0.15, color='darkorange')

for i, t in enumerate(targets):
    d_cos = np.linalg.norm(p_cos_r - t, axis=1)
    s_t   = s_norm[np.argmin(d_cos)]
    ax2.axvline(s_t, color='red', lw=0.8, linestyle=':', alpha=0.7)
    ax2.text(s_t + 0.01, diff.max()*1000*0.9, f'T{i+1}', color='red', fontsize=8)

ax2.set_xlabel('Normalized arc-length s/L')
ax2.set_ylabel('Deviation Cosserat − Masse-spring [mm]')
ax2.set_title(f'Shape deviation (mean={e_mean*1000:.2f} mm, max={e_max*1000:.2f} mm)')
ax2.legend(fontsize=9)
ax2.set_xlim(0, 1)
ax2.set_ylim(bottom=0)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cosserat_vs_mass_spring.png', dpi=150)
plt.show()