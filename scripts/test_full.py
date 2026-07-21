"""Minimal 1-sample generation with constraint tracing."""
import sys, numpy as np, torch
sys.path.insert(0, '/data/masteryip/kimodo/kimodo/scripts')
sys.path.insert(0, '/data/masteryip/kimodo/kimodo')

from locomotion_framework.constraints import build_root2d_constraint
from kimodo import load_model
from kimodo.constraints import load_constraints_lst

vx, vy, wz = 0.5, 0.0, 0.0
dur, fps = 5.0, 30

c_dict = build_root2d_constraint({'vx': vx, 'vy': vy, 'wz': wz}, dur, fps=30)
root_c = np.array(c_dict['smooth_root_2d'])
print(f'Constraint: {len(root_c)} frames, X_end={root_c[-1,0]:.3f} Z_end={root_c[-1,1]:.3f}')
print(f'  X=lateral (should be 0), Z=forward (should be ~{vx*dur:.1f})')

device = 'cuda:0'
model, name = load_model('kimodo-g1-rp', device=device, return_resolved_name=True)
k_constraints = load_constraints_lst([c_dict], model.skeleton, device=device)

num_frames = int(dur * fps)
output = model(
    "a robot walks forward normally.",
    num_frames,
    constraint_lst=k_constraints,
    num_denoising_steps=100,
    num_samples=1,
    return_numpy=True,
)

rp = output['root_positions']   # (T, 3)
sr = output['smooth_root_pos']  # (T, 3)

k2m = np.array([[0,0,1],[1,0,0],[0,1,0]], dtype=np.float32)
rp_mj = rp @ k2m.T
sr_mj = sr @ k2m.T

dx_rp = rp_mj[-1,0] - rp_mj[0,0]
dy_rp = rp_mj[-1,1] - rp_mj[0,1]
dx_sr = sr_mj[-1,0] - sr_mj[0,0]
dy_sr = sr_mj[-1,1] - sr_mj[0,1]

print(f'\nGenerated T={rp.shape[0]}')
print(f'  root_positions MJ:  start=({rp_mj[0,0]:.3f},{rp_mj[0,1]:.3f}) end=({rp_mj[-1,0]:.3f},{rp_mj[-1,1]:.3f})')
print(f'  smooth_root MJ:     start=({sr_mj[0,0]:.3f},{sr_mj[0,1]:.3f}) end=({sr_mj[-1,0]:.3f},{sr_mj[-1,1]:.3f})')
print(f'  root DX_fwd={dx_rp:.3f}m DY_left={dy_rp:.3f}m')
print(f'  smooth_root DX_fwd={dx_sr:.3f}m DY_left={dy_sr:.3f}m')
print(f'  Expected:             DX_fwd={vx*dur:.1f}m DY_left=0.0m')

# Check smooth_root follows constraint
t_arr = np.arange(rp.shape[0]) * (1/fps)
expected_x = vx * t_arr
sr_resid = (sr_mj[:,0] - sr_mj[0,0]) - expected_x
print(f'\nSmooth root constraint residual X: max={np.abs(sr_resid).max():.4f}m')
print(f'Constraint is', 'FOLLOWED' if np.abs(sr_resid).max() < 0.15 else 'NOT FOLLOWED')

del model; torch.cuda.empty_cache()
