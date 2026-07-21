import sys, numpy as np
sys.path.insert(0, '/data/masteryip/kimodo/kimodo/scripts')
from locomotion_framework.constraints import build_root2d_constraint

c = build_root2d_constraint({'vx': 0.5, 'vy': 0.0, 'wz': 0.0}, 5.0, fps=30)
root = np.array(c['smooth_root_2d'])
print(f'T={len(root)} X_end(lat)={root[-1,0]:.3f} Z_end(fwd)={root[-1,1]:.3f}')
print(f'Expected: X=0.0 (no lateral), Z=2.5 (forward at 0.5m/s * 5s)')
print(f'CORRECT' if abs(root[-1,0])<0.01 and abs(root[-1,1]-2.5)<0.01 else 'WRONG')
