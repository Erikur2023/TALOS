import numpy as np

verts = []
faces = []
v_indices = set()

obj_path = 'unitree_rl_gym/resources/robots/g1_description/meshes/seaside_villa.obj'
with open(obj_path, 'r', encoding='utf-8', errors='ignore') as f:
    cur_g = 'default'
    for line in f:
        if line.startswith('v '):
            verts.append([float(x) for x in line.strip().split()[1:4]])
        elif line.startswith('o ') or line.startswith('g '):
            cur_g = line.strip().split()[1]
        elif line.startswith('f '):
            if cur_g not in ['cocos', 'Color200', 'Color201', 'Color202', 'Color203', 'Color09', 'zip08']:
                f_pts = []
                for p in line.strip().split()[1:]:
                    v_idx = int(p.split('/')[0]) - 1
                    f_pts.append(v_idx)
                    v_indices.add(v_idx)
                faces.append(f_pts)

verts = np.array(verts)

# Compute bounds
used_v_list = sorted(list(v_indices))
old_to_new = {old: new for new, old in enumerate(used_v_list)}
filtered_v = verts[used_v_list]

# Scale from cm to meters (0.01)
# Coordinate transform: X_new = -(Z_old - max(Z_old)), Y_new = X_old - center(X_old), Z_new = Y_old - min(Y_old)
z_max = np.max(filtered_v[:, 2]) # Front facade
x_center = (np.min(filtered_v[:, 0]) + np.max(filtered_v[:, 0])) / 2.0
y_min = np.min(filtered_v[:, 1])

new_verts = np.zeros_like(filtered_v)
# In MuJoCo: X is forward into the house (0 to 10m)
new_verts[:, 0] = -(filtered_v[:, 2] - z_max) * 0.01
# Y is lateral (-6.5m to +6.5m)
new_verts[:, 1] = (filtered_v[:, 0] - x_center) * 0.01
# Z is vertical (0 to 7.7m)
new_verts[:, 2] = (filtered_v[:, 1] - y_min) * 0.01

print(f"Processed Real Villa: {len(new_verts)} vertices, {len(faces)} faces.")
print(f"Bounds in meters (X_fwd, Y_lat, Z_up):")
print(f"  X: {np.min(new_verts[:, 0]):.2f}m .. {np.max(new_verts[:, 0]):.2f}m")
print(f"  Y: {np.min(new_verts[:, 1]):.2f}m .. {np.max(new_verts[:, 1]):.2f}m")
print(f"  Z: {np.min(new_verts[:, 2]):.2f}m .. {np.max(new_verts[:, 2]):.2f}m")

# Write out clean, unified Wavefront OBJ
out_obj = 'unitree_rl_gym/resources/robots/g1_description/meshes/real_villa_architecture.obj'
with open(out_obj, 'w', encoding='utf-8') as f:
    f.write("# Real Architectural Seaside Villa from internet (Clara.io / Jason Shoumar)\n")
    for v in new_verts:
        f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
    for f_pts in faces:
        if len(f_pts) == 3:
            f.write(f"f {old_to_new[f_pts[0]]+1} {old_to_new[f_pts[1]]+1} {old_to_new[f_pts[2]]+1}\n")
        elif len(f_pts) == 4:
            f.write(f"f {old_to_new[f_pts[0]]+1} {old_to_new[f_pts[1]]+1} {old_to_new[f_pts[2]]+1}\n")
            f.write(f"f {old_to_new[f_pts[0]]+1} {old_to_new[f_pts[2]]+1} {old_to_new[f_pts[3]]+1}\n")

print(f"Saved {out_obj} successfully.")
