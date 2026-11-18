import trimesh
import numpy as np

scene = trimesh.load('unitree_rl_gym/resources/robots/g1_description/meshes/apartment_1.glb')
mesh = scene.to_geometry()
# Align ground floor to Z = 0
mesh.vertices[:, 2] -= (-1.65)

# Dining room floor bounds: X in [3.0, 5.5], Y in [-7.0, -4.5]
mask_dining = (mesh.vertices[:, 0] > 3.0) & (mesh.vertices[:, 0] < 5.5) & (mesh.vertices[:, 1] > -7.0) & (mesh.vertices[:, 1] < -4.5)
dining_z = mesh.vertices[mask_dining, 2]
floor_dining = np.percentile(dining_z[dining_z < 0.2], 5)
print(f"Dining room floor Z: {floor_dining:.3f}m")

# Corridor floor bounds: X in [2.0, 3.5], Y in [-4.0, -2.0]
mask_corridor = (mesh.vertices[:, 0] > 2.0) & (mesh.vertices[:, 0] < 3.5) & (mesh.vertices[:, 1] > -4.0) & (mesh.vertices[:, 1] < -2.0)
corr_z = mesh.vertices[mask_corridor, 2]
floor_corr = np.percentile(corr_z[corr_z < 0.2], 5)
print(f"Corridor floor Z: {floor_corr:.3f}m")

# Living room floor bounds: X in [1.5, 4.0], Y in [-2.0, 0.5]
mask_living = (mesh.vertices[:, 0] > 1.5) & (mesh.vertices[:, 0] < 4.0) & (mesh.vertices[:, 1] > -2.0) & (mesh.vertices[:, 1] < 0.5)
living_z = mesh.vertices[mask_living, 2]
floor_living = np.percentile(living_z[living_z < 0.2], 5)
print(f"Living room floor Z: {floor_living:.3f}m")
