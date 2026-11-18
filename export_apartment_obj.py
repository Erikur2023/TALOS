import os
import trimesh
import numpy as np

print("Loading apartment_1.glb...")
scene = trimesh.load('unitree_rl_gym/resources/robots/g1_description/meshes/apartment_1.glb')
mesh = scene.to_geometry()

# Center ground floor at Z = 0
# Floor peak is at Z = -1.65m
floor_z = -1.65
mesh.vertices[:, 2] -= floor_z

# Let's save the 4K texture
dest_dir = "unitree_rl_gym/resources/robots/g1_description"
tex_path = os.path.join(dest_dir, "apartment_1_texture.png")
if hasattr(mesh.visual, 'material') and hasattr(mesh.visual.material, 'baseColorTexture') and mesh.visual.material.baseColorTexture:
    print("Saving 4K texture to apartment_1_texture.png...")
    mesh.visual.material.baseColorTexture.save(tex_path)
    print("Saved texture.")

# Simplify mesh slightly for fast physics/rendering if needed (e.g. 50,000 faces) or save full
# Let's see: full has 381k faces, let's decimate to 60k faces for blazing fast 500 Hz simulation in MuJoCo!
print(f"Original faces: {len(mesh.faces)}, vertices: {len(mesh.vertices)}")
print("Decimating to 60,000 faces for smooth 500 Hz simulation in MuJoCo...")
try:
    mesh_opt = mesh.simplify_quadric_decimation(60000)
    print(f"Decimated mesh: {len(mesh_opt.faces)} faces, {len(mesh_opt.vertices)} vertices")
except Exception as e:
    print(f"Decimation failed: {e}, using original")
    mesh_opt = mesh

out_obj = os.path.join(dest_dir, "meshes", "real_apartment_scan.obj")
print(f"Exporting to {out_obj}...")
mesh_opt.export(out_obj)
print("Successfully exported real_apartment_scan.obj!")
