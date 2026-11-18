import os
import trimesh
import numpy as np
from PIL import Image

print("Loading skokloster_castle.glb...")
scene = trimesh.load('unitree_rl_gym/resources/robots/g1_description/meshes/skokloster_castle.glb')
mesh = scene.to_geometry()

# Save downscaled texture for fast loading in MuJoCo (2048x2048)
mat = mesh.visual.material
tex_img = mat.baseColorTexture
print("Texture original size:", tex_img.size)
tex_2k = tex_img.resize((2048, 2048), Image.Resampling.LANCZOS)
dest_tex = "unitree_rl_gym/resources/robots/g1_description/skokloster_texture_2k.png"
tex_2k.save(dest_tex)
print(f"Saved {dest_tex}")

# Export OBJ
dest_obj = "unitree_rl_gym/resources/robots/g1_description/meshes/skokloster_castle.obj"
print("Exporting OBJ...")
mesh.export(dest_obj)
print(f"Exported {dest_obj}")
