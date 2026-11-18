import os
import struct
import numpy as np
from scipy.spatial import ConvexHull
from PIL import Image

MESH_DIR = r"c:\Users\fredb\Documents\Hackathon\unitree_rl_gym\resources\robots\g1_description\rubble_assets"
os.makedirs(MESH_DIR, exist_ok=True)

def write_binary_stl(filepath, points, simplices):
    with open(filepath, 'wb') as f:
        f.write(b'Binary STL generated for Realistic Rubble Scene'.ljust(80, b'\0'))
        f.write(struct.pack('<I', len(simplices)))
        for s in simplices:
            pts = points[s]
            v1 = pts[1] - pts[0]
            v2 = pts[2] - pts[0]
            norm = np.cross(v1, v2)
            l = np.linalg.norm(norm)
            if l > 0:
                norm /= l
            f.write(struct.pack('<3f', float(norm[0]), float(norm[1]), float(norm[2])))
            for p in pts:
                f.write(struct.pack('<3f', float(p[0]), float(p[1]), float(p[2])))
            f.write(struct.pack('<H', 0))

def generate_jagged_rock(filepath, scale=(0.4, 0.4, 0.35), seed=101, n_pts=50):
    np.random.seed(seed)
    phi = np.random.uniform(0, 2*np.pi, n_pts)
    costheta = np.random.uniform(-1, 1, n_pts)
    theta = np.arccos(costheta)
    # irregular fractures
    r = 1.0 + 0.45 * np.random.uniform(-1, 1, n_pts)
    # flatten one side to make it sit naturally on the ground
    x = r * np.sin(theta) * np.cos(phi) * scale[0]
    y = r * np.sin(theta) * np.sin(phi) * scale[1]
    z = (r * np.cos(theta)) * scale[2]
    # Shift so bottom is near 0
    z -= np.min(z)
    points = np.stack([x, y, z], axis=1)
    hull = ConvexHull(points)
    write_binary_stl(filepath, points, hull.simplices)
    print(f"Generated {os.path.basename(filepath)}: {len(hull.simplices)} facets")

def generate_broken_beam(filepath, length=1.2, width=0.25, height=0.25, seed=202):
    np.random.seed(seed)
    pts = []
    # 8 corners of a beam + jagged fractures at the ends
    for x in np.linspace(-length/2, length/2, 6):
        for y in [-width/2, width/2]:
            for z in [0, height]:
                jitter = np.random.uniform(-0.04, 0.04, 3)
                if abs(x) > length * 0.4:  # fractured ends
                    jitter *= 2.5
                pts.append([x + jitter[0], y + jitter[1], z + max(0, jitter[2])])
    pts = np.array(pts)
    hull = ConvexHull(pts)
    write_binary_stl(filepath, pts, hull.simplices)
    print(f"Generated {os.path.basename(filepath)}: {len(hull.simplices)} facets")

def generate_cracked_slab(filepath, size=(0.8, 0.6, 0.18), seed=303):
    np.random.seed(seed)
    pts = []
    for x in [-size[0]/2, size[0]/2]:
        for y in [-size[1]/2, size[1]/2]:
            for z in [0, size[2]]:
                pts.append([x + np.random.uniform(-0.05, 0.05),
                            y + np.random.uniform(-0.05, 0.05),
                            z + np.random.uniform(-0.02, 0.02)])
    # Add random internal fracture edge points
    for _ in range(20):
        pts.append([np.random.uniform(-size[0]/2, size[0]/2),
                    np.random.uniform(-size[1]/2, size[1]/2),
                    np.random.uniform(0, size[2])])
    pts = np.array(pts)
    hull = ConvexHull(pts)
    write_binary_stl(filepath, pts, hull.simplices)
    print(f"Generated {os.path.basename(filepath)}: {len(hull.simplices)} facets")

# Generate 3 variations of realistic debris meshes
generate_jagged_rock(os.path.join(MESH_DIR, "rock_debris_1.stl"), scale=(0.35, 0.32, 0.30), seed=42)
generate_jagged_rock(os.path.join(MESH_DIR, "rock_debris_2.stl"), scale=(0.45, 0.38, 0.40), seed=88)
generate_jagged_rock(os.path.join(MESH_DIR, "rock_debris_3.stl"), scale=(0.28, 0.25, 0.22), seed=133)
generate_broken_beam(os.path.join(MESH_DIR, "broken_beam_1.stl"), length=1.1, width=0.22, height=0.22, seed=77)
generate_broken_beam(os.path.join(MESH_DIR, "broken_beam_2.stl"), length=0.8, width=0.20, height=0.20, seed=99)
generate_cracked_slab(os.path.join(MESH_DIR, "concrete_slab_1.stl"), size=(0.75, 0.55, 0.16), seed=55)
generate_cracked_slab(os.path.join(MESH_DIR, "concrete_slab_2.stl"), size=(0.60, 0.50, 0.14), seed=66)

# Generate photorealistic textures for rubble, concrete, rust, and asphalt
def generate_procedural_textures(out_dir):
    np.random.seed(42)
    # 1. Concrete / Rubble texture (speckled grey with cracks)
    w, h = 512, 512
    base = np.random.normal(140, 15, (w, h)).clip(90, 190).astype(np.uint8)
    # Add darker speckles
    noise = np.random.binomial(1, 0.05, (w, h)) * -40
    concrete = np.clip(base + noise, 0, 255).astype(np.uint8)
    concrete_b = (concrete.astype(np.float32) * 0.95).astype(np.uint8)
    concrete_rgb = np.stack([concrete, concrete, concrete_b], axis=-1)
    Image.fromarray(concrete_rgb).save(os.path.join(out_dir, "concrete_diffuse.png"))

    # 2. Rust / Metal rebar texture (oxidized orange-brown)
    rust_r = np.random.normal(160, 20, (w, h)).clip(100, 210)
    rust_g = (rust_r * 0.45 + np.random.normal(0, 10, (w, h))).clip(30, 120)
    rust_b = (rust_r * 0.20 + np.random.normal(0, 5, (w, h))).clip(15, 60)
    rust_rgb = np.stack([rust_r, rust_g, rust_b], axis=-1).astype(np.uint8)
    Image.fromarray(rust_rgb).save(os.path.join(out_dir, "rust_diffuse.png"))

    # 3. Rubble ground / gravel asphalt (textured ground)
    ground_base = np.random.normal(90, 22, (w, h)).clip(40, 150).astype(np.uint8)
    # Dirt tints
    g_r = ground_base
    g_g = (ground_base.astype(np.float32) * 0.92).astype(np.uint8)
    g_b = (ground_base.astype(np.float32) * 0.82).astype(np.uint8)
    ground_rgb = np.stack([g_r, g_g, g_b], axis=-1)
    Image.fromarray(ground_rgb).save(os.path.join(out_dir, "tactical_ground.png"))

    print("Procedural textures generated successfully!")

generate_procedural_textures(MESH_DIR)
