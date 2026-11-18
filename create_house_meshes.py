import os
import struct
import numpy as np

MESH_DIR = "unitree_rl_gym/resources/robots/g1_description/meshes"
os.makedirs(MESH_DIR, exist_ok=True)

def write_binary_stl(filename, facets):
    filepath = os.path.join(MESH_DIR, filename)
    with open(filepath, 'wb') as f:
        f.write(b'MuJoCo Architectural Mesh'.ljust(80, b'\0'))
        f.write(struct.pack('<I', len(facets)))
        for (p0, p1, p2) in facets:
            p0, p1, p2 = np.array(p0, dtype=np.float32), np.array(p1, dtype=np.float32), np.array(p2, dtype=np.float32)
            v1 = p1 - p0
            v2 = p2 - p0
            norm = np.cross(v1, v2)
            l = np.linalg.norm(norm)
            if l > 1e-6:
                norm /= l
            else:
                norm = np.array([0, 0, 1], dtype=np.float32)
            f.write(struct.pack('<3f', float(norm[0]), float(norm[1]), float(norm[2])))
            f.write(struct.pack('<3f', float(p0[0]), float(p0[1]), float(p0[2])))
            f.write(struct.pack('<3f', float(p1[0]), float(p1[1]), float(p1[2])))
            f.write(struct.pack('<3f', float(p2[0]), float(p2[1]), float(p2[2])))
            f.write(struct.pack('<H', 0))
    print(f"Exported {filename} ({len(facets)} triangles)")

def add_box(facets, cx, cy, cz, sx, sy, sz):
    x0, x1 = cx - sx/2, cx + sx/2
    y0, y1 = cy - sy/2, cy + sy/2
    z0, z1 = cz - sz/2, cz + sz/2
    # Bottom (z0)
    facets.append(((x0, y0, z0), (x1, y1, z0), (x1, y0, z0)))
    facets.append(((x0, y0, z0), (x0, y1, z0), (x1, y1, z0)))
    # Top (z1)
    facets.append(((x0, y0, z1), (x1, y0, z1), (x1, y1, z1)))
    facets.append(((x0, y0, z1), (x1, y1, z1), (x0, y1, z1)))
    # Front (x1)
    facets.append(((x1, y0, z0), (x1, y1, z1), (x1, y0, z1)))
    facets.append(((x1, y0, z0), (x1, y1, z0), (x1, y1, z1)))
    # Back (x0)
    facets.append(((x0, y0, z0), (x0, y0, z1), (x1, y0, z1)))
    facets.append(((x0, y0, z0), (x1, y0, z1), (x0, y1, z0)))
    # Right (y1)
    facets.append(((x0, y1, z0), (x1, y1, z1), (x0, y1, z1)))
    facets.append(((x0, y1, z0), (x1, y1, z0), (x1, y1, z1)))
    # Left (y0)
    facets.append(((x0, y0, z0), (x0, y0, z1), (x1, y0, z1)))
    facets.append(((x0, y0, z0), (x1, y0, z1), (x1, y0, z0)))

def add_cylinder(facets, cx, cy, cz_bot, radius, height, num_sides=16):
    theta = np.linspace(0, 2*np.pi, num_sides, endpoint=False)
    z_top = cz_bot + height
    for i in range(num_sides):
        i_next = (i + 1) % num_sides
        x0 = cx + radius * np.cos(theta[i])
        y0 = cy + radius * np.sin(theta[i])
        x1 = cx + radius * np.cos(theta[i_next])
        y1 = cy + radius * np.sin(theta[i_next])
        # Side quad (2 triangles)
        facets.append(((x0, y0, cz_bot), (x1, y1, z_top), (x0, y0, z_top)))
        facets.append(((x0, y0, cz_bot), (x1, y1, cz_bot), (x1, y1, z_top)))
        # Top cap
        facets.append(((cx, cy, z_top), (x0, y0, z_top), (x1, y1, z_top)))
        # Bottom cap
        facets.append(((cx, cy, cz_bot), (x1, y1, cz_bot), (x0, y0, cz_bot)))

# 1. Open Paneled Entrance Door
door_facets = []
# Main door slab (Width=0.88m, Height=2.10m, Thickness=0.045m)
add_box(door_facets, 0, 0, 1.05, 0.045, 0.88, 2.10)
# 4 raised decorative moldings (architectural paneling)
add_box(door_facets, 0.026, -0.22, 1.55, 0.012, 0.32, 0.70)
add_box(door_facets, 0.026, 0.22, 1.55, 0.012, 0.32, 0.70)
add_box(door_facets, 0.026, -0.22, 0.55, 0.012, 0.32, 0.65)
add_box(door_facets, 0.026, 0.22, 0.55, 0.012, 0.32, 0.65)
# Door handle (brass lever)
add_box(door_facets, 0.040, 0.36, 1.05, 0.025, 0.04, 0.14)
add_box(door_facets, 0.060, 0.31, 1.05, 0.015, 0.12, 0.025)
write_binary_stl("house_front_door.stl", door_facets)

# 2. Hallway Console Table
table_facets = []
# Table top (Length=0.90m, Depth=0.35m, Thickness=0.04m, Height=0.76m)
add_box(table_facets, 0, 0, 0.76, 0.35, 0.90, 0.04)
# 4 Legs
leg_r = 0.022
for lx in [-0.13, 0.13]:
    for ly in [-0.38, 0.38]:
        add_cylinder(table_facets, lx, ly, 0.0, leg_r, 0.74, num_sides=12)
# Decorative vase on table
add_cylinder(table_facets, 0, 0, 0.78, 0.06, 0.22, num_sides=14)
write_binary_stl("house_console_table.stl", table_facets)

# 3. Entrance Banister / Stair Handrail
banister_facets = []
# Handrail slopes from X=0 to X=1.20m, Z from 0.85 to 1.10m
# Newel posts (starting and ending square posts)
add_box(banister_facets, 0.0, 0.0, 0.45, 0.08, 0.08, 0.90)
add_box(banister_facets, 1.20, 0.0, 0.60, 0.08, 0.08, 1.05)
# Handrail bar
add_box(banister_facets, 0.60, 0.0, 0.98, 1.25, 0.06, 0.05)
# 6 baluster spindles
for bx in np.linspace(0.18, 1.02, 6):
    bz = 0.45 + (bx / 1.20) * 0.20
    add_cylinder(banister_facets, bx, 0.0, bz - 0.35, 0.018, 0.70, num_sides=10)
write_binary_stl("house_stair_banister.stl", banister_facets)

# 4. Architectural Ceiling Pendant Chandelier
light_facets = []
add_cylinder(light_facets, 0, 0, 0.30, 0.01, 0.50, num_sides=8) # suspension cord
add_cylinder(light_facets, 0, 0, 0.12, 0.18, 0.18, num_sides=16) # glass/brass drum shade
add_cylinder(light_facets, 0, 0, 0.15, 0.06, 0.09, num_sides=12) # inner bulb
write_binary_stl("house_ceiling_lamp.stl", light_facets)

print("All house architectural meshes created successfully.")
