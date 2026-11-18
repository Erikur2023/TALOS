import struct
import numpy as np
from scipy.spatial import ConvexHull
import mujoco

def write_binary_stl(filename, points, simplices):
    with open(filename, 'wb') as f:
        f.write(b'Binary STL generated for MuJoCo'.ljust(80, b'\0'))
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

pts = np.random.uniform(-0.5, 0.5, (30, 3))
hull = ConvexHull(pts)
write_binary_stl('test_rock_bin.stl', pts, hull.simplices)

xml = '<mujoco><asset><mesh name="rock1" file="test_rock_bin.stl"/></asset><worldbody><geom type="mesh" mesh="rock1" pos="1 0 0" rgba="0.5 0.5 0.5 1"/></worldbody></mujoco>'
m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
mujoco.mj_step(m, d)
print('SUCCESS! MuJoCo loaded binary STL mesh! Geoms:', m.ngeom)
