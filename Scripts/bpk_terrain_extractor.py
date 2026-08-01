"""
Terrain/collision-mesh extractor for Matchbox: Emergency Patrol .BPK world
files (megacity.BPK / megacitys.BPK).

Format confirmed via Ghidra decompilation of renderd3dR.dll:
  - sbox_bsp_partition::sbox_bsp_partition @ 0x10001900 (file header, grid)
  - sbox_bsp_model::init_partition (FUN_10001340) (per-cell sub-array setup)
  - a polygon/plane 3D-reconstruction routine (FUN_10001600) used by the
    BSP collision-raycast code, whose exact formula is reproduced below.

File layout (all little-endian):
  0x00 u32   magic       0x1234EBAF
  0x38 i32   grid_min    signed grid index
  0x3C i32   grid_max    signed grid index
                          grid_dim = (grid_max - grid_min) + 1  (55 in
                          megacity.BPK -> 55x55 = 3025 cells)
  0x40 u32   name_count  number of NUL-terminated strings in the name table
                          immediately following the grid-cell array
                          (material/surface names, shared with .ms2-style
                          texture references)

  Grid-cell array starts at file offset 0x44, grid_dim*grid_dim entries of
  144 bytes each (row-major). Each cell record:
    0x00  f32[3]  bbox_min
    0x0C  f32[3]  bbox_max
    0x18  f32[3]  bbox_center (redundant, == (min+max)/2)
    0x50  u32     collision_verts_off   (purpose still unclear -- NOT a
                                          flat position array; probing it
                                          directly returns non-sequitur
                                          bytes. Left decoded but unused
                                          here.)
    0x54  u32     collision_verts_count
    0x58  u32     ? (another sub-array offset, unused here)
    0x60  u32     render_data_off       D3D draw-call stream (mirrors the
                                          .ms2 face-stream format: material-
                                          indexed groups terminated by
                                          0xFFF0). Not decoded here.
    0x7C  u32     planes_off            offset of this cell's plane array
    0x80  u32     planes_count
    0x84  u32     coords2d_count        (coords2d array immediately follows
                                          the plane array)
    0x88  u32     polys_count           (poly array immediately follows the
                                          coords2d array)
    0x8C  u32     ? (extra count/flags, unused here)

  planes_off / planes_count: array of 20-byte records:
    f32[3] normal, f32 dist, u32 flags
  Immediately after the plane array: coords2d array, coords2d_count records
  of 8 bytes (f32 u, f32 v) -- 2D coordinates in the polygon's own plane.
  Immediately after that: poly array, polys_count records of 28 bytes:
    f32[4] bbox2d (umin, vmin, umax, vmax)
    u16    plane_index      (index into this cell's plane array)
    u16    coords2d_index   (start index into this cell's coords2d array)
    u32    ? (unused)
    u8     vertex_count     (this polygon's fan uses coords2d_index ..
                              coords2d_index + vertex_count)
    (3 more bytes, unused)

  3D reconstruction (dominant-axis projection -- this is the actual BSP
  raycast/collision-query technique, reproduced exactly): for a polygon
  referencing plane P = (nx, ny, nz, d) with flags F:
    au = F & 3; av = (F >> 2) & 3; aw = (F >> 4) & 3   (0=X, 1=Y, 2=Z)
  For each (u, v) in its coords2d run:
    pos[au] = u
    pos[av] = v
    pos[aw] = (d - u*n[au] - v*n[av]) / n[aw]
  Confirmed exact: reconstructed positions land precisely on the cell's own
  bounding box.

This is the BSP *collision* geometry (used for vehicle/ground raycasts),
not necessarily byte-identical to the separate D3D render-data stream --
but for a city/terrain surface the two describe the same physical shape,
so this is a solid, fully-decoded stand-in for "the terrain mesh."
"""
import struct
import sys

MAGIC = 0x1234EBAF
CELL_ARRAY_OFF = 0x44
CELL_STRIDE = 144
PLANE_STRIDE = 20
COORD_STRIDE = 8
POLY_STRIDE = 28


class BpkWorld:
    def __init__(self, path):
        self.path = path
        with open(path, 'rb') as f:
            self.data = f.read()
        magic = struct.unpack('<I', self.data[0:4])[0]
        if magic != MAGIC:
            raise ValueError(f"bad magic 0x{magic:08X}")
        grid_min, grid_max, name_count = struct.unpack('<iiI', self.data[0x38:0x44])
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.grid_dim = (grid_max - grid_min) + 1
        self.name_count = name_count
        self.num_cells = self.grid_dim * self.grid_dim

    def names(self):
        off = CELL_ARRAY_OFF + self.num_cells * CELL_STRIDE
        out = []
        for _ in range(self.name_count):
            end = self.data.index(b'\x00', off)
            out.append(self.data[off:end].decode('ascii', 'replace'))
            off = end + 1
        return out

    def cell_fields(self, i):
        off = CELL_ARRAY_OFF + i * CELL_STRIDE
        c = self.data[off:off + CELL_STRIDE]
        bbox_min = struct.unpack('<3f', c[0:12])
        bbox_max = struct.unpack('<3f', c[12:24])
        (f50, f54, f58, f60, f7c, f80, f84, f88, f8c) = struct.unpack(
            '<IIIIIIIII', c[0x50:0x54] + c[0x54:0x58] + c[0x58:0x5C] +
            c[0x60:0x64] + c[0x7C:0x80] + c[0x80:0x84] + c[0x84:0x88] +
            c[0x88:0x8C] + c[0x8C:0x90])
        return {
            'index': i, 'bbox_min': bbox_min, 'bbox_max': bbox_max,
            'planes_off': f7c, 'planes_count': f80,
            'coords_count': f84, 'polys_count': f88,
            'render_data_off': f60,
        }

    def cell_polygons(self, cell):
        """Yields lists of 3D (x,y,z) points, one list per polygon fan."""
        planes_off = cell['planes_off']
        n_planes = cell['planes_count']
        n_coords = cell['coords_count']
        n_polys = cell['polys_count']
        if n_planes == 0 or n_polys == 0:
            return
        coords_off = planes_off + n_planes * PLANE_STRIDE
        polys_off = coords_off + n_coords * COORD_STRIDE

        planes = []
        for i in range(n_planes):
            p = planes_off + i * PLANE_STRIDE
            nx, ny, nz, d, flags = struct.unpack('<4fI', self.data[p:p + PLANE_STRIDE])
            au, av, aw = flags & 3, (flags >> 2) & 3, (flags >> 4) & 3
            planes.append(((nx, ny, nz), d, au, av, aw))

        for i in range(n_polys):
            p = polys_off + i * POLY_STRIDE
            if p + POLY_STRIDE > len(self.data):
                break
            plane_idx, coord_idx = struct.unpack('<HH', self.data[p + 16:p + 20])
            vcount = self.data[p + 24]
            if plane_idx >= n_planes or vcount < 3:
                continue
            if coord_idx + vcount > n_coords:
                continue
            n, d, au, av, aw = planes[plane_idx]
            if n[aw] == 0:
                continue
            pts = []
            for k in range(vcount):
                cp = coords_off + (coord_idx + k) * COORD_STRIDE
                u, v = struct.unpack('<ff', self.data[cp:cp + COORD_STRIDE])
                w = (d - u * n[au] - v * n[av]) / n[aw]
                pos = [0.0, 0.0, 0.0]
                pos[au], pos[av], pos[aw] = u, v, w
                pts.append(tuple(pos))
            yield pts

    def extract_mesh(self, cell_filter=None):
        """Returns (vertices, triangles) for the whole world (or a filtered
        subset of cell indices)."""
        verts = []
        tris = []
        indices = range(self.num_cells) if cell_filter is None else cell_filter
        for i in indices:
            cell = self.cell_fields(i)
            for poly in self.cell_polygons(cell):
                base = len(verts)
                verts.extend(poly)
                for k in range(1, len(poly) - 1):
                    tris.append((base, base + k, base + k + 1))
        return verts, tris

    def export_obj(self, out_path, cell_filter=None):
        verts, tris = self.extract_mesh(cell_filter)
        with open(out_path, 'w') as f:
            f.write(f"# {self.path} terrain/collision mesh\n")
            f.write(f"# vertices={len(verts)} triangles={len(tris)}\n")
            for x, y, z in verts:
                f.write(f"v {x:.4f} {y:.4f} {z:.4f}\n")
            for a, b, c in tris:
                f.write(f"f {a+1} {b+1} {c+1}\n")
        return len(verts), len(tris)


def main():
    if len(sys.argv) < 3:
        print("usage: bpk_terrain_extractor.py <world.bpk> <out.obj>")
        return
    w = BpkWorld(sys.argv[1])
    print(f"file: {sys.argv[1]}")
    print(f"  grid: {w.grid_dim}x{w.grid_dim} ({w.num_cells} cells), "
          f"name_count={w.name_count}")
    nv, nt = w.export_obj(sys.argv[2])
    print(f"  wrote {nv} vertices, {nt} triangles to {sys.argv[2]}")


if __name__ == '__main__':
    main()
