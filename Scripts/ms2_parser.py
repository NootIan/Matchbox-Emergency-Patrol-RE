"""
Parser for Matchbox: Emergency Patrol (Sandbox engine / XGame40) .ms2 model files.

Format fully confirmed via Ghidra decompilation of
Game Contents/patrol/renderd3dR.dll:
  - sbox_ms2_model::sbox_ms2_model @ 0x10005960  (header / pointer setup)
  - render/face-decode function      @ 0x10005fb0
plus geometric validation of the decoded output (correct bounding boxes,
recognizable silhouettes for e.g. action.ms2 and airplane.ms2).

Header (all little-endian):
  0x00  u32   magic            0xEF349912
  0x04  u32   version          0x10000005
  0x08  u32   frame_data_off   offset (from file start) to frame-descriptor array
  0x0C  u32   num_frames       number of animation frames (1 = static)
  0x10  u32   frame_entries    actual vertex-position records in this frame's blob
  0x14  u32   ?                (seen: 16)
  0x18  u32   ?                (seen: 16)
  0x1C  u32   ?                (seen: 1)
  0x20  u32   ?                (seen: 0)
  0x24  u32   geom_data_off    offset to the index-remap table
  0x28  u32   vbuf_entries     size of the index-remap table (logical index slots;
                               >= frame_entries -- multiple slots can share one
                               real vertex, e.g. for UV seams)
  0x2C  u32   face_data_off    offset to face/triangle-strip section
  0x30  u32   face_data_size   size in bytes of face section
  0x34  u32   material_tbl_off offset to material/texture name table
  0x38  u32   material_count   number of 64-byte material slots (incl. terminator)

Material table: material_count entries of 64 bytes each:
  byte 0:      0x01 = valid entry, 0x00 = terminator
  bytes 1..:   null-terminated ASCII texture/material name, zero padded

Frame descriptor array (at frame_data_off, 32 bytes per frame):
  0x00 u32   rel_offset   offset (from file start) to this frame's vertex blob
  0x04 u32   blob_size    size in bytes of the vertex blob (== vbuf_entries*12)
  0x08 f32   min0, min1, min2   per-axis minimums, axis order (Y, Z, X)
  0x14 f32   scale0,scale1,scale2  per-axis scale = extent/65535, same axis order
Only frame 0 is handled here; multi-frame (animated) models use a different,
still-unconfirmed encoding for subsequent frames.

Vertex blob: vbuf_entries records of 12 bytes (6x uint16):
  u16 0: raw Y   -> Y = min0 + raw*scale0
  u16 1: raw Z   -> Z = min1 + raw*scale1
  u16 2: raw X   -> X = min2 + raw*scale2
  u16 3,4,5: packed normal (signed, approx -32768..32767 -> -1..1)

Index-remap table (at geom_data_off): array of 12-byte records; only the
first uint16 of each record is used by the renderer -- it is the real index
into the vertex blob above. (The remaining 8 bytes hold UV floats used
elsewhere for texture mapping; not needed for pure geometry export.)

Face section (at face_data_off, face_data_size bytes): a stream of uint16
values, structured as a sequence of groups, terminated by matID == 0xFFF0:

  group := matID(u16) stripCount(u16)
           [ if stripCount != 0:
               baseOffset(u16) pad(u16)
               (stripCount) u16 strip indices, decoded as a sliding-window
               triangle strip of (stripCount-2) triangles: for triangle i,
               window = (strip[i], strip[i+1], strip[i+2]), with the first
               two swapped when i is odd (winding alternation). Each of the
               3 raw indices is offset by +baseOffset, then triangles where
               any two of the 3 (offset) indices are equal are degenerate
               and skipped. ]
           secondCount(u16)
           [ if secondCount != 0:
               base2(u16) pad(u16)
               (secondCount+2)//3 triangles, each 3 fresh u16 indices
               (flat triangle list, no strip sharing), each offset by
               +base2, no degenerate check. ]

  Every raw index (after the +base offset) is then resolved through the
  index-remap table: real_vertex_index = remap_table[index].u16_0.
"""
import struct
import os
import sys


class MS2Model:
    def __init__(self, path):
        self.path = path
        with open(path, 'rb') as f:
            self.data = f.read()
        self._parse_header()

    def _parse_header(self):
        d = self.data
        if len(d) < 0x3C:
            raise ValueError("file too small")
        magic, version = struct.unpack('<II', d[0:8])
        if magic != 0xEF349912:
            raise ValueError(f"bad magic 0x{magic:08X}")
        self.version = version
        (self.frame_data_off, self.num_frames, self.frame_entries,
         f14, f18, f1c, f20,
         self.geom_data_off, self.vbuf_entries,
         self.face_data_off, self.face_data_size,
         self.material_tbl_off, self.material_count) = struct.unpack(
            '<IIIIIIIIIIIII', d[8:0x3C])

    def materials(self):
        mats = []
        off = self.material_tbl_off
        for i in range(self.material_count):
            entry = self.data[off:off + 64]
            if not entry or entry[0] != 1:
                break
            name = entry[1:].split(b'\x00')[0].decode('ascii', 'replace')
            mats.append(name)
            off += 64
        return mats

    def frame_descriptor(self, frame_idx=0):
        fd_off = self.frame_data_off + frame_idx * 32
        (rel_offset, blob_size, min0, min1, min2,
         s0, s1, s2) = struct.unpack('<2I6f', self.data[fd_off:fd_off + 32])
        return {
            'vertex_blob_offset': rel_offset,
            'vertex_blob_size': blob_size,
            'min': (min0, min1, min2),   # axis order Y, Z, X
            'scale': (s0, s1, s2),
        }

    def vertices(self, frame_idx=0):
        fd = self.frame_descriptor(frame_idx)
        m0, m1, m2 = fd['min']
        s0, s1, s2 = fd['scale']
        voff = fd['vertex_blob_offset']
        # frame_entries = actual position records present in the blob;
        # vbuf_entries is the (possibly larger) index-remap table size.
        n = min(self.frame_entries, fd['vertex_blob_size'] // 12)
        verts = []
        normals = []
        for i in range(n):
            p = voff + i * 12
            a, b, c, nx, ny, nz = struct.unpack('<6H', self.data[p:p + 12])
            X = m2 + c * s2
            Y = m0 + a * s0
            Z = m1 + b * s1
            verts.append((X, Y, Z))

            def to_snorm(v):
                sv = v - 65536 if v >= 32768 else v
                return sv / 32767.0
            # Same per-field -> axis mapping as position (field0->Y, field1->Z,
            # field2->X) applied to the normal's 3 fields, but negated -- fit
            # empirically against the mesh's own face normals (~80-85% direct
            # per-vertex agreement across a multi-file sample).
            normals.append((-to_snorm(nz), -to_snorm(nx), -to_snorm(ny)))
        return verts, normals

    def _remap(self, logical_index):
        off = self.geom_data_off + logical_index * 12
        if off < 0 or off + 2 > len(self.data):
            return None
        return struct.unpack('<H', self.data[off:off + 2])[0]

    def _uv(self, logical_index):
        # Re-derived from Ghidra decompilation of the vertex-buffer-build
        # function (0x100088d0 in renderd3dR.dll): tracing the exact
        # pre/post-increment loop-counter arithmetic resolves to reading
        # from the *same* 12-byte record used for the position remap --
        # i.e. each geom_data_off record is [u16 remap_idx, u16 pad,
        # f32 u, f32 v], keyed by the same logical/strip index as _remap().
        off = self.geom_data_off + logical_index * 12 + 4
        if off < 0 or off + 8 > len(self.data):
            return None
        u, v = struct.unpack('<ff', self.data[off:off + 8])
        return (u, v)

    def triangles_by_material(self):
        """Returns a list of (material_index, (a,uva), (b,uvb), (c,uvc)) where
        a/b/c are vertex indices and uv* are (u, v) tuples or None. Winding is
        the direct Ghidra-derived order (v0, v1, v2) + baseOffset, with no
        extra reversal. Verified against the mesh's own decoded vertex
        normals: face normal (right-hand rule on a,b,c) agrees with the
        vertex-a normal on ~80-85% of triangles across a multi-file sample,
        for both the strip and flat-list sub-sections, with no reversal
        needed for either."""
        off = self.face_data_off
        size = self.face_data_size
        count = size // 2
        u16s = struct.unpack(f'<{count}H', self.data[off:off + count * 2])
        n = len(u16s)

        def corner(logical_index):
            r = self._remap(logical_index)
            if r is None:
                return None
            return (r, self._uv(logical_index))

        tris = []
        pos = 0
        while pos < n:
            matid = u16s[pos]
            if matid == 0xFFF0:
                break
            stripcount = u16s[pos + 1]
            p = pos + 2
            if stripcount != 0:
                tricount = stripcount - 2
                base = u16s[p]
                p += 2
                for i in range(tricount):
                    if i % 2 == 0:
                        v0, v1, v2 = u16s[p], u16s[p + 1], u16s[p + 2]
                    else:
                        v0, v1, v2 = u16s[p + 1], u16s[p], u16s[p + 2]
                    a, b, c = v0 + base, v1 + base, v2 + base
                    if a != b and b != c and a != c:
                        ca, cb, cc = corner(a), corner(b), corner(c)
                        if ca is not None and cb is not None and cc is not None:
                            tris.append((matid, ca, cb, cc))
                    p += 1
                p += 2
            else:
                p = pos + 2
            secondcount = u16s[p]
            nextpos = p + 1
            if secondcount != 0:
                base2 = u16s[p + 1]
                q = p + 3
                tricount2 = (secondcount + 2) // 3
                for i in range(tricount2):
                    v0, v1, v2 = u16s[q], u16s[q + 1], u16s[q + 2]
                    ca = corner(v0 + base2)
                    cb = corner(v1 + base2)
                    cc = corner(v2 + base2)
                    if ca is not None and cb is not None and cc is not None:
                        tris.append((matid, ca, cb, cc))
                    q += 3
                nextpos = q
            pos = nextpos
        return tris

    def triangles(self):
        return [(ca[0], cb[0], cc[0]) for (_matid, ca, cb, cc) in self.triangles_by_material()]

    # Fallback placeholder colors, cycled across materials that have no
    # matching extracted texture image.
    _PALETTE = [
        (0.75, 0.75, 0.75), (0.80, 0.30, 0.25), (0.25, 0.55, 0.80),
        (0.35, 0.70, 0.35), (0.85, 0.70, 0.20), (0.55, 0.35, 0.75),
        (0.80, 0.50, 0.20), (0.30, 0.75, 0.75), (0.65, 0.35, 0.55),
        (0.45, 0.45, 0.20),
    ]

    @staticmethod
    def _to_export_space(v):
        # Native decode axes are (X, Y, Z) with Z as "up" (height). OBJ/Blender
        # expect Y-up. This is a pure -90deg rotation about X (new_Y=old_Z,
        # new_Z=-old_Y), so it preserves handedness/winding -- it does not
        # interact with the separate winding-direction fix in triangles_by_material().
        x, y, z = v
        return (x, z, -y)

    def to_ursina_submeshes(self, frame_idx=0, flip_u=False, flip_v=True):
        """Direct in-memory geometry for constructing Ursina Mesh objects,
        bypassing the OBJ -> Blender -> glTF round trip entirely (that
        pipeline introduced multiple axis/UV bugs across a modding session
        -- a mirrored terrain theory that turned out false, and a real but
        hard-to-isolate horizontal UV flip). Building meshes directly from
        this decode gives one well-understood coordinate transform
        (_to_export_space, verified as a proper rotation) instead of
        several opaque ones chained together.

        Returns a list of (material_name, vertices, uvs, normals) tuples,
        one per material, each a flat (duplicated per face-corner) list
        matching Ursina's Mesh(vertices=..., uvs=..., normals=...)
        constructor -- no separate index buffer needed."""
        verts, normals = self.vertices(frame_idx)
        tris = self.triangles_by_material()
        mats = self.materials()

        by_material = {}
        for matid, ca, cb, cc in tris:
            mname = mats[matid] if matid < len(mats) else f"_mat{matid}"
            by_material.setdefault(mname, []).append((ca, cb, cc))

        submeshes = []
        for mname, tri_list in by_material.items():
            sub_verts, sub_uvs, sub_normals = [], [], []
            for corners in tri_list:
                for idx, uv in corners:
                    sub_verts.append(self._to_export_space(verts[idx]))
                    sub_normals.append(self._to_export_space(normals[idx]))
                    u, v = uv if uv is not None else (0.0, 0.0)
                    sub_uvs.append((1.0 - u if flip_u else u, 1.0 - v if flip_v else v))
            submeshes.append((mname, sub_verts, sub_uvs, sub_normals))
        return submeshes

    def export_mdd(self, out_path, fps=30.0):
        """Lightwave Point Cache (.mdd) export of all frames' vertex positions,
        for Blender's built-in MDD importer (File > Import > Lightwave Point
        Cache) applied on top of the frame-0 OBJ topology. MDD is big-endian:
        header = numFrames(u32), numPoints(u32); then numFrames floats of
        frame time; then numFrames * numPoints * 3 floats of (x,y,z).
        Only valid for models whose vertex count is constant across frames
        (true for 68/69 animated .ms2 files found; the one exception --
        varying-point-count "firetest.ms2" -- is not exportable this way)."""
        n0 = min(self.frame_entries, self.frame_descriptor(0)['vertex_blob_size'] // 12)
        frames = []
        for i in range(self.num_frames):
            fd = self.frame_descriptor(i)
            n = min(self.frame_entries, fd['vertex_blob_size'] // 12)
            if n != n0:
                raise ValueError(f"vertex count varies across frames ({n0} vs {n} at frame {i}), not MDD-exportable")
            verts, _ = self.vertices(i)
            frames.append(verts)
        with open(out_path, 'wb') as f:
            f.write(struct.pack('>2I', self.num_frames, n0))
            f.write(struct.pack(f'>{self.num_frames}f', *[i / fps for i in range(self.num_frames)]))
            for verts in frames:
                for v in verts:
                    x, y, z = self._to_export_space(v)
                    f.write(struct.pack('>3f', x, y, z))

    def export_obj(self, out_path, frame_idx=0, mtl_path=None, textures_dir=None, native_axes=False):
        """textures_dir: folder of <material_name>.png files (e.g. produced by
        tex_extractor.py from textures_all2.tex) to wire up as map_Kd. Falls
        back to a flat placeholder color per material when no image matches
        or textures_dir is None.
        native_axes: skip the Y-up _to_export_space rotation and write raw
        (X,Y,Z) as decoded by vertices() -- use this when the model needs to
        line up with other raw-space exports (e.g. bpk_terrain_extractor.py's
        terrain OBJ, or LevelFile.mef placement positions, neither of which
        apply any axis swap). Leave False for standalone viewing in
        Blender/other Y-up-expecting tools."""
        verts, normals = self.vertices(frame_idx)
        tris = self.triangles_by_material()
        mats = self.materials()
        name = os.path.splitext(os.path.basename(self.path))[0]

        if mtl_path is None:
            mtl_path = os.path.splitext(out_path)[0] + '.mtl'
        mtl_name = os.path.basename(mtl_path)
        mtl_dir = os.path.dirname(mtl_path) or '.'

        with open(mtl_path, 'w') as f:
            for i, mname in enumerate(mats):
                r, g, b = self._PALETTE[i % len(self._PALETTE)]
                f.write(f"newmtl {mname}\n")
                f.write(f"Kd {r:.3f} {g:.3f} {b:.3f}\n")
                f.write("Ka 0.05 0.05 0.05\nKs 0.05 0.05 0.05\nd 1.0\nillum 1\n")
                if textures_dir is not None:
                    tex_path = os.path.join(textures_dir, f"{mname}.png")
                    if os.path.isfile(tex_path):
                        rel = os.path.relpath(tex_path, mtl_dir).replace(os.sep, '/')
                        f.write(f"map_Kd {rel}\n")
                        # Same image's alpha channel doubles as the opacity
                        # map -- palette index 0 (magenta colorkey) is
                        # exported fully transparent by tex_extractor.py.
                        f.write(f"map_d {rel}\n")
                f.write("\n")

        # One vt per face-corner use (simple and always correct, if a little
        # redundant) -- UV is per-corner here, not per-vertex.
        vts = []

        def vt_index(uv):
            u, v = uv if uv is not None else (0.0, 0.0)
            # OBJ v=0 is bottom; flip to match typical image-space v=0-at-top.
            # u is also flipped: textures come out horizontally mirrored
            # otherwise (confirmed via strsally.ms2 -- the in-game
            # "SALLY" street sign texture reads correctly on its own but
            # rendered backwards in-editor with u unflipped).
            vts.append((1.0 - u, 1.0 - v))
            return len(vts)

        face_lines = []
        current_mat = None
        for matid, ca, cb, cc in tris:
            mname = mats[matid] if matid < len(mats) else None
            usemtl = None
            if mname is not None and mname != current_mat:
                usemtl = mname
                current_mat = mname
            a, uva = ca
            b, uvb = cb
            c, uvc = cc
            ta, tb, tc = vt_index(uva), vt_index(uvb), vt_index(uvc)
            face_lines.append((usemtl, a, ta, b, tb, c, tc))

        with open(out_path, 'w') as f:
            f.write(f"# {name}.ms2 -> OBJ\n")
            f.write(f"# vertices={len(verts)} triangles={len(tris)} "
                    f"frames={self.num_frames} materials={mats}\n")
            f.write(f"mtllib {mtl_name}\n")
            xform = (lambda v: v) if native_axes else self._to_export_space
            for v in verts:
                x, y, z = xform(v)
                f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
            for nvec in normals:
                nx, ny, nz = xform(nvec)
                f.write(f"vn {nx:.6f} {ny:.6f} {nz:.6f}\n")
            for u, v in vts:
                f.write(f"vt {u:.6f} {v:.6f}\n")
            for usemtl, a, ta, b, tb, c, tc in face_lines:
                if usemtl is not None:
                    f.write(f"usemtl {usemtl}\n")
                f.write(f"f {a+1}/{ta}/{a+1} {b+1}/{tb}/{b+1} {c+1}/{tc}/{c+1}\n")


def main():
    if len(sys.argv) < 2:
        print("usage: ms2_parser.py <file.ms2> [out.obj] [textures_dir]")
        return
    in_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(in_path)[0] + '.obj'
    textures_dir = sys.argv[3] if len(sys.argv) > 3 else None
    m = MS2Model(in_path)
    print(f"file: {in_path}")
    print(f"  version=0x{m.version:08X} frames={m.num_frames} "
          f"vbuf_entries={m.vbuf_entries} material_count={m.material_count}")
    print(f"  materials: {m.materials()}")
    verts, _ = m.vertices()
    tris = m.triangles()
    print(f"  decoded {len(verts)} vertices, {len(tris)} triangles")
    m.export_obj(out_path, textures_dir=textures_dir)
    print(f"  wrote {out_path}")
    if m.num_frames != 1:
        mdd_path = os.path.splitext(out_path)[0] + '.mdd'
        try:
            m.export_mdd(mdd_path)
            print(f"  animated ({m.num_frames} frames): wrote {mdd_path} "
                  f"(Blender: import the OBJ, then File > Import > Lightwave Point Cache)")
        except ValueError as e:
            print(f"  animated but NOT MDD-exportable: {e}")


if __name__ == '__main__':
    main()
