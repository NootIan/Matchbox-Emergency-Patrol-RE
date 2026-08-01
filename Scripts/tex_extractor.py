"""
Extractor for Matchbox: Emergency Patrol packed texture archives
(textures_all2.tex / textures_all2s.tex / textures_all4s.tex).

Format confirmed via Ghidra decompilation of renderd3dR.dll
(sbox_renderer::register_texture_set @ 0x1000f264) plus exact arithmetic
validation (pixel-data offsets between consecutive textures match the sum
of a full mip chain: w^2 + (w/2)^2 + ... + 1).

File layout (all little-endian):
  0x00 u32  magic     0x81 (fixed format-version tag)
  0x04 u32  dir_off   absolute file offset of the directory

  bytes [8 .. dir_off): raw pixel data pool. Every texture's full mip chain
  (base level down to 1x1, 8bpp palette-indexed, no padding between levels
  or between textures) is packed back-to-back in this pool.

  at dir_off:
    u32 num_palettes
    u32 num_textures        (== num_palettes in every archive observed)
    num_palettes x 768-byte palette (256 x RGB, palette index 0 is
      conventionally the colorkey/transparent magenta FF00FF)
    num_textures x 276-byte texture record:
      bytes 0..255   null-terminated ASCII name, zero padded
      +256 u32  size          texture is size x size (square, power of 2)
      +260 u32  pixel_offset  absolute file offset of the base mip level
      +264 u32  ? (0 normally; seen 1 once on a 1x1 texture)
      +268 u32  palette_index index into the palette table above
      +272 u32  ? (1 in every sample seen -- possibly bytes-per-pixel)
"""
import struct
import os
import sys

MAGIC = 0x81
DIR_HEADER_SIZE = 8
PALETTE_SIZE = 768  # 256 * 3
ENTRY_SIZE = 276
NAME_FIELD_SIZE = 256


class TexArchive:
    def __init__(self, path):
        self.path = path
        with open(path, 'rb') as f:
            self.data = f.read()
        magic, dir_off = struct.unpack('<II', self.data[0:8])
        if magic != MAGIC:
            raise ValueError(f"bad magic 0x{magic:X} (expected 0x{MAGIC:X})")
        self.dir_off = dir_off
        num_palettes, num_textures = struct.unpack('<II', self.data[dir_off:dir_off + 8])
        self.num_palettes = num_palettes
        self.num_textures = num_textures
        self.pal_start = dir_off + 8
        self.tex_start = self.pal_start + num_palettes * PALETTE_SIZE

    def palette(self, index):
        off = self.pal_start + index * PALETTE_SIZE
        raw = self.data[off:off + PALETTE_SIZE]
        return [tuple(raw[i:i + 3]) for i in range(0, PALETTE_SIZE, 3)]

    def entries(self):
        out = []
        for i in range(self.num_textures):
            off = self.tex_start + i * ENTRY_SIZE
            entry = self.data[off:off + ENTRY_SIZE]
            name = entry[:NAME_FIELD_SIZE].split(b'\x00')[0].decode('ascii', 'replace')
            size, pixel_offset, flag0, palette_index, flag1 = struct.unpack(
                '<5I', entry[NAME_FIELD_SIZE:NAME_FIELD_SIZE + 20])
            out.append({
                'index': i, 'name': name, 'size': size,
                'pixel_offset': pixel_offset, 'palette_index': palette_index,
                'flag0': flag0, 'flag1': flag1,
            })
        return out

    def pixels(self, entry):
        size = entry['size']
        n = size * size
        off = entry['pixel_offset']
        return self.data[off:off + n]

    COLORKEY_RGB = (255, 0, 255)

    def to_image(self, entry):
        from PIL import Image
        size = entry['size']
        pal = self.palette(entry['palette_index'])
        px = self.pixels(entry)
        # Palette index 0 is the engine's colorkey/transparency marker (its
        # RGB is always the magenta FF00FF placeholder -- confirmed across
        # every palette in the archive). Export as RGBA with those texels
        # made fully transparent, rather than baking visible magenta pixels.
        transparent_index = 0 if pal[0] == self.COLORKEY_RGB else None
        img = Image.new('RGBA', (size, size))
        if transparent_index is None:
            img.putdata([pal[b] + (255,) for b in px])
        else:
            img.putdata([pal[b] + (0 if b == transparent_index else 255,) for b in px])
        return img

    def extract_all(self, out_dir):
        from PIL import Image
        os.makedirs(out_dir, exist_ok=True)
        n = 0
        for entry in self.entries():
            try:
                img = self.to_image(entry)
            except Exception as e:
                print(f"  skip {entry['name']!r}: {e}")
                continue
            safe_name = entry['name'] or f"tex_{entry['index']}"
            img.save(os.path.join(out_dir, f"{safe_name}.png"))
            n += 1
        return n


def main():
    if len(sys.argv) < 3:
        print("usage: tex_extractor.py <archive.tex> <out_dir>")
        return
    arc = TexArchive(sys.argv[1])
    print(f"{sys.argv[1]}: {arc.num_textures} textures, {arc.num_palettes} palettes")
    n = arc.extract_all(sys.argv[2])
    print(f"wrote {n} PNGs to {sys.argv[2]}")


if __name__ == '__main__':
    main()
