import struct
from typing import Optional

def image_dimensions(header: bytes) -> Optional[tuple]:
    """Return (width, height) from the first bytes of a JPEG, PNG, or TIFF file."""
    if header[:8] == b'\x89PNG\r\n\x1a\n':
        w, h = struct.unpack('>II', header[16:24])
        return w, h
    if header[:2] == b'\xff\xd8':
        i = 2
        while i < len(header) - 8:
            if header[i] != 0xff:
                break
            marker = header[i + 1]
            if marker in (0xC0, 0xC1, 0xC2):
                h, w = struct.unpack('>HH', header[i + 5:i + 9])
                return w, h
            seg_len = struct.unpack('>H', header[i + 2:i + 4])[0]
            i += 2 + seg_len
    if header[:2] in (b'II', b'MM'):
        bo = '<' if header[:2] == b'II' else '>'
        ifd_off = struct.unpack(bo + 'I', header[4:8])[0]
        if ifd_off + 2 > len(header):
            return None
        n = struct.unpack(bo + 'H', header[ifd_off:ifd_off + 2])[0]
        w = h = 0
        for j in range(n):
            off = ifd_off + 2 + j * 12
            if off + 12 > len(header):
                break
            tag, typ = struct.unpack(bo + 'HH', header[off:off + 4])
            if tag in (256, 257):
                fmt = bo + ('I' if typ == 4 else 'H')
                val = struct.unpack(fmt, header[off + 8:off + 8 + struct.calcsize(fmt)])[0]
                if tag == 256: w = val
                else: h = val
        if w and h:
            return w, h
    return None