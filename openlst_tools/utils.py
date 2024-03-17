import struct

def unpack_cint(data: bytes, size: int, signed: bool) -> int:
    if size == 1:
        fmt = "<b"
    elif size == 2:
        fmt = "<h"
    elif size == 4:
        fmt = "<i"
    elif size == 8:
        fmt = "<q"
    else:
        raise ValueError("Invalid size")

    if not signed:
        fmt = fmt.upper()

    return struct.unpack(fmt, data)[0]


def pack_cint(data: int, size: int, signed: bool) -> bytes:
    if size == 1:
        fmt = "<b"
    elif size == 2:
        fmt = "<h"
    elif size == 4:
        fmt = "<i"
    elif size == 8:
        fmt = "<q"
    else:
        raise ValueError("Invalid size")

    if not signed:
        fmt = fmt.upper()

    return struct.pack(fmt, data)
