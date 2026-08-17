from array import array


def floats_to_bytes(values: list[float]) -> bytes:
    """Pack a Python float list into the FLOAT32 byte layout Redis vector fields expect."""
    return array("f", values).tobytes()
