"""The device-memory primitives of a C-ABI runtime (`nbxtensor.memory.device_memory.AbiMemory`): each primitive calls the
runtime symbol its table names, with the runtime's own argument shapes, and returns the runtime's status code beside the
value; a primitive the table does not name is not supported. Checked against a fake runtime that records its calls —
no device."""
import ctypes

import pytest

from nbxtensor.memory.device_memory import AbiMemory

TABLE = {"malloc": "xMalloc", "free": "xFree", "mem_get_info": "xMemGetInfo", "memset": "xMemset",
         "memcpy": "xMemcpy", "malloc_host": "xMallocHost", "free_host": "xFreeHost", "set_device": "xSetDevice",
         "get_device": "xGetDevice", "device_count": "xGetDeviceCount"}


class FakeRuntime:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def call(*args):
            self.calls.append((name, args))
            if name in ("xMalloc", "xMallocHost"):
                ctypes.cast(args[0], ctypes.POINTER(ctypes.c_void_p))[0] = 0x1000 + args[1].value
            elif name == "xMemGetInfo":
                ctypes.cast(args[0], ctypes.POINTER(ctypes.c_size_t))[0] = 7
                ctypes.cast(args[1], ctypes.POINTER(ctypes.c_size_t))[0] = 9
            elif name == "xGetDevice":
                ctypes.cast(args[0], ctypes.POINTER(ctypes.c_int))[0] = 3
            elif name == "xGetDeviceCount":
                ctypes.cast(args[0], ctypes.POINTER(ctypes.c_int))[0] = 4
            return 0
        return call


def test_each_primitive_calls_its_symbol_and_returns_the_code_beside_the_value():
    rt = FakeRuntime()
    m = AbiMemory(rt, TABLE)
    assert all(m.supports(p) for p in ("malloc", "free", "mem_get_info", "memset", "memcpy", "host_alloc", "host_free",
                                       "set_device", "get_device", "device_count"))
    assert m.malloc(16) == (0, 0x1010)
    assert m.free(0x1010) == 0
    assert m.mem_get_info() == (0, 7, 9)
    assert m.memset(0x1010, 0, 16) == 0
    assert m.memcpy(1, 2, 3, 3) == 0
    assert m.host_alloc(8) == (0, 0x1008)
    assert m.host_free(0x1008) == 0
    assert m.set_device(2) == 0
    assert m.get_device() == (0, 3)
    assert m.device_count() == (0, 4)
    assert m.budget() == (0, 9)
    names = [c[0] for c in rt.calls]
    assert names == ["xMalloc", "xFree", "xMemGetInfo", "xMemset", "xMemcpy", "xMallocHost", "xFreeHost", "xSetDevice",
                     "xGetDevice", "xGetDeviceCount", "xMemGetInfo"], names
    kinds = dict(rt.calls)["xMemcpy"]
    assert [a.value for a in kinds] == [1, 2, 3, 3]


def test_a_primitive_the_table_does_not_name_is_not_supported():
    m = AbiMemory(FakeRuntime(), {k: v for k, v in TABLE.items() if k not in ("malloc_host", "free_host")})
    assert m.supports("malloc") and not m.supports("host_alloc") and not m.supports("host_free")
    with pytest.raises(KeyError):
        m.host_alloc(8)
