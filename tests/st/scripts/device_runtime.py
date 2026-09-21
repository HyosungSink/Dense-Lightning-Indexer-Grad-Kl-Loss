"""CANN ACLNN bindings used only by the local test process.

All numerical operator work runs in the library built from code/.  This module
does not import a reference implementation or provide an output fallback.
"""

from __future__ import annotations

import ctypes as ct
from pathlib import Path

import numpy as np


PTR = ct.c_void_p
SIZE = ct.c_size_t


class DeviceError(RuntimeError):
    def __init__(self, stage: str, status: int):
        self.stage = stage
        self.status = int(status)
        super().__init__(f"{stage}: ret={status}")


def check(status: int, stage: str) -> None:
    if status:
        raise DeviceError(stage, status)


def bind(lib, name, arguments, result=ct.c_int):
    function = getattr(lib, name)
    function.argtypes = arguments
    function.restype = result
    return function


# Bound device waits so a stalled kernel cannot hang the test process forever.
DEFAULT_OP_WAIT_SECONDS = 5


class Runtime:
    def __init__(self, build_dir: Path, device: int = 0,
                 op_wait_seconds: int = DEFAULT_OP_WAIT_SECONDS):
        self.acl = ct.CDLL("libascendcl.so", mode=ct.RTLD_GLOBAL)
        self.nn = ct.CDLL("libnnopbase.so", mode=ct.RTLD_GLOBAL)
        self.op = ct.CDLL(str(build_dir / "libcust_opapi.so"), mode=ct.RTLD_GLOBAL)
        for name, arguments in {
            "aclInit": [ct.c_char_p], "aclFinalize": [],
            "aclrtSetDevice": [ct.c_int32],
            "aclrtCreateStream": [ct.POINTER(PTR)], "aclrtDestroyStream": [PTR],
            "aclrtSynchronizeStream": [PTR],
            "aclrtMalloc": [ct.POINTER(PTR), SIZE, ct.c_int], "aclrtFree": [PTR],
            "aclrtMemcpy": [PTR, SIZE, PTR, SIZE, ct.c_int],
        }.items():
            bind(self.acl, name, arguments)
        dims = ct.POINTER(ct.c_int64)
        bind(self.nn, "aclCreateTensor", [dims, ct.c_uint64, ct.c_int, dims,
             ct.c_int64, ct.c_int, dims, ct.c_uint64, PTR], PTR)
        bind(self.nn, "aclDestroyTensor", [PTR])
        self.workspace_query = bind(self.op, "aclnnDenseLightningIndexerGradKlLossGetWorkspaceSize",
            [PTR] * 5 + [ct.c_double] + [PTR] * 4 + [ct.POINTER(ct.c_uint64), ct.POINTER(PTR)])
        self.launch = bind(self.op, "aclnnDenseLightningIndexerGradKlLoss",
            [PTR, ct.c_uint64, PTR, PTR])
        check(self.acl.aclInit(None), "acl_init")
        check(self.acl.aclrtSetDevice(device), "set_device")
        self.op_wait_seconds = op_wait_seconds
        self.op_timeout_apis = []
        if op_wait_seconds:
            # aclrtSetOpWaitTimeout bounds a notify/event wait; a kernel that
            # spins on an unreachable cross-core barrier is only stopped by the
            # operator execution timeout.  Both are set, and whichever the
            # installed runtime exports is used.
            for name in ("aclrtSetOpWaitTimeout", "aclrtSetOpExecuteTimeOut"):
                try:
                    setter = bind(self.acl, name, [ct.c_uint32])
                except AttributeError:
                    continue
                status = setter(ct.c_uint32(op_wait_seconds))
                if status == 0:
                    self.op_timeout_apis.append(name)
            if not self.op_timeout_apis:
                self.op_wait_seconds = None
        self.stream = PTR()
        check(self.acl.aclrtCreateStream(ct.byref(self.stream)), "create_stream")
        self.allocations = []
        self.tensors = []
        self.workspace = None
        self.workspace_size = 0

    def allocate(self, size):
        pointer = PTR()
        check(self.acl.aclrtMalloc(ct.byref(pointer), max(size, 32), 0), "malloc")
        self.allocations.append(pointer)
        return pointer.value

    def tensor(self, array, dtype):
        raw = np.ascontiguousarray(array)
        # Keep a 64-byte guard immediately adjacent to each tensor on both sides.
        guarded = np.full(raw.nbytes + 128, 0xA5, dtype=np.uint8)
        guarded[64:-64] = raw.view(np.uint8).reshape(-1)
        allocation = self.allocate(guarded.nbytes)
        check(self.acl.aclrtMemcpy(allocation, guarded.nbytes, guarded.ctypes.data,
                                  guarded.nbytes, 1), "copy_input")
        dimensions = (ct.c_int64 * raw.ndim)(*raw.shape)
        strides = (ct.c_int64 * raw.ndim)(*[v // raw.itemsize for v in raw.strides])
        tensor = self.nn.aclCreateTensor(dimensions, raw.ndim,
            {"float32": 0, "float16": 1, "bfloat16": 27}[dtype], strides, 0, 2,
            dimensions, raw.ndim, allocation + 64)
        if not tensor:
            raise RuntimeError("aclCreateTensor returned null")
        self.tensors.append(tensor)
        return {"tensor": tensor, "allocation": allocation, "raw": raw, "guarded": guarded}

    def read(self, tensor):
        data = np.empty_like(tensor["guarded"])
        check(self.acl.aclrtMemcpy(data.ctypes.data, data.nbytes, tensor["allocation"],
                                  data.nbytes, 2), "copy_output")
        guard_ok = bool(np.all(data[:64] == 0xA5) and np.all(data[-64:] == 0xA5))
        raw = tensor["raw"]
        return data[64:-64].view(raw.dtype).reshape(raw.shape).copy(), guard_ok

    def reset(self, tensor):
        original = tensor["guarded"]
        check(self.acl.aclrtMemcpy(tensor["allocation"], original.nbytes, original.ctypes.data,
                                  original.nbytes, 1), "reset_output")

    def stage(self, inputs, outputs, scale):
        size, executor = ct.c_uint64(), PTR()
        check(self.workspace_query(*[t["tensor"] for t in inputs], scale,
            *[t["tensor"] for t in outputs], ct.byref(size), ct.byref(executor)), "workspace_query")
        if size.value > self.workspace_size:
            self.workspace = self.allocate(size.value)
            self.workspace_size = size.value
        return size.value, executor

    def execute(self, staged):
        size, executor = staged
        check(self.launch(self.workspace, size, executor, self.stream), "kernel_launch")
        check(self.acl.aclrtSynchronizeStream(self.stream), "stream_sync")

    def close(self):
        for tensor in reversed(self.tensors):
            self.nn.aclDestroyTensor(tensor)
        for allocation in reversed(self.allocations):
            self.acl.aclrtFree(allocation)
        self.acl.aclrtDestroyStream(self.stream)
        self.acl.aclFinalize()
