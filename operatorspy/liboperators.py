import os
import platform
import ctypes
from ctypes import c_int, c_int64, c_uint64, Structure, POINTER
from .data_layout import *
from .devices import *

Device = c_int
Optype = c_int

LIB_OPERATORS_DIR = os.path.join(os.environ.get("HOME"), ".infini", "lib")

class TensorDescriptor(Structure):
    """定义TensorDescriptor结构体，该结构体的成员与 C 中同名的结构体相同
    """
    _fields_ = [
        ("dt", DataLayout),
        ("ndim", c_uint64),
        ("shape", POINTER(c_uint64)),
        ("strides", POINTER(c_int64)),
    ]

    def invalidate(self):
        for i in range(self.ndim):
            self.shape[i] = 0
            self.strides[i] = 0


infiniopTensorDescriptor_t = ctypes.POINTER(TensorDescriptor)
"""定义 infiniopTensorDescriptor_t 类型是同名 C 类型的指针
"""

class CTensor:
    """包含 infiniopTensorDescriptor_t 张量描述符 descriptor 和指向实际数据的指针 data
    """
    def __init__(self, desc, data):
        self.descriptor = desc
        self.data = data


class Handle(Structure):
    """定义和 C 中同名同结构的的 infiniop 句柄结构体
    """
    _fields_ = [("device", c_int)]

# 定义 infiniopHandle_t 类型是同名 C 类型的指针
infiniopHandle_t = POINTER(Handle)


# 加载算子的动态库文件，返回值为所加载的动态库对象，其中包含了动态库暴露的所有成员，Open operators library
def open_lib():
    def find_library_in_ld_path(library_name):
        ld_library_path = LIB_OPERATORS_DIR
        paths = ld_library_path.split(os.pathsep)
        for path in paths:
            full_path = os.path.join(path, library_name)
            if os.path.isfile(full_path):
                return full_path
        return None

    system_name = platform.system()
    # Load the library
    # 根据不同的系统，加载不同的库文件
    if system_name == "Windows":
        library_path = find_library_in_ld_path("infiniop.dll")
    elif system_name == "Linux":
        library_path = find_library_in_ld_path("libinfiniop.so")

    assert (
        library_path is not None
    ), f"Cannot find infiniop.dll or libinfiniop.so. Check if INFINI_ROOT is set correctly."

    # 加载动态库文件
    lib = ctypes.CDLL(library_path)

    # 要使用的动态库函数需要显式指定动态库中暴露函数的参数和返回值类型
    # 这里只给 infiniop 和 tensor 句柄指定了，具体算子的句柄在测试文件中再指定具体的参数和返回值类型
    lib.infiniopCreateTensorDescriptor.argtypes = [
        POINTER(infiniopTensorDescriptor_t),
        c_uint64,
        POINTER(c_uint64),
        POINTER(c_int64),
        DataLayout,
    ]
    lib.infiniopCreateHandle.argtypes = [POINTER(infiniopHandle_t), c_int, c_int]
    lib.infiniopCreateHandle.restype = c_int
    lib.infiniopDestroyHandle.argtypes = [infiniopHandle_t]
    lib.infiniopDestroyHandle.restype = c_int

    return lib
