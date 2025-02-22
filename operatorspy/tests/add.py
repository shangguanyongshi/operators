from ctypes import POINTER, Structure, c_int32, c_void_p
import ctypes
import sys
import os

# 将当前文件 ../../ 添加到环境变量中，这样就能使用 operatorspy 模块了
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from operatorspy import (
    # operatorspy/devices.py 中的类型
    DeviceEnum,
    # operatorspy/operators.py 中的函数和类型
    open_lib,
    infiniopHandle_t,
    infiniopTensorDescriptor_t,
    # 下面是 operatorspy/utils.py 中的函数
    to_tensor,
    create_handle,
    destroy_handle,
    check_error,
)

from operatorspy.tests.test_utils import get_args
from enum import Enum, auto
import torch


class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE_A = auto()
    INPLACE_B = auto()


class AddDescriptor(Structure):
    """定义与 C 语言中的 AddCpuDescriptor 类型相同的结构体
    """
    _fields_ = [("device", c_int32)]


# infiniopAddDescriptor_t 类型是 C 语言中指向 AddDescriptor 类型的指针
infiniopAddDescriptor_t = POINTER(AddDescriptor)


def add(x, y):
    """调用 torch 中的 add 操作符执行加法运算

    Args:
        x: 第一个操作数
        y: 第二个操作数

    Returns:
        计算结果
    """
    return torch.add(x, y)


def test(
    lib,
    handle,
    torch_device,
    c_shape, 
    a_shape, 
    b_shape,
    tensor_dtype=torch.float16,
    inplace=Inplace.OUT_OF_PLACE,
):
    """使用所加载的动态库，执行实际的算子计算操作

    Args:
        lib: 所加载的 infiniop 动态库
        handle: infiniop 句柄
        torch_device: torch 设备类型
        c_shape: 结果张量的形状
        a_shape: 第一个操作数的形状
        b_shape: 第二个操作数的形状
        tensor_dtype: 张量的数据类型. 默认为 torch.float16.
        inplace: 是否使用 inplace 操作. 默认为 Inplace.OUT_OF_PLACE.
    """
    print(
        f"Testing Add on {torch_device} with c_shape:{c_shape} a_shape:{a_shape} b_shape:{b_shape} dtype:{tensor_dtype} inplace: {inplace.name}"
    )
    if a_shape != b_shape and inplace != Inplace.OUT_OF_PLACE:
        print("Unsupported test: broadcasting does not support in-place")
        return

    a = torch.rand(a_shape, dtype=tensor_dtype).to(torch_device)
    b = torch.rand(b_shape, dtype=tensor_dtype).to(torch_device)
    c = torch.rand(c_shape, dtype=tensor_dtype).to(torch_device) if inplace == Inplace.OUT_OF_PLACE else (a if inplace == Inplace.INPLACE_A else b)

    # 使用 torch 计算的正确结果
    ans = add(a, b)

    # 将 torch 张量转换为 infiniop 张量
    a_tensor = to_tensor(a, lib)
    b_tensor = to_tensor(b, lib)
    c_tensor = to_tensor(c, lib) if inplace == Inplace.OUT_OF_PLACE else (a_tensor if inplace == Inplace.INPLACE_A else b_tensor)
    descriptor = infiniopAddDescriptor_t()

    check_error(
        lib.infiniopCreateAddDescriptor(
            handle,
            ctypes.byref(descriptor),
            c_tensor.descriptor,
            a_tensor.descriptor,
            b_tensor.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    # 实际执行时，每个操作数张量描述符中的信息已经在调用 infiniopCreateAddDescriptor 时被保存到 add 算子描述符中了，
    # 实际实现时不应该依赖具体操作张量的信息，所以这里将张量描述符中的 shape 和 strides 信息置为无效，以防止运算时直接使用这些信息
    c_tensor.descriptor.contents.invalidate()
    a_tensor.descriptor.contents.invalidate()
    b_tensor.descriptor.contents.invalidate()

    check_error(
        lib.infiniopAdd(descriptor, c_tensor.data, a_tensor.data, b_tensor.data, None)
    )
    assert torch.allclose(c, ans, atol=0, rtol=1e-3)
    check_error(lib.infiniopDestroyAddDescriptor(descriptor))


def test_cpu(lib, test_cases):
    """测试 CPU 上的给定算子

    Args:
        lib: 要测试的 CPU 算子动态库
        test_cases: 要执行的测试用例
    """
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)
    for c_shape, a_shape, b_shape, inplace in test_cases:
        # 测试数据类型为 float16 的数据
        test(lib, handle, "cpu", c_shape, a_shape, b_shape, tensor_dtype=torch.float16, inplace=inplace)
        # 测试数据类型为 float32 的数据
        test(lib, handle, "cpu", c_shape, a_shape, b_shape, tensor_dtype=torch.float32, inplace=inplace)
    destroy_handle(lib, handle)


def test_cuda(lib, test_cases):
    device = DeviceEnum.DEVICE_CUDA
    handle = create_handle(lib, device)
    for c_shape, a_shape, b_shape, inplace in test_cases:
        test(lib, handle, "cuda", c_shape, a_shape, b_shape, tensor_dtype=torch.float16, inplace=inplace)
        test(lib, handle, "cuda", c_shape, a_shape, b_shape, tensor_dtype=torch.float32, inplace=inplace)
    destroy_handle(lib, handle)


def test_bang(lib, test_cases):
    import torch_mlu

    device = DeviceEnum.DEVICE_BANG
    handle = create_handle(lib, device)
    for c_shape, a_shape, b_shape, inplace in test_cases:
        test(lib, handle, "mlu", c_shape, a_shape, b_shape, tensor_dtype=torch.float16, inplace=inplace)
        test(lib, handle, "mlu", c_shape, a_shape, b_shape, tensor_dtype=torch.float32, inplace=inplace)
    destroy_handle(lib, handle)

def test_musa(lib, test_cases):
    import torch_musa

    device = DeviceEnum.DEVICE_MUSA
    handle = create_handle(lib, device)
    for c_shape, a_shape, b_shape, inplace in test_cases:
        test(lib, handle, "musa", c_shape, a_shape, b_shape, tensor_dtype=torch.float16, inplace=inplace)
        test(lib, handle, "musa", c_shape, a_shape, b_shape, tensor_dtype=torch.float32, inplace=inplace)
    destroy_handle(lib, handle)


if __name__ == "__main__":
    test_cases = [
        # c_shape, a_shape, b_shape, inplace
        # ((32, 150, 512000), (32, 150, 512000), (32, 150, 512000), Inplace.OUT_OF_PLACE),
        # ((32, 150, 51200), (32, 150, 51200), (32, 150, 1), Inplace.OUT_OF_PLACE),
        # ((32, 150, 51200), (32, 150, 51200), (32, 150, 51200), Inplace.OUT_OF_PLACE),
        ((1, 3), (1, 3), (1, 3), Inplace.OUT_OF_PLACE),
        ((), (), (), Inplace.OUT_OF_PLACE),
        ((3, 3), (3, 3), (3, 3), Inplace.OUT_OF_PLACE),
        ((2, 20, 3), (2, 1, 3), (2, 20, 3), Inplace.OUT_OF_PLACE),
        ((32, 20, 512), (32, 20, 512), (32, 20, 512), Inplace.INPLACE_A),
        ((32, 20, 512), (32, 20, 512), (32, 20, 512), Inplace.INPLACE_B),
        ((32, 256, 112, 112), (32, 256, 112, 1), (32, 256, 112, 112), Inplace.OUT_OF_PLACE),
        ((32, 256, 112, 112), (32, 256, 112, 112), (32, 256, 112, 112), Inplace.OUT_OF_PLACE),
        ((2, 4, 3), (2, 1, 3), (4, 3), Inplace.OUT_OF_PLACE),
        ((2, 3, 4, 5), (2, 3, 4, 5), (5,), Inplace.OUT_OF_PLACE),
        ((3, 2, 4, 5), (4, 5), (3, 2, 1, 1), Inplace.OUT_OF_PLACE),
    ]
    args = get_args()
    lib = open_lib()
    lib.infiniopCreateAddDescriptor.restype = c_int32
    lib.infiniopCreateAddDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopAddDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]
    lib.infiniopAdd.restype = c_int32
    lib.infiniopAdd.argtypes = [
        infiniopAddDescriptor_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroyAddDescriptor.restype = c_int32
    lib.infiniopDestroyAddDescriptor.argtypes = [
        infiniopAddDescriptor_t,
    ]

    if args.cpu:
        test_cpu(lib, test_cases)
    if args.cuda:
        test_cuda(lib, test_cases)
    if args.bang:
        test_bang(lib, test_cases)
    if args.musa:
        test_musa(lib, test_cases)
    if not (args.cpu or args.cuda or args.bang or args.musa):
        test_cpu(lib, test_cases)
    print("\033[92mTest passed!\033[0m")
