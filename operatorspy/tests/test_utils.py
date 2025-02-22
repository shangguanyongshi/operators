def get_args():
    """获取命令行运行时的参数参数

    Returns:
        _type_: 返回所解析的参数值
    """
    import argparse

    parser = argparse.ArgumentParser(description="Test Operator")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Whether profile tests",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Run CPU test",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Run CUDA test",
    )
    parser.add_argument(
        "--bang",
        action="store_true",
        help="Run BANG test",
    )
    parser.add_argument(
        "--ascend",
        action="store_true",
        help="Run ASCEND NPU test",
    )
    parser.add_argument(
        "--maca",
        action="store_true",
        help="Run ASCEND NPU test",
    )
    parser.add_argument(
        "--musa",
        action="store_true",
        help="Run MUSA test",
    )

    return parser.parse_args()


def synchronize_device(torch_device):
    import torch
    if torch_device == "cuda":
        torch.cuda.synchronize()
    elif torch_device == "npu":
        torch.npu.synchronize()
    elif torch_device == "mlu":
        torch.mlu.synchronize()
