#ifndef __CPU_REDUCE_H__
#define __CPU_REDUCE_H__

#include "../../../devices/cpu/common_cpu.h"
#include "operators.h"

struct ReduceCpuDescriptor {
    Device device;
    DT dtype;
    uint64_t data_ndim; // 输入数据的维度
    uint64_t data_size; // 输入数据的总元素个数
    uint64_t *data_shape; // 输入数据的形状
    uint64_t const *data_strides; // 输入数据的步长
    uint64_t *data_indices; // 用于根据 reduced_indices 遍历 data 的指定维度

    
    uint64_t reduced_ndim; // 输出数据的维度
    uint64_t reduced_size; // 输出数据的总元素个数（初始化为 传入 reduced 元素个数，实际计算时根据参数确定值）
    uint64_t *reduced_shape; // 输出数据的形状（初始化为全 0，实际计算时根据参数确定值）
    uint64_t *reduced_indices; // 用于遍历输出数据的索引（初始化为全 0，实际计算时根据参数确定值）

    int keepdims; // 是否保留减少后的维度，1 表示保留，0 表示不保留
    int noop_with_empty_axes; // 指定输入参数 axes 为空时的行为，1 表示返回输入数据，0 表示返回对所有维度进行操作
    int reduce_type; // 1: max, 2: min, 3: mean
};

typedef struct ReduceCpuDescriptor *ReduceCpuDescriptor_t;

infiniopStatus_t cpuCreateReduceDescriptor(infiniopHandle_t handle,
                                           ReduceCpuDescriptor_t *desc_ptr,
                                           infiniopTensorDescriptor_t reduced,
                                           infiniopTensorDescriptor_t data,
                                           infiniopTensorDescriptor_t axes,
                                           int keepdims,
                                           int noop_with_empty_axes,
                                           int reduce_type);

infiniopStatus_t cpuGetReduceWorkspaceSize(ReduceCpuDescriptor_t desc, uint64_t *size);

infiniopStatus_t cpuReduce(ReduceCpuDescriptor_t desc,
                           void *workspace,
                           uint64_t workspace_size,
                           void *reduced,
                           void const *data,
                           void const *axes,
                           void *stream);


infiniopStatus_t cpuDestroyReduceDescriptor(ReduceCpuDescriptor_t desc); 
#endif