#include "../utils.h"
#include "operators.h"
#include "reduce.h"

#ifdef ENABLE_CPU
#include "cpu/reduce_cpu.h"
#endif

__C infiniopStatus_t infiniopCreateReduceDescriptor(
    infiniopHandle_t handle,
    infiniopReduceDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t reduced,
    infiniopTensorDescriptor_t data,
    infiniopTensorDescriptor_t axes,
    int keepdims,
    int noop_with_empty_axes,
    int reduce_type) {

    switch (handle->device) {
#ifdef ENABLE_CPU
    case DevCpu:
        return cpuCreateReduceDescriptor(handle,
                                         (ReduceCpuDescriptor_t *)desc_ptr,
                                         reduced,
                                         data,
                                         axes,
                                         keepdims,
                                         noop_with_empty_axes,
                                         reduce_type);
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopGetReduceWorkspaceSize(infiniopReduceDescriptor_t desc, uint64_t *size) {
    switch (desc->device) {
#ifdef ENABLE_CPU
    case DevCpu:
        return cpuGetReduceWorkspaceSize((ReduceCpuDescriptor_t)desc, size);
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopReduce(
    infiniopReduceDescriptor_t desc,
    void *workspace,
    uint64_t workspace_size,
    void *reduced,
    void const *data,
    void const *axes,
    void *stream) {

    switch (desc->device) {
#ifdef ENABLE_CPU
    case DevCpu:
        return cpuReduce((ReduceCpuDescriptor_t)desc, workspace, workspace_size, reduced, data, axes, stream);
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopDestroyReduceDescriptor(infiniopReduceDescriptor_t desc) {
    switch (desc->device) {
#ifdef ENABLE_CPU
    case DevCpu:
        return cpuDestroyReduceDescriptor((ReduceCpuDescriptor_t)desc);
#endif
    }
    return STATUS_BAD_DEVICE;
}