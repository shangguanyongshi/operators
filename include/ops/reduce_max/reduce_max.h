#ifndef __REDUCE_MAX_H__
#define __REDUCE_MAX_H__

#include "../../export.h"
#include "../../operators.h"

typedef struct ReduceMaxDescriptor {
    Device device;
} ReduceMaxDescriptor;

typedef ReduceMaxDescriptor *infiniopReduceMaxDescriptor_t;

__C __export infiniopStatus_t infiniopCreateReduceMaxDescriptor(infiniopHandle_t handle,
                                                                infiniopReduceMaxDescriptor_t *desc_ptr,
                                                                infiniopTensorDescriptor_t reduced,
                                                                infiniopTensorDescriptor_t data,
                                                                infiniopTensorDescriptor_t axes,
                                                                int keepdims,
                                                                int noop_with_empty_axes);

__C __export infiniopStatus_t infiniopGetReduceMaxWorkspaceSize(infiniopReduceMaxDescriptor_t desc, uint64_t *size);

__C __export infiniopStatus_t infiniopReduceMax(infiniopReduceMaxDescriptor_t desc,
                                                void *workspace,
                                                uint64_t workspace_size,
                                                void *reduced,
                                                void const *data,
                                                void const *axes,
                                                void *stream);

__C __export infiniopStatus_t infiniopDestroyReduceMaxDescriptor(infiniopReduceMaxDescriptor_t desc);

#endif