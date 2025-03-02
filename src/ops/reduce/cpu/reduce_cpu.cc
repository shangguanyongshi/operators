#include "reduce_cpu.h"
#include "../../utils.h"


/**
 * @brief 将 indices 表示的索引按照字典序递增 1，即在当前索引的基础上，转换为下一个索引
 * @param indices 当前的索引
 * @param shape 索引所确定张量的维度
 * @param ndim 张量的阶数
 */
inline void incrementOne(uint64_t *indices, uint64_t const *shape, uint64_t ndim) {
    // 每次优先从最后一维开始递增
    for (int64_t i = ndim - 1; i >= 0; --i) {
        // 如果当前维度递增后没有超过该维度的最大值，则直接递增并返回
        if (++indices[i] != shape[i]) {
            return;
        }
        // 如果递增后等于了最大值，则将该维度的索引置为 0，继续递增前一个维度
        indices[i] = 0;
    }
}

/**
 * @brief 根据给定的索引和步长，计算该索引确定的元素在一维数组中的位置
 * @param indices 给定的索引
 * @param strides 每个维度的偏移步长
 * @param ndim 索引的总维度
 * @return  返回该索引确定的元素在一维数组中的位置
 */
inline uint64_t compactToFlat(uint64_t const *indices, uint64_t const *strides, uint64_t ndim) {
    return std::inner_product(indices, indices + ndim, strides, uint64_t(0));
}


infiniopStatus_t cpuCreateReduceDescriptor(infiniopHandle_t handle,
                                           ReduceCpuDescriptor_t *desc_ptr,
                                           infiniopTensorDescriptor_t reduced,
                                           infiniopTensorDescriptor_t data,
                                           infiniopTensorDescriptor_t axes,
                                           int keepdims,
                                           int noop_with_empty_axes,
                                           int reduce_type) {
  
    // 检查参数的合法性
    // 1. 检查 axes 非空时应该为一个标量，且类型为 int64
    if (axes != nullptr) {
        if (axes->ndim != 0) {
            return STATUS_BAD_TENSOR_SHAPE;
        }
        if (axes->dt != I64) {
            return STATUS_BAD_TENSOR_DTYPE;
        }
    }
    // 2. 检查 data 的类型和形状都正确
    if (data->dt != F16 && data->dt != F32) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (!is_contiguous(data)) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    // 3. 检查 reduced 的类型和形状都正确
    if (reduced->dt != data->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    // 如果要保留维度，reduced 应该可以广播到 data
    if (keepdims && !isValidBroadcastShape(data, reduced)) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    // 如果不保留维度
    if (keepdims == 0) {
        // 如果 axes 为空，noop_with_empty_axes 为 0，reduced 应该是个标量
        if (axes == nullptr && noop_with_empty_axes == 0 && reduced->ndim != 0) {
            return STATUS_BAD_TENSOR_SHAPE;
        }
        // 如果 axes 非空，且 data 非标量时，reduced 应该比 data 少一个维度
        if (axes != nullptr && reduced->ndim != 0 && reduced->ndim != data->ndim - 1) {
            return STATUS_BAD_TENSOR_SHAPE;
        }
    }
    // 计算创建 ReduceCpuDescriptor 所需的参数
    // 1. 计算 data 的总元素个数
    uint64_t data_size =
        std::accumulate(data->shape, data->shape + data->ndim, uint64_t(1), std::multiplies<uint64_t>());
    // 2. 保存 data 的形状
    uint64_t *data_shape = new uint64_t[data->ndim];
    std::copy(data->shape, data->shape + data->ndim, data_shape);
    // 3. 保存 data 的步长
    uint64_t *data_strides = new uint64_t[data->ndim];
    std::copy(data->strides, data->strides + data->ndim, data_strides);
    // 4. 初始化 data_indices
    uint64_t *data_indices = new uint64_t[data->ndim];
    std::fill(data_indices, data_indices + data->ndim, 0);
    // 5. 初始化 reduced 的形状为全 0 （运行时传入的 axes 是否为空决定了 reduced 的实际维度，因此推迟到运行时计算）
    uint64_t *reduced_shape = new uint64_t[reduced->ndim];
    std::fill(reduced_shape, reduced_shape + reduced->ndim, 0);
    // 6. 初始时根据 reduced 的形状计算总元素个数
    uint64_t reduced_size =
        std::accumulate(reduced->shape, reduced->shape + reduced->ndim, uint64_t(1), std::multiplies<uint64_t>());
    // 7. 初始化 reduced_indices
    uint64_t *reduced_indices = new uint64_t[reduced->ndim];
    std::fill(reduced_indices, reduced_indices + reduced->ndim, 0);

    // 创建 ReduceCpuDescriptor
    *desc_ptr = new ReduceCpuDescriptor{
        DevCpu,
        data->dt,
        data->ndim,
        data_size,
        data_shape,
        data_strides,
        data_indices,
        reduced->ndim,
        reduced_size,
        reduced_shape,
        reduced_indices,
        keepdims,
        noop_with_empty_axes,
        reduce_type,
    };
    return STATUS_SUCCESS;
}

infiniopStatus_t cpuGetReduceWorkspaceSize(ReduceCpuDescriptor_t desc, uint64_t *size) {
    // 执行 mean 运算，且数据类型为 F16 时，使用额外的 float 类型的内存空间，避免 F32 转换为 F16 导致精度下降
    if (desc->reduce_type == 3 && desc->dtype == F16) {
        // std::cout << desc->reduced_size << std::endl;
        *size = desc->reduced_size * sizeof(float);
    }
    return STATUS_SUCCESS;
}

template<typename Tdata>
infiniopStatus_t reduce_cpu(ReduceCpuDescriptor_t desc,
                            void *workspace,
                            uint64_t workspace_size,
                            void *reduced,
                            void const *data,
                            void const *axes,
                            void *stream) {
    
    // 1. 先将输入转换为对应类型的数据
    auto reduced_data = reinterpret_cast<Tdata *>(reduced);
    // 当执行 mean 且 reduced_data 的类型为 F16 时，累加相处的计算过程使用 workspace_data，每个维度的结果直接保存到 reduced_data 中
    auto workspace_data = reinterpret_cast<float *>(workspace);
    auto data_data = reinterpret_cast<Tdata const *>(data);
    auto axes_data = reinterpret_cast<int64_t const *>(axes);

    if (desc->data_size == 1) {
        // 如果只有一个元素，表示是一个标量，直接复制返回
        reduced_data[0] = data_data[0];
        return STATUS_SUCCESS;
    } else if (desc->data_size == 0) {
        // 如果元素个数为 0，表示是空集合，应该根据类型，返回无穷大
        if constexpr (std::is_same<Tdata, uint16_t>::value) {
            if (desc->reduce_type == 1) {
                // 返回 float32 的负无穷小
                reduced_data[0] = f32_to_f16(-std::numeric_limits<float>::infinity());
            } else if (desc->reduce_type == 2) {
                // 返回 float32 的正无穷大
                reduced_data[0] = f32_to_f16(std::numeric_limits<float>::infinity());
            } else if (desc->reduce_type == 3) {
                // 返回 float32 的 NaN
                reduced_data[0] = f32_to_f16(std::numeric_limits<float>::quiet_NaN());
            }
        } else {
            if (desc->reduce_type == 1) {
                // 返回 float32 的负无穷小
                reduced_data[0] = -std::numeric_limits<float>::infinity();
            } else if (desc->reduce_type == 2) {
                // 返回 float32 的正无穷大
                reduced_data[0] = std::numeric_limits<float>::infinity();
            } else if (desc->reduce_type == 3) {
                // 返回 float32 的 NaN
                reduced_data[0] = std::numeric_limits<float>::quiet_NaN();
            }
        }
        return STATUS_SUCCESS;
    }

    // 2. 计算所操作的维度
    int64_t op_axes = -1; // 记录实际操作的正数维度，默认 -1 表示对所有维度进行操作
    if (axes_data == nullptr) {
        if (desc->noop_with_empty_axes != 0) {
            // 如果 noop_with_empty_axes 为 true，则将 data 直接复制到 reduced 中后返回
            std::copy(data_data, data_data + desc->data_size, reduced_data);
            return STATUS_SUCCESS;
        }
        // 如果 noop_with_empty_axes 为 false，op_axes = -1，对所有维度进行操作
    } else {
        // 判断指定的维度是否在 data 阶数范围内 [-data.ndim, data.ndim - 1]
        if ((*axes_data) < -static_cast<int64_t>(desc->data_ndim) ||
            (*axes_data) >= static_cast<int64_t>(desc->data_ndim)) {
            return STATUS_BAD_PARAM;
        }
        // 将指定的维度转换为正数
        op_axes = (*axes_data) < 0 ? (*axes_data) + static_cast<int64_t>(desc->data_ndim) : (*axes_data);
    }

    // 3. 计算输出张量的信息，只有对指定维度操作时才需要计算 reduced 的形状和大小
    if (op_axes != -1) {
        desc->reduced_size = 1; // 先设置为 0，在下面通过累乘得到最终值
        if (desc->keepdims == 0) {
            // 不保留维度时，reduced 的维度比 data 少 1
            desc->reduced_ndim = desc->data_ndim - 1;
            // reduced 的形状中 op_axes 对应的维度被删除
            for (uint64_t i = 0, j = 0; i < desc->data_ndim; ++i) {
                if (i != op_axes) {
                    desc->reduced_shape[j++] = desc->data_shape[i];
                    desc->reduced_size *= desc->data_shape[i];
                }
            }
        } else {
            // 保留维度时，reduced 的维度和 data 相同
            desc->reduced_ndim = desc->data_ndim;
            // reduced 的形状中 op_axes 对应的维度被设置为 1
            for (uint64_t i = 0; i < desc->data_ndim; ++i) {
                if (i == op_axes) {
                    desc->reduced_shape[i] = 1;
                    continue;
                }
                desc->reduced_shape[i] = desc->data_shape[i];
                desc->reduced_size *= desc->data_shape[i];
            }
        }
    }

    // 4. 如果是对所有维度进行操作，直接遍历 data 的每个元素，计算后保存到 reduced_data[0] 中，直接返回
    if (op_axes == -1) {
        // 预保存第一个元素，避免 reduced_data 或 workspace_data 的原有值对结果造成影响
        if (desc->reduce_type == 3 && desc->dtype == F16) {
            workspace_data[0] = f16_to_f32(data_data[0]);
        } else {
            reduced_data[0] = data_data[0];
        }
        for (uint64_t i = 1; i < desc->data_size; ++i) {
            // 根据 reduce_type 选择不同的操作
            if (desc->reduce_type == 1) {
                // 最大值
                if constexpr (std::is_same<Tdata, uint16_t>::value) {
                    reduced_data[0] = f32_to_f16(std::max(f16_to_f32(reduced_data[0]), f16_to_f32(data_data[i])));
                } else {
                    reduced_data[0] = std::max(reduced_data[0], data_data[i]);
                }
            } else if (desc->reduce_type == 2) {
                // 最小值
                if constexpr (std::is_same<Tdata, uint16_t>::value) {
                    reduced_data[0] = f32_to_f16(std::min(f16_to_f32(reduced_data[0]), f16_to_f32(data_data[i])));
                } else {
                    reduced_data[0] = std::min(reduced_data[0], data_data[i]);
                }
            } else if (desc->reduce_type == 3) {
                // 平均值
                if constexpr (std::is_same<Tdata, uint16_t>::value) {
                    workspace_data[0] = workspace_data[0] + f16_to_f32(data_data[i]);
                } else {
                    reduced_data[0] = reduced_data[0] + data_data[i];
                }
            }
        }
        // 如果是平均值，需要除以元素个数
        if (desc->reduce_type == 3) {
            if constexpr (std::is_same<Tdata, uint16_t>::value) {
                // 将结果转换为 F16 保存到 reduced_data 中
                reduced_data[0] = f32_to_f16(workspace_data[0] / desc->data_size);
            } else {
                reduced_data[0] = reduced_data[0] / desc->data_size;
            }
        }
        return STATUS_SUCCESS;
    }
    
    // 5. 对指定的维度进行操作，遍历 reduced_data 的每个索引，根据索引，从 data_data 中获取对应元素
    const auto &indices = desc->reduced_indices;
    const auto &data_indices = desc->data_indices;
    for (uint64_t i = 0; i < desc->reduced_size; ++i, incrementOne(indices, desc->reduced_shape, desc->reduced_ndim)) {
        // 从 indices 获取对应的 data_indices 索引
        std::copy(indices, indices + desc->reduced_ndim, data_indices);
        if (desc->keepdims == 0) {
            // 如果不保留维度，indices 不包含第 op_axes 维度，需要将 op_axes 对应的维度插入
            for (uint64_t j = desc->data_ndim - 1; j > op_axes; --j) {
                data_indices[j] = data_indices[j - 1];
            }
            data_indices[op_axes] = 0;
        }

        // 遍历 data 的 op_axes 维度的所有元素，计算结果保存到 reduced 的 indices 位置
        // 先将 reduced_data[i] 的值设置为当前维度的第一个元素
        data_indices[op_axes] = 0;
        if (desc->reduce_type == 3 && desc->dtype == F16) {
            workspace_data[i] = f16_to_f32(data_data[compactToFlat(data_indices, desc->data_strides, desc->data_ndim)]);
        } else {
            reduced_data[i] = data_data[compactToFlat(data_indices, desc->data_strides, desc->data_ndim)];
        }

        // 遍历当前维度的剩余元素
        for (uint64_t j = 1; j < desc->data_shape[op_axes]; ++j) {
            data_indices[op_axes] = j;
            // 获取在 data 中的索引
            auto data_index = compactToFlat(data_indices, desc->data_strides, desc->data_ndim);
            // 根据 reduce_type 选择不同的操作
            if (desc->reduce_type == 1) {
                // 最大值
                if constexpr (std::is_same<Tdata, uint16_t>::value) {
                    reduced_data[i] = f32_to_f16(std::max(f16_to_f32(reduced_data[i]), f16_to_f32(data_data[data_index])));
                } else {
                    reduced_data[i] = std::max(reduced_data[i], data_data[data_index]);
                }
            } else if (desc->reduce_type == 2) {
                // 最小值
                if constexpr (std::is_same<Tdata, uint16_t>::value) {
                    reduced_data[i] = f32_to_f16(std::min(f16_to_f32(reduced_data[i]), f16_to_f32(data_data[data_index])));
                } else {
                    reduced_data[i] = std::min(reduced_data[i], data_data[data_index]);
                }
            } else if (desc->reduce_type == 3) {
                // 平均值
                if constexpr (std::is_same<Tdata, uint16_t>::value) {
                    workspace_data[i] = workspace_data[i] + f16_to_f32(data_data[data_index]);
                } else {
                    reduced_data[i] = reduced_data[i] + data_data[data_index];
                }
            }
        }
        if (desc->reduce_type == 3) {
            // 如果是平均值，需要除以元素个数
            if constexpr (std::is_same<Tdata, uint16_t>::value) {
                reduced_data[i] = f32_to_f16(workspace_data[i] / desc->data_shape[op_axes]);
            } else {
                reduced_data[i] = reduced_data[i] / desc->data_shape[op_axes];
            }
        }
    }

    return STATUS_SUCCESS;
}

infiniopStatus_t cpuReduce(ReduceCpuDescriptor_t desc,
                           void *workspace,
                           uint64_t workspace_size,
                           void *reduced,
                           void const *data,
                           void const *axes,
                           void *stream) {
    
      if (desc->dtype == F16) {
          return reduce_cpu<uint16_t>(desc, workspace, workspace_size, reduced, data, axes, stream);
      }
      if (desc->dtype == F32) {
          return reduce_cpu<float>(desc, workspace, workspace_size, reduced, data, axes, stream);
      }
      return STATUS_BAD_TENSOR_DTYPE;
}

infiniopStatus_t cpuDestroyReduceDescriptor(ReduceCpuDescriptor_t desc) {
    delete desc;
    delete[] desc->data_shape;
    delete[] desc->data_strides;
    delete[] desc->data_indices;
    delete[] desc->reduced_shape;
    delete[] desc->reduced_indices;

    return STATUS_SUCCESS;
}