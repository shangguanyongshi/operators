#include "add_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
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

infiniopStatus_t cpuCreateAddDescriptor(infiniopHandle_t,
                                        AddCpuDescriptor_t *desc_ptr,
                                        infiniopTensorDescriptor_t c,
                                        infiniopTensorDescriptor_t a,
                                        infiniopTensorDescriptor_t b) {
    uint64_t ndim = c->ndim;
    if (!isValidBroadcastShape(a, b, c)) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    if (!is_contiguous(a) || !is_contiguous(b) || !is_contiguous(c)) {
        return STATUS_BAD_TENSOR_STRIDES;
    }
    // 只计算 F16 和 F32 数据类型的相加
    if (c->dt != F16 && c->dt != F32) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    // 保证所有张量中元素的类型都相同
    if (c->dt != a->dt || c->dt != b->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }

    // 将 c 的各个维度相乘，得到 c 的元素数量
    uint64_t c_data_size = std::accumulate(c->shape, c->shape + c->ndim, 1ULL, std::multiplies<uint64_t>());

    // get the adjusted strides for a and b
    uint64_t *a_strides = new uint64_t[ndim];
    uint64_t *b_strides = new uint64_t[ndim];
    // 遍历每个维度，计算步长
    for (size_t i = 0; i < ndim; ++i) {
        // 如果 c 的阶数大于 a 和 b，则以 c 的后几个阶数与 a 和 b 的所有阶进行匹配，a 前面的维度都看作是 1
        // 如 c 的形状为 [3, 4, 5, 6]，a 的形状为 [4, 5, 6]，实际上通过广播，a 的形状可以看作是 [1, 4, 5, 6]
        // 此时表示 a 的步长时，其前几个维度为 0，后几个维度为 a 的步长
        // i < ndim - a->ndim 是为了索引 c 中比 a 多的前几个维度，i + a->ndim - ndim 是为了索引 a 中的维度
        a_strides[i] = (i < ndim - a->ndim || c->shape[i] != a->shape[i + a->ndim - ndim]) ? 0 : a->strides[i + a->ndim - ndim];
        b_strides[i] = (i < ndim - b->ndim || c->shape[i] != b->shape[i + b->ndim - ndim]) ? 0 : b->strides[i + b->ndim - ndim];
    }

    // 索引初始化为全 0 的数组
    uint64_t *c_indices = new uint64_t[ndim];
    std::fill(c_indices, c_indices + ndim, 0);
    // 根据 c 的形状，创建一个新的数组，用于存储 c 的形状
    uint64_t *c_shape = new uint64_t[ndim];
    std::copy(c->shape, c->shape + ndim, c_shape);

    // 创建 AddCpuDescriptor 时传入的参数都是在当前作用域中新定义并分配的
    *desc_ptr = new AddCpuDescriptor{
        DevCpu,
        c->dt,
        ndim,
        c_data_size,
        c_shape,
        a_strides,
        b_strides,
        c_indices,
    };

    return STATUS_SUCCESS;
}

infiniopStatus_t cpuDestroyAddDescriptor(AddCpuDescriptor_t desc) {
    delete[] desc->c_shape;
    delete[] desc->a_strides;
    delete[] desc->b_strides;
    delete[] desc->c_indices;
    delete desc;
    return STATUS_SUCCESS;
}

template<typename Tdata>
infiniopStatus_t add_cpu(AddCpuDescriptor_t desc, void *c, void const *a, void const *b) {
    // 先将数据转换为对应类型的指针
    auto a_ = reinterpret_cast<Tdata const *>(a);
    auto b_ = reinterpret_cast<Tdata const *>(b);
    auto c_ = reinterpret_cast<Tdata *>(c);
    const auto &indices = desc->c_indices;

    for (uint64_t i = 0; i < desc->c_data_size; ++i, incrementOne(indices, desc->c_shape, desc->ndim)) {
        // 获取 a 和 b 数组中要相加的元素的索引
        auto a_index = compactToFlat(indices, desc->a_strides, desc->ndim);
        auto b_index = compactToFlat(indices, desc->b_strides, desc->ndim);
        // a 和 b 中对应索引的元素相加，结果存储在 c 的第 i 个索引处
        // if constexpr 在编译时就根据常量表达式判断执行哪一段代码
        if constexpr (std::is_same<Tdata, uint16_t>::value) {
            // 如果是 F16 类型，则先将其转换为 F32 类型，相加后再转换为 F16 类型
            c_[i] = f32_to_f16(f16_to_f32(a_[a_index]) + f16_to_f32(b_[b_index]));
        } else {
            c_[i] = a_[a_index] + b_[b_index];
        }
    }
    return STATUS_SUCCESS;
}

infiniopStatus_t cpuAdd(AddCpuDescriptor_t desc,
                        void *c, void const *a, void const *b,
                        void *stream) {
    if (desc->dtype == F16) {
        return add_cpu<uint16_t>(desc, c, a, b);
    }
    if (desc->dtype == F32) {
        return add_cpu<float>(desc, c, a, b);
    }
    return STATUS_BAD_TENSOR_DTYPE;
}
