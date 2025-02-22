#ifndef __CPU_ADD_H__
#define __CPU_ADD_H__

#include "operators.h"
#include <numeric>
#include <type_traits>

/**
 * @brief CPU 加法操作的描述符
 */
struct AddCpuDescriptor {
    Device device;  // 设备类型
    DT dtype;  // 数据中元素的类型
    uint64_t ndim;  // 结果张量的维度
    uint64_t c_data_size;  // 结果张量的元素数量
    uint64_t const *c_shape;  // 结果张量的形状
    uint64_t const *a_strides;  // 第一个操作数的偏移量步长
    uint64_t const *b_strides;  // 第二个操作数的偏移量步长
    uint64_t *c_indices;  // 结果张量的索引
};

/**
 * @brief CPU 加法操作描述符的指针类型
 */
typedef struct AddCpuDescriptor *AddCpuDescriptor_t;

/**
 * @brief 创建一个用于对两个张量执行 CPU 加法操作的描述符
 * @param infiniopHandle_t 函数中未使用
 * @param desc_ptr 存储创建好的描述符
 * @param c 结果张量描述符
 * @param a 第一个操作数的张量描述符
 * @param b 第二个操作数的张量描述符
 * @return 标识是否创建成功
 */
infiniopStatus_t cpuCreateAddDescriptor(infiniopHandle_t,
                                        AddCpuDescriptor_t *,
                                        infiniopTensorDescriptor_t c,
                                        infiniopTensorDescriptor_t a,
                                        infiniopTensorDescriptor_t b);
/**
 * @brief 根据给定的 CPU 加法操作描述符，对传入的参数执行加法运算
 * @param desc 指定了加法操作指向方式的描述符
 * @param c 结果张量
 * @param a 第一个操作数
 * @param b 第二个操作数
 * @param stream 未使用参数
 * @return 返回操作执行的结果
 */
infiniopStatus_t cpuAdd(AddCpuDescriptor_t desc,
                        void *c, void const *a, void const *b,
                        void *stream);

/**
 * @brief 销毁指定的 CPU 加法操作描述符
 * @param desc 要销毁的加法操作描述符
 * @return 返回表示是否销毁成功的状态
 */
infiniopStatus_t cpuDestroyAddDescriptor(AddCpuDescriptor_t desc);

#endif
