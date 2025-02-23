/**
 * @file random_sample_cpu.h
 * @brief 大模型联合采样算子（https://blog.csdn.net/qq_43243579/article/details/136331123）
 * @author 上官永石
 * @date 2025-02-23
 */
#ifndef __CPU_RANDOM_SAMPLE_H__
#define __CPU_RANDOM_SAMPLE_H__

#include "operators.h"
struct RandomSampleCpuDescriptor {
    Device device; // 设备类型
    DT dtype; // 执行计算所依赖的评分张量(probs)的元素的类型
    int voc; // 执行计算所依赖的评分张量(probs)中元素的个数
    DT rDtype; // 结果张量中元素的类型
    int rLength; // 结果的长度
};

typedef struct RandomSampleCpuDescriptor *RandomSampleCpuDescriptor_t;

/**
 * @brief 根据 result 和 probs 创建 RandomSample 描述符
 * @param result 结果张量的信息
 * @param probs 执行 RandomSample 计算时，所需要的评分张量的描述符
 * @return 返回是否创建成功
 */
infiniopStatus_t cpuCreateRandomSampleDescriptor(infiniopHandle_t,
                                                 RandomSampleCpuDescriptor_t *,
                                                 infiniopTensorDescriptor_t result,
                                                 infiniopTensorDescriptor_t probs);

/**
 * @brief 获取执行该操作所需要的额外内存空间
 * @param desc RandomSample 的描述符
 * @param size 计算的所需要的内存大小所保存的位置
 * @return 返回是否计算成功
 */
infiniopStatus_t cpuGetRandomSampleWorkspaceSize(RandomSampleCpuDescriptor_t desc, uint64_t *size);

/**
 * @brief 执行实际的 RandomSample 计算
 * @param desc RandomSample 描述符
 * @param workspace 用户根据 cpuGetRandomSampleWorkspaceSize 手动分配的所需要的额外内存空间地址
 * @param workspace_size 所分配的额外内存空间的大小
 * @param result 结果数据保存的地址（实际的结果是一个值，保存在 result 的第一个位置）
 * @param probs 执行 RandomSample 计算时，所需要的评分张量的地址
 * @param random_val 执行 RandomSample 计算时，所需要的随机数
 * @param topp 进行筛选时，保留概率总和不超过 topp 的元素
 * @param topk 进行筛选时，选出前 k 个最大的元素
 * @param temperature 执行 RandomSample 计算时，所需要的温度值
 * @param stream 未使用
 * @return 
 */
infiniopStatus_t cpuRandomSample(RandomSampleCpuDescriptor_t desc,
                                 void *workspace,
                                 uint64_t workspace_size,
                                 void *result,
                                 void const *probs,
                                 float random_val,
                                 float topp,
                                 int topk,
                                 float temperature,
                                 void *stream);

/** 
 * @brief 销毁 RandomSample 描述符
 * @param desc RandomSample 描述符
 * @return 返回是否销毁成功
 */
infiniopStatus_t cpuDestroyRandomSampleDescriptor(RandomSampleCpuDescriptor_t desc);

#endif
