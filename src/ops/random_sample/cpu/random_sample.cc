#include "../../../devices/cpu/common_cpu.h"
#include "../../utils.h"
#include "random_sample_cpu.h"
#include <cmath>


infiniopStatus_t cpuCreateRandomSampleDescriptor(infiniopHandle_t,
                                                 RandomSampleCpuDescriptor_t *desc_ptr,
                                                 infiniopTensorDescriptor_t result,
                                                 infiniopTensorDescriptor_t probs) {
    // 先对输入条件过滤：
    // 1. porbs 的维度要是 1
    // 2. probs 的数据类型要是 F16
    // 3. result 的数据类型要是 U64
    // 4. result 的维度要是 1 且该维度的大小应该为 1
    int ndim = probs->ndim;
    if (ndim != 1) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    if (!dtype_eq(probs->dt, F16)) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (!dtype_eq(result->dt, U64))
        return STATUS_BAD_TENSOR_DTYPE;
    int voc = probs->shape[0];
    int rLength = result->shape[0];
    if (result->ndim != 1 && rLength != 1) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    *desc_ptr = new RandomSampleCpuDescriptor{
        DevCpu,
        probs->dt,
        voc, // probs->shape[0],
        result->dt,
        rLength};

    return STATUS_SUCCESS;
}

infiniopStatus_t cpuGetRandomSampleWorkspaceSize(RandomSampleCpuDescriptor_t desc, uint64_t *size) {
    // 所需额外空间的大小为 元素个数 * (结果元素大小 + probs 元素大小)
    // 需要中间的结果空间用于选择前 k 个最大值
    // 需要额外个与 probs 相同大小的空间用于冒泡排序（probs 是 const 的）
    *size = desc->voc * (sizeof(uint64_t) + sizeof(desc->dtype));
    return STATUS_SUCCESS;
}

infiniopStatus_t cpuDestroyRandomSampleDescriptor(RandomSampleCpuDescriptor_t desc) {
    delete desc;
    return STATUS_SUCCESS;
}


void random_sample_cpu_f16(RandomSampleCpuDescriptor_t desc,
                           void *workspace,
                           void *result,
                           void const *probs,
                           float random_val,
                           float topp,
                           int topk,
                           float temperature) {
    int voc = desc->voc;
    char *origin = reinterpret_cast<char *>(workspace);
    //排序得到前k个最大值，按照从大到小顺序存储在logits_前k个位置里面
    char *logitsTmp = origin + voc * sizeof(uint64_t);
    uint64_t *indexTmp = (uint64_t *) origin;
    uint16_t *logits_ = (uint16_t *) logitsTmp;


    auto source = reinterpret_cast<const uint16_t *>(probs);

    std::copy(source, source + voc, logits_);
    auto index_ = reinterpret_cast<uint64_t *>(result);

    /* 以上操作对多个内存空间进行了变换
    workspace->origin->indexTmp(int64_t，为了保存前 k 个最大值的索引))[0,...,voc-1]
                       indexTmp 数组后为 logits_ 数组(int16_t，保存从 probs 中复制来的值，用于排序时保存中间结果)
    result->index_ 数组(int64_t，保存最终结果的索引)
    probs->source 数组(int16_t，保存输入数据)
    */
    // 如果k大于voc，调整k为voc
    if (topk > voc) {
        topk = voc;
    }

    for (int i = 0; i < voc; i++) {
        indexTmp[i] = i;
    }
    // 根据 logits_ 进行排序，选出前 k 个最大的元素，按照从大到小的顺序将索引保存在 indexTmp 中
    for (int i = 0; i < topk; i++) {
        for (int j = i + 1; j < voc; j++) {
            if (f16_to_f32(logits_[i]) < f16_to_f32(logits_[j])) {
                float M = f16_to_f32(logits_[i]);
                logits_[i] = logits_[j];
                logits_[j] = f32_to_f16(M);


                int index = indexTmp[i];
                indexTmp[i] = indexTmp[j];
                indexTmp[j] = index;
            }
        }
    }

    //做类似于softmax的temperature变换
    // 变换公式（https://www.zhihu.com/question/649717092/answer/3440074767）
    float reduceM = f16_to_f32(logits_[0]); // 保存最大值
    float reduceS = 0.0f;
    for (int i = 0; i < voc; i++) {
        reduceS += std::exp((f16_to_f32(logits_[i]) - reduceM) / temperature);
    }
    for (int i = 0; i < voc; i++) {
        logits_[i] = f32_to_f16(std::exp((f16_to_f32(logits_[i]) - reduceM) / temperature) / reduceS);
    }
    //在前k个元素里面利用topp选取不超过topp的元素作为数据集
    float tmp = 0.0f;
    int end = 0;
    for (end = 0; end < topk; end++) {
        tmp += f16_to_f32(logits_[end]);
        if (tmp >= topp) {
            break;
        }
    }
    //printf("%d\n", end);
    if (end < topk - 1) {
        end += 1;
    } else {
        end = topk;
    }
    //利用随机数随机输出满足同时满足topk,topp的某个元素在原始向量的index

    float sum_s = 0.0f;
    for (int i = 0; i < end; i++) {
        sum_s += f16_to_f32(logits_[i]);
    }
    random_val *= sum_s;

    sum_s = 0.0f;
    // 经过变换后根据 logits 变换的结果，根据规则选择一个索引，保存到 result 的第一个位置
    for (int i = 0; i < end; i++) {
        sum_s += f16_to_f32(logits_[i]);
        if (random_val < sum_s) {
            index_[0] = indexTmp[i];
            break;
        }
    }
}
/**
 * @brief 返回 probs 中值最大的索引，保存在数组 result[0] 中
 * @param desc RandomSample 操作的描述符
 * @param workspace 选择或排序时可以使用的额外空间
 * @param result 最大索引保存在 result 数组（U32 类型）的第一个位置
 * @param probs 选择索引的依据
 */
void random_sample_cpu_f16(RandomSampleCpuDescriptor_t desc,
                           void *workspace,
                           void *result,
                           void const *probs) {
    // 
    int voc = desc->voc;
    // result -> index_
    auto index_ = reinterpret_cast<uint64_t *>(result);
    // porbs -> source
    auto source = reinterpret_cast<const uint16_t *>(probs);
    
    // 将 workspace 地址转换为 uint16_t 类型的指针 logits_，probs 中的数据以执行
    char *origin = reinterpret_cast<char *>(workspace);
    uint16_t *logits_ = (uint16_t *) origin;

    std::copy(source, source + voc, logits_);
    
    // 选择 logits_(即 probs) 中值最大的索引
    float M = f16_to_f32(logits_[0]);
    int index = 0;
    for (int j = 1; j < voc; j++) {
        if (M < f16_to_f32(logits_[j])) {
            M = f16_to_f32(logits_[j]);
            index = j;
        }
    }

    index_[0] = index;
}

infiniopStatus_t cpuRandomSample(RandomSampleCpuDescriptor_t desc,
                                 void *workspace,
                                 uint64_t workspace_size,
                                 void *result,
                                 void const *probs,
                                 float random_val,
                                 float topp,
                                 int topk,
                                 float temperature,
                                 void *stream) {
    if (dtype_eq(desc->dtype, F16)) {
        if (topp > 0 && topk > 1) {
            random_sample_cpu_f16(desc,
                                  workspace,
                                  result,
                                  probs,
                                  random_val,
                                  topp,
                                  topk,
                                  temperature);
        } else {
            // 没有指定 topp 和 topk，直接选择 probs 中值最大的索引
            random_sample_cpu_f16(desc,
                                  workspace,
                                  result,
                                  probs);
        }
        return STATUS_SUCCESS;
    }

    return STATUS_BAD_TENSOR_DTYPE;
}
