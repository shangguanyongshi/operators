/**
 * @file data_type.h
 * @brief 定义保存数据布局的类型 DT，及部分 static const 表示的数据类型结构
 * @author 上官永石
 * @date 2025-02-20
 */
#ifndef __DATA_TYPE_H__
#define __DATA_TYPE_H__

typedef struct DataLayout {
    // 共 4 字节
    // packed、sign、size 通过位域限制，一同保存前 16 位中
    // mantissa 和 exponent 一起保存在后 16 位中
    unsigned short
        packed : 8, 
        sign : 1, // 是否有符号
        size : 7, // 数据类型的字节个数
        mantissa : 8, // 尾数的位数
        exponent : 8; // 阶数的位数

#ifdef __cplusplus
    bool operator==(const DataLayout &other) const {
        union TypePun {
            DataLayout layout;
            unsigned int i;
        } pun;
        // 将当前对象保存到 pun.layout 中
        pun.layout = *this;
        // 以 unsigned int 的方式获取 4 个字节中的数据
        auto a_ = pun.i;
        pun.layout = other;
        auto b_ = pun.i;
        // 比较两个 unsigned int 的值是否相等就可以得到两个对象是否相同
        return a_ == b_;
    }

    bool operator!=(const DataLayout &other) const {
        return !(*this == other);
    }
#endif
} DataLayout;

typedef struct DataLayout DT;

// clang-format off
const static struct DataLayout
    I8   = {1, 1, 1,  7,  0},
    I16  = {1, 1, 2, 15,  0},
    I32  = {1, 1, 4, 31,  0},
    I64  = {1, 1, 8, 63,  0},
    U8   = {1, 0, 1,  8,  0},
    U16  = {1, 0, 2, 16,  0},
    U32  = {1, 0, 4, 32,  0},
    U64  = {1, 0, 8, 64,  0},
    F16  = {1, 1, 2, 10,  5},
    BF16 = {1, 1, 2,  7,  8},
    F32  = {1, 1, 4, 23,  8},
    F64  = {1, 1, 8, 52, 11};
// clang-format on

#endif// __DATA_TYPE_H__
