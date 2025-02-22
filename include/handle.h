/**
 * @file handle.h
 * @brief 定义表示设备的句柄的结构体与对应的指针
 * @author 上官永石
 * @date 2025-02-21
 */
#ifndef INFINIOP_HANDLE_H
#define INFINIOP_HANDLE_H

#include "device.h"

typedef struct HandleStruct {
    Device device;
} HandleStruct;

/**
 * @brief 表示设备句柄的指针
 */
typedef HandleStruct *infiniopHandle_t;

#endif
