/**
 * @file cpu_handle.h
 * @brief 定义表示 CPU 设备的句柄，及创建 CPU 设备句柄的函数
 * @author 上官永石
 * @date 2025-02-21
 */
#ifndef CPU_HANDLE_H
#define CPU_HANDLE_H

#include "device.h"
#include "status.h"

struct CpuContext {
    Device device;
};
typedef struct CpuContext *CpuHandle_t;

infiniopStatus_t createCpuHandle(CpuHandle_t *handle_ptr);

#endif
