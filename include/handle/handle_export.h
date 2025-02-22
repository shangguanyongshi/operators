/**
 * @file handle_export.h
 * @brief 定义 infiniop 句柄创建和销毁的接口，提供给外部使用
 * @author 上官永石
 * @date 2025-02-21
 */
#ifndef INFINIOP_HANDLE_EXPORT_H
#define INFINIOP_HANDLE_EXPORT_H
#include "../status.h"
#include "../handle.h"
#include "../export.h"
#include "../device.h"

__C __export infiniopStatus_t infiniopCreateHandle(infiniopHandle_t *handle_ptr, Device device, int device_id);

__C __export infiniopStatus_t infiniopDestroyHandle(infiniopHandle_t handle);

#endif // INFINIOP_HANDLE_EXPORT_H
