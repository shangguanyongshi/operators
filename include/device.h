/**
 * @file device.h
 * @brief 定义设备类型相关的枚举值
 * @author 上官永石
 * @date 2025-02-14
 */
#ifndef __DEVICE_H__
#define __DEVICE_H__

enum DeviceEnum {
    DevCpu = 0,
    DevNvGpu = 1,
    DevCambriconMlu = 2,
    DevAscendNpu = 3,
    DevMetaxGpu = 4,
    DevMthreadsGpu = 5,
};

typedef enum DeviceEnum Device;

#endif// __DEVICE_H__
