/**
 * @file export.h
 * @brief 定义导出符号的宏，以在不同的平台上实现不同的导出方式
 * @author 上官永石
 * @date 2025-02-21
 */
#ifndef __EXPORT_H__
#define __EXPORT_H__

// 根据不同的平台，将 __export__ 宏定义为不同的导出符号，以在动态库中显式指定要导出哪些函数给外部使用
#if defined(_WIN32)
// 默认情况下，Windows 下动态库中的函数必须显式导出才能被外部使用
#define __export __declspec(dllexport)
#elif defined(__GNUC__) && ((__GNUC__ >= 4) || (__GNUC__ == 3 && __GNUC_MINOR__ >= 3))
// 高版本的 gcc，使用 __attribute__ 显式指定要暴露给外部的函数，以避免被编译参数 -fvisibility=hidden 隐藏符号
#define __export __attribute__((visibility("default")))
#else
// 默认情况下，linux下，动态库的符号可以直接在外部使用
#define __export
#endif

#ifdef __cplusplus
#define __C extern "C"
#else
#define __C
#endif

#endif// __EXPORT_H__
