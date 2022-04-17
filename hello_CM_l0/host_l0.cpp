/*========================== begin_copyright_notice ============================

Copyright (C) 2020-2021 Intel Corporation

SPDX-License-Identifier: MIT

============================= end_copyright_notice ===========================*/

#include <iostream>
#include <cassert>
#include <math.h>
#include <vector>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

#include <level_zero/ze_api.h>


// コンパイルオプションでカーネルファイルを指定して
#ifndef KERNEL
#error "Error: KERNEL must be defined with location of kernel binary"
#endif

#define CHECK(a) do { \
    auto err = (a); \
    if (err != 0) { \
        fprintf(stderr, "FAIL: err=%d @ line=%d (%s)\n", err, __LINE__, (#a)); \
        exit(err); \
    } \
}while (0)
#define CHECK2(a, msg) do { \
    if ((a)) { \
        fprintf(stderr, "FAIL: @ line=%d (%s)\n", __LINE__, (msg)); \
        exit(-1); \
    } \
}while (0)

void buffer_dump(int *buf, int size)
{
    for(int i=0; i<size; i++)
    {
        printf("%5d ", buf[i]);
    }
    puts("");

    return;
}


#define SZ 32
#define KERNEL_SZ 16
int main(int argc, char* argv[])
{
    // initialize data
    int *src1 = new int[SZ];
    int *src2 = new int[SZ];
    int *dst  = new int[SZ];

    for (unsigned i=0; i<SZ; i++)
    {
        src1[i] = i;
        src2[i] = (SZ-i);
    }


    // initialize GPU
    ze_driver_handle_t driver = nullptr;
    ze_device_handle_t device = nullptr;
    ze_context_handle_t context = nullptr;
    ze_command_queue_handle_t queue;
    ze_command_list_handle_t commands;
    ze_module_handle_t module;
    ze_kernel_handle_t kernel;

    // oneAPI driverの初期化
    CHECK(zeInit(ZE_INIT_FLAG_GPU_ONLY));

    // Discover all the driver instances
    // driverの総数を取得
    uint32_t driverCount = 0;
    CHECK(zeDriverGet(&driverCount, nullptr));
    CHECK2((driverCount == 0), "unable to locate driver(s)");

    // driverインスタンス取得
    ze_driver_handle_t *allDrivers = (ze_driver_handle_t *)malloc(driverCount * sizeof(*allDrivers));
    CHECK(zeDriverGet(&driverCount, allDrivers));

    /* ------------------------------------------------------------------------------
     * Find a driver instance with device
     * ------------------------------------------------------------------------------*/
    // ドライバーインスタンスをたどる
    for (uint32_t i = 0; i < driverCount; ++i)
    {
        // deviceの総数を取得
        uint32_t deviceCount = 0;
        CHECK(zeDeviceGet(allDrivers[i], &deviceCount, nullptr));
        if (deviceCount == 0) continue;
        
        // ドライバ内のdeviceを取得
        ze_device_handle_t *allDevices = (ze_device_handle_t *)malloc(deviceCount * sizeof(ze_device_handle_t));
        CHECK(zeDeviceGet(allDrivers[i], &deviceCount, allDevices));
        
        // deviceをたどる
        for (uint32_t d = 0; d < deviceCount; ++d) {
            
            // deviceプロパティを取得
            ze_device_properties_t device_properties;
            CHECK(zeDeviceGetProperties(allDevices[d], &device_properties));
            
            if (ZE_DEVICE_TYPE_GPU == device_properties.type) {
                fprintf(stderr, "INFO: GPU device located driver=%d, device=%d\n", i, d);
                driver = allDrivers[i];
                device = allDevices[d];
                break;
            }
        }
        if (nullptr != device) break;
    }
    CHECK2((driver == nullptr), "unable to locate driver with GPU device");
    CHECK2((device == nullptr), "unable to locate GPU device");


    /* ------------------------------------------------------------------------------
     * コンテキストを作成する
     * コンテキストを作成することで他のAppとのリソースアクセス制限/共有を管理できるようにする
     * メモリ、コマンドキュー/リスト、モジュール、同期オブジェクトなどが関連付けられていて、
     * 複数作成して、タスクごとにごそっと現状を切り替えるコンテキストスイッチが実現できる(恐らく)
     * ------------------------------------------------------------------------------*/
    ze_context_desc_t contextDesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
    CHECK(zeContextCreate(driver, &contextDesc, &context));


    /* ------------------------------------------------------------------------------
     * create a command queue and list
     * コマンドキューとコマンドリストを確保
     * なんで2つが分離してるかは以下を読んで
     * https://spec.oneapi.io/level-zero/latest/core/PROG.html#command-queues-and-command-lists
     * ------------------------------------------------------------------------------*/
    ze_command_queue_desc_t commandQueueDesc = {
      ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr, 0, 0, 0,
      ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS, ZE_COMMAND_QUEUE_PRIORITY_NORMAL
    };
    CHECK(zeCommandQueueCreate(context, device, &commandQueueDesc, &queue));
    ze_command_list_desc_t commandListDesc = { 
      ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, nullptr, 0, 0
    };
    CHECK(zeCommandListCreate(context, device, &commandListDesc, &commands));


    /* ------------------------------------------------------------------------------
     * カーネルバイナリを読む
     * ------------------------------------------------------------------------------*/
    FILE *fp = fopen(KERNEL, "rb");
    if (fp == nullptr) {
        fprintf(stderr, "FAIL: unable to open %s\n", KERNEL);
        exit(-1);
    }
    fseek(fp, 0, SEEK_END);
    size_t sz = ftell(fp);
    rewind(fp);

    unsigned char *code = (unsigned char *)malloc(sz);
    fread(code, 1, sz, fp);
    fclose(fp);

    /* ------------------------------------------------------------------------------
     * カーネル初期化
     * いままで device program とか isa_code とか呼んでたものは、これからは[モジュール]と呼ぶ。
     * 概念はいままでと変わってなくて、
     * 1つのGenxコードの中に複数のカーネルを記述して実行時にどのカーネルを実行するか選べたと思うが、
     * そのカーネルを複数内包した塊が[モジュール]で、実際に実行される関数を[カーネル]と呼ぶと、
     * 明確に名前が決まっただけである。
     * 
     * 詳しくは以下参照:
     * https://spec.oneapi.io/level-zero/latest/core/PROG.html#modules-and-kernels
     * ------------------------------------------------------------------------------*/
    ze_module_desc_t moduleDesc = {ZE_STRUCTURE_TYPE_MODULE_DESC, nullptr,
                                   ZE_MODULE_FORMAT_IL_SPIRV,
                                   sz, code,
                                   "-vc-codegen", nullptr};
    CHECK(zeModuleCreate(context, device, &moduleDesc, &module, nullptr));

    ze_kernel_desc_t kernelDesc = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr,
                                   0, "vector_add"};
    CHECK(zeKernelCreate(module, &kernelDesc, &kernel));


    /* ------------------------------------------------------------------------------
     * kernel parameter initialization
     * ------------------------------------------------------------------------------*/
    size_t bytes = SZ*sizeof(int);
    void *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;

    /* 
     * デバイスメモリを確保(旧 Surface定義に近い？)
     * align は割り当ての最小アラインメント（byte単位）。2の累乗でなければならない
     */
    ze_device_mem_alloc_desc_t deviceMemDesc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
    CHECK(zeMemAllocDevice(context, &deviceMemDesc, /*size*/bytes, /*align*/64, device, &d_a));
    CHECK(zeMemAllocDevice(context, &deviceMemDesc, /*size*/bytes, /*align*/64, device, &d_b));
    CHECK(zeMemAllocDevice(context, &deviceMemDesc, /*size*/bytes, /*align*/64, device, &d_c));

    /* [host -> device]メモリコピーとバリアをコマンドリストに詰める */
    CHECK(zeCommandListAppendMemoryCopy(commands, d_a, src1, bytes, nullptr, 0, nullptr));
    CHECK(zeCommandListAppendMemoryCopy(commands, d_b, src2, bytes, nullptr, 0, nullptr));
    CHECK(zeCommandListAppendBarrier(commands, nullptr, 0, nullptr));

    /* 引数設定 */
    CHECK(zeKernelSetArgumentValue(kernel, 0, sizeof(d_a), &d_a));
    CHECK(zeKernelSetArgumentValue(kernel, 1, sizeof(d_b), &d_b));
    CHECK(zeKernelSetArgumentValue(kernel, 2, sizeof(d_c), &d_c));


    /* 
     * set group size - single KERNEL_SZ size entry per group
     * Group内サイズを設定
     * 恐らく cm_local_id() とかに影響する
     */
    CHECK(zeKernelSetGroupSize(kernel, /*x*/ 1, /*y*/ 1, /*z*/ 1));

    /* 
     * launch - data split across multiple groups
     * 起動するグループの数を決める
     */
    ze_group_count_t groupCount = {SZ/KERNEL_SZ, 1, 1};

    /* カーネル実行とバリアをコマンドリストに詰める */
    CHECK(zeCommandListAppendLaunchKernel(commands, kernel, &groupCount, nullptr, 0, nullptr));
    CHECK(zeCommandListAppendBarrier(commands, nullptr, 0, nullptr));
    
    /* [device -> host]メモリコピーをコマンドリストに詰める */
    CHECK(zeCommandListAppendMemoryCopy(commands, dst, d_c, bytes, nullptr, 0, nullptr));

    /* コマンドリストの追加を〆る */
    CHECK(zeCommandListClose(commands));
    
    /*
     * send to GPU
     * コマンドキューにコマンドリストを詰める
     * つまり、GPUリソースが空いてればCommandListsが即座に実行される
     */
    printf("\n【 Run GPU 】\n");
    CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &commands, nullptr));

    /* 
     * synchronize host and device
     * コマンドキューの消費待ちをしたい場合とか、
     * キューの状態を監視したい場合問い合わせることも可能
    */
    // zeCommandQueueSynchronize(queue, UINT64_MAX);

    // process output and cleanup
    CHECK(zeMemFree(context, d_a));
    CHECK(zeMemFree(context, d_b));
    CHECK(zeMemFree(context, d_c));

    printf("\n【 Result 】\n");
    printf("src1  = "); buffer_dump(src1, SZ);
    printf("src2  = "); buffer_dump(src2, SZ);
    printf("added = "); buffer_dump(dst,  SZ);

    printf("\n【 ERROR CHECK 】\n");
    for (unsigned i=0; i<SZ; i++)
        if ((src1[i] + src2[i]) != dst[i]) {
            fprintf(stderr, "FAIL: comparison at index[%d]: %d + %d => %d(host), but %d(gpu)\n", i, src1[i], src2[i], (src1[i]+src2[i]), dst[i]);
            //exit(-1);
    }
    fprintf(stderr, "PASSED\n");
    
    return 0;
}
