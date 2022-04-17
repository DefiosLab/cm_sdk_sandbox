/* ********************************************************************************
 * Level 0 API sample
 *
 * Level 0 APIの仕様は以下を読めば完全に理解できる
 * https://spec.oneapi.io/level-zero/latest/index.html
 * ******************************************************************************** */
#include <iostream>
#include <level_zero/ze_api.h>

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


int main()
{
    puts("Hello!! Intel(R) Level zero");

    // oneAPI driverの初期化
    // oneAPI L0の構想だとGPU, VPU, FPGAとかあるけど、今サポートされてるのは恐らく[GPU]だけ
    CHECK(zeInit(ZE_INIT_FLAG_GPU_ONLY));

    /* ------------------------------------------------------------------------------
     * Discover all the driver instances
     * Level 0 APIは、Driverをベースにいろんなオブジェクトにアクセスできる
     * 
     * 図: https://spec.oneapi.io/level-zero/latest/_images/core_device.png
     * ------------------------------------------------------------------------------*/
    // driverの総数を取得
    uint32_t driverCount = 0;
    CHECK(zeDriverGet(&driverCount, nullptr));
    CHECK2((driverCount == 0), "unable to locate driver(s)");

    // driverインスタンス取得
    // (driverインスタンスハンドルの配列が生成される)
    ze_driver_handle_t *allDrivers = (ze_driver_handle_t *)malloc(driverCount * sizeof(*allDrivers));
    CHECK(zeDriverGet(&driverCount, allDrivers));


    /* ------------------------------------------------------------------------------
     * Find a driver instance with device
     * ------------------------------------------------------------------------------*/

    // ドライバーインスタンスをたどる
    for(int i = 0; i < driverCount; ++i)
    {
        // deviceの総数を取得
        uint32_t deviceCount = 0;
        zeDeviceGet(allDrivers[i], &deviceCount, nullptr);

        // ドライバ内のdeviceを取得
        // (deviceハンドルの配列が生成される)
        ze_device_handle_t *allDevices = (ze_device_handle_t *)malloc(deviceCount * sizeof(ze_device_handle_t));
        zeDeviceGet(allDrivers[i], &deviceCount, allDevices);

        // deviceをたどる
        for(int d = 0; d < deviceCount; ++d)
        {

            // deviceプロパティを取得
            ze_device_properties_t device_properties;
            zeDeviceGetProperties(allDevices[d], &device_properties);

            ze_device_compute_properties_t device_compute_properties;
            zeDeviceGetComputeProperties(allDevices[d], &device_compute_properties);

            // Typeは[GPU, CPU, FPGA, MCA, VPU] とか居るけどいまのところGPUしか認識しない
            if(ZE_DEVICE_TYPE_GPU == device_properties.type)
            {
                printf("\n=== Device Info =============================================================================\n");
                printf("type     : GPU\n");
                printf("name     : %s\n", device_properties.name               );  // [out] generic device type
                printf("vendorId : %d\n", (uint32_t)device_properties.vendorId );  // [out] vendor id from PCI configuration
                printf("deviceId : %d\n", (uint32_t)device_properties.deviceId );  // [out] device id from PCI configuration
                printf("flags    : 0x%02X\n", (ze_device_property_flags_t)device_properties.flags            );  // [out] 0 (none) or a valid combination of ::ze_device_property_flag_t
                printf("subdeviceId             : %d\n" , (uint32_t)device_properties.subdeviceId            );  // [out] sub-device id. Only valid if ::ZE_DEVICE_PROPERTY_FLAG_SUBDEVICE is set.
                printf("coreClockRate           : %d\n" , (uint32_t)device_properties.coreClockRate          );  // [out] Clock rate for device core.
                printf("maxMemAllocSize         : %ld\n", (uint64_t)device_properties.maxMemAllocSize        );  // [out] Maximum memory allocation size.
                printf("maxHardwareContexts     : %d\n" , (uint32_t)device_properties.maxHardwareContexts    );  // [out] Maximum number of logical hardware contexts.
                printf("maxCommandQueuePriority : %d\n" , (uint32_t)device_properties.maxCommandQueuePriority);  // [out] Maximum priority for command queues. Higher value is higher priority.
                printf("numThreadsPerEU         : %d\n" , (uint32_t)device_properties.numThreadsPerEU        );  // [out] Number of threads per EU.
                printf("physicalEUSimdWidth     : %d\n" , (uint32_t)device_properties.physicalEUSimdWidth    );  // [out] The physical EU simd width.
                printf("numEUsPerSubslice       : %d\n" , (uint32_t)device_properties.numEUsPerSubslice      );  // [out] Number of EUs per sub-slice.
                printf("numSubslicesPerSlice    : %d\n" , (uint32_t)device_properties.numSubslicesPerSlice   );  // [out] Number of sub-slices per slice.
                printf("numSlices               : %d\n" , (uint32_t)device_properties.numSlices              );  // [out] Number of slices.
                printf("timerResolution         : %ld\n", (uint64_t)device_properties.timerResolution        );  // [out] Returns the resolution of device timer used for profiling,

                printf("\n--- Device Compute Info ---------------------------------------------------------------------\n");
                printf("maxTotalGroupSize     : %d\n", (uint32_t)device_compute_properties.maxTotalGroupSize    );  // [out] Maximum items per compute group. (groupSizeX * groupSizeY * groupSizeZ) <= maxTotalGroupSize
                printf("maxGroupSizeX         : %d\n", (uint32_t)device_compute_properties.maxGroupSizeX        );  // [out] Maximum items for X dimension in group
                printf("maxGroupSizeY         : %d\n", (uint32_t)device_compute_properties.maxGroupSizeY        );  // [out] Maximum items for Y dimension in group
                printf("maxGroupSizeZ         : %d\n", (uint32_t)device_compute_properties.maxGroupSizeZ        );  // [out] Maximum items for Z dimension in group
                printf("maxGroupCountX        : %d\n", (uint32_t)device_compute_properties.maxGroupCountX       );  // [out] Maximum groups that can be launched for x dimension
                printf("maxGroupCountY        : %d\n", (uint32_t)device_compute_properties.maxGroupCountY       );  // [out] Maximum groups that can be launched for y dimension
                printf("maxGroupCountZ        : %d\n", (uint32_t)device_compute_properties.maxGroupCountZ       );  // [out] Maximum groups that can be launched for z dimension
                printf("maxSharedLocalMemory  : %d\n", (uint32_t)device_compute_properties.maxSharedLocalMemory );  // [out] Maximum shared local memory per group.
                printf("numSubGroupSizes      : %d\n", (uint32_t)device_compute_properties.numSubGroupSizes     );  // [out] Number of subgroup sizes supported. This indicates number of entries in subGroupSizes.
                
                printf("group sizes supported : [");
                for(int sg=0; sg<device_compute_properties.numSubGroupSizes; sg++)
                {
                    printf(" %3d", (uint32_t)device_compute_properties.subGroupSizes[sg]);  // [out] Size group sizes supported.
                }
                puts(" ]");

                break;
            }
        }

        free(allDevices);
    }

    puts("[Succes]");

    return 0;
}