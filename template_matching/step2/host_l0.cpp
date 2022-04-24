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
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <sys/time.h>
#include <float.h>
#include "l0_rt_helpers.h"
#include <level_zero/ze_api.h>
#include "cpu_zncc.h"
#include "tools.h"

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


int main(int argc, char* argv[])
{
    double st,ed;
    cv::Mat src = cv::imread("../../images/mini_cat.jpg");
    cv::Mat src_g;
    cv::Mat temp_g;
    cv::cvtColor(src, src_g, cv::COLOR_BGR2GRAY);
    int32_t img_w = src_g.size().width;
    int32_t img_h = src_g.size().height;
    cv::Mat temp = cv::imread("../../images/mini_cat_eye.jpg");
    cv::cvtColor(temp, temp_g, cv::COLOR_BGR2GRAY);
    int32_t temp_w = temp_g.size().width;
    int32_t temp_h = temp_g.size().height;
    float *psrc = new float[img_w*img_h];
    float *ptemp = new float[temp_w*temp_h];
    float *gpu_out_score = new float[(img_w-temp_w)*(img_h-temp_h)];
    float *cpu_out_score = new float[(img_w-temp_w)*(img_h-temp_h)];

    Mat2Float(src_g,psrc);
    Mat2Float(temp_g,ptemp);
    
    auto [sum_temp, sum_temp_pw] = calculate_temp(temp_h, temp_w, ptemp);
    st = get_time();
    cpu_zncc(img_h,img_w,temp_h,temp_w,sum_temp, sum_temp_pw,psrc,ptemp,cpu_out_score);
    ed = get_time();
    std::cout << "CPU:"<< (ed-st)*1000 << "msec" << std::endl;
    
    auto [driver, device, context] = findDriverAndDevice();
    auto [queue,commands] = createCommandQueueAndList(context, device);
    
    auto kernel = createKernel(context, device,"kernel.spv.skl", "zncc");
    
    ze_image_format_t fmt = {ZE_IMAGE_FORMAT_LAYOUT_32, ZE_IMAGE_FORMAT_TYPE_FLOAT};
    auto g_src = createImage2D(context, device, commands, fmt, img_w, img_h, psrc);
    auto g_temp = createImage2D(context, device, commands, fmt, temp_w, temp_h, ptemp);
    auto g_out_score = createImage2D(context, device, commands, fmt,img_w-temp_w, img_h-temp_h);
    setKernelArgs(kernel, &g_src, &g_temp, &g_out_score, &img_h, &img_w, &temp_h, &temp_w, &sum_temp, &sum_temp_pw);

    CHECK(zeKernelSetGroupSize(kernel, 1, 1, 1));
    
    uint32_t block_x = 8;
    uint32_t block_y = 8;

    ze_group_count_t groupCount = {(img_w-temp_w) / block_x, (img_h-temp_h) / block_y, 1};

    CHECK(zeCommandListAppendLaunchKernel(commands, kernel, &groupCount, nullptr, 0, nullptr));
    CHECK(zeCommandListAppendBarrier(commands, nullptr, 0, nullptr));
    CHECK(zeCommandListAppendImageCopyToMemory(commands, gpu_out_score, g_out_score, nullptr, nullptr, 0, nullptr));

    CHECK(zeCommandListClose(commands));
    printf("\n【 Run GPU 】\n");
    st = get_time();
    CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &commands, nullptr));
    ed = get_time();
    std::cout << "GPU:"<< (ed-st)*1000 << "msec" << std::endl;


    
    destroy(g_src);
    destroy(g_temp);
    destroy(g_out_score);
    printf("\n【 Result 】\n");
    float lmax = -INFINITY;

    // for(int i=0;i<32;i++){
    //     std::cout << "cpu:" << cpu_out_score[i] << "gpu:" << gpu_out_score[i] << std::endl;
    // }
    for (unsigned i=0; i<(img_h-temp_h)*(img_w-temp_h); i++){
        if(fabs(cpu_out_score[i] -gpu_out_score[i]) / fabs(cpu_out_score[i]) > 1){
            std::cout << "i=:" << i << std::endl;
            std::cout << "cpu:" << cpu_out_score[i] << std::endl;
            std::cout << "gpu:" << gpu_out_score[i] << std::endl;
        }
        lmax = fmaxf(fabs(cpu_out_score[i] - gpu_out_score[i]) / fabs(cpu_out_score[i]),lmax);
    }
    std::cout << "absolute max error:" << lmax << std::endl;
    WriteImage(gpu_out_score,src,(char*)"out.jpg",img_h,img_w,temp_h,temp_w);
    return 0;
}
