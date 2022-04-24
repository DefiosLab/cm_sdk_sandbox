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
#define ITER 100

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
    // cv::Mat src = cv::imread("../../images/cat.jpg");
    cv::Mat src_g;
    cv::Mat temp_g;
    cv::cvtColor(src, src_g, cv::COLOR_BGR2GRAY);
    int32_t img_w = src_g.size().width;
    int32_t img_h = src_g.size().height;
    cv::Mat temp = cv::imread("../../images/mini_cat_eye.jpg");
    // cv::Mat temp = cv::imread("../../images/cat_eye.jpg");
    cv::cvtColor(temp, temp_g, cv::COLOR_BGR2GRAY);
    int32_t temp_w = temp_g.size().width;
    int32_t temp_h = temp_g.size().height;
    uint32_t score_h = img_h-temp_h;
    uint32_t score_w = img_w-temp_w;
    float *psrc = new float[img_w*img_h];
    float *ptemp = new float[temp_w*temp_h];
    float *gpu_out_score = new float[score_w*score_h];
    float *cpu_out_score = new float[score_w*score_h];

    Mat2Float(src_g,psrc);
    Mat2Float(temp_g,ptemp);
    
    auto [sum_temp, sum_temp_pw] = calculate_temp(temp_h, temp_w, ptemp);
    double tcpu = 0;
    for(int i=0;i<ITER;i++){
        st = get_time();
        cpu_zncc(img_h,img_w,temp_h,temp_w,sum_temp, sum_temp_pw,psrc,ptemp,cpu_out_score);
        ed = get_time();
        tcpu += ed-st;
    }
    std::cout << "CPU:"<< tcpu*1000/ITER << "msec" << std::endl;
    
    auto [driver, device, context] = findDriverAndDevice();
    auto commands = createImmCommandList(context, device);
    ze_event_handle_t event = createEvent(context, device);
    auto kernel = createKernel(context, device,"kernel.spv.skl", "zncc");
    
    ze_image_format_t fmt = {ZE_IMAGE_FORMAT_LAYOUT_32, ZE_IMAGE_FORMAT_TYPE_FLOAT};

    uint32_t block_x = 8;
    uint32_t block_y = 8;
    unsigned long long kernel_ns = 0;
    double thost = 0.0f;
    for(int i=0;i<ITER;i++){
        st = get_time();
        auto g_src = createImage2D(context, device, commands, fmt, img_w, img_h, psrc);
        auto g_temp = createImage2D(context, device, commands, fmt, temp_w, temp_h, ptemp);
        auto g_out_score = createImage2D(context, device, commands, fmt,img_w-temp_w, img_h-temp_h);
        setKernelArgs(kernel, &g_src, &g_temp, &g_out_score, &img_h, &img_w, &temp_h, &temp_w, &sum_temp, &sum_temp_pw);
        CHECK(zeKernelSetGroupSize(kernel, 1, 1, 1));
        ze_group_count_t groupCount = {score_w / block_x, score_h / block_y, 1};
        

        appendLaunchKernel(commands, kernel, &groupCount, event);
        zeEventHostSynchronize(event, std::numeric_limits<uint32_t>::max());


        
        ze_kernel_timestamp_result_t timestamp;
        zeEventQueryKernelTimestamp(event, &timestamp);        
        kernel_ns += (timestamp.context.kernelEnd - timestamp.context.kernelStart);

        reset(event);
        copyToMemory(commands, gpu_out_score, g_out_score, event);
        zeEventHostSynchronize(event, std::numeric_limits<uint32_t>::max());
        ed = get_time();
        thost +=ed-st;
        destroy(g_src);
        destroy(g_temp);
        destroy(g_out_score);
    }
    printf("\n【 Run GPU 】\n");
    std::cout << "copy+GPU execute:"<< thost*1000/ITER << "msec" << std::endl;
    std::cout << "GPU execute:"<< kernel_ns / 1000000.0f / ITER << "msec" << std::endl;



    

    printf("\n【 Result 】\n");
    ErrorCheck(cpu_out_score,gpu_out_score,score_h*score_w);
    WriteImage(gpu_out_score,src,(char*)"out.jpg",img_h,img_w,temp_h,temp_w);
    return 0;
}
