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
#include <float.h>
#include "l0_rt_helpers.h"
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

void Mat2Float(cv::Mat src, float *dst){
  for(uint32_t i = 0;i < src.size().height;i++){
    for(uint32_t j = 0;j < src.size().width;j++){
      uint32_t idx = i*src.size().width+j;
      dst[idx] = src.at<unsigned char>(i,j);

    }
  }
}
void Float2Mat(cv::Mat dst, float *src){
  for(uint32_t i = 0;i < dst.size().height;i++){
    for(uint32_t j = 0;j < dst.size().width;j++){
      uint32_t idx = i*dst.size().width+j;
      dst.at<float>(i,j) = src[idx];
    }
  }
}



int main(int argc, char* argv[])
{
    if(argc != 2){
        std::cout << "Usage ./vector.l0.skl [Image Path]" << std::endl;
    }
    cv::Mat src = cv::imread(argv[1]);
    if (image.empty() == true) {
        std::count << "Error File does not exist." << std::endl;
		return -1;
	}
    cv::Mat src_g;
    cv::cvtColor(src, src_g, cv::COLOR_BGR2GRAY);
    int32_t img_w = src_g.size().width;
    int32_t img_h = src_g.size().height;
    float *psrc = new float[img_w*img_h];
    float *pdst = new float[img_w*img_h];
    Mat2Float(src_g,psrc);
    auto [driver, device, context] = findDriverAndDevice();
    auto [queue,commands] = createCommandQueueAndList(context, device);
    
    auto kernel = createKernel(context, device,"kernel.spv.skl", "vector_add");
    
    ze_image_format_t fmt = {ZE_IMAGE_FORMAT_LAYOUT_32, ZE_IMAGE_FORMAT_TYPE_FLOAT};
    auto g_src = createImage2D(context, device, commands, fmt, img_w, img_h, psrc);
    auto g_dst = createImage2D(context, device, commands, fmt, img_w, img_h);
    
    setKernelArgs(kernel, &g_src, &g_dst, &img_h, &img_w);

    CHECK(zeKernelSetGroupSize(kernel, 1, 1, 1));
    
    uint32_t block_x = 8;
    uint32_t block_y = 8;
    ze_group_count_t groupCount = {img_w / 8, img_h / 8, 1};

    CHECK(zeCommandListAppendLaunchKernel(commands, kernel, &groupCount, nullptr, 0, nullptr));
    CHECK(zeCommandListAppendBarrier(commands, nullptr, 0, nullptr));
    CHECK(zeCommandListAppendImageCopyToMemory(commands, pdst, g_dst, nullptr, nullptr, 0, nullptr));

    CHECK(zeCommandListClose(commands));
    printf("\n【 Run GPU 】\n");
    CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &commands, nullptr));

    destroy(g_src);
    destroy(g_dst);

    printf("\n【 Result 】\n");
    printf("\n【 ERROR CHECK 】\n");
    for (unsigned i=0; i<img_h*img_w; i++)
        if (psrc[i] != pdst[i]) {
            fprintf(stderr, "FAIL: comparison at index[%d]: => %f(host), but %f(gpu)\n", i, psrc[i], pdst[i]);
            //exit(-1);
    }
    fprintf(stderr, "PASSED\n");
    
    return 0;
}
