/*========================== begin_copyright_notice ============================

Copyright (C) 2020-2021 Intel Corporation

SPDX-License-Identifier: MIT

============================= end_copyright_notice ===========================*/

#include <opencv2/opencv.hpp>
#include "l0_rt_helpers.h"
#include <sys/time.h>
#include "define.h"

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


inline double get_time() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec + t.tv_usec * 1e-6;
}


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
        return -1;
    }

    cv::Mat src = cv::imread(argv[1]);
    if (src.empty() == true) {
        std::cout << "Error File does not exist." << std::endl;
        return -1;
    }

    cv::Mat src_g;
    cv::cvtColor(src, src_g, cv::COLOR_BGR2GRAY);
    cv::imwrite("input.jpg", src_g);

    int32_t img_w = src_g.size().width;
    int32_t img_h = src_g.size().height;
    if(img_w % BLOCK_W != 0 && img_h % BLOCK_H != 0){
        fprintf(stderr, "Error: required Image Width and Height %% %d == 0\n", BLOCK_W);
        return -1;
    }
    float *psrc = new float[img_w*img_h];
    float *pdst = new float[img_w*img_h];
    Mat2Float(src_g,psrc);
    auto [driver, device, context] = findDriverAndDevice();
    auto [queue,commands] = createCommandQueueAndList(context, device);
    
    auto kernel = createKernel(context, device, "kernel.spv.skl", "image_coler_inv");
    
    ze_image_format_t fmt = {ZE_IMAGE_FORMAT_LAYOUT_32, ZE_IMAGE_FORMAT_TYPE_FLOAT};
    auto g_src = createImage2D(context, device, commands, fmt, img_w, img_h, psrc);
    auto g_dst = createImage2D(context, device, commands, fmt, img_w, img_h);
    
    setKernelArgs(kernel, &g_src, &g_dst);
    
    CHECK(zeKernelSetGroupSize(kernel, 1, 1, 1));

    uint32_t groupSizeX = (img_w / BLOCK_W);
    uint32_t groupSizeY = (img_h / BLOCK_H);
    ze_group_count_t groupCount = {groupSizeX, groupSizeY, 1};

    CHECK(zeCommandListAppendLaunchKernel(commands, kernel, &groupCount, nullptr, 0, nullptr));
    CHECK(zeCommandListAppendBarrier(commands, nullptr, 0, nullptr));
    CHECK(zeCommandListAppendImageCopyToMemory(commands, pdst, g_dst, nullptr, nullptr, 0, nullptr));

    CHECK(zeCommandListClose(commands));
    printf("\n【 Run GPU 】\n");
    double start = get_time();
    CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &commands, nullptr));
    zeCommandQueueSynchronize(queue, UINT64_MAX);
    double end = get_time();

    destroy(g_src);
    destroy(g_dst);

    printf("\n【 Result 】\n");
    std::cout << "GPU:"<< (end-start)*1000 << "msec" << std::endl;
    
    cv::Mat dst_im = cv::Mat::zeros(src_g.size(), CV_32F);
    Float2Mat(dst_im, pdst);
    cv::imwrite("output.jpg", dst_im);

    return 0;
}
