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
    uint32_t img_w = 1920;
    uint32_t img_h = 1080;
    cv::Mat temp = cv::imread("rect.png");
    if (temp.empty() == true) {
        std::cout << "Error File does not exist." << std::endl;
        std::cout << "Run get_temp to generate rect.png." << std::endl;
		return -1;
    }
    cv::Mat src_g;
    cv::Mat temp_g;
    cv::cvtColor(temp, temp_g, cv::COLOR_BGR2GRAY);
    int32_t temp_w = temp_g.size().width;
    int32_t temp_h = temp_g.size().height;
    float *ptemp = new float[temp_w*temp_h];
    Mat2Float(temp_g,ptemp);
    auto [sum_temp, sum_temp_pw] = calculate_temp(temp_h, temp_w, ptemp);
    double st,ed;
    uint32_t score_h = img_h-temp_h;
    uint32_t score_w = img_w-temp_w;
    float *psrc = new float[img_w*img_h];
    float *gpu_out_score = new float[score_w*score_h];

    
    auto [driver, device, context] = findDriverAndDevice();
    auto commands = createImmCommandList(context, device);
    ze_event_handle_t event = createEvent(context, device);
    auto kernel = createKernel(context, device,KERNEL, "zncc");
    
    ze_image_format_t fmt = {ZE_IMAGE_FORMAT_LAYOUT_32, ZE_IMAGE_FORMAT_TYPE_FLOAT};
    auto g_src = createImage2D(context, device, commands, fmt, img_w, img_h);
    auto g_temp = createImage2D(context, device, commands, fmt, temp_w, temp_h, ptemp);
    auto g_out_score = createImage2D(context, device, commands, fmt,img_w-temp_w, img_h-temp_h);
    double thost = 0.0f;
    cv::Mat src;
    double mMin, mMax;
    cv::Point minP, maxP;
    char fps[256];
    cv::Mat result= cv::Mat::zeros(img_h-temp_h,img_w-temp_w, CV_32F);

    cv::VideoCapture cap(0);
    cap.set(cv::CAP_PROP_FOURCC ,CV_FOURCC('M', 'J', 'P', 'G') );
    cap.set(cv::CAP_PROP_FRAME_WIDTH, img_w);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, img_h);
    setKernelArgs(kernel, &g_src, &g_temp, &g_out_score, &img_h, &img_w, &temp_h, &temp_w, &sum_temp, &sum_temp_pw);
    CHECK(zeKernelSetGroupSize(kernel, 1, 1, 1));
    ze_group_count_t groupCount = {score_w / (uint32_t)BLOCK_X, score_h / (uint32_t)BLOCK_Y, 1};    
    while(cap.read(src)){
        st = get_time();
        cv::cvtColor(src, src_g, cv::COLOR_BGR2GRAY);
        Mat2Float(src_g,psrc);
        L0_SAFE_CALL(zeCommandListAppendImageCopyFromMemory(commands, g_src,
                                                            psrc, nullptr, nullptr, 0, nullptr));

        
        appendLaunchKernel(commands, kernel, &groupCount, event);
        zeEventHostSynchronize(event, std::numeric_limits<uint32_t>::max());
        
        reset(event);
        copyToMemory(commands, gpu_out_score, g_out_score, event);
        zeEventHostSynchronize(event, std::numeric_limits<uint32_t>::max());
        
        Float2Mat(result, gpu_out_score);
        cv::minMaxLoc(result, &mMin, &mMax, &minP, &maxP);
        if(mMax > 0.0){
            cv::rectangle(src,maxP,maxP+cv::Point(temp_w,temp_h),cv::Scalar(0,0,255),1,cv::LINE_4);
        }
        ed = get_time();
        thost = ed-st;
        sprintf(fps, "%.2f FPS",1/thost);
        cv::putText(
                    src,
                    fps,
                    cv::Point(25,75),
                    cv::FONT_HERSHEY_SIMPLEX,
                    2.5,
                    cv::Scalar(0,0,0),
                    3
                    );
        cv::imshow("img", src);
        const int key = cv::waitKey(1);
        if(key == 'q')
        {
            break;
        }        
    }
    destroy(g_src);
    destroy(g_temp);
    destroy(g_out_score);




    
    return 0;
}
