#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <float.h>
#include "cpu_zncc.h"

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

int main(){
  cv::Mat src = cv::imread("./images/mini_cat.bmp");
  cv::Mat src_g;
  cv::Mat rect_g;
  cv::cvtColor(src, src_g, cv::COLOR_BGR2GRAY);
  int32_t img_w = src_g.size().width;
  int32_t img_h = src_g.size().height;
  cv::Mat rect = cv::imread("./images/mini_rect.bmp");
  cv::cvtColor(rect, rect_g, cv::COLOR_BGR2GRAY);
  int32_t rect_w = rect_g.size().width;
  int32_t rect_h = rect_g.size().height;

  float *psrc = new float[img_w*img_h];
  float *prect = new float[rect_w*rect_h];
  float *out_score = new float[(img_w-rect_w)*(img_h-rect_h)];
  Mat2Float(src_g,psrc);
  Mat2Float(rect_g,prect);
  // for(int i=0;i<rect_h;i++){
  //   for(int j=0;j<rect_w;j++){
  //     prect[i*rect_w+j]=psrc[(i+16)*img_w+(j+78)];
  //   }
  // }
  cpu_zncc(img_h,img_w,rect_h,rect_w,psrc,prect,out_score);

  cv::Mat result= cv::Mat::zeros(img_h-rect_h,img_w-rect_w, CV_32F);
  Float2Mat(result, out_score);
 
  double mMin, mMax;
  cv::Point minP, maxP;
  cv::minMaxLoc(result, &mMin, &mMax, &minP, &maxP);
  std::cout << "min: " << mMin << ", point " << minP << std::endl;
  std::cout << "max: " << mMax << ", point " << maxP << std::endl;
}
