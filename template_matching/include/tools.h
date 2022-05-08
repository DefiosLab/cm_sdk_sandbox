
inline double get_time() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec + t.tv_usec * 1e-6;
}

inline void Mat2Float(cv::Mat src, float *dst){
  for(uint32_t i = 0;i < src.size().height;i++){
    for(uint32_t j = 0;j < src.size().width;j++){
      uint32_t idx = i*src.size().width+j;
      dst[idx] = src.at<unsigned char>(i,j);

    }
  }
}
inline void Float2Mat(cv::Mat dst, float *src){
  for(uint32_t i = 0;i < dst.size().height;i++){
    for(uint32_t j = 0;j < dst.size().width;j++){
      uint32_t idx = i*dst.size().width+j;
      dst.at<float>(i,j) = src[idx];
    }
  }
}
inline void ErrorCheck(float *CPU,float *GPU,uint32_t size){
    float lmax = -INFINITY;
    float amax = -INFINITY;
    for (unsigned i=0; i<size; i++){
        lmax = fmaxf(fabs(CPU[i] - GPU[i]) / fmaxf(fabs(CPU[i]),fabs(GPU[i])),lmax);
        amax = fmaxf(fabs(CPU[i] - GPU[i]),amax);
    }
    std::cout << "relative max error:" << lmax << std::endl;
    std::cout << "absolute max error:" << amax << std::endl;
}
inline void WriteImage(float *score,cv::Mat image, char *filename, uint32_t img_h, uint32_t img_w,
                       uint32_t temp_h,uint32_t temp_w){
    cv::Mat result= cv::Mat::zeros(img_h-temp_h,img_w-temp_w, CV_32F);
    Float2Mat(result, score);
    double mMin, mMax;
    cv::Point minP, maxP;
    cv::minMaxLoc(result, &mMin, &mMax, &minP, &maxP);
    std::cout << "max: " << mMax << ", point " << maxP << std::endl;
    cv::rectangle(image,maxP,maxP+cv::Point(temp_w,temp_h),cv::Scalar(0,0,255),5);
    cv::imwrite(filename,image);
}
