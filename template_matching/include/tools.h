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
