std::tuple<float,float> calculate_temp(int32_t temp_h, int32_t temp_w, float *tempImg);
void cpu_zncc(int32_t in_h,int32_t in_w,
              int32_t rec_h,int32_t rec_w,
              float sum_temp, float sum_temp_pw,
              float *srcImg,float *recImg,
              float *out_score);
