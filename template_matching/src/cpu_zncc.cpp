#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <iostream>
#include <omp.h>
#include <tuple>

std::tuple<float,float> calculate_temp(int32_t temp_h, int32_t temp_w, float *tempImg){
    float sum_temp=0;
    float sum_temp_pw=0;
    for(uint32_t i=0;i<temp_h;i++){
        for(uint32_t j=0;j<temp_w;j++){
            uint32_t tempidx=i*temp_w+j;
            sum_temp += tempImg[tempidx];
            sum_temp_pw += tempImg[tempidx] * tempImg[tempidx]; 
        }
    }
    return {sum_temp,sum_temp_pw};
      
}

void cpu_zncc(int32_t src_h,int32_t src_w,
              int32_t temp_h,int32_t temp_w,
              float sum_temp, float sum_temp_pw,
              float *srcImg,float *tempImg,
              float *out_score){
    uint32_t temp_size = temp_h*temp_w;
    uint32_t looph=src_h-temp_h;
    uint32_t loopw=src_w-temp_w;
#pragma omp parallel for
    for(uint32_t i=0;i<looph;i++){
        for(uint32_t j=0;j<loopw;j++){

            float sum_src=0;
            float sum_mul=0;
            float sum_src_pw=0;

            for(uint32_t m=0;m<temp_h;m++){
                for(uint32_t n=0;n<temp_w;n++){
                    uint32_t tempidx = m*temp_w+n;
                    uint32_t idx = (i+m)*src_w+(j+n);
                    sum_src +=srcImg[idx];
                    sum_mul +=srcImg[idx]*tempImg[tempidx];
                    sum_src_pw += srcImg[idx]*srcImg[idx];
                }
            }
            float m =temp_size*sum_mul - sum_src * sum_temp;
            float d = sqrt(abs((temp_size*sum_src_pw - sum_src*sum_src) *
                               (temp_size*sum_temp_pw - sum_temp*sum_temp)));
            if(d==0){
                out_score[i*loopw+j]=0;
            }else{
                out_score[i*loopw+j]=m/d;
            }
        }
    }
    return;
}
