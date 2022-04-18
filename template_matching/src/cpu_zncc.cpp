#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <omp.h>
void cpu_zncc(int32_t in_h,int32_t in_w,
              int32_t rec_h,int32_t rec_w,
              float *srcImg,float *recImg,
              float *out_score){

  float sum_rect=0;
  float sum_mul_rc=0;
  for(int i=0;i<rec_h;i++){
    for(int j=0;j<rec_w;j++){
      int recidx=i*rec_w+j;
      sum_rect+=recImg[recidx];
      sum_mul_rc += recImg[recidx] * recImg[recidx]; 
    }
  }
  

  const int looph=in_h-rec_h;
  const int loopw=in_w-rec_w;
#pragma omp parallel for
  for(int i=0;i<looph;i++){
    for(int j=0;j<loopw;j++){

      float sum_src=0;
      float sum_mul=0;
      float sum_mul_in=0;

      for(int m=0;m<rec_h;m++){
        for(int n=0;n<rec_w;n++){
          int recidx = m*rec_w+n;
          int idx = (i+m)*in_w+(j+n);
          sum_src +=srcImg[idx];
          sum_mul +=srcImg[idx]*recImg[recidx];
          sum_mul_in += srcImg[idx]*srcImg[idx];
        }
      }
      int size = rec_h*rec_w;
      // float m =size*sum_mul - sum_src * sum_rect;
      // float d = sqrt(abs((size*sum_mul_in - sum_src*sum_src) * (size*sum_mul_rc - sum_rect*sum_rect)));
      float m =sum_mul - sum_src * (sum_rect/size);
      float d = sqrt(abs((sum_mul_in - sum_src*(sum_src/size)) * (sum_mul_rc - sum_rect*(sum_rect/size))));
      if(d==0){
        out_score[i*loopw+j]=0;
      }else{
        out_score[i*loopw+j]=m/d;
      }
    }
  }
  return;
}
