/*========================== begin_copyright_notice ============================

Copyright (C) 2020-2021 Intel Corporation

SPDX-License-Identifier: MIT

============================= end_copyright_notice ===========================*/

#include <cm/cm.h>

#ifdef SHIM
#include "shim_support.h"
#endif  // SHIM

#define BLOCK_SIZE 

_GENX_MAIN_ void zncc(
	SurfaceIndex sbuf [[type("image2d_t float")]],
    SurfaceIndex tbuf [[type("image2d_t float")]],
	SurfaceIndex obuf [[type("image2d_t float")]],
    uint32_t img_h,uint32_t img_w,
    uint32_t temp_h,uint32_t temp_w,
    float sum_temp, float sum_temp_pw
	)
{
    const uint32_t block_x = 8;
    const uint32_t block_y = 8;
    const uint32_t reg_x = block_x * 2;
    const uint32_t reg_y = block_y * 2;
    const uint32_t temp_size = temp_h*temp_w;
    int idx = cm_group_id(0)*cm_local_size(0) + cm_local_id(0);
    int idy = cm_group_id(1)*cm_local_size(1) + cm_local_id(1);
    matrix<float, reg_y, reg_x> in;
    matrix<float, block_y, block_x> temp;
    matrix<float, block_y, block_x> out;
    matrix<float, block_y, block_x> sum_src(0.0f);
    matrix<float, block_y, block_x> sum_mul(0.0f);
    matrix<float, block_y, block_x> sum_src_pw(0.0f);
    matrix<float, block_y, block_x> m;
    matrix<float, block_y, block_x> d;

   
   
    for(uint32_t i=0;i<temp_h / 8;i++){
        for(uint32_t j=0;j<temp_w / 8;j++){
            int32_t offset_x = idx*block_x+j*8;
            int32_t offset_y = idy*block_y+i*8;
            read(sbuf, (offset_x)          *sizeof(float), offset_y,         in.select<block_y,1,block_x,1>(0,0));
            read(sbuf, (offset_x + block_x)*sizeof(float), offset_y,         in.select<block_y,1,block_x,1>(0,block_x));
            read(sbuf, (offset_x)          *sizeof(float), offset_y+block_y, in.select<block_y,1,block_x,1>(block_y,0));
            read(sbuf, (offset_x + block_x)*sizeof(float), offset_y+block_y, in.select<block_y,1,block_x,1>(block_y,block_x));

            read(tbuf, j*8*sizeof(float), i*8,temp);
            
#pragma unroll
            for(uint32_t ri = 0; ri < block_y; ri++){
#pragma unroll
                for(uint32_t rj = 0; rj < block_x; rj++){
                    sum_src +=in.select<block_y,1,block_x,1>(ri,rj);
                    sum_mul +=in.select<block_y,1,block_x,1>(ri,rj)*temp(ri,rj);
                    sum_src_pw += in.select<block_y,1,block_x,1>(ri,rj)*in.select<block_y,1,block_x,1>(ri,rj);            
                }
            }
        }
    }
    m = temp_size * sum_mul - sum_src * sum_temp;
    d = cm_sqrt(cm_abs<float>((temp_size*sum_src_pw - sum_src*sum_src) *
                       (temp_size*sum_temp_pw - sum_temp*sum_temp)));

    out = m/d;
    write(obuf, idx*block_x*sizeof(float), idy*block_y, out);

}

#ifdef SHIM
EXPORT_SIGNATURE(vector_add);
#endif  // SHIM
