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
    const uint32_t temp_size = temp_h*temp_w;
    int idx = cm_group_id(0)*cm_local_size(0) + cm_local_id(0);
    int idy = cm_group_id(1)*cm_local_size(1) + cm_local_id(1);
    matrix<float, 1, 1> in;
    matrix<float, 1, 1> temp;
    matrix<float, 1, 1> out;
    matrix<float, 1, 1> sum_src(0.0f);
    matrix<float, 1, 1> sum_mul(0.0f);
    matrix<float, 1, 1> sum_src_pw(0.0f);
    matrix<float, 1, 1> m;
    matrix<float, 1, 1> d;

   
#pragma unroll
    for(uint32_t i=0;i<32;i++){
#pragma unroll
        for(uint32_t j=0;j<32;j++){
            read(sbuf, (idx+j)*sizeof(float), idy+i, in);
            read(tbuf, j*sizeof(float), i, temp);
            sum_src +=in;
            sum_mul +=in*temp;
            sum_src_pw += in*in;            
        }
    }

    m = temp_size * sum_mul - sum_src * sum_temp;
    d = cm_sqrt(cm_abs<float>((temp_size*sum_src_pw - sum_src*sum_src) *
                       (temp_size*sum_temp_pw - sum_temp*sum_temp)));
    out = m/d;
    write(obuf, idx*sizeof(float), idy, out);

}

#ifdef SHIM
EXPORT_SIGNATURE(vector_add);
#endif  // SHIM
