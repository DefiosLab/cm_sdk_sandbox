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
    float sum_src=0;
    float sum_mul=0;
    float sum_src_pw=0;
    float m;
    float d;

   

    for(uint32_t i=0;i<temp_h;i++){
        for(uint32_t j=0;j<temp_w;j++){
            read(sbuf, (idx+j)*sizeof(float), idy+i, in);
            read(tbuf, j*sizeof(float), i, temp);
            sum_src +=in(0,0);
            sum_mul +=in(0,0)*temp(0,0);
            sum_src_pw += in(0,0)*in(0,0);            
        }
    }

    m = temp_size * sum_mul - sum_src * sum_temp;
    d = cm_sqrt((temp_size*sum_src_pw - sum_src*sum_src) *
                (temp_size*sum_temp_pw - sum_temp*sum_temp));
    out = m/d;
    write(obuf, idx*sizeof(float), idy, out);

}

#ifdef SHIM
EXPORT_SIGNATURE(vector_add);
#endif  // SHIM
