/*========================== begin_copyright_notice ============================

Copyright (C) 2020-2021 Intel Corporation

SPDX-License-Identifier: MIT

============================= end_copyright_notice ===========================*/

#include <cm/cm.h>

#ifdef SHIM
#include "shim_support.h"
#endif  // SHIM

#define BLOCK_SIZE 16

/* --------------------------------------------------------------------------------
 * memory type:
 * CMCは入出力の形式が 1D buffer か 2D surface なんて判別できないので、
 * 2Dにしか使えない関数に 1D buffer 渡しても動いちゃったり(未定義動作)するかもで危険だったから
 * 属性を明示的に記述できるようにしたよ。
 * パフォーマンスを向上させるため? にさらに 2D surface では R/W アクセス属性も記述できるよ。
 * 
 * 参考: cm_sdk_20211028/docs/external/cmoclrt/cmocl.html
 * -------------------------------------------------------------------------------- */
_GENX_MAIN_ void vector_add(
	SurfaceIndex ibuf [[type("image2d_t float")]],
	SurfaceIndex obuf [[type("image2d_t float")]],
    uint32_t img_h,uint32_t img_w
	)
{
    int idx = cm_group_id(0)*cm_local_size(0) + cm_local_id(0);
    int idy = cm_group_id(1)*cm_local_size(1) + cm_local_id(1);
    matrix<float, 8, 8> in;
    matrix<float, 8, 8> out;
   

    read(ibuf, idx*8*sizeof(float), idy*8, in);
    out = in;
    write (obuf, idx*8*sizeof(float), idy*8, out);
}

#ifdef SHIM
EXPORT_SIGNATURE(vector_add);
#endif  // SHIM
