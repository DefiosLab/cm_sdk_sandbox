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
	SurfaceIndex ibuf1 [[type("buffer_t")]],
	SurfaceIndex ibuf2 [[type("buffer_t")]],
	SurfaceIndex obuf  [[type("buffer_t")]]
	)
{
    printf("gid(0)=%d, gid(1)=%d, lid(0)=%d, lid(1)=%d\n", cm_group_id(0), cm_group_id(1), cm_local_id(0), cm_local_id(1));

    vector<int, BLOCK_SIZE> in_vec1;
    vector<int, BLOCK_SIZE> in_vec2;
    vector<int, BLOCK_SIZE> out_vec;

    unsigned offset = sizeof(unsigned) * BLOCK_SIZE * cm_group_id(0);

    /* --------------------------------------------------------------------------------
     * read():
     * 128 byte 以下のブロックデータが読める
     * offset が OWord(16byte) aligned だったら以下の通りでOK
     * offset が DWord( 4byte) aligned だったら DWALIGNED(ibuf) マクロを使う必要あり?
     * 
     * 例: char を8つずつ(8byte)読みたいときとかは恐らく DWALIGNED(ibuf)
     * -------------------------------------------------------------------------------- */
    read(ibuf1, offset, in_vec1);
    read(ibuf2, offset, in_vec2);
    
    // add
    out_vec = in_vec1 + in_vec2;
    
    /* --------------------------------------------------------------------------------
     * write():
     * 128 byte 以下のブロックデータが書ける
     * offset は必ず OWord(16byte) aligned で揃える必要がある
     * 
     * 出力 vector size は 1, 2, 4, 8 OWords からしか選べないので、
     * 例えば float(4byte) の場合次のようにブロックサイズを選ぶ
     * 下限は 1 OWords( 16byte) なので float(4byte) *  4 =  16byte のブロック
     * 上限は 8 OWords(128byte) なので float(4byte) * 32 = 128byte のブロック
     * -------------------------------------------------------------------------------- */
    write (obuf, offset, out_vec);
}

/* 
 * shim layer (CM kernel, OpenCL runtime, GPU)
 * シミュレーションするレイヤーを登録しよう
 */
#ifdef SHIM
EXPORT_SIGNATURE(vector_add);
#endif  // SHIM