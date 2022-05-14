#include <cm/cm.h>
#include "define.h"

#ifdef SHIM
#include "shim_support.h"
#endif  // SHIM

_GENX_MAIN_ void image_coler_inv(
    SurfaceIndex ibuf [[type("image2d_t float")]],
    SurfaceIndex obuf [[type("image2d_t float")]]
    )
{
    if(cm_group_id(0)==0 && cm_group_id(1) == 0){
        printf("cm_group_count %dx%d\n", cm_group_count(0), cm_group_count(1));
    }

    int id_x = cm_group_id(0) * cm_local_size(0) + cm_local_id(0);
    int id_y = cm_group_id(1) * cm_local_size(1) + cm_local_id(1);
    matrix<float, BLOCK_W, BLOCK_H> in;
    matrix<float, BLOCK_W, BLOCK_H> out;
   
    read(ibuf, id_x * BLOCK_W * sizeof(float), id_y * BLOCK_H, in);

    out = 255.0f - in;

    write (obuf, id_x * BLOCK_W * sizeof(float), id_y * BLOCK_H, out);
}

#ifdef SHIM
EXPORT_SIGNATURE(vector_add);
#endif  // SHIM
