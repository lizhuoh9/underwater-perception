#include "mono_map.h"

#include <assert.h>
#include <string.h>

int main(void)
{
    static mm_map_t map;
    static mvo_t vo;
    const float near_point[1][3] = {{0.0f, -1.0f, 8.0f}};
    const float far_points[7][3] = {
        {0.0f, -3.5f, 12.0f}, {0.0f, -2.5f, 12.0f}, {0.0f, -0.5f, 12.0f},
        {0.0f, 0.5f, 12.0f}, {0.0f, 1.5f, 12.0f}, {0.0f, 2.5f, 12.0f}, {0.0f, 3.5f, 12.0f}
    };
    int col, row, i;

    mm_init(&map);
    memset(&vo, 0, sizeof vo);
    map.cell = 1.0f;
    map.origin[0] = map.origin[1] = -MM_SIZE / 2;
    assert(mm_add(&map, &vo, near_point, 1) == 1);
    assert(mm_cell_of(&map, 0.0f, 8.0f, &col, &row));
    assert(map.odds[row * MM_SIZE + col] == MM_HIT);
    assert(mm_add(&map, &vo, far_points, 7) == 7);
    assert(map.odds[row * MM_SIZE + col] == MM_HIT);
    for (i = 0; i < MM_SIZE * MM_SIZE; ++i) {
        assert(map.odds[i] >= 0);
    }

    vo.world_centre[1] = 0.5f;
    assert(mm_add(&map, &vo, NULL, 0) == 0);
    assert(map.height_resets == 0 && map.odds[row * MM_SIZE + col] == MM_HIT);
    vo.world_centre[1] = 10.0f;
    assert(mm_add(&map, &vo, NULL, 0) == 0);
    assert(map.height_resets == 1 && map.slice_y == 10.0f);
    for (i = 0; i < MM_SIZE * MM_SIZE; ++i) {
        assert(map.odds[i] == 0);
    }
    assert(mm_observe(&vo, NULL, 0, 0, NULL, 1) == 0);
    return 0;
}
