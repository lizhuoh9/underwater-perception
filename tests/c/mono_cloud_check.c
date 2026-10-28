#include "mono_cloud.h"
#include "mono_map.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

static mc_map_t map, saved, reordered;
static mvo_t vo;
static mvo_step_t step;

static void setup(void)
{
    memset(&vo, 0, sizeof vo);
    memset(&step, 0, sizeof step);
    mc_init(&map);
    step.kind = MVO_ESSENTIAL;
    vo.world_rotation[2] = 1.0f;
    vo.world_rotation[4] = 1.0f;
    vo.world_rotation[6] = -1.0f;
    vo.world_centre[0] = 10.0f;
    vo.world_centre[1] = 20.0f;
    vo.world_centre[2] = 30.0f;
}

static void set_feature(int index, uint32_t id)
{
    mvo_feature_t *feature = &vo.feature[index];
    memset(feature, 0, sizeof *feature);
    feature->id = id;
    feature->has_point = 1;
    feature->spread = 0.01f;
    feature->point[0] = 1.0f;
    feature->point[1] = 2.0f;
    feature->point[2] = 3.0f;
    if (id != UINT32_MAX && id >= vo.next_id) vo.next_id = id + 1;
}

static int has_id(uint32_t id)
{
    uint32_t i;
    for (i = 0; i < map.count; ++i) if (map.points[i].id == id) return 1;
    return 0;
}

static void transform_and_update(void)
{
    setup();
    assert(mc_map_size() == sizeof map && sizeof map <= 13000);
    assert(MC_MIN_SPREAD == MM_MIN_SPREAD);
    mc_init(NULL);
    assert(mc_update(NULL, &vo, &step) == 0 && mc_update(&map, &vo, NULL) == 0);
    assert(map.continuous == 1 && map.count == 0 && map.dropped == 0);
    vo.count = 1;
    set_feature(0, 7);
    assert(mc_update(&map, &vo, &step) == 1);
    assert(map.count == 1 && map.points[0].id == 7 && map.points[0].observations == 1);
    assert(map.points[0].xyz[0] == 13.0f && map.points[0].xyz[1] == 22.0f && map.points[0].xyz[2] == 29.0f);

    vo.feature[0].point[0] = 2.0f;
    vo.feature[0].spread = 0.001f;
    assert(mc_update(&map, &vo, &step) == 1);
    assert(map.count == 1 && map.points[0].observations == 2 && map.points[0].xyz[2] == 29.0f);
    vo.feature[0].spread = 0.04f;
    assert(mc_update(&map, &vo, &step) == 1);
    assert(map.count == 1 && map.points[0].observations == 3 && map.points[0].xyz[2] == 28.0f);
    assert(map.points[0].spread == 0.04f);

    saved = map;
    step.kind = MVO_ROTATION;
    assert(mc_update(&map, &vo, &step) == 0);
    assert(memcmp(&map, &saved, sizeof map) == 0);
    step.kind = MVO_ESSENTIAL;
    set_feature(0, 8);
    vo.feature[0].spread = MC_MIN_SPREAD;
    assert(mc_update(&map, &vo, &step) == 1 && map.count == 2);
}

static void reject_bad_points(void)
{
    int test;
    setup();
    vo.count = 1;
    for (test = 0; test < 9; ++test) {
        set_feature(0, 7);
        switch (test) {
        case 0: vo.feature[0].has_point = 0; break;
        case 1: vo.feature[0].spread = 0.00029f; break;
        case 2: vo.feature[0].spread = NAN; break;
        case 3: vo.feature[0].spread = INFINITY; break;
        case 4: vo.feature[0].point[2] = 0.0f; break;
        case 5: vo.feature[0].point[2] = -1.0f; break;
        case 6: vo.feature[0].point[0] = NAN; break;
        case 7: vo.feature[0].point[1] = INFINITY; break;
        case 8: vo.world_centre[0] = INFINITY; break;
        }
        assert(mc_update(&map, &vo, &step) == 0);
        assert(map.count == 0 && map.dropped == 0);
    }
    vo.world_centre[0] = 10.0f;
    set_feature(0, 7);
    vo.feature[0].point[0] = FLT_MAX;
    vo.world_rotation[6] = 2.0f; /* finite inputs can still overflow the transform */
    assert(mc_update(&map, &vo, &step) == 0 && map.count == 0);
    vo.count = MVO_MAX_FEATURES + 1;
    assert(mc_update(&map, &vo, &step) == 0 && map.count == 0);
    vo.count = -1;
    assert(mc_update(&map, &vo, &step) == 0 && map.count == 0);
}

static void loss_and_reset(void)
{
    setup();
    vo.count = 1;
    set_feature(0, 7);
    assert(mc_update(&map, &vo, &step) == 1);
    saved = map;
    step.kind = MVO_LOST;
    assert(mc_update(&map, NULL, &step) == 0 && map.continuous == 0);
    assert(map.count == saved.count && map.dropped == saved.dropped);
    assert(memcmp(map.points, saved.points, sizeof map.points) == 0);
    step.kind = MVO_ESSENTIAL;
    set_feature(0, 9);
    assert(mc_update(&map, &vo, &step) == 0 && map.continuous == 0);
    assert(memcmp(map.points, saved.points, sizeof map.points) == 0);
    mc_init(&map);
    assert(map.continuous == 1 && map.count == 0 && map.dropped == 0);
    assert(mc_update(&map, &vo, &step) == 1 && map.points[0].id == 9);
}

static void capacity(void)
{
    int i, batch;
    uint32_t tracked;
    setup();
    for (batch = 0; batch < 2; ++batch) {
        vo.count = MVO_MAX_FEATURES;
        for (i = 0; i < vo.count; ++i) set_feature(i, (uint32_t)(batch * MVO_MAX_FEATURES + i));
        assert(mc_update(&map, &vo, &step) == MVO_MAX_FEATURES);
    }
    vo.count = 14;
    for (i = 0; i < vo.count; ++i) set_feature(i, (uint32_t)(500 + i));
    assert(mc_update(&map, &vo, &step) == 14);
    assert(map.count == MC_MAX_POINTS && map.dropped == 2);
    for (i = 500; i < 514; ++i) assert(has_id((uint32_t)i));
    vo.count = 2;
    set_feature(0, 2);
    vo.feature[0].spread = 0.2f;
    vo.feature[0].point[0] = 2.0f;
    set_feature(1, 514);
    assert(mc_update(&map, &vo, &step) == 2);
    assert(map.count == MC_MAX_POINTS && map.dropped == 3);
    assert(map.points[2].id == 2 && map.points[2].observations == 2 && map.points[2].xyz[2] == 28.0f);

    for (batch = 0; batch < 4; ++batch) {
        vo.count = MVO_MAX_FEATURES;
        for (i = 0; i < vo.count; ++i) set_feature(i, (uint32_t)(10000 + batch * MVO_MAX_FEATURES + i));
        assert(mc_update(&map, &vo, &step) == MVO_MAX_FEATURES);
        assert(map.count == MC_MAX_POINTS && map.dropped == (uint32_t)(3 + (batch + 1) * MVO_MAX_FEATURES));
        for (i = 0; i < vo.count; ++i) assert(has_id(vo.feature[i].id));
    }
    saved = map;
    step.kind = MVO_LOST;
    assert(mc_update(&map, &vo, &step) == 0 && map.continuous == 0);
    assert(map.count == saved.count && map.dropped == saved.dropped);
    assert(memcmp(map.points, saved.points, sizeof map.points) == 0);
    step.kind = MVO_ESSENTIAL;
    set_feature(0, 80000);
    assert(mc_update(&map, &vo, &step) == 0 && map.dropped == saved.dropped);
    assert(memcmp(map.points, saved.points, sizeof map.points) == 0);
    map = saved;
    tracked = map.points[0].id;
    vo.count = 2;
    set_feature(0, tracked);
    set_feature(1, 90000);
    map.points[0].observations = UINT32_MAX;
    map.dropped = UINT32_MAX;
    assert(mc_update(&map, &vo, &step) == 2);
    assert(map.points[0].observations == UINT32_MAX && map.dropped == UINT32_MAX);
    vo.count = 1;
    set_feature(0, 90001);
    assert(mc_update(&map, &vo, &step) == 1 && has_id(90001));
    set_feature(0, 90002);
    assert(mc_update(&map, &vo, &step) == 1 && has_id(90001) && has_id(90002));
}

static void frame_order_and_protected_ids(void)
{
    mvo_feature_t swap;
    int i, batch;
    setup();
    for (batch = 0; batch < 3; ++batch) {
        vo.count = batch < 2 ? MVO_MAX_FEATURES : MC_MAX_POINTS - 2 * MVO_MAX_FEATURES;
        for (i = 0; i < vo.count; ++i) set_feature(i, (uint32_t)(1000 + batch * MVO_MAX_FEATURES + i));
        assert(mc_update(&map, &vo, &step) == vo.count);
    }
    assert(map.count == MC_MAX_POINTS && map.dropped == 0 && map.points[0].id == 1000);
    saved = map;
    vo.count = 3;
    set_feature(0, 2001); /* two new points need slots before a later tracked old point */
    set_feature(1, 1000);
    vo.feature[1].has_point = 0; /* even a tracked point lacking triangulation stays protected */
    set_feature(2, 2000);
    assert(mc_update(&map, &vo, &step) == 2);
    assert(map.points[0].id == 1000 && has_id(2000) && has_id(2001) && map.dropped == 2);
    reordered = map;
    map = saved;
    swap = vo.feature[0];
    vo.feature[0] = vo.feature[2];
    vo.feature[2] = swap;
    assert(mc_update(&map, &vo, &step) == 2);
    assert(memcmp(&map, &reordered, sizeof map) == 0);
    saved = map;
    vo.feature[2].id = vo.feature[0].id;
    assert(mc_update(&map, &vo, &step) == 0);
    assert(memcmp(&map, &saved, sizeof map) == 0);
}

static void keep_recent_points_while_active_slots_are_protected(void)
{
    int batch, frame, i;
    setup();
    for (batch = 0; batch < 3; ++batch) {
        vo.count = batch < 2 ? MVO_MAX_FEATURES : MC_MAX_POINTS - 2 * MVO_MAX_FEATURES;
        for (i = 0; i < vo.count; ++i) set_feature(i, (uint32_t)(batch * MVO_MAX_FEATURES + i));
        assert(mc_update(&map, &vo, &step) == vo.count);
    }
    for (frame = 0; frame < 6; ++frame) {
        vo.count = MVO_MAX_FEATURES;
        for (i = 0; i < MVO_MAX_FEATURES - 1; ++i) {
            set_feature(i, (uint32_t)i);
            vo.feature[i].has_point = 0; /* still tracked despite no current triangulation */
        }
        set_feature(MVO_MAX_FEATURES - 1, (uint32_t)(MC_MAX_POINTS + frame));
        assert(mc_update(&map, &vo, &step) == 1);
        assert(map.count == MC_MAX_POINTS && map.dropped == (uint32_t)(frame + 1));
        for (i = 0; i < MVO_MAX_FEATURES - 1; ++i) assert(has_id((uint32_t)i));
        for (i = 0; i <= frame; ++i) assert(has_id((uint32_t)(MC_MAX_POINTS + i)));
    }
}

static void eviction_across_id_wrap(void)
{
    uint32_t i, oldest = (uint32_t)(5u - MC_MAX_POINTS);
    setup();
    map.count = MC_MAX_POINTS;
    for (i = 0; i < MC_MAX_POINTS; ++i) {
        map.points[i].id = oldest + i;
        map.points[i].spread = 0.01f;
        map.points[i].observations = 1;
    }
    vo.count = 2;
    set_feature(0, oldest);
    vo.feature[0].has_point = 0;
    set_feature(1, 5);
    vo.next_id = 6; /* id 5 was just allocated after uint32 wrap */
    assert(mc_update(&map, &vo, &step) == 1);
    assert(has_id(oldest));
    assert(!has_id(oldest + 1));
    assert(has_id(5) && map.dropped == 1);
}

int main(void)
{
    transform_and_update();
    reject_bad_points();
    loss_and_reset();
    capacity();
    frame_order_and_protected_ids();
    keep_recent_points_while_active_slots_are_protected();
    eviction_across_id_wrap();
    return 0;
}
