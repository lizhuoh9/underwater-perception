/* The voxel map's contract (core/voxel_map.h): what is filed, through LOST and ROTATION steps, with and without a mask,
 * how a new side re-keys and merges, which voxels make room, and that every voxel held stays findable. */
#include "voxel_map.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

static vm_map_t map, other, saved;
static mvo_t vo;
static mvo_step_t step;
static uint8_t mask[60 * 80];

static void setup(void)
{
    memset(&vo, 0, sizeof vo);
    memset(&step, 0, sizeof step);
    vm_init(&map);
    vo.world_rotation[0] = vo.world_rotation[4] = vo.world_rotation[8] = 1.0f;
    step.kind = MVO_ESSENTIAL;
}

/* Feature k sees point (x, y, z) in the camera's axes at pixel (160, 120). */
static void put(int k, float x, float y, float z)
{
    mvo_feature_t *f = &vo.feature[k];
    memset(f, 0, sizeof *f);
    f->has_point = 1;
    f->spread = 0.01f;
    f->point[0] = x;
    f->point[1] = y;
    f->point[2] = z;
    f->x = 160.0f;
    f->y = 120.0f;
    f->id = (uint32_t)k;
    if (k >= vo.count) {
        vo.count = k + 1;
    }
}

/* Every voxel held is found at its own key, no key twice, and the index names exactly the voxels held. */
static void check_index(const vm_map_t *m)
{
    uint32_t k, named = 0;
    for (k = 0; k < m->count; ++k) {
        assert(m->voxel[k].hits > 0 && m->voxel[k].frames > 0);
        assert(vm_find(m, m->voxel[k].key) == &m->voxel[k]);
    }
    for (k = 0; k < VM_SLOTS; ++k) {
        if (m->index[k]) {
            assert(m->index[k] <= m->count);
            ++named;
        }
    }
    assert(named == m->count);
}

/* A frame of 20 features at depth 8 on a row at height y: the side becomes 1. */
static void first_frame_at(float y)
{
    int k;
    vo.count = 0;
    for (k = 0; k < VM_MIN_POINTS; ++k) {
        put(k, (float)k, y, 8.0f);
    }
    assert(vm_update(&map, &vo, &step, NULL, 0, 0) == VM_MIN_POINTS);
    assert(map.side == 1.0f && map.count == VM_MIN_POINTS);
}

static void first_frame(void) { first_frame_at(0.0f); }

static void files_observations_and_goes_on_after_lost(void)
{
    const float inside[3] = {3.5f, 0.5f, 8.5f}, empty[3] = {3.5f, 1.5f, 8.5f};
    int k;
    setup();
    assert(vm_map_size() == sizeof map && sizeof map < 20000);
    /* refused, the map as it was */
    saved = map;
    assert(vm_update(NULL, &vo, &step, NULL, 0, 0) == 0 && vm_update(&map, &vo, NULL, NULL, 0, 0) == 0);
    assert(vm_update(&map, NULL, &step, NULL, 0, 0) == 0 && vm_update(&map, &vo, &step, mask, 0, 60) == 0);
    vo.count = MVO_MAX_FEATURES + 1;
    assert(vm_update(&map, &vo, &step, NULL, 0, 0) == 0);
    assert(memcmp(&map, &saved, sizeof map) == 0);
    /* too few observations to set the side: nothing filed yet */
    vo.count = 0;
    for (k = 0; k < VM_MIN_POINTS - 1; ++k) {
        put(k, (float)k, 0.0f, 8.0f);
    }
    assert(vm_update(&map, &vo, &step, NULL, 0, 0) == 0 && map.side == 0.0f && map.frames == 1);
    first_frame();
    assert(map.voxel[3].key[0] == 3 && map.voxel[3].key[1] == 0 && map.voxel[3].key[2] == 8);
    assert(map.voxel[3].hits == 1 && map.voxel[3].frames == 1 && !vm_occupied(&map, inside));
    /* a feature without a good point is no observation */
    vo.feature[0].spread = VM_MIN_SPREAD * 0.5f;
    vo.feature[1].has_point = 0;
    vo.feature[2].point[2] = -1.0f;
    assert(vm_update(&map, &vo, &step, NULL, 0, 0) == VM_MIN_POINTS - 3);
    assert(map.voxel[3].frames == 2 && vm_occupied(&map, inside) && !vm_occupied(&map, empty));
    /* rotation steps add nothing; lost steps add nothing and are counted; the map goes on after them */
    saved = map;
    step.kind = MVO_ROTATION;
    assert(vm_update(&map, NULL, &step, NULL, 0, 0) == 0);
    step.kind = MVO_LOST;
    assert(vm_update(&map, NULL, &step, NULL, 0, 0) == 0);
    assert(map.lost == 1 && map.frames == saved.frames + 2 && map.count == saved.count);
    assert(memcmp(map.voxel, saved.voxel, sizeof map.voxel) == 0);
    step.kind = MVO_ESSENTIAL;
    vo.world_centre[0] = 100.0f; /* moved on: new voxels */
    assert(vm_update(&map, &vo, &step, NULL, 0, 0) == VM_MIN_POINTS - 3);
    assert(map.count == saved.count + VM_MIN_POINTS - 3);
    check_index(&map);
    /* the mask: only cells that are not water */
    memset(mask, 0, sizeof mask);
    vo.world_centre[0] = 200.0f;
    assert(vm_update(&map, &vo, &step, mask, 80, 60) == 0);
    mask[30 * 80 + 40] = 1; /* the cell under pixel (160, 120) */
    assert(vm_update(&map, &vo, &step, mask, 80, 60) == VM_MIN_POINTS - 3);
    printf("filing ok voxels=%u frames=%u lost=%u\n", (unsigned)map.count, (unsigned)map.frames, (unsigned)map.lost);
}

static void a_new_side_rekeys_and_merges(void)
{
    const float far_point[3] = {0.5f, 0.5f, 8.5f};
    int k;
    setup();
    first_frame(); /* voxels x = 0..19 at side 1 */
    vo.count = 0;
    for (k = 0; k < VM_MIN_POINTS; ++k) { /* the scene three times as deep: side 3 */
        put(k, 30.0f + (float)k, 0.0f, 24.0f);
    }
    assert(vm_update(&map, &vo, &step, NULL, 0, 0) == VM_MIN_POINTS);
    assert(map.side == 3.0f && map.rescales == 1);
    /* old centres x + 0.5 over 3: 0..19 fall in 7 voxels, so 13 merged; then 20 points over x 30..49 in 7 more */
    assert(map.merged == 13 && map.count == 7 + 7);
    check_index(&map);
    assert(vm_occupied(&map, far_point) == 0); /* each merged voxel had one frame: max, not sum */
    /* a side within VM_RESCALE of the current one stays */
    vo.count = 0;
    for (k = 0; k < VM_MIN_POINTS; ++k) {
        put(k, 30.0f + (float)k, 0.0f, 40.0f);
    }
    vm_update(&map, &vo, &step, NULL, 0, 0);
    assert(map.side == 3.0f && map.rescales == 1);
    printf("rescale ok voxels=%u merged=%u\n", (unsigned)map.count, (unsigned)map.merged);
}

#define FILL 245 /* voxels a fill frame adds */
#define FILLS ((VM_CAPACITY - VM_MIN_POINTS) / FILL) /* fill frames that fit after the first frame's 20 */

/* Fill frame f (1 to FILLS + 1, at most 10) adds FILL voxels within VM_RADIUS: x -24..24, five rows of y from
 * 5 (f - 1) - 20. */
static void fill_frame(int f)
{
    int k;
    vo.count = 0;
    for (k = 0; k < FILL; ++k) {
        put(k, (float)(k % 49 - 24) + 0.5f, (float)(k / 49 + 5 * (f - 1) - 20) + 0.5f, 8.0f);
    }
}

/* The fill frames' voxels go oldest first, and within a frame in the order they were first seen: of fill frames 1 to
 * FILLS, exactly the first `dropped` (in that order) are gone and the rest are held. */
static void check_dropped_in_order(uint32_t dropped)
{
    uint32_t k, held = 0;
    for (k = 0; k < map.count; ++k) {
        const int x = map.voxel[k].key[0], y = map.voxel[k].key[1], f = (y + 20) / 5 + 1;
        if (y >= -20 && f <= FILLS) {
            const uint32_t place = (uint32_t)(f - 1) * FILL + (uint32_t)((y + 20 - 5 * (f - 1)) * 49 + x + 24);
            assert(place >= dropped);
            ++held;
        }
    }
    assert(held + dropped == (uint32_t)FILLS * FILL);
}

static void room_is_made_far_first_then_oldest(void)
{
    uint32_t k;
    int f;
    setup();
    first_frame_at(40.0f); /* 20 voxels, the oldest, on a row the fill frames never reach */
    for (f = 1; f <= FILLS; ++f) {
        fill_frame(f);
        assert(vm_update(&map, &vo, &step, NULL, 0, 0) == FILL);
    }
    assert(map.count == VM_MIN_POINTS + FILLS * FILL && map.evicted == 0);
    fill_frame(FILLS + 1); /* full part-way: room comes VM_EVICT at a time, the first frame's 20 first */
    assert(vm_update(&map, &vo, &step, NULL, 0, 0) == FILL);
    assert(map.evicted > VM_MIN_POINTS && map.evicted % VM_EVICT == 0);
    assert(map.count == VM_MIN_POINTS + (FILLS + 1) * FILL - map.evicted);
    for (k = 0; k < map.count; ++k) {
        assert(map.voxel[k].key[1] != 40);
    }
    check_dropped_in_order(map.evicted - VM_MIN_POINTS);
    check_index(&map);
    /* far first: move the camera 100 voxels away; the next frame's voxels push out everything behind */
    vo.world_centre[0] = 100.0f;
    fill_frame(1);
    for (f = 0; f < FILL; ++f) {
        vo.feature[f].point[1] = (float)(f / 49 + 60) + 0.5f; /* new rows */
    }
    assert(vm_update(&map, &vo, &step, NULL, 0, 0) == FILL);
    check_index(&map);
    for (k = 0; k < map.count; ++k) { /* everything near the old place was far, and went in one batch */
        assert(map.voxel[k].key[0] >= 100 - 24);
    }
    assert(map.count == FILL);
    printf("eviction ok voxels=%u evicted=%u\n", (unsigned)map.count, (unsigned)map.evicted);
}

static void obstacles_outlive_single_sightings(void)
{
    uint32_t k, kept = 0;
    int f;
    setup();
    first_frame_at(40.0f);
    assert(vm_update(&map, &vo, &step, NULL, 0, 0) == VM_MIN_POINTS); /* seen twice: 20 obstacles, the oldest */
    for (f = 1; f <= FILLS; ++f) {
        fill_frame(f);
        assert(vm_update(&map, &vo, &step, NULL, 0, 0) == FILL);
    }
    fill_frame(FILLS + 1); /* room comes from the oldest single sightings (fill frame 1's), not the older obstacles */
    assert(vm_update(&map, &vo, &step, NULL, 0, 0) == FILL);
    assert(map.evicted > 0 && map.evicted % VM_EVICT == 0);
    assert(map.count == VM_MIN_POINTS + (FILLS + 1) * FILL - map.evicted);
    for (k = 0; k < map.count; ++k) {
        kept += map.voxel[k].key[1] == 40;
    }
    assert(kept == VM_MIN_POINTS);
    check_dropped_in_order(map.evicted);
    check_index(&map);
}

/* After a long time without new voxels every age is past what the classes count: room is still made VM_EVICT at a
 * time, the first seen first, never the whole map at once. */
static void a_long_pause_drops_only_a_batch(void)
{
    int f;
    setup();
    first_frame_at(40.0f);
    for (f = 1; f <= FILLS; ++f) {
        fill_frame(f);
        assert(vm_update(&map, &vo, &step, NULL, 0, 0) == FILL);
    }
    step.kind = MVO_ROTATION;
    for (f = 0; f < 200; ++f) { /* 200 frames that file nothing: every voxel ages past 127 */
        vm_update(&map, NULL, &step, NULL, 0, 0);
    }
    step.kind = MVO_ESSENTIAL;
    fill_frame(FILLS + 1);
    assert(vm_update(&map, &vo, &step, NULL, 0, 0) == FILL);
    assert(map.evicted == VM_EVICT || map.evicted == 2 * VM_EVICT);
    assert(map.count == VM_MIN_POINTS + (FILLS + 1) * FILL - map.evicted);
    check_index(&map);
}

static void same_input_same_bytes(void)
{
    int f;
    setup();
    first_frame();
    for (f = 1; f <= 8; ++f) {
        fill_frame(f);
        vm_update(&map, &vo, &step, NULL, 0, 0);
    }
    other = map;
    setup();
    first_frame();
    for (f = 1; f <= 8; ++f) {
        fill_frame(f);
        vm_update(&map, &vo, &step, NULL, 0, 0);
    }
    assert(memcmp(&map, &other, sizeof map) == 0);
}

int main(void)
{
    files_observations_and_goes_on_after_lost();
    a_new_side_rekeys_and_merges();
    room_is_made_far_first_then_oldest();
    obstacles_outlive_single_sightings();
    a_long_pause_drops_only_a_batch();
    same_input_same_bytes();
    printf("voxel_map_check ok map_bytes=%u\n", (unsigned)vm_map_size());
    return 0;
}
