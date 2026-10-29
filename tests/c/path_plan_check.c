/* The planner's contract (core/path_plan.h): ranges in steps from the map, the mask's veto lifted only by the map,
 * the goal, creeping, turning and stopping. */
#include "path_plan.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

static vm_map_t map;
static mvo_t vo;
static od_layout_t layout;
static od_state_t dirs;
static pp_plan_t plan, saved;

/* The lake clip's camera: 110 degrees across a 960x544 frame stretched to 320x240. */
static const mvo_camera_t lake = {112.0302963f, 148.2792358f, 160.0f, 120.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

/* A camera at the origin looking along world z, one step being half a unit (so PP_SAFE_STEPS is 4 units), an empty
 * map whose cells (PP_GROUP voxels to a side) are units, every direction seen and, from a first mask, none blocked. */
static void setup(void)
{
    memset(&vo, 0, sizeof vo);
    vo.world_rotation[0] = vo.world_rotation[4] = vo.world_rotation[8] = 1.0f;
    vo.recent_count = 3;
    vo.recent_length[0] = 0.4f;
    vo.recent_length[1] = 0.5f;
    vo.recent_length[2] = 0.9f;
    vm_init(&map);
    map.side = 1.0f / (float)PP_GROUP;
    assert(od_layout(&layout, &lake, 80, 60) == 1);
    od_init(&dirs);
    dirs.frames = 1;
    pp_init(&plan);
}

/* An obstacle voxel (seen in `frames` frames) holding world point (x, y, z); the planner measures to the centre of
 * its unit cell. */
static void obstacle(float x, float y, float z, int frames)
{
    vm_voxel_t *v = &map.voxel[map.count++];
    x *= (float)PP_GROUP;
    y *= (float)PP_GROUP;
    z *= (float)PP_GROUP;
    v->key[0] = (int16_t)(x < 0.0f ? (int)x - 1 : (int)x);
    v->key[1] = (int16_t)(y < 0.0f ? (int)y - 1 : (int)y);
    v->key[2] = (int16_t)(z < 0.0f ? (int)z - 1 : (int)z);
    v->hits = v->frames = (uint8_t)frames;
}

static void contract_and_stop(void)
{
    setup();
    assert(pp_plan_size() == sizeof plan);
    assert(plan.direction == -1 && plan.goal == OD_CENTRE && plan.range[0] == -1.0f && plan.nearest == -1.0f);
    saved = plan;
    assert(pp_update(NULL, &map, &vo, &layout, &dirs, NULL) == -1 && pp_update(&plan, NULL, &vo, NULL, NULL, NULL) == -1);
    assert(pp_update(&plan, &map, NULL, NULL, NULL, NULL) == -1);
    map.count = VM_CAPACITY + 1;
    assert(pp_update(&plan, &map, &vo, NULL, NULL, NULL) == -1 && memcmp(&plan, &saved, sizeof plan) == 0);
    map.count = 0;
    /* nothing known (no speed yet, no mask) and nothing seen: stop; with no speed but a mask, the mask decides */
    vo.recent_count = 0;
    memset(layout.cells, 0, sizeof layout.cells);
    assert(pp_update(&plan, &map, &vo, &layout, NULL, NULL) == PP_STOP && plan.direction == -1 && plan.speed == 0.0f);
    setup();
    vo.recent_count = 0;
    dirs.blocked[OD_CENTRE] = 1;
    assert(pp_update(&plan, &map, &vo, &layout, &dirs, NULL) == PP_GO && plan.direction == OD_CENTRE - 1);
    assert(plan.known == 0 && plan.passable == (uint16_t)(((1u << OD_DIRECTIONS) - 1) & ~(1u << OD_CENTRE)));
    /* a direction state that has taken no mask yet is no mask: with no speed either, stop */
    setup();
    vo.recent_count = 0;
    dirs.frames = 0;
    memset(layout.cells, 0, sizeof layout.cells);
    assert(pp_update(&plan, &map, &vo, &layout, &dirs, NULL) == PP_STOP);
    printf("stop ok\n");
}

static void map_ranges_in_steps(void)
{
    setup();
    obstacle(0.2f, 0.3f, 3.0f, 2);
    obstacle(-0.3f, 0.1f, 3.5f, 2); /* ahead within 4 units: the centre is not passable */
    obstacle(0.0f, 0.0f, 1.0f, 1);  /* a single sighting: no obstacle */
    assert(pp_update(&plan, &map, &vo, &layout, &dirs, NULL) == PP_GO);
    assert(plan.speed == 0.5f && plan.direction == OD_CENTRE - 1 && !((plan.passable >> OD_CENTRE) & 1u));
    assert((plan.known >> OD_CENTRE) & 1u && plan.range[OD_CENTRE] > 3.4f && plan.range[OD_CENTRE] < 3.7f);
    assert(plan.nearest > 3.5f && plan.nearest < 3.6f); /* centre (0.5, 0.5, 3.5), not the single sighting's */
    /* one obstacle alone leaves the range unknown, so the mask decides */
    setup();
    obstacle(0.2f, 0.3f, 3.0f, 2);
    assert(pp_update(&plan, &map, &vo, &layout, &dirs, NULL) == PP_GO && plan.direction == OD_CENTRE && plan.known == 0);
    /* the mask blocks the centre and the map sees free water there to 6.5 units (13 steps): the mask's veto stands */
    setup();
    dirs.blocked[OD_CENTRE] = 1;
    obstacle(0.2f, 0.3f, 6.0f, 3);
    obstacle(-0.3f, 0.1f, 6.5f, 2);
    assert(pp_update(&plan, &map, &vo, &layout, &dirs, NULL) == PP_GO && plan.direction == OD_CENTRE - 1);
    assert((plan.known >> OD_CENTRE) & 1u && plan.range[OD_CENTRE] > 6.5f && !((plan.passable >> OD_CENTRE) & 1u));
    /* beyond the horizon (40 steps, 20 units) obstacles do not count */
    setup();
    obstacle(0.0f, 0.0f, 25.0f, 2);
    obstacle(0.5f, 0.0f, 25.5f, 2);
    assert(pp_update(&plan, &map, &vo, &layout, &dirs, NULL) == PP_GO && plan.known == 0 && plan.nearest == -1.0f);
    printf("ranges ok\n");
}

static void goal_and_previous_choice(void)
{
    const float right_ahead[3] = {1.0f, 0.0f, 1.0f}, behind_left[3] = {-1.0f, 0.0f, -1.0f};
    const float zero[3] = {0.0f, 0.0f, 0.0f}, up[3] = {0.0f, -1.0f, 0.0f}, down_behind[3] = {0.1f, 1.0f, -0.002f};
    setup();
    assert(pp_update(&plan, &map, &vo, &layout, &dirs, right_ahead) == PP_GO);
    assert(plan.goal == OD_CENTRE + 2 && plan.direction == OD_CENTRE + 2); /* 45 degrees right: the hard right sector */
    assert(pp_update(&plan, &map, &vo, &layout, &dirs, behind_left) == PP_GO);
    assert(plan.goal == OD_COLUMNS && plan.direction == OD_COLUMNS); /* behind: the hard sector on its side */
    assert(pp_update(&plan, &map, &vo, &layout, &dirs, zero) == PP_GO && plan.goal == OD_CENTRE);
    assert(pp_update(&plan, &map, &vo, &layout, &dirs, up) == PP_GO && plan.goal == OD_COLUMNS / 2);
    assert(pp_update(&plan, &map, &vo, &layout, &dirs, down_behind) == PP_GO && plan.goal == 2 * OD_COLUMNS + 2);
    /* the camera turned 90 degrees right (its x is world -z): a goal straight ahead in the world is now to its left */
    {
        const float ahead[3] = {0.0f, 0.0f, 1.0f};
        setup();
        memset(vo.world_rotation, 0, sizeof vo.world_rotation);
        vo.world_rotation[2] = 1.0f;  /* world x = camera z */
        vo.world_rotation[4] = 1.0f;  /* world y = camera y */
        vo.world_rotation[6] = -1.0f; /* world z = -camera x */
        assert(pp_update(&plan, &map, &vo, &layout, &dirs, ahead) == PP_GO && plan.goal == OD_COLUMNS);
    }
    /* left and right equally good: the one planned last stays */
    setup();
    dirs.blocked[OD_CENTRE] = 1;
    assert(pp_update(&plan, &map, &vo, &layout, &dirs, NULL) == PP_GO && plan.direction == OD_CENTRE - 1);
    plan.direction = OD_CENTRE + 1;
    assert(pp_update(&plan, &map, &vo, &layout, &dirs, NULL) == PP_GO && plan.direction == OD_CENTRE + 1);
    printf("goal ok\n");
}

static void creep_turn(void)
{
    int d;
    setup();
    for (d = 0; d < OD_DIRECTIONS; ++d) { /* everything blocked by the mask */
        dirs.blocked[d] = 1;
    }
    obstacle(-2.5f, 0.0f, 2.5f, 2); /* hard left, level, 3.6 units: 7.1 steps, too near to go, far enough to creep */
    obstacle(-2.6f, -0.4f, 2.6f, 2); /* the cell above, as far */
    assert(pp_update(&plan, &map, &vo, &layout, &dirs, NULL) == PP_CREEP && plan.direction == OD_COLUMNS);
    /* nothing to creep into: turn away from the side the map fills (right), towards the left */
    setup();
    for (d = 0; d < OD_DIRECTIONS; ++d) {
        dirs.blocked[d] = 1;
    }
    obstacle(0.5f, 0.0f, 0.2f, 2); /* right in front, 0.9 units: too near even to creep */
    obstacle(0.6f, 0.1f, 0.3f, 2);
    obstacle(2.0f, 0.0f, -1.0f, 2); /* behind on the right counts too */
    assert(pp_update(&plan, &map, &vo, &layout, &dirs, NULL) == PP_TURN && plan.turn == -1 && plan.direction == -1);
    /* an empty map: the mask's level band decides (more not-water on the left: turn right) */
    setup();
    for (d = 0; d < OD_DIRECTIONS; ++d) {
        dirs.blocked[d] = 1;
        dirs.share[d] = 200;
    }
    dirs.share[OD_COLUMNS] = 255;
    assert(pp_update(&plan, &map, &vo, &layout, &dirs, NULL) == PP_TURN && plan.turn == 1);
    printf("creep and turn ok\n");
}

/* The water directions' bound: PP_CLEAR_SHARE of the median range of the voxels filed in this frame, in front and within
 * the horizon, once there are PP_MIN_VIEW of them; seen in an earlier frame or behind the camera, they do not count. */
static void water_is_clear_to_the_surfaces_in_view(void)
{
    int z;
    float expected;
    setup();
    map.frames = 5;
    for (z = 2; z <= 10; ++z) { /* nine surfaces in view: not enough */
        obstacle(0.0f, 0.0f, (float)z, 1);
        map.voxel[map.count - 1].seen = 5;
    }
    obstacle(0.0f, 0.0f, 1.0f, 2); /* filed in an earlier frame */
    map.voxel[map.count - 1].seen = 4;
    obstacle(0.0f, 0.0f, -3.0f, 1); /* behind */
    map.voxel[map.count - 1].seen = 5;
    assert(pp_update(&plan, &map, &vo, &layout, &dirs, NULL) == PP_GO && plan.clear == -1.0f);
    obstacle(0.0f, 0.0f, 11.0f, 1); /* the tenth: centres (0.5, 0.5, z + 0.5), z = 2..11 */
    map.voxel[map.count - 1].seen = 5;
    assert(pp_update(&plan, &map, &vo, &layout, &dirs, NULL) == PP_GO);
    expected = PP_CLEAR_SHARE * 0.5f * (mvo_sqrt(0.5f + 6.5f * 6.5f) + mvo_sqrt(0.5f + 7.5f * 7.5f));
    assert(plan.clear > expected - 1e-4f && plan.clear < expected + 1e-4f);
    /* without a speed there is no horizon, and no bound */
    vo.recent_count = 0;
    assert(pp_update(&plan, &map, &vo, &layout, &dirs, NULL) == PP_GO && plan.clear == -1.0f);
    printf("clear ok %.3f\n", (double)expected);
}

#if PP_GROUP > 1
/* The plan reads the voxels PP_GROUP to a side (path_plan.h): a cell's frames are its voxels' added, it is in view when
 * any of its voxels was filed in this frame, and voxel indices divide rounding down (-1 and -2 share a cell, -3 not). */
static void voxels_are_read_in_cells(void)
{
    int z;
    float expected;
    /* two single sightings in one cell make an obstacle: with a second cell ahead the range is known */
    setup();
    obstacle(0.2f, 0.3f, 3.0f, 1);
    obstacle(0.7f, 0.8f, 3.7f, 1); /* another voxel of the unit cell (0, 0, 3) */
    obstacle(-0.3f, 0.1f, 3.5f, 2);
    assert(pp_update(&plan, &map, &vo, &layout, &dirs, NULL) == PP_GO);
    assert((plan.known >> OD_CENTRE) & 1u && plan.range[OD_CENTRE] > 3.4f && plan.range[OD_CENTRE] < 3.7f);
    setup(); /* one single sighting is no obstacle */
    obstacle(0.2f, 0.3f, 3.0f, 1);
    obstacle(-0.3f, 0.1f, 3.5f, 2);
    assert(pp_update(&plan, &map, &vo, &layout, &dirs, NULL) == PP_GO && !((plan.known >> OD_CENTRE) & 1u));
    /* negative indices: voxels -1 and -2 along x share cell -1 */
    setup();
    obstacle(-0.25f, 0.3f, 3.0f, 1);
    obstacle(-0.75f, 0.3f, 3.2f, 1);
    obstacle(0.2f, 0.3f, 3.0f, 2);
    assert(pp_update(&plan, &map, &vo, &layout, &dirs, NULL) == PP_GO && (plan.known >> OD_CENTRE) & 1u);
    setup(); /* voxels -2 and -3 lie in cells -1 and -2: two single sightings, no obstacle */
    obstacle(-0.75f, 0.3f, 3.0f, 1);
    obstacle(-1.25f, 0.3f, 3.2f, 1);
    obstacle(0.2f, 0.3f, 3.0f, 2);
    assert(pp_update(&plan, &map, &vo, &layout, &dirs, NULL) == PP_GO && !((plan.known >> OD_CENTRE) & 1u));
    /* in view when any voxel of the cell was filed in this frame: ten cells, each a voxel seen before and one now */
    setup();
    map.frames = 5;
    for (z = 2; z <= 11; ++z) {
        obstacle(0.0f, 0.0f, (float)z, 1);
        map.voxel[map.count - 1].seen = 4;
        obstacle(0.6f, 0.6f, (float)z + 0.6f, 1);
        map.voxel[map.count - 1].seen = 5;
    }
    assert(pp_update(&plan, &map, &vo, &layout, &dirs, NULL) == PP_GO);
    expected = PP_CLEAR_SHARE * 0.5f * (mvo_sqrt(0.5f + 6.5f * 6.5f) + mvo_sqrt(0.5f + 7.5f * 7.5f));
    assert(plan.clear > expected - 1e-4f && plan.clear < expected + 1e-4f);
    printf("cells ok\n");
}
#endif

static void same_input_same_bytes(void)
{
    pp_plan_t other;
    setup();
    obstacle(0.2f, 0.3f, 3.0f, 2);
    obstacle(-0.3f, 0.1f, 3.5f, 2);
    pp_update(&plan, &map, &vo, &layout, &dirs, NULL);
    other = plan;
    setup();
    obstacle(0.2f, 0.3f, 3.0f, 2);
    obstacle(-0.3f, 0.1f, 3.5f, 2);
    pp_update(&plan, &map, &vo, &layout, &dirs, NULL);
    assert(memcmp(&plan, &other, sizeof plan) == 0);
}

int main(void)
{
    contract_and_stop();
    map_ranges_in_steps();
    goal_and_previous_choice();
    creep_turn();
    water_is_clear_to_the_surfaces_in_view();
#if PP_GROUP > 1
    voxels_are_read_in_cells();
#endif
    same_input_same_bytes();
    printf("path_plan_check ok plan_bytes=%u\n", (unsigned)pp_plan_size());
    return 0;
}
