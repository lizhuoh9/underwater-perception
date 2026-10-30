/* The metric scale (core/mono_vo_scale.c) on a hand-built state, without images: each made-up step moves the world
 * centre and calls mvo_scale_step, the world's up settled by hand along -y (the first camera's y points down). One line
 * per check; the first that does not hold ends the process with a failure. */
#include "mono_vo_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TRUE_SCALE 0.5f /* metres per odometry unit */

static mvo_t state;
static const mvo_camera_t camera = {200.0f, 200.0f, 160.0f, 120.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

static void check(int condition, const char *what)
{
    if (!condition) {
        printf("FAIL %s\n", what);
        exit(1);
    }
    printf("ok %s\n", what);
}

/* A fresh run, its up settled, starting at `depth` (NaN: without the first frame's depth). */
static void fresh(float depth)
{
    mvo_init(&state, &camera, 1u, NULL, NULL);
    state.gyro.up_sum[1] = -1.0f;
    state.gyro.up_steps = MVO_IMU_UP_STEPS;
    if (depth == depth) {
        mvo_depth_reading(&state, depth);
    }
    mvo_scale_start(&state);
}

/* One step: the odometry moved by (x, y, z), the world's axes in its unit, with this frame's depth unless NaN. */
static void step(float x, float y, float z, float depth)
{
    const float moved[3] = {x, y, z};
    int i;
    for (i = 0; i < 3; ++i) {
        state.world_centre[i] += moved[i];
    }
    if (depth == depth) {
        mvo_depth_reading(&state, depth);
    }
    state.scale.step_time = MVO_SCALE_STEP_TIME;
    mvo_scale_step(&state, moved);
}

/* `count` steps diving 0.1 a step while moving 0.1 forward, the depth TRUE_SCALE times the odometry's. */
static void dive(int count)
{
    int k;
    for (k = 0; k < count; ++k) {
        step(0.1f, 0.1f, 0.0f, 2.0f + TRUE_SCALE * (state.world_centre[1] + 0.1f));
    }
}

int main(void)
{
    float centre[3], scale, variance, before;
    int k;

    fresh(2.0f);
    dive(60); /* past MVO_DEPTH_RING readings: the ring wraps */
    check(mvo_metric(&state, centre, &scale, &variance), "a steady dive gives the scale");
    check(fabsf(scale / TRUE_SCALE - 1.0f) < 0.01f, "the scale is the depth's over the odometry's");
    check(variance > 0.0f && sqrtf(variance) < 0.2f * scale, "its variance is known and small");
    check(fabsf(centre[1] - (state.scale.depth - 2.0f)) < 1e-4f, "the vertical is the depth since the first frame");
    check(fabsf(centre[0] - TRUE_SCALE * state.world_centre[0]) < 1e-3f, "the horizontal is the moves times the scale");

    /* A later observation is weighed in by its variance, not gated: a step the odometry overshoots by a unit while the
     * depth goes on as before observes half the scale, and the scale follows it part of the way. */
    before = state.scale.scale;
    step(0.1f, 1.1f, 0.0f, state.scale.depth + 0.05f);
    check(state.scale.scale < before && state.scale.scale > 0.5f * before, "a later observation is weighed in");

    /* A weak first window: the odometry swims 0.2 a step and climbs 0.0004 while the depth falls 0.012: 30 m a unit,
     * against an error of 7% of the 2 units swum. Taken, it would scale the path so far by 30. */
    fresh(2.0f);
    for (k = 0; k < 30; ++k) {
        step(0.2f, 0.0004f, 0.0f, 2.0f + 0.012f * (float)(k + 1));
    }
    check(!mvo_metric(&state, centre, &scale, &variance), "a weak first observation is not taken");

    /* Without the first frame's depth the vertical stays the odometry's, scaled. */
    fresh(NAN);
    dive(40);
    check(mvo_metric(&state, centre, &scale, NULL), "the depth's changes still give the scale");
    check(fabsf(centre[1] - scale * state.world_centre[1]) < 0.05f * fabsf(centre[1]), "the vertical is not replaced");

    /* A range before the first frame has no points to compare with: it is dropped. */
    {
        const float axis[3] = {0.0f, 0.0f, 1.0f};
        mvo_init(&state, &camera, 1u, NULL, NULL);
        check(mvo_range_setup(&state, axis, 0.99f) && mvo_range_reading(&state, 2.0f), "a range is taken");
        mvo_scale_start(&state);
        check(!state.scale.has_range && state.scale.has_beam, "mvo_start drops it and keeps the beam");
    }
    return 0;
}
