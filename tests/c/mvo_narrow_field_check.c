/* Correct correspondences confined to a narrow calibrated field must still permit motion selection. */
#include "mono_vo_internal.h"
#include <stdio.h>

#define COUNT 36
static float before[COUNT][2], after[COUNT][2];
void mvo_build_pyramid(const mvo_t *s, uint8_t *p) { (void)s; (void)p; }
void mvo_detect(mvo_t *s) { (void)s; }
int mvo_track_features(mvo_t *s, const uint8_t *a, const uint8_t *b)
{
    int k;
    (void)a; (void)b;
    for (k = 0; k < COUNT; ++k) {
        mvo_pa[k][0] = before[k][0]; mvo_pa[k][1] = before[k][1];
        s->feature[k].x = after[k][0]; s->feature[k].y = after[k][1];
        s->feature[k].has_point = 0;
    }
    return s->count = COUNT;
}

int main(void)
{
    static mvo_t state, probe;
    const mvo_camera_t camera = {4000, 4000, 160, 120};
    uint32_t rng = 1u;
    float rotation[9];
    int k, h, support;
    mvo_step_t step;
    for (k = 0; k < COUNT; ++k) {
        float depth;
        rng = rng * 1664525u + 1013904223u;
        depth = 1.5f + 4.5f * (float)(rng >> 8) / 16777216.0f;
        before[k][0] = 230.0f + 8.6f * ((float)(k % 6) - 2.5f);
        before[k][1] = 120.0f + 8.6f * ((float)(k / 6) - 2.5f);
        after[k][0] = before[k][0] - 4000.0f * 0.08f / depth;
        after[k][1] = before[k][1];
        mvo_pa[k][0] = before[k][0]; mvo_pa[k][1] = before[k][1];
        mvo_pb[k][0] = after[k][0]; mvo_pb[k][1] = after[k][1];
        to_ray(&camera, mvo_pa[k], mvo_ra[k]); to_ray(&camera, mvo_pb[k], mvo_rb[k]);
    }
    mvo_init(&probe, &camera, 12345u, 0, 0);
    h = mvo_pure_rotation(&probe, COUNT, rotation);
    support = mvo_rotation_model(&probe, COUNT, rotation);
    mvo_init(&state, &camera, 12345u, 0, 0);
    mvo_step(&state, &step);
    printf("h=%d rotation_support=%d kind=%d x=%.9g y=%.9g z=%.9g\n", h, support,
           (int)step.kind, (double)step.centre[0], (double)step.centre[1], (double)step.centre[2]);
    return 0;
}
