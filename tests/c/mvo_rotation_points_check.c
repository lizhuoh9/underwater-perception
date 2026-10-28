/* Rotation steps keep only 3D points whose tracks agree with the rotation. */
#include "mono_vo_internal.h"
#include <stdio.h>

#define COUNT 36
#define OUTLIERS 6
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
    }
    return s->count = COUNT;
}

int main(void)
{
    static mvo_t s, probe;
    const mvo_camera_t camera = {150, 150, 160, 120};
    int scenario, k;
    for (scenario = 0; scenario < 2; ++scenario) {
        float h[9], parallax;
        float max_error = 0.0f;
        int hc, kept = 0, bad = 0;
        mvo_step_t step;
        mvo_init(&s, &camera, 12345u, 0, 0);
        for (k = 0; k < COUNT; ++k) {
            float x = ((float)(k % 6) - 2.5f) * 0.13f;
            float y = ((float)(k / 6) - 2.5f) * 0.08f;
            before[k][0] = camera.fx * x + camera.cx;
            before[k][1] = camera.fy * y + camera.cy;
            float den = -0.08715574f * x + 0.99619470f;
            after[k][0] = camera.fx * (0.99619470f * x + 0.08715574f) / den + camera.cx
                          + (k < OUTLIERS ? 30.0f : scenario ? (k % 2 ? 1.3f : -1.3f) : 0.0f);
            after[k][1] = camera.fy * y / den + camera.cy;
            s.feature[k].has_point = 1;
            s.feature[k].point[0] = x * 4.0f; s.feature[k].point[1] = y * 4.0f; s.feature[k].point[2] = 4.0f;
            mvo_pa[k][0] = before[k][0]; mvo_pa[k][1] = before[k][1];
            mvo_pb[k][0] = after[k][0]; mvo_pb[k][1] = after[k][1];
            to_ray(&camera, mvo_pa[k], mvo_ra[k]); to_ray(&camera, mvo_pb[k], mvo_rb[k]);
        }
        mvo_init(&probe, &camera, 12345u, 0, 0);
        hc = mvo_pure_rotation(&probe, COUNT, h);
        parallax = mvo_parallax(&probe, COUNT, h);
        mvo_step(&s, &step);
        for (k = 0; k < COUNT; ++k) {
            if (k < OUTLIERS) bad += s.feature[k].has_point != 0;
            else {
                float x = ((float)(k % 6) - 2.5f) * 0.13f, y = ((float)(k / 6) - 2.5f) * 0.08f;
                float truth[3] = {4.0f * (0.99619470f * x + 0.08715574f), 4.0f * y,
                                  4.0f * (-0.08715574f * x + 0.99619470f)};
                int axis;
                kept += s.feature[k].has_point != 0;
                for (axis = 0; axis < 3; ++axis) {
                    float error = mvo_abs(s.feature[k].point[axis] - truth[axis]);
                    if (error > max_error) max_error = error;
                }
            }
        }
        printf("scenario=%d h=%d parallax=%.5g kind=%d outlier_points=%d kept=%d max_error=%.5g\n",
               scenario, hc, (double)parallax, (int)step.kind, bad, kept, (double)max_error);
    }
    return 0;
}
