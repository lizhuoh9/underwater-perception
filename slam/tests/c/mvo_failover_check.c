/* Exercise the conservative model choice with exact 3D correspondences and noisy turns. */
#include "mono_vo_internal.h"

#include <stdio.h>

#define PAIRS 25

static float before[PAIRS][2], after[PAIRS][2];

static float uniform(uint32_t *state)
{
    *state = *state * 1664525u + 1013904223u;
    return (float)(*state >> 8) / 16777216.0f;
}

/* The test injects tracks rather than depending on the separate image tracker. */
void mvo_build_pyramid(const mvo_t *s, uint8_t *pyramid) { (void)s; (void)pyramid; }
void mvo_detect(mvo_t *s) { (void)s; }
int mvo_track_features(mvo_t *s, const uint8_t *previous, const uint8_t *current)
{
    int k;
    (void)previous;
    (void)current;
    for (k = 0; k < PAIRS; ++k) {
        mvo_pa[k][0] = before[k][0];
        mvo_pa[k][1] = before[k][1];
        s->feature[k].x = after[k][0];
        s->feature[k].y = after[k][1];
        s->feature[k].has_point = 0;
    }
    return s->count = PAIRS;
}

int main(void)
{
    static mvo_t state, probe;
    const mvo_camera_t camera = {150.0f, 150.0f, 160.0f, 120.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    mvo_step_t step;
    uint32_t rng = 42u;
    float rotation[9], direction[3];
    int k, homography, essential, trial, outliers, false_turns = 0, weak_false = 0, weak_cases = 0, rng_bad = 0;

    for (k = 0; k < PAIRS; ++k) {
        const float x = 1.2f * uniform(&rng) - 0.6f;
        const float y = 0.8f * uniform(&rng) - 0.4f;
        const float z = 1.0f + 5.0f * uniform(&rng);
        before[k][0] = camera.fx * x + camera.cx;
        before[k][1] = camera.fy * y + camera.cy;
        after[k][0] = camera.fx * (x - 0.3f / z) + camera.cx;
        after[k][1] = before[k][1];
        mvo_pa[k][0] = before[k][0];
        mvo_pa[k][1] = before[k][1];
        mvo_pb[k][0] = after[k][0];
        mvo_pb[k][1] = after[k][1];
        to_ray(&camera, mvo_pa[k], mvo_ra[k]);
        to_ray(&camera, mvo_pb[k], mvo_rb[k]);
    }
    mvo_init(&probe, &camera, 12345u, NULL, NULL);
    homography = mvo_pure_rotation(&probe, PAIRS, rotation);
    essential = mvo_essential_pose(&probe, PAIRS, rotation, direction, NULL);
    mvo_init(&state, &camera, 12345u, NULL, NULL);
    mvo_step(&state, &step);
    printf("homography=%d essential=%d kind=%d inliers=%d x=%.8g y=%.8g z=%.8g\n",
           homography, essential, (int)step.kind, (int)step.inliers,
           (double)step.centre[0], (double)step.centre[1], (double)step.centre[2]);
    /* Unrelated tracks must not be promoted to a translation merely because the fallback runs. */
    for (k = 0; k < PAIRS; ++k) {
        after[k][0] = camera.fx * (1.2f * uniform(&rng) - 0.6f) + camera.cx;
        after[k][1] = camera.fy * (0.8f * uniform(&rng) - 0.4f) + camera.cy;
    }
    mvo_init(&state, &camera, 12345u, NULL, NULL);
    mvo_step(&state, &step);
    printf("unrelated_kind=%d unrelated_x=%.8g unrelated_y=%.8g unrelated_z=%.8g\n",
           (int)step.kind, (double)step.centre[0], (double)step.centre[1], (double)step.centre[2]);
    /* Record how the existing strong-homography path behaves on a noisy turn with outliers. */
    rng = 6u;
    for (k = 0; k < PAIRS; ++k) {
        const float x = 1.1f * uniform(&rng) - 0.55f;
        const float y = 0.7f * uniform(&rng) - 0.35f;
        const float den = -0.08715574f * x + 0.99619470f;
        before[k][0] = camera.fx * x + camera.cx;
        before[k][1] = camera.fy * y + camera.cy;
        if (k < 10) {
            after[k][0] = camera.fx * (1.1f * uniform(&rng) - 0.55f) + camera.cx;
            after[k][1] = camera.fy * (0.7f * uniform(&rng) - 0.35f) + camera.cy;
        } else {
            after[k][0] = camera.fx * ((0.99619470f * x + 0.08715574f) / den) + camera.cx
                          + 2.0f * (uniform(&rng) - 0.5f);
            after[k][1] = camera.fy * y / den + camera.cy + 2.0f * (uniform(&rng) - 0.5f);
        }
    }
    mvo_init(&state, &camera, 12345u, NULL, NULL);
    mvo_step(&state, &step);
    printf("turn_kind=%d turn_x=%.8g turn_y=%.8g turn_z=%.8g\n",
           (int)step.kind, (double)step.centre[0], (double)step.centre[1], (double)step.centre[2]);
    for (outliers = 0; outliers <= 10; outliers += 5) {
      for (trial = 1; trial <= 80; ++trial) {
        rng = (uint32_t)trial;
        for (k = 0; k < PAIRS; ++k) {
            const float x = 1.1f * uniform(&rng) - 0.55f;
            const float y = 0.7f * uniform(&rng) - 0.35f;
            const float den = -0.08715574f * x + 0.99619470f;
            before[k][0] = camera.fx * x + camera.cx;
            before[k][1] = camera.fy * y + camera.cy;
            if (k < outliers) {
                after[k][0] = camera.fx * (1.1f * uniform(&rng) - 0.55f) + camera.cx;
                after[k][1] = camera.fy * (0.7f * uniform(&rng) - 0.35f) + camera.cy;
            } else {
                after[k][0] = camera.fx * ((0.99619470f * x + 0.08715574f) / den) + camera.cx
                              + 2.0f * (uniform(&rng) - 0.5f);
                after[k][1] = camera.fy * y / den + camera.cy + 2.0f * (uniform(&rng) - 0.5f);
            }
            mvo_pa[k][0] = before[k][0];
            mvo_pa[k][1] = before[k][1];
            mvo_pb[k][0] = after[k][0];
            mvo_pb[k][1] = after[k][1];
        }
        mvo_init(&probe, &camera, 12345u, NULL, NULL);
        homography = mvo_pure_rotation(&probe, PAIRS, rotation);
        weak_cases += homography < MVO_MIN_INLIERS;
        mvo_init(&state, &camera, 12345u, NULL, NULL);
        mvo_step(&state, &step);
        false_turns += step.kind == MVO_ESSENTIAL;
        weak_false += homography < MVO_MIN_INLIERS && step.kind == MVO_ESSENTIAL;
        rng_bad += homography < MVO_MIN_INLIERS && step.kind == MVO_LOST && state.rng != probe.rng;
      }
    }
    printf("false_turns=%d weak_false=%d weak_cases=%d rng_bad=%d\n", false_turns, weak_false, weak_cases, rng_bad);
    return 0;
}
