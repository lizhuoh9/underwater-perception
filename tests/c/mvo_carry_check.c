/* Known geometry injected after tracking (as mvo_lost_turn_check.c does): does a LOST step carry the scale on, and only
 * when it should? The camera comes in with the points of an odometry that has been moving at the true scale (120
 * points 1 to 6 units ahead) and a recent speed of 0.1 or `speed`. It then moves by `forward` (0: it holds still) while
 * turning by yaw, with noisy tracks and outliers.
 *   direct: mvo_essential_pose and mvo_carry_depths on that pair. Per trial: the pose's inliers, whether the depths were
 *           carried, and the log2 error of the carried points' median depth against their true depth in the new frame.
 *   steps:  two mvo_step calls: that step, then one 0.1 forward and 0.02 right with clean tracks from the true
 *           positions, so that only the points the first step left carry its noise on. Per trial: both kinds and the
 *           log2 error of the second step's length.
 *   hovers: `forward` is a count instead: that many steps hold the camera still (turning by yaw on the first), each
 *           from the true positions with fresh noise and outliers, then the clean move as in steps. Per trial: the
 *           first and the last hover's kinds, the move's kind and the log2 error of its length. */
#include "mono_vo_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define POINTS 120

static float world[POINTS][3], next[POINTS][2], start[POINTS][2];
static int from_start;

static float uniform(uint32_t *state)
{
    *state = *state * 1664525u + 1013904223u;
    return (float)(*state >> 8) * (1.0f / 16777216.0f);
}

/* Sum of twelve uniforms: zero mean, unit variance. */
static float normal(uint32_t *state)
{
    int k;
    float total = 0.0f;
    for (k = 0; k < 12; ++k) {
        total += uniform(state);
    }
    return total - 6.0f;
}

/* The tracked correspondences are injected: this does not test the image tracker. Every slot keeps its point. */
void mvo_build_pyramid(const mvo_t *s, uint8_t *pyramid) { (void)s; (void)pyramid; }
void mvo_detect(mvo_t *s) { (void)s; }
int mvo_track_features(mvo_t *s, const uint8_t *previous, const uint8_t *current)
{
    int k;
    (void)previous;
    (void)current;
    for (k = 0; k < POINTS; ++k) {
        mvo_pa[k][0] = from_start ? start[k][0] : s->feature[k].x;
        mvo_pa[k][1] = from_start ? start[k][1] : s->feature[k].y;
        s->feature[k].x = next[k][0];
        s->feature[k].y = next[k][1];
    }
    return s->count = POINTS;
}

/* World point k in the frame of a camera at centre c turned by yaw (X_camera = R_y(yaw) (X - c)). */
static void in_camera(int k, const float c[3], float yaw, float out[3])
{
    const float a[3] = {world[k][0] - c[0], world[k][1] - c[1], world[k][2] - c[2]};
    out[0] = cosf(yaw) * a[0] - sinf(yaw) * a[2];
    out[1] = a[1];
    out[2] = sinf(yaw) * a[0] + cosf(yaw) * a[2];
}

static void project(const mvo_camera_t *camera, int k, const float c[3], float yaw, float out[2])
{
    float p[3];
    in_camera(k, c, yaw, p);
    out[0] = camera->fx * p[0] / p[2] + camera->cx;
    out[1] = camera->fy * p[1] / p[2] + camera->cy;
}

/* An odometry that has been moving: points at the true scale, the recent speed `speed`. */
static void prepare(mvo_t *state, const mvo_camera_t *camera, uint32_t *rng, float speed)
{
    int k;
    mvo_init(state, camera, 12345u, NULL, NULL);
    for (k = 0; k < POINTS; ++k) {
        const float x = 1.1f * uniform(rng) - 0.55f, y = 0.7f * uniform(rng) - 0.35f;
        const float z = 1.0f + 5.0f * uniform(rng);
        world[k][0] = x * z;
        world[k][1] = y * z;
        world[k][2] = z;
        state->feature[k].x = camera->fx * x + camera->cx;
        state->feature[k].y = camera->fy * y + camera->cy;
        state->feature[k].has_point = 1;
        memcpy(state->feature[k].point, world[k], sizeof world[k]);
        state->feature[k].spread = 0.01f;
    }
    state->count = POINTS;
    for (k = 0; k < MVO_SPEED_STEPS; ++k) {
        state->recent_length[k] = speed;
    }
    state->recent_count = MVO_SPEED_STEPS;
    state->recent_next = 0;
}

/* The tracks of the first move: noisy, with outliers. */
static void first_tracks(const mvo_camera_t *camera, uint32_t *rng, const float c[3], float yaw, float sigma,
                         float outliers)
{
    int k;
    for (k = 0; k < POINTS; ++k) {
        project(camera, k, c, yaw, next[k]);
        start[k][0] = next[k][0];
        start[k][1] = next[k][1];
        if ((float)k < outliers * (float)POINTS) {
            next[k][0] = camera->cx + 160.0f * (uniform(rng) - 0.5f);
            next[k][1] = camera->cy + 120.0f * (uniform(rng) - 0.5f);
        } else {
            next[k][0] += sigma * normal(rng);
            next[k][1] += sigma * normal(rng);
        }
    }
}

int main(int argc, char **argv)
{
    static mvo_t state;
    const mvo_camera_t camera = {150.0f, 150.0f, 160.0f, 120.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float sigma, outliers, yaw, forward, speed;
    int direct, hovers = 1, first_seed, trials, trial, k;
    if (argc != 9 || (strcmp(argv[1], "direct") != 0 && strcmp(argv[1], "steps") != 0
                      && strcmp(argv[1], "hovers") != 0)) {
        fprintf(stderr, "usage: direct|steps|hovers sigma outliers yaw_deg forward|count speed first_seed trials\n");
        return 2;
    }
    direct = strcmp(argv[1], "direct") == 0;
    sigma = (float)atof(argv[2]);
    outliers = (float)atof(argv[3]);
    yaw = (float)atof(argv[4]) * (3.14159265358979323846f / 180.0f);
    forward = (float)atof(argv[5]);
    if (strcmp(argv[1], "hovers") == 0) {
        hovers = atoi(argv[5]);
        forward = 0.0f;
    }
    speed = (float)atof(argv[6]);
    first_seed = atoi(argv[7]);
    trials = atoi(argv[8]);
    if (sigma < 0.0f || outliers < 0.0f || outliers > 1.0f || speed <= 0.0f || trials < 1 || hovers < 1) {
        fprintf(stderr, "invalid experiment parameters\n");
        return 2;
    }
    printf(direct ? "seed,inliers,carried,depth_error\n" : "seed,kind1,kind_last,kind2,length_error\n");
    for (trial = 0; trial < trials; ++trial) {
        uint32_t rng = (uint32_t)(first_seed + trial);
        const float after1[3] = {0.0f, 0.0f, forward};
        prepare(&state, &camera, &rng, speed);
        first_tracks(&camera, &rng, after1, yaw, sigma, outliers);
        if (direct) {
            float r[9], t[3], ratios[POINTS];
            int count, carried = 0, used = 0;
            for (k = 0; k < POINTS; ++k) {
                mvo_pa[k][0] = state.feature[k].x;
                mvo_pa[k][1] = state.feature[k].y;
                mvo_pb[k][0] = next[k][0];
                mvo_pb[k][1] = next[k][1];
                to_ray(&camera, mvo_pa[k], mvo_ra[k]);
                to_ray(&camera, mvo_pb[k], mvo_rb[k]);
            }
            count = mvo_essential_pose(&state, POINTS, r, t, NULL);
            if (count >= MVO_MIN_INLIERS) {
                carried = mvo_carry_depths(&state, POINTS, r, t);
            }
            for (k = 0; carried && k < POINTS; ++k) {
                float truth[3];
                if (state.feature[k].has_point) {
                    in_camera(k, after1, yaw, truth);
                    ratios[used++] = state.feature[k].point[2] / truth[2];
                }
            }
            printf("%d,%d,%d,%.4f\n", first_seed + trial, count, carried,
                   used ? log2((double)mvo_median(ratios, used)) : -99.0);
        } else {
            mvo_step_t step1, last, step2;
            float after2[3], length, truth;
            int h;
            from_start = 0;
            mvo_step(&state, &step1);
            last = step1;
            for (h = 1; h < hovers; ++h) { /* still, from the true positions, with fresh noise */
                from_start = 1;
                for (k = 0; k < POINTS; ++k) {
                    if ((float)k < outliers * (float)POINTS) {
                        next[k][0] = camera.cx + 160.0f * (uniform(&rng) - 0.5f);
                        next[k][1] = camera.cy + 120.0f * (uniform(&rng) - 0.5f);
                    } else {
                        next[k][0] = start[k][0] + sigma * normal(&rng);
                        next[k][1] = start[k][1] + sigma * normal(&rng);
                    }
                }
                mvo_step(&state, &last);
            }
            after2[0] = after1[0] + 0.1f * sinf(yaw) + 0.02f * cosf(yaw); /* 0.1 forward, 0.02 right when turned */
            after2[1] = after1[1];
            after2[2] = after1[2] + 0.1f * cosf(yaw) - 0.02f * sinf(yaw);
            from_start = 1;
            for (k = 0; k < POINTS; ++k) {
                project(&camera, k, after2, yaw, next[k]);
                next[k][0] += 0.2f * normal(&rng);
                next[k][1] += 0.2f * normal(&rng);
            }
            mvo_step(&state, &step2);
            length = sqrtf(step2.centre[0] * step2.centre[0] + step2.centre[1] * step2.centre[1]
                           + step2.centre[2] * step2.centre[2]);
            truth = sqrtf(0.1f * 0.1f + 0.02f * 0.02f);
            printf("%d,%d,%d,%d,%.4f\n", first_seed + trial, (int)step1.kind, (int)last.kind, (int)step2.kind,
                   length > 0.0f ? log2((double)length / (double)truth) : -99.0);
        }
    }
    return 0;
}
