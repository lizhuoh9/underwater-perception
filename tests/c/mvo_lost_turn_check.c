/* Known two-view geometry injected after tracking (as mvo_motion_selection_check.c does): the rotation a lost step
 * takes. The points lie 1 to 6 units ahead; the camera travels by (centre_x, centre_y, centre_z) and turns by yaw,
 * then pitch. One CSV row per trial: the step's kind, the support of the independent rotation fit and of the
 * homography (fresh probes see the same random draws as the step), whether that step falls back on the homography
 * (the fit lacks MVO_MIN_TURN_INLIERS or half of the pairs, the homography has both), and the angle in degrees between
 * the true rotation and the step's, the homography's and no turn. */
#include "mono_vo_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static float before[MVO_MAX_FEATURES][2], after[MVO_MAX_FEATURES][2];
static int pair_count;

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

/* The tracked correspondences are injected: this does not test the image tracker. */
void mvo_build_pyramid(const mvo_t *s, uint8_t *pyramid) { (void)s; (void)pyramid; }
void mvo_detect(mvo_t *s) { (void)s; }
int mvo_track_features(mvo_t *s, const uint8_t *previous, const uint8_t *current)
{
    int k;
    (void)previous;
    (void)current;
    for (k = 0; k < pair_count; ++k) {
        mvo_pa[k][0] = before[k][0];
        mvo_pa[k][1] = before[k][1];
        s->feature[k].x = after[k][0];
        s->feature[k].y = after[k][1];
        s->feature[k].has_point = 0;
    }
    return s->count = pair_count;
}

/* Angle in degrees between a camera-B-to-A rotation and the true one, r being the points' (X_B = r (X_A - c)). */
static double error_degrees(const float rotation_ab[9], const float r[9])
{
    float m[9];
    double cosine;
    mat3_mul(rotation_ab, r, m); /* the identity when rotation_ab is r transposed */
    cosine = 0.5 * ((double)m[0] + m[4] + m[8] - 1.0);
    cosine = cosine > 1.0 ? 1.0 : (cosine < -1.0 ? -1.0 : cosine);
    return acos(cosine) * 180.0 / 3.14159265358979323846;
}

int main(int argc, char **argv)
{
    static mvo_t state, probe;
    static const float identity[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    const mvo_camera_t camera = {150.0f, 150.0f, 160.0f, 120.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float sigma, outliers, centre[3], yaw, pitch, ry[9], rx[9], r[9];
    int first_seed, trials, trial, k;
    if (argc != 11) {
        fprintf(stderr, "usage: pairs sigma outliers centre_x centre_y centre_z yaw_deg pitch_deg first_seed trials\n");
        return 2;
    }
    pair_count = atoi(argv[1]);
    sigma = (float)atof(argv[2]);
    outliers = (float)atof(argv[3]);
    centre[0] = (float)atof(argv[4]);
    centre[1] = (float)atof(argv[5]);
    centre[2] = (float)atof(argv[6]);
    yaw = (float)atof(argv[7]) * (3.14159265358979323846f / 180.0f);
    pitch = (float)atof(argv[8]) * (3.14159265358979323846f / 180.0f);
    first_seed = atoi(argv[9]);
    trials = atoi(argv[10]);
    if (pair_count < 8 || pair_count > MVO_MAX_FEATURES || sigma < 0.0f || outliers < 0.0f || outliers > 1.0f
        || trials < 1) {
        fprintf(stderr, "invalid experiment parameters\n");
        return 2;
    }
    ry[0] = cosf(yaw); ry[1] = 0.0f; ry[2] = -sinf(yaw);
    ry[3] = 0.0f; ry[4] = 1.0f; ry[5] = 0.0f;
    ry[6] = sinf(yaw); ry[7] = 0.0f; ry[8] = cosf(yaw);
    rx[0] = 1.0f; rx[1] = 0.0f; rx[2] = 0.0f;
    rx[3] = 0.0f; rx[4] = cosf(pitch); rx[5] = -sinf(pitch);
    rx[6] = 0.0f; rx[7] = sinf(pitch); rx[8] = cosf(pitch);
    mat3_mul(ry, rx, r);
    printf("seed,kind,inliers,fit_support,homography,fallback,step_error,homography_error,no_turn_error\n");
    for (trial = 0; trial < trials; ++trial) {
        mvo_step_t step;
        uint32_t rng = (uint32_t)(first_seed + trial);
        float homography_rotation[9], fit_rotation[9];
        int homography, fit_support, fallback;
        for (k = 0; k < pair_count; ++k) {
            const float x = 1.1f * uniform(&rng) - 0.55f, y = 0.7f * uniform(&rng) - 0.35f;
            const float z = 1.0f + 5.0f * uniform(&rng);
            const float nx = normal(&rng), ny = normal(&rng);
            const float out_x = 1.1f * uniform(&rng) - 0.55f, out_y = 0.7f * uniform(&rng) - 0.35f;
            const float a[3] = {x * z - centre[0], y * z - centre[1], z - centre[2]};
            float b[3];
            mat3_vec(r, a, b);
            before[k][0] = camera.fx * x + camera.cx;
            before[k][1] = camera.fy * y + camera.cy;
            if ((float)k < outliers * (float)pair_count || b[2] <= 0.05f) {
                after[k][0] = camera.fx * out_x + camera.cx;
                after[k][1] = camera.fy * out_y + camera.cy;
            } else {
                after[k][0] = camera.fx * b[0] / b[2] + camera.cx + sigma * nx;
                after[k][1] = camera.fy * b[1] / b[2] + camera.cy + sigma * ny;
            }
            mvo_pa[k][0] = before[k][0];
            mvo_pa[k][1] = before[k][1];
            mvo_pb[k][0] = after[k][0];
            mvo_pb[k][1] = after[k][1];
            to_ray(&camera, mvo_pa[k], mvo_ra[k]);
            to_ray(&camera, mvo_pb[k], mvo_rb[k]);
        }
        mvo_init(&probe, &camera, 12345u, NULL, NULL); /* the step's own draws: the homography, then the fit */
        homography = mvo_pure_rotation(&probe, pair_count, homography_rotation);
        fit_support = mvo_rotation_model(&probe, pair_count, fit_rotation);
        fallback = !(fit_support >= MVO_MIN_TURN_INLIERS && 2 * fit_support >= pair_count)
                   && homography >= MVO_MIN_TURN_INLIERS && 2 * homography >= pair_count;
        mvo_init(&state, &camera, 12345u, NULL, NULL);
        mvo_step(&state, &step);
        printf("%d,%d,%d,%d,%d,%d,%.4f,%.4f,%.4f\n", first_seed + trial, (int)step.kind, (int)step.inliers,
               fit_support, homography, fallback, error_degrees(step.rotation, r),
               homography > 0 ? error_degrees(homography_rotation, r) : -1.0, error_degrees(identity, r));
    }
    return 0;
}
