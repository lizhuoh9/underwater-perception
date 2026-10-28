/* Known two-view geometry injected after tracking, for motion-selection experiments. */
#include "mono_vo_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static float before[MVO_MAX_FEATURES][2], after[MVO_MAX_FEATURES][2];
static int pair_count;

static float uniform(uint32_t *state)
{
    *state = *state * 1664525u + 1013904223u;
    return (float)(*state >> 8) * (1.0f / 16777216.0f);
}

/* Sum of twelve uniforms: zero mean, unit variance, finite tails and no extra library dependency. */
static float normal(uint32_t *state)
{
    int k;
    float total = 0.0f;
    for (k = 0; k < 12; ++k) {
        total += uniform(state);
    }
    return total - 6.0f;
}

/* Kabsch/Wahba fit on the homography's inlier rays, independently scored on all matches. */
static void rotation_diagnostics(const mvo_camera_t *camera, int count, int *support,
                                 float *median_all, float *median_inliers)
{
    float covariance[9] = {0.0f}, u[9], singular[3], v[9], vt[9], r[9];
    float errors[MVO_MAX_FEATURES], inlier_errors[MVO_MAX_FEATURES];
    int k, i, j, used = 0, valid_errors = 0, valid_inlier_errors = 0;
    *support = 0;
    *median_all = *median_inliers = -1.0f;
    for (k = 0; k < count; ++k) {
        float a[3], b[3], anorm, bnorm;
        if (!mvo_inlier[k]) {
            continue;
        }
        to_ray(camera, before[k], a);
        to_ray(camera, after[k], b);
        a[2] = b[2] = 1.0f;
        anorm = sqrtf(dot3(a, a));
        bnorm = sqrtf(dot3(b, b));
        for (i = 0; i < 3; ++i) {
            for (j = 0; j < 3; ++j) {
                covariance[3 * i + j] += (b[i] / bnorm) * (a[j] / anorm);
            }
        }
        ++used;
    }
    if (used < 4) {
        return;
    }
    mvo_svd3(covariance, u, singular, v);
    mat3_transpose(v, vt);
    mat3_mul(u, vt, r);
    if (mat3_det(r) < 0.0f) {
        u[2] = -u[2];
        u[5] = -u[5];
        u[8] = -u[8];
        mat3_mul(u, vt, r);
    }
    for (k = 0; k < count; ++k) {
        float a[3], projected[3], dx, dy, error;
        to_ray(camera, before[k], a);
        a[2] = 1.0f;
        mat3_vec(r, a, projected);
        if (projected[2] <= 1e-6f) {
            error = 1e9f;
        } else {
            dx = camera->fx * projected[0] / projected[2] + camera->cx - after[k][0];
            dy = camera->fy * projected[1] / projected[2] + camera->cy - after[k][1];
            error = sqrtf(dx * dx + dy * dy);
        }
        *support += error <= 2.0f;
        errors[valid_errors++] = error;
        if (mvo_inlier[k]) {
            inlier_errors[valid_inlier_errors++] = error;
        }
    }
    *median_all = mvo_median(errors, valid_errors);
    *median_inliers = mvo_median(inlier_errors, valid_inlier_errors);
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

int main(int argc, char **argv)
{
    static mvo_t state, probe;
    const mvo_camera_t camera = {150.0f, 150.0f, 160.0f, 120.0f};
    float sigma_x, sigma_y, outlier_fraction, centre_x, centre_z, yaw_degrees, angle, c, s;
    int first_seed, trials, trial, k;

    if (argc != 11) {
        fprintf(stderr, "usage: pairs sigma_x sigma_y outlier_fraction centre_x centre_z yaw_degrees first_seed trials tag\n");
        return 2;
    }
    pair_count = atoi(argv[1]);
    sigma_x = (float)atof(argv[2]);
    sigma_y = (float)atof(argv[3]);
    outlier_fraction = (float)atof(argv[4]);
    centre_x = (float)atof(argv[5]);
    centre_z = (float)atof(argv[6]);
    yaw_degrees = (float)atof(argv[7]);
    first_seed = atoi(argv[8]);
    trials = atoi(argv[9]);
    if (pair_count < 8 || pair_count > MVO_MAX_FEATURES || sigma_x < 0.0f || sigma_y < 0.0f
        || outlier_fraction < 0.0f || outlier_fraction > 1.0f || trials < 1) {
        fprintf(stderr, "invalid experiment parameters\n");
        return 2;
    }
    angle = yaw_degrees * (3.14159265358979323846f / 180.0f);
    c = cosf(angle);
    s = sinf(angle);
    printf("tag,seed,pairs,sigma_x,sigma_y,outlier_fraction,centre_x,centre_z,yaw_deg,homography,kind,inliers,"
           "estimate_x,estimate_y,estimate_z,direction_cos,rotation_support_2px,rotation_median_px,"
           "rotation_h_inlier_median_px,rotation_model_support,cpu_ms\n");

    for (trial = 0; trial < trials; ++trial) {
        mvo_step_t step;
        uint32_t rng = (uint32_t)(first_seed + trial);
        float rotation[9], length, truth_length, direction_cos = 0.0f, rotation_median, rotation_inlier_median;
        int homography, rotation_support, rotation_model_support;
        clock_t started;

        for (k = 0; k < pair_count; ++k) {
            const float x = 1.1f * uniform(&rng) - 0.55f;
            const float y = 0.7f * uniform(&rng) - 0.35f;
            const float z = 1.0f + 5.0f * uniform(&rng);
            const float nx = normal(&rng), ny = normal(&rng);
            const float out_x = 1.1f * uniform(&rng) - 0.55f;
            const float out_y = 0.7f * uniform(&rng) - 0.35f;
            const float moved_x = x * z - centre_x, moved_z = z - centre_z;
            const float bx = c * moved_x - s * moved_z;
            const float bz = s * moved_x + c * moved_z;
            before[k][0] = camera.fx * x + camera.cx;
            before[k][1] = camera.fy * y + camera.cy;
            if ((float)k < outlier_fraction * (float)pair_count) {
                after[k][0] = camera.fx * out_x + camera.cx;
                after[k][1] = camera.fy * out_y + camera.cy;
            } else {
                after[k][0] = camera.fx * bx / bz + camera.cx + sigma_x * nx;
                after[k][1] = camera.fy * y * z / bz + camera.cy + sigma_y * ny;
            }
            mvo_pa[k][0] = before[k][0];
            mvo_pa[k][1] = before[k][1];
            mvo_pb[k][0] = after[k][0];
            mvo_pb[k][1] = after[k][1];
            to_ray(&camera, mvo_pa[k], mvo_ra[k]);
            to_ray(&camera, mvo_pb[k], mvo_rb[k]);
        }
        mvo_init(&probe, &camera, 12345u, NULL, NULL);
        homography = mvo_pure_rotation(&probe, pair_count, rotation);
        if (homography >= 4) {
            rotation_diagnostics(&camera, pair_count, &rotation_support, &rotation_median, &rotation_inlier_median);
        } else {
            rotation_support = 0;
            rotation_median = rotation_inlier_median = -1.0f;
        }
        rotation_model_support = mvo_rotation_model(&probe, pair_count, rotation);
        mvo_init(&state, &camera, 12345u, NULL, NULL);
        started = clock();
        mvo_step(&state, &step);
        length = sqrtf(step.centre[0] * step.centre[0] + step.centre[1] * step.centre[1]
                       + step.centre[2] * step.centre[2]);
        truth_length = sqrtf(centre_x * centre_x + centre_z * centre_z);
        if (length > 0.0f && truth_length > 0.0f) {
            direction_cos = (step.centre[0] * centre_x + step.centre[2] * centre_z) / (length * truth_length);
        }
        printf("%s,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%d,%d,%d,%.9g,%.9g,%.9g,%.9g,%d,%.9g,%.9g,%d,%.3f\n",
               argv[10], first_seed + trial, pair_count, (double)sigma_x, (double)sigma_y, (double)outlier_fraction,
               (double)centre_x, (double)centre_z, (double)yaw_degrees, homography, (int)step.kind, (int)step.inliers,
               (double)step.centre[0], (double)step.centre[1], (double)step.centre[2], (double)direction_cos,
               rotation_support, (double)rotation_median, (double)rotation_inlier_median, rotation_model_support,
               1000.0 * (double)(clock() - started) / (double)CLOCKS_PER_SEC);
    }
    return 0;
}
