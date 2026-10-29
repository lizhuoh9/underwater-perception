/* Which shared points set a step length (shared_ratios in core/mono_vo.c), on exact known geometry: the pose's
 * epipolar inliers that the new pair puts in front of both cameras nearer than MVO_RATIO_DEPTH baselines, whether or
 * not mvo_best has them (that cut is MVO_MAX_DEPTH); never the points beyond, nor the outliers. Each group of points
 * carries its own depth ratio, so the length a lost step's carry gives tells which groups counted. */
#include "mono_vo_internal.h"

#include <stdio.h>
#include <string.h>

static mvo_t state;

/* Point k at (x, y, 1) * depth in frame A, frame B one unit to its right (X_B = X_A + t, t = (-1, 0, 0)); the
 * previous step left it at `ratio` times its depth. */
static void add(int k, float x, float y, float depth, float ratio, int epipolar, int best)
{
    mvo_feature_t *f = &state.feature[k];
    memset(f, 0, sizeof *f);
    f->x = 160.0f;
    f->y = 120.0f;
    f->has_point = 1;
    f->point[0] = ratio * x * depth;
    f->point[1] = ratio * y * depth;
    f->point[2] = ratio * depth;
    mvo_ra[k][0] = x;
    mvo_ra[k][1] = y;
    mvo_rb[k][0] = x - 1.0f / depth;
    mvo_rb[k][1] = y;
    mvo_epipolar[k] = (uint8_t)epipolar;
    mvo_best[k] = (uint8_t)best;
}

int main(void)
{
    static const float r[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    static const float t[3] = {-1.0f, 0.0f, 0.0f};
    const mvo_camera_t camera = {220.0f, 220.0f, 160.0f, 120.0f, 0, 0, 0, 0, 0};
    int k, n = 0;
    float carried;
    mvo_init(&state, &camera, 1u, NULL, NULL);
    for (k = 0; k < 10; ++k, ++n) { /* within MVO_MAX_DEPTH: mvo_best */
        add(n, 0.01f * (float)k, 0.02f, 40.0f, 2.0f, 1, 1);
    }
    for (k = 0; k < 14; ++k, ++n) { /* beyond MVO_MAX_DEPTH, within MVO_RATIO_DEPTH: not mvo_best, still counted */
        add(n, 0.01f * (float)k, -0.02f, 55.0f, 2.4f, 1, 0);
    }
    for (k = 0; k < 30; ++k, ++n) { /* beyond MVO_RATIO_DEPTH */
        add(n, 0.01f * (float)k, 0.05f, 70.0f, 5.0f, 1, 0);
    }
    for (k = 0; k < 30; ++k, ++n) { /* outliers of the pose */
        add(n, 0.01f * (float)k, -0.05f, 30.0f, 9.0f, 0, 0);
    }
    state.count = n;
    state.recent_count = 0; /* no recent speed: no clamp */
    if (!(MVO_MAX_DEPTH < 55.0f && 55.0f < MVO_RATIO_DEPTH && MVO_RATIO_DEPTH < 70.0f)) {
        printf("the groups' depths no longer straddle MVO_MAX_DEPTH and MVO_RATIO_DEPTH\n");
        return 1;
    }
    if (!mvo_carry_depths(&state, n, r, t)) {
        printf("not carried\n");
        return 1;
    }
    /* The 24 counted ratios: ten 2.0, fourteen 2.4; median 2.4. mvo_best alone would give 2.0, no limit 5.0, the
     * outliers 9.0. Only the mvo_best points get a point, at that length times their depth in B (40). */
    carried = state.feature[0].point[2] / 40.0f;
    if (carried < 2.399f || carried > 2.401f) {
        printf("carried length %.4f, expected 2.4\n", (double)carried);
        return 1;
    }
    for (k = 0; k < n; ++k) {
        if (state.feature[k].has_point != (k < 10)) {
            printf("point %d: has_point %d\n", k, state.feature[k].has_point);
            return 1;
        }
    }
    printf("mvo_ratio_check ok\n");
    return 0;
}
