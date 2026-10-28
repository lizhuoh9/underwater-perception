/*
 * mono_vo_internal.h - what the units of the monocular visual odometry share; not part of its interface (mono_vo.h).
 * It holds the scratch that one step passes from unit to unit, the small helpers every unit inlines and each unit's
 * entry points. A build compiles all five units:
 *
 *   mono_vo.c             the public functions and the step
 *   mono_vo_numerics.c    Jacobi eigenvectors and the 3x3 SVD, Gaussian elimination, Householder QR and one-sided
 *                         Jacobi, the Cayley rotation, RANSAC helpers
 *   mono_vo_tracking.c    image pyramid, Shi-Tomasi corners, fixed-point pyramidal Lucas-Kanade
 *   mono_vo_homography.c  the rotation of a pure turn and the parallax test
 *   mono_vo_essential.c   the motion from the essential matrix: eight-point MSAC, decomposition and triangulation,
 *                         robust Levenberg-Marquardt, the direction search
 */
#ifndef MONO_VO_INTERNAL_H
#define MONO_VO_INTERNAL_H

#include <string.h>

#include "mono_vo.h"

#define MVO_RANSAC_MAX 500

/* 2 ln(4 n) for n = MVO_MAX_FEATURES, to three decimals (no libm on the board): the size term of the translation
 * evidence's penalty (mono_vo_essential.c). */
#if MVO_MAX_FEATURES == 250
#define MVO_EVIDENCE_LOG 13.816f
#elif MVO_MAX_FEATURES == 300
#define MVO_EVIDENCE_LOG 14.180f
#else
#error "MVO_EVIDENCE_LOG needs 2 ln(4 n) for this MVO_MAX_FEATURES"
#endif

/* Scratch the units share (defined in mono_vo.c), not thread-safe (see mono_vo.h). */
extern float mvo_pa[MVO_MAX_FEATURES][2], mvo_pb[MVO_MAX_FEATURES][2]; /* pixels in the previous / current frame */
extern float mvo_ra[MVO_MAX_FEATURES][2], mvo_rb[MVO_MAX_FEATURES][2]; /* the same as calibrated rays (x, y, 1) */
extern uint8_t mvo_inlier[MVO_MAX_FEATURES]; /* a model's inliers while it is fitted */
extern uint8_t mvo_best[MVO_MAX_FEATURES];   /* the essential pose's inliers in front, near enough (mvo_essential_pose) */
extern uint8_t mvo_epipolar[MVO_MAX_FEATURES]; /* all the essential pose's inliers, far ones too */
extern float mvo_values[MVO_MAX_FEATURES];   /* per-pair values to take the median of */

/* 1 / sqrt(v) for v > 0 without a division, which costs about twenty multiplications on the S3: three Newton steps
 * from the classic bit-level guess, near float precision and the same on every IEEE-754 platform when no multiply-add
 * is fused. Values below 1e-30 are scaled by 2^64 first, so that the guess sees a normal exponent. */
static inline float mvo_rsqrt(float v)
{
    union {
        float f;
        uint32_t u;
    } guess;
    float half, y, scale = 1.0f;
    if (v < 1.0e-30f) {
        v *= 1.8446744e19f;
        scale = 4294967296.0f;
    }
    half = 0.5f * v;
    guess.f = v;
    guess.u = 0x5F375A86u - (guess.u >> 1);
    y = guess.f;
    y = y * (1.5f - half * y * y);
    y = y * (1.5f - half * y * y);
    y = y * (1.5f - half * y * y);
    return y * scale;
}

/* mvo_sqrt, inline so that the units' inner loops make no call for it. */
static inline float mvo_root(float v)
{
    if (v <= 0.0f) {
        return 0.0f;
    }
    if (v > 3.0e38f) { /* infinity: v times its zero reciprocal root would give NaN */
        return v;
    }
    return v * mvo_rsqrt(v);
}

static inline float mvo_abs(float v) { return v < 0.0f ? -v : v; }

static inline void mat3_mul(const float a[9], const float b[9], float out[9])
{
    float r[9];
    int i, j;
    for (i = 0; i < 3; ++i) {
        for (j = 0; j < 3; ++j) {
            r[3 * i + j] = a[3 * i] * b[j] + a[3 * i + 1] * b[3 + j] + a[3 * i + 2] * b[6 + j];
        }
    }
    memcpy(out, r, sizeof r);
}

static inline void mat3_transpose(const float a[9], float out[9])
{
    float r[9];
    int i, j;
    for (i = 0; i < 3; ++i) {
        for (j = 0; j < 3; ++j) {
            r[3 * i + j] = a[3 * j + i];
        }
    }
    memcpy(out, r, sizeof r);
}

static inline void mat3_vec(const float a[9], const float v[3], float out[3])
{
    float r[3];
    int i;
    for (i = 0; i < 3; ++i) {
        r[i] = a[3 * i] * v[0] + a[3 * i + 1] * v[1] + a[3 * i + 2] * v[2];
    }
    memcpy(out, r, sizeof r);
}

static inline float mat3_det(const float a[9])
{
    return a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) + a[2] * (a[3] * a[7] - a[4] * a[6]);
}

static inline float dot3(const float a[3], const float b[3]) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

static inline void to_ray(const mvo_camera_t *c, const float p[2], float r[2])
{
    r[0] = (p[0] - c->cx) / c->fx;
    r[1] = (p[1] - c->cy) / c->fy;
}

/* mono_vo.c */
/* Carries the depth chain through a LOST step with the essential pose (r, t), or returns 0 changing no point. */
int mvo_carry_depths(mvo_t *s, int n, const float r[9], const float t[3]);

/* mono_vo_numerics.c */
void mvo_smallest_vector(float *a, int n, float *out);
void mvo_svd3(const float m[9], float u[9], float s[3], float v[9]);
int mvo_solve(float *m, int n, float *x);
int mvo_null_vector8(float *m, float out[9]);
void mvo_householder_r(float (*rows)[9], int m, float r[81]);
void mvo_smallest_right_vector(float *a, float out[9]);
void mvo_cayley(const float c[3], float out[9]);
float mvo_median(float *v, int n);
int mvo_ransac_needed(int inliers, int n, int sample, float confidence, int cap);
void mvo_pick(uint32_t *rng, int n, int count, int *out);

/* mono_vo_tracking.c */
void mvo_build_pyramid(const mvo_t *s, uint8_t *pyramid);
void mvo_detect(mvo_t *s);
int mvo_track_features(mvo_t *s, const uint8_t *previous, const uint8_t *current);

/* mono_vo_homography.c */
int mvo_pure_rotation(mvo_t *s, int n, float rotation_ab[9]);
float mvo_parallax(const mvo_t *s, int n, const float rotation_ab[9]);
int mvo_rotation_model(const mvo_t *s, int n, float rotation_ab[9]); /* -1 without a model, otherwise support */
int mvo_rotation_inlier(const mvo_t *s, const float rotation[9], int k);

/* mono_vo_essential.c */
int mvo_essential_pose(mvo_t *s, int n, float r[9], float t[3], const float turn[9]);
int mvo_translation_evidence(const mvo_t *s, int n, const float rotation_ab[9], const float r[9], const float t[3]);
int mvo_triangulate(const float r[9], const float t[3], const float ra[2], const float rb[2], float out[3]);
float mvo_ray_spread(const float r[9], int k);

#endif
