/* The motion of a step with parallax, from the essential matrix: eight-point MSAC with least-squares refits on
 * Sampson distances, the decomposition and triangulation, robust Levenberg-Marquardt, the search over travel
 * directions, the cheirality cost and a fast turn's pull (see mono_vo_internal.h). */
#include "mono_vo_internal.h"

#include <string.h>

#define MVO_RANSAC_MIN 50 /* essential MSAC samples drawn even when the adaptive count is reached sooner */
#define MVO_RANSAC_BATCH 8 /* essential samples drawn, then scored together: half of them on each core */
#define MVO_REFINE_ITERATIONS 10
#define MVO_DIRECTIONS 81 /* travel directions tried by the direction search */
#define MVO_SEARCH_POINTS 48 /* pairs the direction search fits */
#define MVO_SEARCH_STEPS 3 /* Gauss-Newton steps on the rotation per searched direction */

#define MVO_REFINE_TOLERANCE 1e-4f /* an iteration gaining less than this share of the cost ends the refinement */

/* Scratch, not thread-safe (see mono_vo.h). */
typedef struct { /* a pose refinement's: two may run at the same time */
    float residual[MVO_MAX_FEATURES], moved[MVO_MAX_FEATURES], jacobian[MVO_MAX_FEATURES][5];
} mvo_refiner_t;
typedef struct { /* a direction fit's: the two halves of the search may run at the same time */
    float g[MVO_SEARCH_POINTS][3], c[MVO_SEARCH_POINTS], weight[MVO_SEARCH_POINTS], d2[MVO_SEARCH_POINTS];
} mvo_searcher_t;
static mvo_refiner_t s_refiner[2];
static mvo_searcher_t s_searcher[2];
static uint8_t s_marks[2][MVO_MAX_FEATURES]; /* inlier marks of the samples a batch half scores */
static float s_lsq[MVO_MAX_FEATURES][9]; /* constraint rows of a least-squares essential fit */
/* A fast turn's pull (see mono_vo.h), set for the step by mvo_essential_pose: the pure turn's rotation (x_B = R x_A),
 * its unit axis and the pull's weight, 0 for none; only read while the two refinements run at once. */
static float s_turn[9], s_turn_axis[3], s_turn_weight;

/* Small-angle vector of r turn^T, half its skew part: how far the rotation r is from the pure turn's. */
static void turn_offset(const float r[9], float d[3])
{
    float tt[9], m[9];
    mat3_transpose(s_turn, tt);
    mat3_mul(r, tt, m);
    d[0] = 0.5f * (m[7] - m[5]);
    d[1] = 0.5f * (m[2] - m[6]);
    d[2] = 0.5f * (m[3] - m[1]);
}

/* The pull's cost at r, on the offset along the turn's axis only, a = axis . d: weight MVO_TURN_SOFT u / (1 + u)
 * with u = a^2 / MVO_TURN_SOFT (Geman-McClure), about weight a^2 near the pure turn and at most weight MVO_TURN_SOFT. */
static float pull_cost(const float r[9])
{
    float d[3], along, u;
    if (s_turn_weight <= 0.0f) {
        return 0.0f;
    }
    turn_offset(r, d);
    along = dot3(d, s_turn_axis);
    u = along * along / MVO_TURN_SOFT;
    return s_turn_weight * MVO_TURN_SOFT * u / (1.0f + u);
}

/* Hartley normalisation of the marked (or indexed) ray pairs: centroids, and scales giving a mean distance of sqrt 2;
 * the number of pairs, or 0 below eight. */
static int normalisation(int n, const uint8_t *mark, const int *idx, int count, float ca[2], float cb[2], float *sa,
                         float *sb)
{
    int k, used = 0, total = idx ? count : n;
    ca[0] = ca[1] = cb[0] = cb[1] = 0.0f;
    *sa = *sb = 0.0f;
    for (k = 0; k < total; ++k) {
        const int at = idx ? idx[k] : k;
        if (idx || mark[at]) {
            ca[0] += mvo_ra[at][0]; ca[1] += mvo_ra[at][1]; cb[0] += mvo_rb[at][0]; cb[1] += mvo_rb[at][1];
            ++used;
        }
    }
    if (used < 8) {
        return 0;
    }
    ca[0] /= (float)used; ca[1] /= (float)used; cb[0] /= (float)used; cb[1] /= (float)used;
    for (k = 0; k < total; ++k) {
        const int at = idx ? idx[k] : k;
        if (idx || mark[at]) {
            const float ax = mvo_ra[at][0] - ca[0], ay = mvo_ra[at][1] - ca[1], bx = mvo_rb[at][0] - cb[0], by = mvo_rb[at][1] - cb[1];
            *sa += mvo_root(ax * ax + ay * ay);
            *sb += mvo_root(bx * bx + by * by);
        }
    }
    *sa = *sa > 0.0f ? 1.41421356f * (float)used / *sa : 1.0f;
    *sb = *sb > 0.0f ? 1.41421356f * (float)used / *sb : 1.0f;
    return used;
}

/* The epipolar constraint b^T En a = 0 of normalised pair `at` as a row over En (row-major). */
static void epipolar_row(int at, const float ca[2], const float cb[2], float sa, float sb, float row[9])
{
    const float ax = (mvo_ra[at][0] - ca[0]) * sa, ay = (mvo_ra[at][1] - ca[1]) * sa;
    const float bx = (mvo_rb[at][0] - cb[0]) * sb, by = (mvo_rb[at][1] - cb[1]) * sb;
    row[0] = bx * ax; row[1] = bx * ay; row[2] = bx; row[3] = by * ax; row[4] = by * ay; row[5] = by;
    row[6] = ax; row[7] = ay; row[8] = 1.0f;
}

/* E = Tb^T En Ta, projected to singular values (1, 1, 0). */
static void denormalise_essential(const float en[9], const float ca[2], const float cb[2], float sa, float sb, float e[9])
{
    float ta[9], tbt[9], tmp[9], u[9], sv[3], v[9], vt[9], d[9];
    ta[0] = sa; ta[1] = 0.0f; ta[2] = -sa * ca[0]; ta[3] = 0.0f; ta[4] = sa; ta[5] = -sa * ca[1]; ta[6] = 0.0f; ta[7] = 0.0f; ta[8] = 1.0f;
    tbt[0] = sb; tbt[1] = 0.0f; tbt[2] = 0.0f; tbt[3] = 0.0f; tbt[4] = sb; tbt[5] = 0.0f; tbt[6] = -sb * cb[0]; tbt[7] = -sb * cb[1]; tbt[8] = 1.0f;
    mat3_mul(en, ta, tmp);
    mat3_mul(tbt, tmp, e);
    mvo_svd3(e, u, sv, v);
    memset(d, 0, sizeof d);
    d[0] = 1.0f;
    d[4] = 1.0f;
    mat3_transpose(v, vt);
    mat3_mul(d, vt, tmp);
    mat3_mul(u, tmp, e);
}

/* Least-squares essential matrix of the marked ray pairs: QR of the normalised constraint rows, then the smallest
 * right singular vector of R (float keeps the precision that the normal equations A^T A would square away). */
static int eight_point_fit(int n, const uint8_t *mark, float e[9])
{
    float ca[2], cb[2], sa, sb, r[81], en[9];
    int k, m = 0;
    if (!normalisation(n, mark, 0, 0, ca, cb, &sa, &sb)) {
        return 0;
    }
    for (k = 0; k < n; ++k) {
        if (mark[k]) {
            epipolar_row(k, ca, cb, sa, sb, s_lsq[m++]);
        }
    }
    mvo_householder_r(s_lsq, m, r);
    mvo_smallest_right_vector(r, en);
    denormalise_essential(en, ca, cb, sa, sb, e);
    return 1;
}

/* Essential matrix of eight indexed ray pairs (a RANSAC sample). */
static int eight_point_sample(const int *idx, float e[9])
{
    float ca[2], cb[2], sa, sb, m[8 * 9], en[9];
    int k;
    if (!normalisation(0, 0, idx, 8, ca, cb, &sa, &sb)) {
        return 0;
    }
    for (k = 0; k < 8; ++k) {
        epipolar_row(idx[k], ca, cb, sa, sb, m + 9 * k);
    }
    if (!mvo_null_vector8(m, en)) {
        return 0;
    }
    denormalise_essential(en, ca, cb, sa, sb, e);
    return 1;
}

/* F = K^-T E K^-1: the essential matrix between pixels. */
static void fundamental(const mvo_camera_t *c, const float e[9], float f[9])
{
    float k_inv[9], k_inv_t[9], tmp[9];
    k_inv[0] = 1.0f / c->fx; k_inv[1] = 0.0f; k_inv[2] = -c->cx / c->fx;
    k_inv[3] = 0.0f; k_inv[4] = 1.0f / c->fy; k_inv[5] = -c->cy / c->fy;
    k_inv[6] = 0.0f; k_inv[7] = 0.0f; k_inv[8] = 1.0f;
    mat3_transpose(k_inv, k_inv_t);
    mat3_mul(e, k_inv, tmp);
    mat3_mul(k_inv_t, tmp, f);
}

/* The Sampson distance (pixels) of a pixel pair to F as num / sqrt(den): its signed numerator b^T F a and its
 * denominator (den 0 where it is undefined). The callers square it or shrink it without a square root of their own
 * where they can: a root costs several divisions on the S3. */
static void sampson_parts(const float f[9], const float pa[2], const float pb[2], float *num, float *den)
{
    const float a[3] = {pa[0], pa[1], 1.0f}, b[3] = {pb[0], pb[1], 1.0f};
    float fa[3], ftb0, ftb1;
    mat3_vec(f, a, fa);
    ftb0 = f[0] * b[0] + f[3] * b[1] + f[6];
    ftb1 = f[1] * b[0] + f[4] * b[1] + f[7];
    *num = dot3(b, fa);
    *den = fa[0] * fa[0] + fa[1] * fa[1] + ftb0 * ftb0 + ftb1 * ftb1;
}

/* Squared Sampson distance (pixels squared) of a pixel pair to F; 1e18 where it is undefined. */
static float sampson2(const float f[9], const float pa[2], const float pb[2])
{
    float num, den;
    sampson_parts(f, pa, pb, &num, &den);
    return den > 1e-20f ? num * num / den : 1e18f;
}

/* Marks the pairs within `threshold` pixels (Sampson) of E and returns their count; with `score`, also the MSAC score
 * (squared distances, each capped at the threshold's square: lower is better). */
static int essential_inliers(const mvo_t *s, const float e[9], int n, float threshold, uint8_t *mark, float *score)
{
    float f[9], sum = 0.0f;
    int k, count = 0;
    fundamental(&s->camera, e, f);
    for (k = 0; k < n; ++k) {
        const float d2 = sampson2(f, mvo_pa[k], mvo_pb[k]);
        const int ok = d2 < threshold * threshold;
        mark[k] = (uint8_t)ok;
        count += ok;
        sum += ok ? d2 : threshold * threshold;
    }
    if (score) {
        *score = sum;
    }
    return count;
}

/* [v]x: the matrix of the cross product v x . */
static void cross_matrix(const float v[3], float m[9])
{
    m[0] = 0.0f; m[1] = -v[2]; m[2] = v[1];
    m[3] = v[2]; m[4] = 0.0f; m[5] = -v[0];
    m[6] = -v[1]; m[7] = v[0]; m[8] = 0.0f;
}

/* E = [t]x R of the pose x_B = R x_A + t. */
static void essential_of(const float r[9], const float t[3], float e[9])
{
    float tx[9];
    cross_matrix(t, tx);
    mat3_mul(tx, r, e);
}

/* Evidence for translation over an independently fitted rotation, on the same epipolar inliers.
 * Five times the median Sampson distance of all pairs supplies a conservative noise scale, floored at 0.5 pixels;
 * above 4 pixels it rejects translation rather than understating noise. Fitting E can suppress its own residual.
 * shortcut: the noise range is heuristic; validate or calibrate it for the actual camera and tracker. */
static int translation_evidence_baseline(const mvo_t *s, int n, const float rotation_ab[9], const float r[9], const float t[3])
{
    float e[9], f[9], rotation[9], difference[4] = {0}, inv_variance[4], variance, penalty;
    int k, j, used = 0;
    essential_of(r, t, e);
    fundamental(&s->camera, e, f);
    mat3_transpose(rotation_ab, rotation);
    for (k = 0; k < n; ++k) mvo_values[k] = sampson2(f, mvo_pa[k], mvo_pb[k]);
    variance = 25.0f * mvo_median(mvo_values, n);
    if (variance > 16.0f) return 0;
    if (variance < 0.25f) variance = 0.25f;
    inv_variance[0] = 4.0f;
    inv_variance[3] = 1.0f / variance;
    inv_variance[1] = (2.0f * inv_variance[0] + inv_variance[3]) / 3.0f;
    inv_variance[2] = (inv_variance[0] + 2.0f * inv_variance[3]) / 3.0f;
    for (k = 0; k < n; ++k) {
        const float a[3] = {mvo_ra[k][0], mvo_ra[k][1], 1.0f};
        float b[3], dx, dy, rotation_d2, essential_d2;
        if (!mvo_epipolar[k]) continue;
        ++used;
        mat3_vec(rotation, a, b);
        if (b[2] > 1e-6f) {
            dx = s->camera.fx * b[0] / b[2] + s->camera.cx - mvo_pb[k][0];
            dy = s->camera.fy * b[1] / b[2] + s->camera.cy - mvo_pb[k][1];
            rotation_d2 = dx * dx + dy * dy;
        } else {
            rotation_d2 = 1e18f;
        }
        essential_d2 = sampson2(f, mvo_pa[k], mvo_pb[k]);
        for (j = 0; j < 4; ++j) {
            float rotation_cost = 0.5f * rotation_d2 * inv_variance[j];
            float essential_cost = essential_d2 * inv_variance[j];
            difference[j] += (rotation_cost < 4.0f ? rotation_cost : 4.0f)
                             - (essential_cost < 2.0f ? essential_cost : 2.0f);
        }
    }
    if (used < MVO_MIN_INLIERS) return 0;
    /* 13.816 >= 2 ln(4*n) for n <= MVO_MAX_FEATURES, avoiding libm on the board. */
    penalty = 1.38629436f * (float)used + 13.816f + 9.21f;
    for (j = 0; j < 4; ++j) if (difference[j] <= penalty) return 0;
    return 1;
}

/* Score the same epipolar inliers at several noise levels; a negative value rejects the model. */
static float small_match_score(int n, const float *rotation_errors, const float *essential_errors, float variance, int used)
{
    float minimum = 1e30f;
    int j, k;
    if (variance > 16.0f) return -1e6f;
    if (variance < 0.25f) variance = 0.25f;
    for (j = 0; j < 4; ++j) {
        float inv_variance = 4.0f + (1.0f / variance - 4.0f) * (float)j / 3.0f;
        float score = -(1.38629436f * (float)used + 13.816f + 9.21f);
        for (k = 0; k < n; ++k) if (mvo_epipolar[k]) {
            float a = 0.5f * rotation_errors[k] * inv_variance, b = essential_errors[k] * inv_variance;
            score += (a < 4.0f ? a : 4.0f) - (b < 2.0f ? b : 2.0f);
        }
        if (score < minimum) minimum = score;
    }
    return minimum;
}

int mvo_translation_evidence(const mvo_t *s, int n, const float rotation_ab[9], const float r[9], const float t[3])
{
    float e[9], f[9], rotation[9], essential_errors[50], rotation_errors[50];
    float median_e, upper_e, median_r, score4, score5, score_upper;
    int k, used = 0, front = 0;
    if (n >= 50) return translation_evidence_baseline(s, n, rotation_ab, r, t);
    essential_of(r, t, e);
    fundamental(&s->camera, e, f);
    mat3_transpose(rotation_ab, rotation);
    for (k = 0; k < n; ++k) {
        const float a[3] = {mvo_ra[k][0], mvo_ra[k][1], 1.0f};
        float b[3], dx, dy;
        essential_errors[k] = sampson2(f, mvo_pa[k], mvo_pb[k]);
        mvo_values[k] = essential_errors[k];
        mat3_vec(rotation, a, b);
        if (b[2] > 1e-6f) {
            dx = s->camera.fx * b[0] / b[2] + s->camera.cx - mvo_pb[k][0];
            dy = s->camera.fy * b[1] / b[2] + s->camera.cy - mvo_pb[k][1];
            rotation_errors[k] = dx * dx + dy * dy;
        } else {
            rotation_errors[k] = 1e18f;
        }
        used += mvo_epipolar[k];
        front += mvo_best[k];
    }
    if (used < MVO_MIN_INLIERS) return 0;
    median_e = mvo_median(mvo_values, n); /* mvo_median sorts, so the upper quartile is also available */
    upper_e = mvo_values[(3 * n) / 4];
    for (k = 0; k < n; ++k) mvo_values[k] = rotation_errors[k];
    median_r = mvo_median(mvo_values, n);
    score4 = small_match_score(n, rotation_errors, essential_errors, 16.0f * median_e, used);
    score5 = small_match_score(n, rotation_errors, essential_errors, 25.0f * median_e, used);
    score_upper = small_match_score(n, rotation_errors, essential_errors, 4.0f * upper_e, used);
    /* A sharp residual tail signals an overfitted E; very large tails may instead be outliers.
     * Compare squared errors: a 4x residual ratio matches the conservative score4 noise scale. */
    if (upper_e < 16.0f && upper_e > 16.0f * median_e && upper_e > 0.0625f * median_r) return 0;
    /* shortcut: these empirical evidence margins need calibration on actual tracked camera data. */
    return score5 > 1.0f || (score5 > 0.0f && score_upper > 0.0f)
           || (score4 > 0.0f && score_upper > 0.0f && median_r > 9.0f && 3 * front >= 2 * used);
}

/* Midpoint of the closest points of the two rays (P_A = [I|0], P_B = [R|t]); 0 for parallel rays. */
int mvo_triangulate(const float r[9], const float t[3], const float ra[2], const float rb[2], float out[3])
{
    float rt[9], da[3] = {ra[0], ra[1], 1.0f}, bray[3] = {rb[0], rb[1], 1.0f}, db[3], c[3], a, b, cc, d, e, den, lambda, mu;
    int i;
    mat3_transpose(r, rt);
    mat3_vec(rt, bray, db);
    mat3_vec(rt, t, c);
    c[0] = -c[0];
    c[1] = -c[1];
    c[2] = -c[2];
    a = dot3(da, da);
    b = dot3(da, db);
    cc = dot3(db, db);
    d = dot3(da, c);
    e = dot3(db, c);
    den = a * cc - b * b;
    if (mvo_abs(den) < 1e-12f) {
        return 0;
    }
    lambda = (d * cc - b * e) / den;
    mu = (b * d - a * e) / den;
    for (i = 0; i < 3; ++i) {
        out[i] = 0.5f * (lambda * da[i] + c[i] + mu * db[i]);
    }
    return 1;
}

/* Squared sine of the angle between the rays of pair k once rotated into one frame (R maps A to B). */
float mvo_ray_spread(const float r[9], int k)
{
    const float a[3] = {mvo_ra[k][0], mvo_ra[k][1], 1.0f}, b[3] = {mvo_rb[k][0], mvo_rb[k][1], 1.0f};
    float rt[9], db[3], c[3];
    mat3_transpose(r, rt);
    mat3_vec(rt, b, db);
    c[0] = a[1] * db[2] - a[2] * db[1];
    c[1] = a[2] * db[0] - a[0] * db[2];
    c[2] = a[0] * db[1] - a[1] * db[0];
    return dot3(c, c) / (dot3(a, a) * dot3(db, db));
}

/* Whether pair k triangulates in front of both cameras and nearer than MVO_MAX_DEPTH baselines. */
static int in_front(const float r[9], const float t[3], int k)
{
    float p[3], q[3];
    if (!mvo_triangulate(r, t, mvo_ra[k], mvo_rb[k], p)) {
        return 0;
    }
    mat3_vec(r, p, q);
    q[2] += t[2];
    return p[2] > 0.0f && q[2] > 0.0f && p[2] < MVO_MAX_DEPTH && q[2] < MVO_MAX_DEPTH;
}

/* Rotation and unit translation (x_B = R x_A + t) from E: of the four decompositions, the one with the most marked
 * pairs in front of both cameras. */
static void recover_pose(int n, const uint8_t *mark, const float e[9], float r_out[9], float t_out[3])
{
    static const float w[9] = {0, -1, 0, 1, 0, 0, 0, 0, 1};
    float u[9], sv[3], v[9], vt[9], wt[9], r1[9], r2[9], tmp[9];
    float candidates_r[4][9], candidates_t[4][3];
    int best = 0, best_count = -1, c, k, i;
    mvo_svd3(e, u, sv, v);
    if (mat3_det(u) < 0.0f) {
        for (i = 0; i < 9; ++i) {
            u[i] = -u[i];
        }
    }
    if (mat3_det(v) < 0.0f) {
        for (i = 0; i < 9; ++i) {
            v[i] = -v[i];
        }
    }
    mat3_transpose(v, vt);
    mat3_transpose(w, wt);
    mat3_mul(w, vt, tmp);
    mat3_mul(u, tmp, r1);
    mat3_mul(wt, vt, tmp);
    mat3_mul(u, tmp, r2);
    for (c = 0; c < 4; ++c) {
        memcpy(candidates_r[c], c < 2 ? r1 : r2, sizeof r1);
        for (i = 0; i < 3; ++i) {
            candidates_t[c][i] = (c % 2 ? -1.0f : 1.0f) * u[3 * i + 2];
        }
    }
    for (c = 0; c < 4; ++c) {
        int count = 0;
        for (k = 0; k < n; ++k) {
            count += mark[k] && in_front(candidates_r[c], candidates_t[c], k);
        }
        if (count > best_count) {
            best_count = count;
            best = c;
        }
    }
    memcpy(r_out, candidates_r[best], sizeof r1);
    memcpy(t_out, candidates_t[best], sizeof candidates_t[0]);
}

/* Two unit vectors spanning the plane perpendicular to the unit vector t. */
static void tangent_basis(const float t[3], float basis[6])
{
    float axis[3] = {0.0f, 0.0f, 0.0f}, norm;
    int i, smallest = 0;
    for (i = 1; i < 3; ++i) {
        if (mvo_abs(t[i]) < mvo_abs(t[smallest])) {
            smallest = i;
        }
    }
    axis[smallest] = 1.0f;
    basis[0] = t[1] * axis[2] - t[2] * axis[1];
    basis[1] = t[2] * axis[0] - t[0] * axis[2];
    basis[2] = t[0] * axis[1] - t[1] * axis[0];
    norm = mvo_root(dot3(basis, basis));
    for (i = 0; i < 3; ++i) {
        basis[i] /= norm;
    }
    basis[3] = t[1] * basis[2] - t[2] * basis[1];
    basis[4] = t[2] * basis[0] - t[0] * basis[2];
    basis[5] = t[0] * basis[1] - t[1] * basis[0];
}

/* The pose moved by p: R' = mvo_cayley(p0..p2) R, t' = t + p3 b1 + p4 b2 back on the unit sphere. */
static void perturb(const float r[9], const float t[3], const float basis[6], const float p[5], float r_out[9],
                    float t_out[3])
{
    float turn[9], norm;
    int i;
    mvo_cayley(p, turn);
    mat3_mul(turn, r, r_out);
    for (i = 0; i < 3; ++i) {
        t_out[i] = t[i] + p[3] * basis[i] + p[4] * basis[3 + i];
    }
    norm = mvo_root(dot3(t_out, t_out));
    for (i = 0; i < 3; ++i) {
        t_out[i] /= norm;
    }
}

/* Robust residuals of the marked pairs to the pose, in order, into out: the signed Sampson distance d (pixels) shrunk
 * to d / sqrt(1 + d^2), so that their squares follow the Geman-McClure loss d^2 / (1 + d^2) (scale MVO_RANSAC_PIXELS =
 * 1 pixel) and a pair far off costs at most 1; returns the sum of squares. With d = num / sqrt(den) that is
 * num / sqrt(den + num^2): one reciprocal root per pair and no division. */
static float pose_residuals(const mvo_t *s, const float r[9], const float t[3], int n, const uint8_t *mark, float *out)
{
    float e[9], f[9], sum = 0.0f;
    int k, used = 0;
    essential_of(r, t, e);
    fundamental(&s->camera, e, f);
    for (k = 0; k < n; ++k) {
        if (mark[k]) {
            float num, den, shrunk;
            sampson_parts(f, mvo_pa[k], mvo_pb[k], &num, &den);
            shrunk = den > 1e-20f ? num * mvo_rsqrt(den + num * num) : 1.0f;
            out[used++] = shrunk;
            sum += shrunk * shrunk;
        }
    }
    return sum;
}

/* The derivatives of F (see fundamental) in the pose parameters of perturb at 0: mvo_cayley's derivative there is
 * twice the cross matrix, so dE = [t]x 2 [e_j]x R for the rotation, and dE = [b_j]x R for the two steps of travel. */
static void pose_derivatives(const mvo_camera_t *c, const float r[9], const float t[3], const float basis[6],
                             float df[5][9])
{
    float tx[9], ex[9], m[9], de[9];
    int j;
    cross_matrix(t, tx);
    for (j = 0; j < 3; ++j) {
        float axis[3] = {0.0f, 0.0f, 0.0f};
        axis[j] = 2.0f;
        cross_matrix(axis, ex);
        mat3_mul(ex, r, m);
        mat3_mul(tx, m, de);
        fundamental(c, de, df[j]);
    }
    for (j = 0; j < 2; ++j) {
        cross_matrix(basis + 3 * j, ex);
        mat3_mul(ex, r, de);
        fundamental(c, de, df[3 + j]);
    }
}

typedef struct {
    int n, used;
    const uint8_t *mark;
    float f[9], df[5][9];   /* F at the pose and its derivatives */
    mvo_refiner_t *scratch; /* the Jacobian goes here */
} mvo_jacobian_job_t;

/* Half `part` of the Jacobian's rows (the marked pairs in order): the derivatives of pose_residuals' residuals. With
 * D = den + num^2 and q = 1 / sqrt(D), the residual num q moves by q (dnum - num q^2 (dden / 2 + num dnum)). */
static void jacobian_rows(void *context, int part)
{
    mvo_jacobian_job_t *job = context;
    const float *f = job->f;
    const int begin = part * job->used / 2, end = (part + 1) * job->used / 2;
    int k, j, row = 0;
    for (k = 0; k < job->n && row < end; ++k) {
        const float *pa = mvo_pa[k], *pb = mvo_pb[k];
        float fa[3], ftb0, ftb1, num, den, q, *out;
        if (!job->mark[k] || row++ < begin) {
            continue;
        }
        out = job->scratch->jacobian[row - 1];
        fa[0] = f[0] * pa[0] + f[1] * pa[1] + f[2];
        fa[1] = f[3] * pa[0] + f[4] * pa[1] + f[5];
        fa[2] = f[6] * pa[0] + f[7] * pa[1] + f[8];
        ftb0 = f[0] * pb[0] + f[3] * pb[1] + f[6];
        ftb1 = f[1] * pb[0] + f[4] * pb[1] + f[7];
        num = pb[0] * fa[0] + pb[1] * fa[1] + fa[2];
        den = fa[0] * fa[0] + fa[1] * fa[1] + ftb0 * ftb0 + ftb1 * ftb1;
        if (!(den > 1e-20f)) { /* the residual is the constant 1 there */
            for (j = 0; j < 5; ++j) {
                out[j] = 0.0f;
            }
            continue;
        }
        q = mvo_rsqrt(den + num * num);
        for (j = 0; j < 5; ++j) {
            const float *d = job->df[j];
            const float da0 = d[0] * pa[0] + d[1] * pa[1] + d[2], da1 = d[3] * pa[0] + d[4] * pa[1] + d[5];
            const float da2 = d[6] * pa[0] + d[7] * pa[1] + d[8];
            const float db0 = d[0] * pb[0] + d[3] * pb[1] + d[6], db1 = d[1] * pb[0] + d[4] * pb[1] + d[7];
            const float dnum = pb[0] * da0 + pb[1] * da1 + da2;
            const float half_dden = fa[0] * da0 + fa[1] * da1 + ftb0 * db0 + ftb1 * db1;
            out[j] = q * (dnum - num * q * q * (half_dden + num * dnum));
        }
    }
}

/* Levenberg-Marquardt on the Sampson distances of the marked pairs over the rotation (a Cayley vector) and the
 * direction of travel (two steps on the tangent plane of the unit sphere), with analytic derivatives, plus a fast
 * turn's pull (pull_cost, its Geman-McClure weight held for each linearisation); returns the final sum of squared
 * distances and the pull (1e30 with fewer than eight pairs, left as they are). With split the Jacobian's rows go in two
 * halves, at the same time when the caller gave s->parallel. */
static float refine_pose(const mvo_t *s, int n, const uint8_t *mark, float r[9], float t[3], mvo_refiner_t *scratch,
                         int split)
{
    float lambda = 1e-3f, cost, basis[6], e[9];
    int iteration, used = 0, k;
    mvo_jacobian_job_t job;
    for (k = 0; k < n; ++k) {
        used += mark[k];
    }
    if (used < 8) {
        return 1e30f;
    }
    cost = pose_residuals(s, r, t, n, mark, scratch->residual) + pull_cost(r);
    for (iteration = 0; iteration < MVO_REFINE_ITERATIONS; ++iteration) {
        float jtj[25], jtr[5], m[5 * 6], delta[5], trial_r[9], trial_t[3], previous = cost;
        int i, j, accepted = 0;
        tangent_basis(t, basis);
        job.n = n;
        job.used = used;
        job.mark = mark;
        job.scratch = scratch;
        essential_of(r, t, e);
        fundamental(&s->camera, e, job.f);
        pose_derivatives(&s->camera, r, t, basis, job.df);
        if (split && s->parallel) {
            s->parallel(jacobian_rows, &job);
        } else {
            jacobian_rows(&job, 0);
            jacobian_rows(&job, 1);
        }
        memset(jtj, 0, sizeof jtj);
        memset(jtr, 0, sizeof jtr);
        for (k = 0; k < used; ++k) {
            for (i = 0; i < 5; ++i) {
                jtr[i] += scratch->jacobian[k][i] * scratch->residual[k];
                for (j = 0; j < 5; ++j) {
                    jtj[i * 5 + j] += scratch->jacobian[k][i] * scratch->jacobian[k][j];
                }
            }
        }
        if (s_turn_weight > 0.0f) { /* the pull's residual sqrt(w) axis . (d + 2 p), p the rotation steps (mvo_cayley) */
            float d[3], along, u, w;
            turn_offset(r, d);
            along = dot3(d, s_turn_axis);
            u = along * along / MVO_TURN_SOFT;
            w = s_turn_weight / ((1.0f + u) * (1.0f + u));
            for (i = 0; i < 3; ++i) {
                for (j = 0; j < 3; ++j) {
                    jtj[i * 5 + j] += 4.0f * w * s_turn_axis[i] * s_turn_axis[j];
                }
                jtr[i] += 2.0f * w * along * s_turn_axis[i];
            }
        }
        while (!accepted && lambda < 1e8f) {
            for (i = 0; i < 5; ++i) {
                for (j = 0; j < 5; ++j) {
                    m[i * 6 + j] = jtj[i * 5 + j] + (i == j ? lambda * jtj[i * 5 + i] : 0.0f);
                }
                m[i * 6 + 5] = -jtr[i];
            }
            if (mvo_solve(m, 5, delta)) {
                float trial;
                perturb(r, t, basis, delta, trial_r, trial_t);
                trial = pose_residuals(s, trial_r, trial_t, n, mark, scratch->moved) + pull_cost(trial_r);
                if (trial < cost) {
                    memcpy(r, trial_r, sizeof trial_r);
                    memcpy(t, trial_t, sizeof trial_t);
                    memcpy(scratch->residual, scratch->moved, (size_t)used * sizeof scratch->moved[0]);
                    cost = trial;
                    accepted = 1;
                    lambda = lambda * 0.1f > 1e-7f ? lambda * 0.1f : 1e-7f;
                }
            }
            if (!accepted) {
                lambda *= 10.0f;
            }
        }
        if (!accepted || previous - cost < MVO_REFINE_TOLERANCE * previous) {
            break;
        }
    }
    return cost;
}

/* Travel directions tried by direction_search: the upper half (each direction up to sign) of a twice-subdivided
 * icosahedron, neighbours at most 16.4 degrees apart. */
static const float kDirections[MVO_DIRECTIONS][3] = {
    {-0.5257311f, 0.8506508f, 0.0000000f},
    {0.5257311f, 0.8506508f, 0.0000000f},
    {0.0000000f, -0.5257311f, 0.8506508f},
    {0.0000000f, 0.5257311f, 0.8506508f},
    {0.8506508f, 0.0000000f, 0.5257311f},
    {-0.8506508f, 0.0000000f, 0.5257311f},
    {-0.8090170f, 0.5000000f, 0.3090170f},
    {-0.5000000f, 0.3090170f, 0.8090170f},
    {-0.3090170f, 0.8090170f, 0.5000000f},
    {0.3090170f, 0.8090170f, 0.5000000f},
    {0.0000000f, 1.0000000f, 0.0000000f},
    {0.5000000f, 0.3090170f, 0.8090170f},
    {0.8090170f, 0.5000000f, 0.3090170f},
    {-0.5000000f, -0.3090170f, 0.8090170f},
    {0.0000000f, 0.0000000f, 1.0000000f},
    {-0.8090170f, -0.5000000f, 0.3090170f},
    {0.8090170f, -0.5000000f, 0.3090170f},
    {0.5000000f, -0.3090170f, 0.8090170f},
    {0.3090170f, -0.8090170f, 0.5000000f},
    {-0.3090170f, -0.8090170f, 0.5000000f},
    {1.0000000f, 0.0000000f, 0.0000000f},
    {-0.6937805f, 0.7020464f, 0.1606220f},
    {-0.5877853f, 0.6881910f, 0.4253254f},
    {-0.4338886f, 0.8626685f, 0.2598919f},
    {-0.7020464f, 0.1606220f, 0.6937805f},
    {-0.6881910f, 0.4253254f, 0.5877853f},
    {-0.8626685f, 0.2598919f, 0.4338886f},
    {-0.1606220f, 0.6937805f, 0.7020464f},
    {-0.4253254f, 0.5877853f, 0.6881910f},
    {-0.2598919f, 0.4338886f, 0.8626685f},
    {-0.1624598f, 0.9510565f, 0.2628656f},
    {-0.2732665f, 0.9619384f, 0.0000000f},
    {0.1606220f, 0.6937805f, 0.7020464f},
    {0.0000000f, 0.8506508f, 0.5257311f},
    {0.2732665f, 0.9619384f, 0.0000000f},
    {0.1624598f, 0.9510565f, 0.2628656f},
    {0.4338886f, 0.8626685f, 0.2598919f},
    {-0.8506508f, 0.5257311f, 0.0000000f},
    {-0.9510565f, 0.2628656f, 0.1624598f},
    {-0.9619384f, 0.0000000f, 0.2732665f},
    {0.5877853f, 0.6881910f, 0.4253254f},
    {0.6937805f, 0.7020464f, 0.1606220f},
    {0.2598919f, 0.4338886f, 0.8626685f},
    {0.4253254f, 0.5877853f, 0.6881910f},
    {0.8626685f, 0.2598919f, 0.4338886f},
    {0.6881910f, 0.4253254f, 0.5877853f},
    {0.7020464f, 0.1606220f, 0.6937805f},
    {-0.2628656f, 0.1624598f, 0.9510565f},
    {0.0000000f, 0.2732665f, 0.9619384f},
    {-0.7020464f, -0.1606220f, 0.6937805f},
    {-0.5257311f, 0.0000000f, 0.8506508f},
    {0.0000000f, -0.2732665f, 0.9619384f},
    {-0.2628656f, -0.1624598f, 0.9510565f},
    {-0.2598919f, -0.4338886f, 0.8626685f},
    {-0.9510565f, -0.2628656f, 0.1624598f},
    {-0.8626685f, -0.2598919f, 0.4338886f},
    {-0.6937805f, -0.7020464f, 0.1606220f},
    {0.6937805f, -0.7020464f, 0.1606220f},
    {0.5877853f, -0.6881910f, 0.4253254f},
    {0.4338886f, -0.8626685f, 0.2598919f},
    {0.7020464f, -0.1606220f, 0.6937805f},
    {0.6881910f, -0.4253254f, 0.5877853f},
    {0.8626685f, -0.2598919f, 0.4338886f},
    {0.1606220f, -0.6937805f, 0.7020464f},
    {0.4253254f, -0.5877853f, 0.6881910f},
    {0.2598919f, -0.4338886f, 0.8626685f},
    {0.1624598f, -0.9510565f, 0.2628656f},
    {-0.1606220f, -0.6937805f, 0.7020464f},
    {0.0000000f, -0.8506508f, 0.5257311f},
    {-0.1624598f, -0.9510565f, 0.2628656f},
    {-0.4338886f, -0.8626685f, 0.2598919f},
    {0.9510565f, -0.2628656f, 0.1624598f},
    {0.9619384f, 0.0000000f, 0.2732665f},
    {0.2628656f, -0.1624598f, 0.9510565f},
    {0.5257311f, 0.0000000f, 0.8506508f},
    {0.2628656f, 0.1624598f, 0.9510565f},
    {-0.5877853f, -0.6881910f, 0.4253254f},
    {-0.4253254f, -0.5877853f, 0.6881910f},
    {-0.6881910f, -0.4253254f, 0.5877853f},
    {0.9510565f, 0.2628656f, 0.1624598f},
    {0.8506508f, 0.5257311f, 0.0000000f}
};

static float cross_dot(const float a[3], const float b[3], const float c[3]) /* a . (b x c) */
{
    return a[0] * (b[1] * c[2] - b[2] * c[1]) + a[1] * (b[2] * c[0] - b[0] * c[2]) + a[2] * (b[0] * c[1] - b[1] * c[0]);
}

static void cross3(const float a[3], const float b[3], float out[3])
{
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

/* Which sign of t puts pair k in front of both cameras for the pose (R, t): 1 for t, -1 for -t, 0 for neither. With
 * y = R a and u = b x y, the depths along the rays are z_A = -(b x t) . u / |u|^2 and z_B = (t x y) . u / |u|^2. */
static int front_sign(const float r[9], const float t[3], int k)
{
    const float a[3] = {mvo_ra[k][0], mvo_ra[k][1], 1.0f}, b[3] = {mvo_rb[k][0], mvo_rb[k][1], 1.0f};
    float y[3], u[3], bt[3], ty[3], za, zb;
    mat3_vec(r, a, y);
    cross3(b, y, u);
    cross3(b, t, bt);
    cross3(t, y, ty);
    za = -dot3(bt, u);
    zb = dot3(ty, u);
    return za > 0.0f && zb > 0.0f ? 1 : (za < 0.0f && zb < 0.0f ? -1 : 0);
}

/* Geman-McClure cost of the marked pairs (see pose_residuals), a pair behind either camera costing 1 as an outlier
 * does: on fast turns several poses fit the epipolar constraint about equally well, and cheirality tells them apart.
 * Plus a fast turn's pull (pull_cost). */
static float pose_cost(const mvo_t *s, const float r[9], const float t[3], int n, const uint8_t *mark)
{
    float e[9], f[9], cost = 0.0f;
    int k;
    essential_of(r, t, e);
    fundamental(&s->camera, e, f);
    for (k = 0; k < n; ++k) {
        if (mark[k]) {
            const float d2 = sampson2(f, mvo_pa[k], mvo_pb[k]);
            cost += front_sign(r, t, k) > 0 ? d2 / (1.0f + d2) : 1.0f;
        }
    }
    return cost + pull_cost(r);
}

/* For travel direction +-t: the rotation that fits the listed pairs best, by MVO_SEARCH_STEPS reweighted Gauss-Newton
 * steps from r0 (into r_out), and the sign of t that puts more of them in front of both cameras (into sign); returns
 * the Geman-McClure cost of the pairs at the last step, a pair within a pixel but behind a camera costing 1. Each
 * step linearises about the current R: with R' a = y + w x y for y = R a, the residual b . (t x R' a) is c + w . g where
 * c = b . (t x y) and g = y x (b x t). Weighted by the Sampson denominators at R and by Geman-McClure weights
 * 1 / (1 + d^2 / sigma^2)^2 whose sigma shrinks from step to step (kSearchInvSigma2 holds 1 / sigma^2), the
 * least-squares w moves R to mvo_cayley(w / 2) R: plain least squares first, so a start several degrees off is not cut
 * off, then robust, so a stray pair cannot pull the rotation. */
static float direction_fit(const mvo_t *s, const float r0[9], const float t[3], const int *subset,
                           const float (*turned)[3], int count, mvo_searcher_t *scratch, float r_out[9], int *sign)
{
    static const float kSearchInvSigma2[MVO_SEARCH_STEPS] = {0.0f, 0.25f, 1.0f}; /* per pixel squared */
    float ahead = 0.0f, behind = 0.0f;
    int k, i, j, step;
    memcpy(r_out, r0, 9 * sizeof r0[0]);
    for (step = 0; step < MVO_SEARCH_STEPS; ++step) {
        float e[9], f[9], m[3 * 4], w[3] = {0.0f, 0.0f, 0.0f}, half[3], turn[9];
        essential_of(r_out, t, e);
        fundamental(&s->camera, e, f);
        memset(m, 0, sizeof m);
        for (k = 0; k < count; ++k) {
            const int at = subset[k];
            const float a[3] = {mvo_ra[at][0], mvo_ra[at][1], 1.0f}, b[3] = {mvo_rb[at][0], mvo_rb[at][1], 1.0f};
            const float pa[3] = {mvo_pa[at][0], mvo_pa[at][1], 1.0f};
            float y[3], bt[3], fa[3], ftb0, ftb1, den, weight;
            if (step) {
                mat3_vec(r_out, a, y);
            } else { /* r_out is still r0 */
                memcpy(y, turned[k], sizeof y);
            }
            cross3(b, t, bt);
            cross3(y, bt, scratch->g[k]);
            scratch->c[k] = cross_dot(b, t, y);
            mat3_vec(f, pa, fa);
            ftb0 = f[0] * mvo_pb[at][0] + f[3] * mvo_pb[at][1] + f[6];
            ftb1 = f[1] * mvo_pb[at][0] + f[4] * mvo_pb[at][1] + f[7];
            den = fa[0] * fa[0] + fa[1] * fa[1] + ftb0 * ftb0 + ftb1 * ftb1;
            scratch->weight[k] = den > 1e-20f ? 1.0f / den : 0.0f;
            weight = scratch->weight[k];
            if (step) { /* the first step is plain least squares */
                const float robust = 1.0f + scratch->c[k] * scratch->c[k] * weight * kSearchInvSigma2[step];
                weight /= robust * robust;
            }
            for (i = 0; i < 3; ++i) { /* the upper triangle of the symmetric normal matrix */
                const float wg = weight * scratch->g[k][i];
                for (j = i; j < 3; ++j) {
                    m[i * 4 + j] += wg * scratch->g[k][j];
                }
                m[i * 4 + 3] -= wg * scratch->c[k];
            }
        }
        for (i = 1; i < 3; ++i) {
            for (j = 0; j < i; ++j) {
                m[i * 4 + j] = m[j * 4 + i];
            }
        }
        if (!mvo_solve(m, 3, w)) {
            w[0] = w[1] = w[2] = 0.0f;
        }
        for (k = 0; k < count; ++k) {
            const float residual = scratch->c[k] + dot3(w, scratch->g[k]);
            scratch->d2[k] = residual * residual * scratch->weight[k];
        }
        half[0] = 0.5f * w[0];
        half[1] = 0.5f * w[1];
        half[2] = 0.5f * w[2];
        mvo_cayley(half, turn);
        mat3_mul(turn, r_out, r_out);
    }
    for (k = 0; k < count; ++k) { /* the cost with t (ahead) and with -t (behind) */
        const float d2 = scratch->d2[k], loss = d2 / (1.0f + d2);
        const int front = d2 < MVO_RANSAC_PIXELS * MVO_RANSAC_PIXELS ? front_sign(r_out, t, subset[k]) : 2;
        ahead += front == 1 || front == 2 ? loss : 1.0f;
        behind += front == -1 || front == 2 ? loss : 1.0f;
    }
    *sign = behind < ahead ? -1 : 1;
    return behind < ahead ? behind : ahead;
}

typedef struct {
    const mvo_t *s;
    const float *r0;
    int subset[MVO_SEARCH_POINTS], count;
    float turned[MVO_SEARCH_POINTS][3];     /* r0 times each listed pair's ray in A */
    float cost[2], rotation[2][9], t[2][3]; /* each half's best */
} mvo_search_job_t;

/* Half `part` of the directions: the first that fits best, with its rotation and sign. */
static void search_half(void *context, int part)
{
    mvo_search_job_t *job = context;
    const int begin = part * MVO_DIRECTIONS / 2, end = (part + 1) * MVO_DIRECTIONS / 2;
    float rotation[9];
    int d, sign;
    for (d = begin; d < end; ++d) {
        const float cost = direction_fit(job->s, job->r0, kDirections[d], job->subset,
                                         (const float (*)[3])job->turned, job->count, &s_searcher[part], rotation, &sign);
        if (d == begin || cost < job->cost[part]) {
            job->cost[part] = cost;
            memcpy(job->rotation[part], rotation, sizeof rotation);
            job->t[part][0] = (float)sign * kDirections[d][0];
            job->t[part][1] = (float)sign * kDirections[d][1];
            job->t[part][2] = (float)sign * kDirections[d][2];
        }
    }
}

/* The travel direction of kDirections (with its sign), and its fitted rotation (see direction_fit), that fits best up
 * to MVO_SEARCH_POINTS of the marked pairs, spread evenly over them; rotations start from r0. A global look at the
 * direction, which the eight-point fit leaves nearly free on short steps. The directions go in two halves (at the
 * same time when the caller gave s->parallel); the first best of all is the second half's only when it is better. */
static void direction_search(mvo_t *s, int n, const uint8_t *mark, const float r0[9], float r_out[9], float t_out[3])
{
    mvo_search_job_t job;
    int used = 0, rank = 0, target, k, part;
    for (k = 0; k < n; ++k) {
        used += mark[k];
    }
    target = used < MVO_SEARCH_POINTS ? used : MVO_SEARCH_POINTS;
    job.s = s;
    job.r0 = r0;
    job.count = 0;
    for (k = 0; k < n && job.count < target; ++k) {
        if (mark[k]) {
            if (rank == job.count * used / target) {
                job.subset[job.count++] = k;
            }
            ++rank;
        }
    }
    for (k = 0; k < job.count; ++k) {
        const float a[3] = {mvo_ra[job.subset[k]][0], mvo_ra[job.subset[k]][1], 1.0f};
        mat3_vec(r0, a, job.turned[k]);
    }
    if (s->parallel) {
        s->parallel(search_half, &job);
    } else {
        search_half(&job, 0);
        search_half(&job, 1);
    }
    part = job.cost[1] < job.cost[0];
    memcpy(r_out, job.rotation[part], sizeof job.rotation[part]);
    memcpy(t_out, job.t[part], sizeof job.t[part]);
}

typedef struct {
    const mvo_t *s;
    int n;
    int idx[MVO_RANSAC_BATCH][8];  /* the batch's samples */
    int valid[MVO_RANSAC_BATCH];   /* whether each gave a model */
    float e[MVO_RANSAC_BATCH][9];  /* and its essential matrix */
    float score[MVO_RANSAC_BATCH]; /* and its MSAC score */
    int count[MVO_RANSAC_BATCH];   /* and inliers */
} mvo_batch_job_t;

/* Half `part` of a batch: each sample's eight-point model and its MSAC score. */
static void score_half(void *context, int part)
{
    mvo_batch_job_t *job = context;
    int b;
    for (b = part * MVO_RANSAC_BATCH / 2; b < (part + 1) * MVO_RANSAC_BATCH / 2; ++b) {
        job->valid[b] = eight_point_sample(job->idx[b], job->e[b]);
        if (job->valid[b]) {
            job->count[b] = essential_inliers(job->s, job->e[b], job->n, MVO_RANSAC_PIXELS, s_marks[part], &job->score[b]);
        }
    }
}

typedef struct {
    const mvo_t *s;
    int n;
    float *r[2], *t[2]; /* the two poses */
    float cost[2];      /* and their costs (pose_cost) once refined */
} mvo_refine_job_t;

/* Pose `part`: refined on the pairs in mvo_best, then costed. */
static void refine_one(void *context, int part)
{
    mvo_refine_job_t *job = context;
    refine_pose(job->s, job->n, mvo_best, job->r[part], job->t[part], &s_refiner[part], 0);
    job->cost[part] = pose_cost(job->s, job->r[part], job->t[part], job->n, mvo_best);
}

/* Eight-point MSAC (at least MVO_RANSAC_MIN samples, drawn in batches of MVO_RANSAC_BATCH and scored at the same time
 * when the caller gave s->parallel, then taken in order) with the Sampson distance, the decomposition of the best
 * sample, then Levenberg-Marquardt on its epipolar inliers, once more if refining changed them. Small steps leave a wide
 * rotation-translation ambiguity: a pose 35 degrees off in direction can still put most pairs within a pixel, so
 * models are ranked by MSAC's capped squared distances rather than by inlier count, and the refit is nonlinear (a
 * linear least-squares refit in float can slide along the same ambiguity). Marks the inliers of the refined pose
 * (mvo_epipolar) and those of them in front of both cameras within MVO_MAX_DEPTH baselines (mvo_best), and returns the
 * latter's count, or 0. turn, when given, is the pure turn's rotation (x_B = turn x_A): when it and the decomposed
 * pose both turn by more than MVO_TURN_GATE, it pulls the refined rotation (see mono_vo.h). */
int mvo_essential_pose(mvo_t *s, int n, float r[9], float t[3], const float turn[9])
{
    float e[9], best_e[9], score, best_score = 0.0f, searched_r[9], searched_t[3], sample_cost, searched_cost;
    int best = -1, iteration, needed = MVO_RANSAC_MAX, count, k, b;
    mvo_batch_job_t batch;
    mvo_refine_job_t job;
    if (n < 8) {
        return 0;
    }
    s_turn_weight = 0.0f;
    batch.s = s;
    batch.n = n;
    for (iteration = 0; iteration < needed || iteration < MVO_RANSAC_MIN; iteration += MVO_RANSAC_BATCH) {
        for (b = 0; b < MVO_RANSAC_BATCH; ++b) {
            mvo_pick(&s->rng, n, 8, batch.idx[b]);
        }
        if (s->parallel) {
            s->parallel(score_half, &batch);
        } else {
            score_half(&batch, 0);
            score_half(&batch, 1);
        }
        for (b = 0; b < MVO_RANSAC_BATCH; ++b) {
            if (!batch.valid[b] || !(best < 0 || batch.score[b] < best_score)) {
                continue;
            }
            /* a new best: a least-squares refit on its inliers (marked again here), kept when it scores better */
            memcpy(e, batch.e[b], sizeof e);
            count = essential_inliers(s, e, n, MVO_RANSAC_PIXELS, mvo_inlier, &score);
            if (count >= 8) {
                float refit[9], refit_score;
                if (eight_point_fit(n, mvo_inlier, refit)) {
                    const int refit_count = essential_inliers(s, refit, n, MVO_RANSAC_PIXELS, mvo_inlier, &refit_score);
                    if (refit_score < score) {
                        memcpy(e, refit, sizeof refit);
                        score = refit_score;
                        count = refit_count;
                    }
                }
            }
            best = count;
            best_score = score;
            memcpy(best_e, e, sizeof e);
            needed = mvo_ransac_needed(best, n, 8, 0.999f, MVO_RANSAC_MAX);
        }
    }
    if (best < 8) {
        return 0;
    }
    essential_inliers(s, best_e, n, MVO_RANSAC_PIXELS, mvo_best, 0);
    recover_pose(n, mvo_best, best_e, r, t);
    if (turn) { /* a fast turn's pull, when the pure turn and the decomposed pose both turn fast */
        const float turn_bend = 0.5f * (3.0f - (turn[0] + turn[4] + turn[8])); /* 1 - cos of the angle */
        const float pose_bend = 0.5f * (3.0f - (r[0] + r[4] + r[8]));
        const float bend = turn_bend < pose_bend ? turn_bend : pose_bend;
        if (bend >= MVO_TURN_GATE) {
            const float ramp = bend / MVO_TURN_FULL; /* about the angle squared, in units of MVO_TURN_FULL's */
            float norm;
            memcpy(s_turn, turn, sizeof s_turn);
            s_turn_axis[0] = turn[7] - turn[5]; /* the axis: the skew part of turn, at least sin 5 degrees long */
            s_turn_axis[1] = turn[2] - turn[6];
            s_turn_axis[2] = turn[3] - turn[1];
            norm = mvo_root(dot3(s_turn_axis, s_turn_axis));
            s_turn_axis[0] /= norm;
            s_turn_axis[1] /= norm;
            s_turn_axis[2] /= norm;
            s_turn_weight = MVO_TURN_PULL * (ramp < MVO_TURN_CAP ? ramp : MVO_TURN_CAP);
        }
    }
    direction_search(s, n, mvo_best, r, searched_r, searched_t);
    job.s = s;
    job.n = n;
    job.r[0] = r;
    job.t[0] = t;
    job.r[1] = searched_r;
    job.t[1] = searched_t;
    if (s->parallel) { /* the decomposed pose and the searched one, refined side by side */
        s->parallel(refine_one, &job);
    } else {
        refine_one(&job, 0);
        refine_one(&job, 1);
    }
    sample_cost = job.cost[0];
    searched_cost = job.cost[1];
    if (searched_cost < sample_cost) {
        memcpy(r, searched_r, sizeof searched_r);
        memcpy(t, searched_t, sizeof searched_t);
    }
    essential_of(r, t, e);
    if (essential_inliers(s, e, n, MVO_RANSAC_PIXELS, mvo_best, 0) != best) { /* once more on the refined inliers */
        refine_pose(s, n, mvo_best, r, t, &s_refiner[0], 1); /* alone: both cores on its Jacobian */
        essential_of(r, t, e);
        essential_inliers(s, e, n, MVO_RANSAC_PIXELS, mvo_best, 0);
    }
    count = 0;
    for (k = 0; k < n; ++k) {
        mvo_epipolar[k] = mvo_best[k];
        mvo_best[k] = (uint8_t)(mvo_best[k] && in_front(r, t, k));
        count += mvo_best[k];
    }
    return count;
}
