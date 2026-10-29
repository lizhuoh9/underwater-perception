/* The rotation of a pure turn, from a RANSAC homography refitted by least squares, and the parallax test that tells
 * such steps apart (see mono_vo_internal.h). */
#include "mono_vo_internal.h"

#include <string.h>

#define MVO_HOMOGRAPHY_BATCH 8 /* samples drawn, then scored together: half of them on each core */

static uint8_t s_marks[2][MVO_MAX_FEATURES]; /* inlier marks of the samples a batch half scores */

typedef struct {
    int n;
    int idx[MVO_HOMOGRAPHY_BATCH][4];  /* the batch's samples */
    int count[MVO_HOMOGRAPHY_BATCH];   /* each one's inliers, or -1 for no model */
    float h[MVO_HOMOGRAPHY_BATCH][9];  /* and its homography */
} mvo_homography_job_t;

/* Homography from four correspondences (h33 = 1); 0 when degenerate. */
static int homography4(const int *idx, float h[9])
{
    float m[8 * 9], x[8];
    int k;
    for (k = 0; k < 4; ++k) {
        const float *a = mvo_pa[idx[k]], *b = mvo_pb[idx[k]];
        float *r0 = m + (2 * k) * 9, *r1 = m + (2 * k + 1) * 9;
        r0[0] = a[0]; r0[1] = a[1]; r0[2] = 1.0f; r0[3] = 0.0f; r0[4] = 0.0f; r0[5] = 0.0f;
        r0[6] = -b[0] * a[0]; r0[7] = -b[0] * a[1]; r0[8] = b[0];
        r1[0] = 0.0f; r1[1] = 0.0f; r1[2] = 0.0f; r1[3] = a[0]; r1[4] = a[1]; r1[5] = 1.0f;
        r1[6] = -b[1] * a[0]; r1[7] = -b[1] * a[1]; r1[8] = b[1];
    }
    if (!mvo_solve(m, 8, x)) {
        return 0;
    }
    for (k = 0; k < 8; ++k) {
        h[k] = x[k];
    }
    h[8] = 1.0f;
    return 1;
}

static int homography_inliers(const float h[9], int n, float threshold, uint8_t *mark)
{
    int k, count = 0;
    for (k = 0; k < n; ++k) {
        const float *a = mvo_pa[k], *b = mvo_pb[k];
        float w = h[6] * a[0] + h[7] * a[1] + h[8], u, v, du, dv;
        int ok = 0;
        if (mvo_abs(w) > 1e-12f) {
            u = (h[0] * a[0] + h[1] * a[1] + h[2]) / w;
            v = (h[3] * a[0] + h[4] * a[1] + h[5]) / w;
            du = u - b[0];
            dv = v - b[1];
            ok = du * du + dv * dv < threshold * threshold;
        }
        mark[k] = (uint8_t)ok;
        count += ok;
    }
    return count;
}

/* Half `part` of a batch: each sample's homography and its inliers. */
static void homography_half(void *context, int part)
{
    mvo_homography_job_t *job = context;
    int b;
    for (b = part * MVO_HOMOGRAPHY_BATCH / 2; b < (part + 1) * MVO_HOMOGRAPHY_BATCH / 2; ++b) {
        job->count[b] = homography4(job->idx[b], job->h[b])
                            ? homography_inliers(job->h[b], job->n, MVO_HOMOGRAPHY_PIXELS, s_marks[part])
                            : -1;
    }
}

/* Least-squares homography on the marked correspondences (normalised DLT, smallest eigenvector of A^T A). */
static int homography_fit(int n, const uint8_t *mark, float h[9])
{
    float ca[2] = {0, 0}, cb[2] = {0, 0}, sa = 0.0f, sb = 0.0f, ata[81], v[9], ta[9], tb_inv[9], hn[9], tmp[9];
    int k, i, j, used = 0;
    for (k = 0; k < n; ++k) {
        if (mark[k]) {
            ca[0] += mvo_pa[k][0]; ca[1] += mvo_pa[k][1]; cb[0] += mvo_pb[k][0]; cb[1] += mvo_pb[k][1];
            ++used;
        }
    }
    if (used < 4) {
        return 0;
    }
    ca[0] /= (float)used; ca[1] /= (float)used; cb[0] /= (float)used; cb[1] /= (float)used;
    for (k = 0; k < n; ++k) {
        if (mark[k]) {
            float ax = mvo_pa[k][0] - ca[0], ay = mvo_pa[k][1] - ca[1], bx = mvo_pb[k][0] - cb[0], by = mvo_pb[k][1] - cb[1];
            sa += mvo_root(ax * ax + ay * ay);
            sb += mvo_root(bx * bx + by * by);
        }
    }
    sa = sa > 0.0f ? 1.41421356f * (float)used / sa : 1.0f;
    sb = sb > 0.0f ? 1.41421356f * (float)used / sb : 1.0f;
    memset(ata, 0, sizeof ata);
    for (k = 0; k < n; ++k) {
        float ax, ay, bx, by, r0[9], r1[9];
        if (!mark[k]) {
            continue;
        }
        ax = (mvo_pa[k][0] - ca[0]) * sa; ay = (mvo_pa[k][1] - ca[1]) * sa;
        bx = (mvo_pb[k][0] - cb[0]) * sb; by = (mvo_pb[k][1] - cb[1]) * sb;
        r0[0] = ax; r0[1] = ay; r0[2] = 1.0f; r0[3] = 0.0f; r0[4] = 0.0f; r0[5] = 0.0f; r0[6] = -bx * ax; r0[7] = -bx * ay; r0[8] = -bx;
        r1[0] = 0.0f; r1[1] = 0.0f; r1[2] = 0.0f; r1[3] = ax; r1[4] = ay; r1[5] = 1.0f; r1[6] = -by * ax; r1[7] = -by * ay; r1[8] = -by;
        for (i = 0; i < 9; ++i) {
            for (j = 0; j < 9; ++j) {
                ata[i * 9 + j] += r0[i] * r0[j] + r1[i] * r1[j];
            }
        }
    }
    mvo_smallest_vector(ata, 9, v);
    /* H = Tb^-1 Hn Ta */
    ta[0] = sa; ta[1] = 0.0f; ta[2] = -sa * ca[0]; ta[3] = 0.0f; ta[4] = sa; ta[5] = -sa * ca[1]; ta[6] = 0.0f; ta[7] = 0.0f; ta[8] = 1.0f;
    tb_inv[0] = 1.0f / sb; tb_inv[1] = 0.0f; tb_inv[2] = cb[0]; tb_inv[3] = 0.0f; tb_inv[4] = 1.0f / sb; tb_inv[5] = cb[1];
    tb_inv[6] = 0.0f; tb_inv[7] = 0.0f; tb_inv[8] = 1.0f;
    for (k = 0; k < 9; ++k) {
        hn[k] = v[k];
    }
    mat3_mul(hn, ta, tmp);
    mat3_mul(tb_inv, tmp, h);
    return 1;
}

/* R_AB of a pure rotation, R = orthonormalised K^-1 H K (x_B = R x_A), and the homography's inlier count. RANSAC
 * draws its samples in batches of MVO_HOMOGRAPHY_BATCH, scored at the same time when the caller gave s->parallel and
 * then taken in order. */
int mvo_pure_rotation(mvo_t *s, int n, float rotation_ab[9])
{
    const float threshold = MVO_HOMOGRAPHY_PIXELS;
    float h[9], best_h[9], k_mat[9], k_inv[9], m[9], u[9], sv[3], v[9], vt[9], r[9];
    int best = -1, iteration, needed = MVO_RANSAC_MAX, count, b;
    mvo_homography_job_t batch;
    if (n < 4) {
        return 0;
    }
    batch.n = n;
    for (iteration = 0; iteration < needed; iteration += MVO_HOMOGRAPHY_BATCH) {
        for (b = 0; b < MVO_HOMOGRAPHY_BATCH; ++b) {
            mvo_pick(&s->rng, n, 4, batch.idx[b]);
        }
        if (s->parallel) {
            s->parallel(homography_half, &batch);
        } else {
            homography_half(&batch, 0);
            homography_half(&batch, 1);
        }
        for (b = 0; b < MVO_HOMOGRAPHY_BATCH; ++b) {
            if (batch.count[b] > best) {
                best = batch.count[b];
                memcpy(best_h, batch.h[b], sizeof best_h);
                needed = mvo_ransac_needed(best, n, 4, 0.995f, MVO_RANSAC_MAX);
            }
        }
    }
    if (best < 4) {
        return 0;
    }
    homography_inliers(best_h, n, threshold, mvo_inlier);
    if (!homography_fit(n, mvo_inlier, h)) {
        return 0;
    }
    count = homography_inliers(h, n, threshold, mvo_inlier);
    k_mat[0] = s->camera.fx; k_mat[1] = 0.0f; k_mat[2] = s->camera.cx;
    k_mat[3] = 0.0f; k_mat[4] = s->camera.fy; k_mat[5] = s->camera.cy;
    k_mat[6] = 0.0f; k_mat[7] = 0.0f; k_mat[8] = 1.0f;
    k_inv[0] = 1.0f / s->camera.fx; k_inv[1] = 0.0f; k_inv[2] = -s->camera.cx / s->camera.fx;
    k_inv[3] = 0.0f; k_inv[4] = 1.0f / s->camera.fy; k_inv[5] = -s->camera.cy / s->camera.fy;
    k_inv[6] = 0.0f; k_inv[7] = 0.0f; k_inv[8] = 1.0f;
    mat3_mul(k_inv, h, m);
    mat3_mul(m, k_mat, m);
    mvo_svd3(m, u, sv, v);
    mat3_transpose(v, vt);
    mat3_mul(u, vt, r);
    if (mat3_det(r) < 0.0f) {
        int i;
        for (i = 0; i < 9; ++i) {
            r[i] = -r[i];
        }
    }
    mat3_transpose(r, rotation_ab);
    return count;
}

/* Median distance (pixels) between the tracks' ends and where the rotation alone takes their starts. */
float mvo_parallax(const mvo_t *s, int n, const float rotation_ab[9])
{
    float rotation[9];
    int k;
    mat3_transpose(rotation_ab, rotation); /* x_B = R x_A */
    for (k = 0; k < n; ++k) {
        float ray[3], turned[3], u, v;
        to_ray(&s->camera, mvo_pa[k], ray);
        ray[2] = 1.0f;
        mat3_vec(rotation, ray, turned);
        u = turned[2] > 1e-6f ? s->camera.fx * turned[0] / turned[2] + s->camera.cx : 1e9f;
        v = turned[2] > 1e-6f ? s->camera.fy * turned[1] / turned[2] + s->camera.cy : 1e9f;
        mvo_values[k] = mvo_root((u - mvo_pb[k][0]) * (u - mvo_pb[k][0]) + (v - mvo_pb[k][1]) * (v - mvo_pb[k][1]));
    }
    return mvo_median(mvo_values, n);
}

/* An orthonormal frame from two separated rays, with the rays as its first two directions. */
static int rotation_basis(const float a[2], const float b[2], float basis[9])
{
    float x[3] = {a[0], a[1], 1.0f}, y[3] = {b[0], b[1], 1.0f}, z[3], norm, projection;
    int k;
    norm = mvo_rsqrt(dot3(x, x));
    for (k = 0; k < 3; ++k) x[k] *= norm;
    projection = dot3(x, y);
    for (k = 0; k < 3; ++k) y[k] -= projection * x[k];
    norm = dot3(y, y);
    if (norm < 0.000001f) return 0; /* near-parallel rays do not define a stable two-ray basis */
    norm = mvo_rsqrt(norm);
    for (k = 0; k < 3; ++k) y[k] *= norm;
    z[0] = x[1] * y[2] - x[2] * y[1];
    z[1] = x[2] * y[0] - x[0] * y[2];
    z[2] = x[0] * y[1] - x[1] * y[0];
    for (k = 0; k < 3; ++k) {
        basis[3 * k] = x[k]; basis[3 * k + 1] = y[k]; basis[3 * k + 2] = z[k];
    }
    return 1;
}

static float rotation_error2(const mvo_t *s, const float r[9], int k)
{
    const float x = mvo_ra[k][0], y = mvo_ra[k][1];
    const float b0 = r[0] * x + r[1] * y + r[2];
    const float b1 = r[3] * x + r[4] * y + r[5];
    const float b2 = r[6] * x + r[7] * y + r[8];
    float dx, dy;
    if (b2 <= 1e-6f) return 1e18f;
    dx = s->camera.fx * b0 / b2 + s->camera.cx - mvo_pb[k][0];
    dy = s->camera.fy * b1 / b2 + s->camera.cy - mvo_pb[k][1];
    return dx * dx + dy * dy;
}

int mvo_rotation_inlier(const mvo_t *s, const float rotation[9], int k)
{
    return rotation_error2(s, rotation, k) < MVO_HOMOGRAPHY_PIXELS * MVO_HOMOGRAPHY_PIXELS;
}

#define MVO_ROTATION_BATCH 8
typedef struct {
    const mvo_t *state;
    int n;
    int idx[MVO_ROTATION_BATCH][2];
    float threshold2, cutoff;
    float r[MVO_ROTATION_BATCH][9], score[MVO_ROTATION_BATCH];
    int valid[MVO_ROTATION_BATCH];
} rotation_batch_t;

/* Candidates in each half read the same tracks and cutoff, and write separate result slots. */
static void rotation_score_half(void *context, int part)
{
    rotation_batch_t *job = context;
    int b, k;
    for (b = part * (MVO_ROTATION_BATCH / 2); b < (part + 1) * (MVO_ROTATION_BATCH / 2); ++b) {
        float a[9], bb[9], at[9], score = 0.0f;
        const int *idx = job->idx[b];
        job->valid[b] = rotation_basis(mvo_ra[idx[0]], mvo_ra[idx[1]], a)
                        && rotation_basis(mvo_rb[idx[0]], mvo_rb[idx[1]], bb);
        if (!job->valid[b]) continue;
        mat3_transpose(a, at);
        mat3_mul(bb, at, job->r[b]);
        for (k = 0; k < job->n; ++k) {
            float d2 = rotation_error2(job->state, job->r[b], k);
            score += d2 < job->threshold2 ? d2 : job->threshold2;
            if (score >= job->cutoff) break; /* nonnegative remaining costs cannot improve this candidate */
        }
        job->score[b] = score;
    }
}

/* Fit a rotation independently of the homography and essential models.  A local RNG keeps a rejected
 * candidate from changing subsequent frames' RANSAC samples. */
int mvo_rotation_model(const mvo_t *s, int n, float rotation_ab[9])
{
    float best_r[9], best_score = 1e30f, threshold2 = 4.0f * MVO_HOMOGRAPHY_PIXELS * MVO_HOMOGRAPHY_PIXELS;
    uint32_t rng = s->rng;
    int iteration, k, support = 0;
    rotation_batch_t batch;
    if (n < 2) return -1;
    batch.state = s;
    batch.n = n;
    batch.threshold2 = threshold2;
    for (iteration = 0; iteration < 128; iteration += MVO_ROTATION_BATCH) {
        int b;
        for (b = 0; b < MVO_ROTATION_BATCH; ++b) mvo_pick(&rng, n, 2, batch.idx[b]);
        batch.cutoff = best_score;
        if (s->parallel) s->parallel(rotation_score_half, &batch);
        else { rotation_score_half(&batch, 0); rotation_score_half(&batch, 1); }
        for (b = 0; b < MVO_ROTATION_BATCH; ++b) {
            if (batch.valid[b] && batch.score[b] < best_score) {
                best_score = batch.score[b];
                memcpy(best_r, batch.r[b], sizeof best_r);
            }
        }
    }
    if (best_score == 1e30f) return -1;
    for (iteration = 0; iteration < 3; ++iteration) {
        float covariance[9] = {0}, u[9], sv[3], v[9], vt[9], r[9];
        int i, j;
        support = 0;
        for (k = 0; k < n; ++k) {
            float a[3] = {mvo_ra[k][0], mvo_ra[k][1], 1.0f};
            float b[3] = {mvo_rb[k][0], mvo_rb[k][1], 1.0f};
            float scale;
            if (rotation_error2(s, best_r, k) >= threshold2) continue;
            ++support;
            scale = mvo_rsqrt(dot3(a, a)) * mvo_rsqrt(dot3(b, b));
            for (i = 0; i < 3; ++i)
                for (j = 0; j < 3; ++j) covariance[3 * i + j] += b[i] * a[j] * scale;
        }
        if (support < 3) break;
        mvo_svd3(covariance, u, sv, v);
        mat3_transpose(v, vt);
        mat3_mul(u, vt, r);
        if (mat3_det(r) < 0.0f) {
            u[2] = -u[2]; u[5] = -u[5]; u[8] = -u[8];
            mat3_mul(u, vt, r);
        }
        { /* Ill-conditioned narrow fields can make the SVD's result non-orthogonal or less accurate. */
            float rt[9], gram[9], score = 0.0f;
            int valid = 1;
            mat3_transpose(r, rt);
            mat3_mul(rt, r, gram);
            for (i = 0; i < 3; ++i)
                for (j = 0; j < 3; ++j)
                    if (mvo_abs(gram[3 * i + j] - (i == j ? 1.0f : 0.0f)) > 0.001f) valid = 0;
            for (k = 0; k < n; ++k) {
                float d2 = rotation_error2(s, r, k);
                score += d2 < threshold2 ? d2 : threshold2;
            }
            if (!valid || !(score < best_score)) break;
            best_score = score;
        }
        memcpy(best_r, r, sizeof best_r);
    }
    support = 0;
    for (k = 0; k < n; ++k) support += mvo_rotation_inlier(s, best_r, k);
    mat3_transpose(best_r, rotation_ab);
    return support;
}
