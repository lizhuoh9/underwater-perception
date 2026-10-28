/* Small numerics of the monocular odometry: Jacobi eigenvectors and the 3x3 SVD, Gaussian elimination, Householder
 * QR and one-sided Jacobi, the Cayley rotation and RANSAC helpers (see mono_vo_internal.h). */
#include "mono_vo_internal.h"

/* Cyclic Jacobi on a symmetric n x n matrix (destroyed): eigenvalues in d, eigenvectors as the columns of v. */
static void jacobi(float *a, int n, float *v, float *d)
{
    int sweep, p, q, k;
    for (p = 0; p < n; ++p) {
        for (q = 0; q < n; ++q) {
            v[p * n + q] = p == q ? 1.0f : 0.0f;
        }
    }
    for (sweep = 0; sweep < 16; ++sweep) {
        float off = 0.0f, scale = 0.0f;
        for (p = 0; p < n; ++p) {
            scale += a[p * n + p] * a[p * n + p];
            for (q = p + 1; q < n; ++q) {
                off += a[p * n + q] * a[p * n + q];
            }
        }
        if (off <= 1e-14f * scale || off == 0.0f) {
            break;
        }
        for (p = 0; p < n; ++p) {
            for (q = p + 1; q < n; ++q) {
                float apq = a[p * n + q], theta, t, c, s;
                if (mvo_abs(apq) < 1e-30f) {
                    continue;
                }
                theta = (a[q * n + q] - a[p * n + p]) / (2.0f * apq);
                if (mvo_abs(theta) > 1e15f) { /* theta^2 would overflow: t ~ 1 / (2 theta) */
                    t = 0.5f / theta;
                } else {
                    t = (theta >= 0.0f ? 1.0f : -1.0f) / (mvo_abs(theta) + mvo_root(theta * theta + 1.0f));
                }
                c = mvo_rsqrt(t * t + 1.0f);
                s = t * c;
                for (k = 0; k < n; ++k) { /* columns p and q */
                    float akp = a[k * n + p], akq = a[k * n + q];
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for (k = 0; k < n; ++k) { /* rows p and q */
                    float apk = a[p * n + k], aqk = a[q * n + k];
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
                for (k = 0; k < n; ++k) {
                    float vkp = v[k * n + p], vkq = v[k * n + q];
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    for (p = 0; p < n; ++p) {
        d[p] = a[p * n + p];
    }
}

/* Unit eigenvector of the smallest eigenvalue of a symmetric n x n matrix (destroyed). */
void mvo_smallest_vector(float *a, int n, float *out)
{
    float v[81], d[9];
    int k, best = 0;
    jacobi(a, n, v, d);
    for (k = 1; k < n; ++k) {
        if (d[k] < d[best]) {
            best = k;
        }
    }
    for (k = 0; k < n; ++k) {
        out[k] = v[k * n + best];
    }
}

/* M = U diag(s) V^T with s descending, U and V rotations or reflections as they come out. */
void mvo_svd3(const float m[9], float u[9], float s[3], float v[9])
{
    float mt[9], b[9], vv[9], d[3];
    int order[3] = {0, 1, 2}, i, j;
    mat3_transpose(m, mt);
    mat3_mul(mt, m, b);
    jacobi(b, 3, vv, d);
    for (i = 0; i < 2; ++i) { /* sort descending */
        for (j = i + 1; j < 3; ++j) {
            if (d[order[j]] > d[order[i]]) {
                int t = order[i];
                order[i] = order[j];
                order[j] = t;
            }
        }
    }
    for (j = 0; j < 3; ++j) {
        for (i = 0; i < 3; ++i) {
            v[3 * i + j] = vv[3 * i + order[j]];
        }
        s[j] = mvo_root(d[order[j]] > 0.0f ? d[order[j]] : 0.0f);
    }
    for (j = 0; j < 2; ++j) {
        float col[3] = {v[j], v[3 + j], v[6 + j]}, mc[3];
        mat3_vec(m, col, mc);
        for (i = 0; i < 3; ++i) {
            u[3 * i + j] = s[j] > 1e-20f ? mc[i] / s[j] : (i == j ? 1.0f : 0.0f);
        }
    }
    /* The third column is first x second, signed like M v3. Dividing M v3 by s3 instead would amplify rounding noise:
     * eigenvalues of M^T M carry ~1e-7 absolute error, so an essential matrix's s3 comes out near 3e-4, not 0. */
    u[2] = u[3] * u[7] - u[6] * u[4];
    u[5] = u[6] * u[1] - u[0] * u[7];
    u[8] = u[0] * u[4] - u[3] * u[1];
    {
        const float col[3] = {v[2], v[5], v[8]};
        float mc[3];
        mat3_vec(m, col, mc);
        if (mc[0] * u[2] + mc[1] * u[5] + mc[2] * u[8] < 0.0f) {
            u[2] = -u[2];
            u[5] = -u[5];
            u[8] = -u[8];
        }
    }
}

/* Gaussian elimination with partial pivoting on n x (n + 1) augmented rows; 0 when singular. */
int mvo_solve(float *m, int n, float *x)
{
    int col, row, k;
    for (col = 0; col < n; ++col) {
        int pivot = col;
        for (row = col + 1; row < n; ++row) {
            if (mvo_abs(m[row * (n + 1) + col]) > mvo_abs(m[pivot * (n + 1) + col])) {
                pivot = row;
            }
        }
        if (mvo_abs(m[pivot * (n + 1) + col]) < 1e-12f) {
            return 0;
        }
        if (pivot != col) {
            for (k = 0; k <= n; ++k) {
                float t = m[col * (n + 1) + k];
                m[col * (n + 1) + k] = m[pivot * (n + 1) + k];
                m[pivot * (n + 1) + k] = t;
            }
        }
        for (row = col + 1; row < n; ++row) {
            float f = m[row * (n + 1) + col] / m[col * (n + 1) + col];
            for (k = col; k <= n; ++k) {
                m[row * (n + 1) + k] -= f * m[col * (n + 1) + k];
            }
        }
    }
    for (row = n - 1; row >= 0; --row) {
        float sum = m[row * (n + 1) + n];
        for (k = row + 1; k < n; ++k) {
            sum -= m[row * (n + 1) + k] * x[k];
        }
        x[row] = sum / m[row * (n + 1) + row];
    }
    return 1;
}

/* Unit null vector of an 8 x 9 matrix (destroyed): Gaussian elimination with full pivoting, the free column set to 1.
 * Solving the sample directly keeps float precision; the normal equations A^T A would square its condition number. */
int mvo_null_vector8(float *m, float out[9])
{
    int column[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8}, k, row, col;
    float x[9], norm = 0.0f;
    for (k = 0; k < 8; ++k) {
        int pivot_row = k, pivot_col = k, swap_index;
        float largest = 0.0f;
        for (row = k; row < 8; ++row) {
            for (col = k; col < 9; ++col) {
                if (mvo_abs(m[row * 9 + col]) > largest) {
                    largest = mvo_abs(m[row * 9 + col]);
                    pivot_row = row;
                    pivot_col = col;
                }
            }
        }
        if (largest < 1e-12f) {
            return 0;
        }
        for (col = 0; col < 9; ++col) { /* rows k and pivot_row */
            const float swap = m[k * 9 + col];
            m[k * 9 + col] = m[pivot_row * 9 + col];
            m[pivot_row * 9 + col] = swap;
        }
        for (row = 0; row < 8; ++row) { /* columns k and pivot_col: the unknowns are renumbered */
            const float swap = m[row * 9 + k];
            m[row * 9 + k] = m[row * 9 + pivot_col];
            m[row * 9 + pivot_col] = swap;
        }
        swap_index = column[k];
        column[k] = column[pivot_col];
        column[pivot_col] = swap_index;
        for (row = k + 1; row < 8; ++row) {
            const float factor = m[row * 9 + k] / m[k * 9 + k];
            for (col = k; col < 9; ++col) {
                m[row * 9 + col] -= factor * m[k * 9 + col];
            }
        }
    }
    x[8] = 1.0f;
    for (row = 7; row >= 0; --row) {
        float sum = 0.0f;
        for (col = row + 1; col < 9; ++col) {
            sum += m[row * 9 + col] * x[col];
        }
        x[row] = -sum / m[row * 9 + row];
    }
    for (k = 0; k < 9; ++k) {
        norm += x[k] * x[k];
    }
    norm = mvo_root(norm);
    for (k = 0; k < 9; ++k) {
        out[column[k]] = x[k] / norm;
    }
    return 1;
}

/* The 9 x 9 upper triangle R (row-major) of the QR decomposition of m x 9 rows (destroyed), by Householder. */
void mvo_householder_r(float (*rows)[9], int m, float r[81])
{
    int k, i, j;
    for (k = 0; k < 9 && k < m; ++k) {
        float norm = 0.0f, alpha, length = 0.0f;
        for (i = k; i < m; ++i) {
            norm += rows[i][k] * rows[i][k];
        }
        norm = mvo_root(norm);
        if (norm == 0.0f) {
            continue;
        }
        alpha = rows[k][k] > 0.0f ? -norm : norm;
        rows[k][k] -= alpha; /* column k from row k down is now the reflection vector v */
        for (i = k; i < m; ++i) {
            length += rows[i][k] * rows[i][k];
        }
        for (j = k + 1; j < 9; ++j) {
            float dot = 0.0f, factor;
            for (i = k; i < m; ++i) {
                dot += rows[i][k] * rows[i][j];
            }
            factor = 2.0f * dot / length;
            for (i = k; i < m; ++i) {
                rows[i][j] -= factor * rows[i][k];
            }
        }
        rows[k][k] = alpha;
    }
    for (i = 0; i < 9; ++i) {
        for (j = 0; j < 9; ++j) {
            r[i * 9 + j] = j >= i && i < m ? rows[i][j] : 0.0f;
        }
    }
}

/* Unit right singular vector of the smallest singular value of a 9 x 9 matrix (destroyed), by one-sided Jacobi. */
void mvo_smallest_right_vector(float *a, float out[9])
{
    float v[81], smallest = 0.0f;
    int sweep, p, q, k, best = 0;
    for (p = 0; p < 81; ++p) {
        v[p] = p % 10 == 0 ? 1.0f : 0.0f;
    }
    for (sweep = 0; sweep < 20; ++sweep) {
        int rotated = 0;
        for (p = 0; p < 8; ++p) {
            for (q = p + 1; q < 9; ++q) {
                float alpha = 0.0f, beta = 0.0f, gamma = 0.0f, zeta, t, c, s;
                for (k = 0; k < 9; ++k) {
                    alpha += a[k * 9 + p] * a[k * 9 + p];
                    beta += a[k * 9 + q] * a[k * 9 + q];
                    gamma += a[k * 9 + p] * a[k * 9 + q];
                }
                if (gamma * gamma <= 1e-14f * alpha * beta) {
                    continue;
                }
                rotated = 1;
                zeta = (beta - alpha) / (2.0f * gamma);
                if (mvo_abs(zeta) > 1e15f) {
                    t = 0.5f / zeta;
                } else {
                    t = (zeta >= 0.0f ? 1.0f : -1.0f) / (mvo_abs(zeta) + mvo_root(zeta * zeta + 1.0f));
                }
                c = mvo_rsqrt(t * t + 1.0f);
                s = t * c;
                for (k = 0; k < 9; ++k) {
                    const float ap = a[k * 9 + p], aq = a[k * 9 + q], vp = v[k * 9 + p], vq = v[k * 9 + q];
                    a[k * 9 + p] = c * ap - s * aq;
                    a[k * 9 + q] = s * ap + c * aq;
                    v[k * 9 + p] = c * vp - s * vq;
                    v[k * 9 + q] = s * vp + c * vq;
                }
            }
        }
        if (!rotated) {
            break;
        }
    }
    for (p = 0; p < 9; ++p) { /* the singular values are the column norms */
        float norm = 0.0f;
        for (k = 0; k < 9; ++k) {
            norm += a[k * 9 + p] * a[k * 9 + p];
        }
        if (p == 0 || norm < smallest) {
            smallest = norm;
            best = p;
        }
    }
    for (k = 0; k < 9; ++k) {
        out[k] = v[k * 9 + best];
    }
}

/* The rotation of Cayley vector c (axis times tan of half the angle): I + 2 ([c]x + [c]x^2) / (1 + |c|^2). */
void mvo_cayley(const float c[3], float out[9])
{
    const float cx[9] = {0.0f, -c[2], c[1], c[2], 0.0f, -c[0], -c[1], c[0], 0.0f};
    float cx2[9];
    const float g = 2.0f / (1.0f + dot3(c, c));
    int i;
    mat3_mul(cx, cx, cx2);
    for (i = 0; i < 9; ++i) {
        out[i] = (i % 4 == 0 ? 1.0f : 0.0f) + g * (cx[i] + cx2[i]);
    }
}

/* See mono_vo.h. OpenCV's model takes normalised (x, y) to x (1 + k1 r^2 + k2 r^4 + k3 r^6) + 2 p1 x y + p2 (r^2 + 2 x^2)
 * and y (1 + k1 r^2 + k2 r^4 + k3 r^6) + p1 (r^2 + 2 y^2) + 2 p2 x y; Newton steps on that map, with its Jacobian, from
 * the distorted point. */
int mvo_undistort(const mvo_camera_t *camera, const float in[2], float out[2])
{
    const float xd = (in[0] - camera->cx) / camera->fx, yd = (in[1] - camera->cy) / camera->fy;
    const float k1 = camera->k1, k2 = camera->k2, k3 = camera->k3, p1 = camera->p1, p2 = camera->p2;
    float x = xd, y = yd;
    int i, ok = 1;
    if (k1 == 0.0f && k2 == 0.0f && k3 == 0.0f && p1 == 0.0f && p2 == 0.0f) {
        out[0] = in[0];
        out[1] = in[1];
        return 1;
    }
    for (i = 0; i < MVO_UNDISTORT_STEPS; ++i) {
        const float r2 = x * x + y * y, xy = x * y;
        const float radial = 1.0f + r2 * (k1 + r2 * (k2 + r2 * k3));
        const float slope = k1 + r2 * (2.0f * k2 + 3.0f * k3 * r2); /* d radial / d r^2 */
        const float ex = x * radial + 2.0f * p1 * xy + p2 * (r2 + 2.0f * x * x) - xd;
        const float ey = y * radial + p1 * (r2 + 2.0f * y * y) + 2.0f * p2 * xy - yd;
        const float a = radial + 2.0f * x * x * slope + 2.0f * p1 * y + 6.0f * p2 * x;
        const float b = 2.0f * xy * slope + 2.0f * p1 * x + 2.0f * p2 * y;
        const float d = radial + 2.0f * y * y * slope + 6.0f * p1 * y + 2.0f * p2 * x;
        const float det = a * d - b * b;
        if (!(radial > 0.0f) || !(det > 1e-6f)) { /* folded over: a radius turned back or through the centre */
            ok = 0;
            break;
        }
        x -= (d * ex - b * ey) / det;
        y -= (a * ey - b * ex) / det;
    }
    out[0] = camera->cx + camera->fx * x;
    out[1] = camera->cy + camera->fy * y;
    return ok;
}

static void sort_floats(float *v, int n) /* insertion sort: n <= MVO_MAX_FEATURES */
{
    int i, j;
    for (i = 1; i < n; ++i) {
        float key = v[i];
        for (j = i - 1; j >= 0 && v[j] > key; --j) {
            v[j + 1] = v[j];
        }
        v[j + 1] = key;
    }
}

float mvo_median(float *v, int n)
{
    sort_floats(v, n);
    return n % 2 ? v[n / 2] : 0.5f * (v[n / 2 - 1] + v[n / 2]);
}

/* RANSAC iterations needed for `confidence` with this inlier share and sample size, capped. */
int mvo_ransac_needed(int inliers, int n, int sample, float confidence, int cap)
{
    float w = (float)inliers / (float)n, all = 1.0f, miss = 1.0f;
    int k;
    for (k = 0; k < sample; ++k) {
        all *= w;
    }
    if (all <= 0.0f) {
        return cap;
    }
    for (k = 0; k < cap; ++k) {
        if (miss <= 1.0f - confidence) {
            return k;
        }
        miss *= 1.0f - all;
    }
    return cap;
}

static uint32_t mvo_next(uint32_t *state)
{
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

void mvo_pick(uint32_t *rng, int n, int count, int *out)
{
    int k = 0, j;
    while (k < count) {
        int candidate = (int)(mvo_next(rng) % (uint32_t)n), fresh = 1;
        for (j = 0; j < k; ++j) {
            fresh &= out[j] != candidate;
        }
        if (fresh) {
            out[k++] = candidate;
        }
    }
}
