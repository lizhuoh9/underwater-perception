/* Image pyramid, Shi-Tomasi corners and the fixed-point pyramidal Lucas-Kanade tracker of the monocular odometry
 * (see mono_vo_internal.h). */
#include "mono_vo_internal.h"

#include <string.h>

#define MVO_MAX_CANDIDATES 1024 /* strongest local maxima kept of the frame, and so by each half of the detector */
#define MVO_DETECT_SPLIT (MVO_HEIGHT / 2) /* the detector's second half looks for corners from this row down */
#define MVO_CELL 8 /* grid cell for the minimum-distance test, pixels (= MVO_MIN_DISTANCE) */
#define MVO_GRID_W (MVO_WIDTH / MVO_CELL)
#define MVO_GRID_H (MVO_HEIGHT / MVO_CELL)
#define MVO_CELL_CAPACITY 6
#define MVO_MIN_EIGEN 1e-4f /* Lucas-Kanade: smallest eigenvalue of the gradient matrix per window pixel */
#define MVO_RAW (MVO_SIDE + 3) /* template plus one pixel each side for derivatives and one for interpolation */
#define MVO_W_BITS 14 /* fixed-point bilinear weights of the tracker */

static const int kWidth[MVO_LEVELS] = {320, 160, 80, 40};
static const int kHeight[MVO_LEVELS] = {240, 120, 60, 30};
static const int kOffset[MVO_LEVELS] = {0, 76800, 96000, 100800};
static const float kScale[MVO_LEVELS] = {1.0f, 0.5f, 0.25f, 0.125f}; /* level pixels per full-resolution pixel */

typedef struct {
    int16_t x, y;
    float r;
} mvo_candidate_t;

/* Scratch, not thread-safe (see mono_vo.h). */
typedef struct { /* one half of the detector: the two may run at the same time */
    mvo_candidate_t heap[MVO_MAX_CANDIDATES]; /* the strongest local maxima so far, the weakest on top */
    int16_t gradients[3][MVO_WIDTH][2];       /* Sobel x and y of three image rows */
    int32_t response[3][MVO_WIDTH];           /* corner response of three rows, as float bits (see detect_half) */
    int size;                                 /* candidates in the heap */
    float best;                               /* strongest response of the half's rows */
} mvo_detector_t;
static mvo_detector_t s_detector[2];
typedef uint16_t mvo_grid_index_t; /* a feature's place in mvo_t.feature */
static mvo_grid_index_t s_grid[MVO_GRID_H][MVO_GRID_W][MVO_CELL_CAPACITY]; /* feature indices */
static uint8_t s_grid_count[MVO_GRID_H][MVO_GRID_W];
typedef struct { /* one tracker's scratch: the two halves of the features may run at the same time */
    mvo_window_t window;
    uint8_t block[MVO_RAW * 32 + 32]; /* a window that crosses the image border, clamped (see mvo_block_c); slack */
    const uint8_t *copied;                 /* the image level the block was copied from (NULL: none this step) */
    int copied_x, copied_y, copied_size;   /* and where */
} mvo_tracker_t;
static mvo_tracker_t s_tracker[2];
static float s_went[MVO_MAX_FEATURES][2]; /* where each feature went */
static uint8_t s_tracked[MVO_MAX_FEATURES]; /* and whether it survived the forward-backward check */

static int mvo_floor(float v)
{
    int i = (int)v;
    return v < (float)i ? i - 1 : i;
}

/* ---- image pyramid ---------------------------------------------------------------------------------------- */

static int reflect101(int i, int n)
{
    if (i < 0) {
        i = -i;
    }
    if (i >= n) {
        i = 2 * n - 2 - i;
    }
    return i;
}

/* The [1 4 6 4 1] filter along one source row at every second column, for every destination column (reflected at
 * the borders, plain indexing inside). */
static void pyramid_row(const uint8_t *row, int sw, int dw, int16_t *out)
{
    int x;
    for (x = 0; x < dw; ++x) {
        const int c = 2 * x;
        if (c >= 2 && c + 2 < sw) {
            out[x] = (int16_t)(row[c - 2] + 4 * row[c - 1] + 6 * row[c] + 4 * row[c + 1] + row[c + 2]);
        } else {
            out[x] = (int16_t)(row[reflect101(c - 2, sw)] + 4 * row[reflect101(c - 1, sw)] + 6 * row[c]
                               + 4 * row[reflect101(c + 1, sw)] + row[reflect101(c + 2, sw)]);
        }
    }
}

/* Rows first .. last - 1 of the Gaussian [1 4 6 4 1]^2 / 256 of every second pixel, borders reflected (OpenCV's
 * pyrDown), separably, with the scratch `rows`. Filtered source rows live in a ring of five slots keyed by row index
 * modulo 5: a destination row needs five consecutive (reflected) source rows, so they never share a slot, and each is
 * filtered once. */
static void pyramid_down(const uint8_t *src, int sw, int sh, uint8_t *dst, int dw, int first, int last,
                         int16_t rows[5][MVO_WIDTH / 2])
{
    int cached[5] = {-1, -1, -1, -1, -1}, slot[5], x, y, j;
    for (y = first; y < last; ++y) {
        for (j = 0; j < 5; ++j) {
            const int r = reflect101(2 * y + j - 2, sh);
            slot[j] = r % 5;
            if (cached[slot[j]] != r) {
                pyramid_row(src + r * sw, sw, dw, rows[slot[j]]);
                cached[slot[j]] = r;
            }
        }
        for (x = 0; x < dw; ++x) {
            const int sum = rows[slot[0]][x] + 4 * rows[slot[1]][x] + 6 * rows[slot[2]][x] + 4 * rows[slot[3]][x]
                            + rows[slot[4]][x];
            dst[y * dw + x] = (uint8_t)((sum + 128) >> 8);
        }
    }
}

static int16_t s_pyramid_rows[2][5][MVO_WIDTH / 2]; /* each half's filtered source rows */

/* Half `part` of the rows of level 1, three quarters of the pyramid's work. */
static void level1_half(void *context, int part)
{
    uint8_t *pyramid = context;
    const int half = kHeight[1] / 2;
    pyramid_down(pyramid, kWidth[0], kHeight[0], pyramid + kOffset[1], kWidth[1], part * half, (part + 1) * half,
                 s_pyramid_rows[part]);
}

void mvo_build_pyramid(const mvo_t *s, uint8_t *pyramid)
{
    int level;
    if (s->parallel) {
        s->parallel(level1_half, pyramid);
    } else {
        level1_half(pyramid, 0);
        level1_half(pyramid, 1);
    }
    for (level = 2; level < MVO_LEVELS; ++level) {
        pyramid_down(pyramid + kOffset[level - 1], kWidth[level - 1], kHeight[level - 1], pyramid + kOffset[level],
                     kWidth[level], 0, kHeight[level], s_pyramid_rows[0]);
    }
}

/* ---- Shi-Tomasi corners ------------------------------------------------------------------------------------ */

static void sobel_row(const uint8_t *img, int y, int16_t out[MVO_WIDTH][2])
{
    int x;
    const uint8_t *up = img + (y - 1) * MVO_WIDTH, *mid = img + y * MVO_WIDTH, *down = img + (y + 1) * MVO_WIDTH;
    for (x = 1; x < MVO_WIDTH - 1; ++x) {
        out[x][0] = (int16_t)((up[x + 1] + 2 * mid[x + 1] + down[x + 1]) - (up[x - 1] + 2 * mid[x - 1] + down[x - 1]));
        out[x][1] = (int16_t)((down[x - 1] + 2 * down[x] + down[x + 1]) - (up[x - 1] + 2 * up[x] + up[x + 1]));
    }
    out[0][0] = out[0][1] = out[MVO_WIDTH - 1][0] = out[MVO_WIDTH - 1][1] = 0;
}

/* The structure tensor's products xx, xy, yy summed down column x of three gradient rows. */
static inline void column_sums(const int16_t (*g0)[2], const int16_t (*g1)[2], const int16_t (*g2)[2], int x,
                               int32_t out[3])
{
    const int32_t a0 = g0[x][0], b0 = g0[x][1], a1 = g1[x][0], b1 = g1[x][1], a2 = g2[x][0], b2 = g2[x][1];
    out[0] = a0 * a0 + a1 * a1 + a2 * a2;
    out[1] = a0 * b0 + a1 * b1 + a2 * b2;
    out[2] = b0 * b0 + b1 * b1 + b2 * b2;
}

/* mvo_root(v) for v = 0 or v >= 1, as sums of squared integers are: the same operations without its branches, so the
 * same result, and four of them can overlap in the FPU's pipeline. At v = 0 the reciprocal guess stays finite and v
 * times it is 0, which mvo_root returns there too. */
static inline float root_of_squares(float v)
{
    union {
        float f;
        uint32_t u;
    } guess;
    const float half = 0.5f * v;
    float y;
    guess.f = v;
    guess.u = 0x5F375A86u - (guess.u >> 1);
    y = guess.f;
    y = y * (1.5f - half * y * y);
    y = y * (1.5f - half * y * y);
    y = y * (1.5f - half * y * y);
    return v * y;
}

static int32_t float_bits(float f)
{
    union {
        float f;
        int32_t i;
    } u;
    u.f = f;
    return u.i;
}

/* Smallest eigenvalue of the 3x3-summed structure tensor for row y (needs gradient rows y-1..y+1 in the ring), as float
 * bits, four pixels at a time. */
static void response_row(const mvo_detector_t *d, int y, int32_t out[MVO_WIDTH])
{
    const int16_t(*g0)[2] = d->gradients[(y - 1) % 3], (*g1)[2] = d->gradients[y % 3];
    const int16_t(*g2)[2] = d->gradients[(y + 1) % 3];
    int32_t column[6][3]; /* the product sums of columns x - 1 .. x + 4 */
    int x, j, k;
    out[0] = out[1] = out[MVO_WIDTH - 2] = out[MVO_WIDTH - 1] = 0;
    column_sums(g0, g1, g2, 1, column[4]);
    column_sums(g0, g1, g2, 2, column[5]);
    for (x = 2; x < MVO_WIDTH - 2; x += 4) { /* MVO_WIDTH - 4 pixels: a multiple of 4 */
        float a[4], b[4], c[4], v[4];
        for (k = 0; k < 3; ++k) {
            column[0][k] = column[4][k];
            column[1][k] = column[5][k];
        }
#pragma GCC unroll 4 /* the four pixels' independent chains, so that they can overlap */
        for (j = 0; j < 4; ++j) {
            column_sums(g0, g1, g2, x + 1 + j, column[2 + j]);
        }
#pragma GCC unroll 4
        for (j = 0; j < 4; ++j) {
            float diff;
            a[j] = (float)(column[j][0] + column[j + 1][0] + column[j + 2][0]);
            b[j] = (float)(column[j][1] + column[j + 1][1] + column[j + 2][1]);
            c[j] = (float)(column[j][2] + column[j + 1][2] + column[j + 2][2]);
            diff = a[j] - c[j];
            v[j] = diff * diff + 4.0f * b[j] * b[j];
        }
#pragma GCC unroll 4
        for (j = 0; j < 4; ++j) {
            out[x + j] = float_bits(0.5f * ((a[j] + c[j]) - root_of_squares(v[j])));
        }
    }
}

static int candidate_less(const mvo_candidate_t *p, const mvo_candidate_t *q) /* weaker first; ties by position */
{
    if (p->r != q->r) {
        return p->r < q->r;
    }
    if (p->y != q->y) {
        return p->y > q->y;
    }
    return p->x > q->x;
}

static void heap_down(mvo_candidate_t *heap, int size, int at)
{
    for (;;) {
        int left = 2 * at + 1, right = left + 1, low = at;
        mvo_candidate_t t;
        if (left < size && candidate_less(&heap[left], &heap[low])) {
            low = left;
        }
        if (right < size && candidate_less(&heap[right], &heap[low])) {
            low = right;
        }
        if (low == at) {
            return;
        }
        t = heap[at];
        heap[at] = heap[low];
        heap[low] = t;
        at = low;
    }
}

/* Sift-down in a heap with the strongest on top (the selection's). */
static void strongest_down(mvo_candidate_t *heap, int size, int at)
{
    for (;;) {
        int left = 2 * at + 1, right = left + 1, high = at;
        mvo_candidate_t t;
        if (left < size && candidate_less(&heap[high], &heap[left])) {
            high = left;
        }
        if (right < size && candidate_less(&heap[high], &heap[right])) {
            high = right;
        }
        if (high == at) {
            return;
        }
        t = heap[at];
        heap[at] = heap[high];
        heap[high] = t;
        at = high;
    }
}

/* Keeps c among the heap's MVO_MAX_CANDIDATES strongest. */
static void heap_add(mvo_candidate_t *heap, int *size, mvo_candidate_t c)
{
    int at;
    if (*size == MVO_MAX_CANDIDATES) {
        if (!candidate_less(&heap[0], &c)) {
            return;
        }
        heap[0] = c;
        heap_down(heap, *size, 0);
        return;
    }
    at = (*size)++;
    heap[at] = c;
    while (at > 0 && candidate_less(&heap[at], &heap[(at - 1) / 2])) {
        mvo_candidate_t t = heap[at];
        heap[at] = heap[(at - 1) / 2];
        heap[(at - 1) / 2] = t;
        at = (at - 1) / 2;
    }
}

static int grid_free(const mvo_t *s, float x, float y)
{
    int gx = (int)(x / MVO_CELL), gy = (int)(y / MVO_CELL), i, j, k;
    for (j = gy - 1; j <= gy + 1; ++j) {
        for (i = gx - 1; i <= gx + 1; ++i) {
            if (i < 0 || j < 0 || i >= MVO_GRID_W || j >= MVO_GRID_H) {
                continue;
            }
            for (k = 0; k < s_grid_count[j][i]; ++k) {
                const mvo_feature_t *f = &s->feature[s_grid[j][i][k]];
                float dx = f->x - x, dy = f->y - y;
                if (dx * dx + dy * dy < MVO_MIN_DISTANCE * MVO_MIN_DISTANCE) {
                    return 0;
                }
            }
        }
    }
    return 1;
}

static void grid_add(int index, float x, float y)
{
    int gx = (int)(x / MVO_CELL), gy = (int)(y / MVO_CELL);
    if (gx < 0 || gy < 0 || gx >= MVO_GRID_W || gy >= MVO_GRID_H || s_grid_count[gy][gx] >= MVO_CELL_CAPACITY) {
        return;
    }
    s_grid[gy][gx][s_grid_count[gy][gx]++] = (mvo_grid_index_t)index;
}

/* Half `part` of the corner search: the responses of its rows, their strongest local maxima (into its heap) and the
 * strongest response. The first half looks for maxima in rows 3 .. MVO_DETECT_SPLIT - 1, the second from
 * MVO_DETECT_SPLIT to MVO_HEIGHT - 4; each computes the responses one row beyond, which both then hold. */
static void detect_half(void *context, int part)
{
    const mvo_t *s = context;
    const uint8_t *img = s->pyramid[s->current];
    mvo_detector_t *d = &s_detector[part];
    const int first = part ? MVO_DETECT_SPLIT : 3, last = part ? MVO_HEIGHT - 3 : MVO_DETECT_SPLIT;
    int32_t best = 0; /* the bits of 0.0f */
    int y, x;
    d->size = 0;
    sobel_row(img, first - 2, d->gradients[(first - 2) % 3]);
    sobel_row(img, first - 1, d->gradients[(first - 1) % 3]);
    /* The responses are compared as their bits, integers: for a positive float r and any other float q, r < q as
     * integers exactly when as floats (a negative q has the sign bit, a negative integer), and a float is above zero
     * exactly when its bits are. */
    for (y = first - 1; y <= last; ++y) {
        const int32_t *row = d->response[y % 3];
        sobel_row(img, y + 1, d->gradients[(y + 1) % 3]);
        response_row(d, y, d->response[y % 3]);
        for (x = 2; x < MVO_WIDTH - 2; ++x) {
            best = row[x] > best ? row[x] : best;
        }
        if (y > first) { /* local maxima of row y - 1: rows y - 2 .. y are in the ring */
            const int32_t *above = d->response[(y - 2) % 3], *centre = d->response[(y - 1) % 3];
            for (x = 3; x < MVO_WIDTH - 3; ++x) {
                const int32_t r = centre[x];
                union {
                    int32_t i;
                    float f;
                } value;
                mvo_candidate_t c;
                if (r <= 0) {
                    continue;
                }
                if (r < centre[x - 1] || r < centre[x + 1] || r < above[x - 1] || r < above[x] || r < above[x + 1]
                    || r < row[x - 1] || r < row[x] || r < row[x + 1]) {
                    continue;
                }
                value.i = r;
                c.x = (int16_t)x;
                c.y = (int16_t)(y - 1);
                c.r = value.f;
                heap_add(d->heap, &d->size, c);
            }
        }
    }
    {
        union {
            int32_t i;
            float f;
        } value;
        value.i = best;
        d->best = value.f;
    }
}

/* New corners at least MVO_MIN_DISTANCE from each other and from the tracked features, strongest first. Each half of
 * the frame keeps its MVO_MAX_CANDIDATES strongest local maxima (at the same time when the caller gave s->parallel);
 * merged, they are the frame's strongest, since a half holds at most all of those. The quality threshold, which needs
 * the strongest response of the whole frame, is applied to the sorted candidates afterwards: that keeps the same
 * corners as thresholding first, since everything above the threshold outranks everything below it. */
void mvo_detect(mvo_t *s)
{
    mvo_candidate_t *heap = s_detector[0].heap;
    float best, threshold;
    int size, k;
    if (s->parallel) {
        s->parallel(detect_half, s);
    } else {
        detect_half(s, 0);
        detect_half(s, 1);
    }
    best = s_detector[1].best > s_detector[0].best ? s_detector[1].best : s_detector[0].best;
    size = s_detector[0].size;
    for (k = 0; k < s_detector[1].size; ++k) {
        heap_add(heap, &size, s_detector[1].heap[k]);
    }
    threshold = MVO_QUALITY * best;
    memset(s_grid_count, 0, sizeof s_grid_count);
    for (k = 0; k < s->count; ++k) {
        grid_add(k, s->feature[k].x, s->feature[k].y);
    }
    for (k = size / 2 - 1; k >= 0; --k) { /* the strongest on top now, then taken one by one */
        strongest_down(heap, size, k);
    }
    while (size > 0 && s->count < MVO_MAX_FEATURES && heap[0].r > threshold) {
        const float x = (float)heap[0].x, y = (float)heap[0].y;
        heap[0] = heap[--size];
        strongest_down(heap, size, 0);
        if (grid_free(s, x, y)) {
            mvo_feature_t *f = &s->feature[s->count];
            f->x = x;
            f->y = y;
            f->has_point = 0;
            f->id = s->next_id++;
            grid_add(s->count, x, y);
            ++s->count;
        }
    }
}

/* ---- pyramidal Lucas-Kanade -------------------------------------------------------------------------------- */

void mvo_block_c(uint8_t *out, int size, const uint8_t *img, int w, int h, const int origin[2])
{
    const int x0 = origin[0], y0 = origin[1], stride = (size + 15) & ~15;
    /* columns left of the image and right of it (size < w at every level, so they never meet) */
    const int left = x0 < 0 ? (-x0 < size ? -x0 : size) : 0;
    const int right = x0 + size > w ? (x0 + size - w < size ? x0 + size - w : size) : 0;
    int i, j;
    for (j = 0; j < size; ++j) {
        const uint8_t *row = img + (y0 + j < 0 ? 0 : (y0 + j >= h ? h - 1 : y0 + j)) * w;
        uint8_t *line = out + j * stride;
        for (i = 0; i < left; ++i) {
            line[i] = row[0];
        }
        memcpy(line + left, row + x0 + left, (size_t)(size - left - right));
        for (i = size - right; i < size; ++i) {
            line[i] = row[w - 1];
        }
    }
}

/* The size x size block of img whose top-left pixel is (x0, y0): a pointer into img when the block lies inside, else
 * a copy in the tracker's block with coordinates clamped to the image (kernels->block, kept while the same block is
 * asked for again); its row stride goes to *stride. */
static const uint8_t *block_at(mvo_tracker_t *tr, const mvo_kernels_t *kernels, const uint8_t *img, int w, int h,
                               int x0, int y0, int size, int *stride)
{
    int origin[2];
    if (x0 >= 0 && y0 >= 0 && x0 + size <= w && y0 + size <= h) {
        *stride = w;
        return img + y0 * w + x0;
    }
    *stride = (size + 15) & ~15;
    if (tr->copied == img && tr->copied_x == x0 && tr->copied_y == y0 && tr->copied_size == size) {
        return tr->block; /* an iteration that stayed on the same whole pixel */
    }
    origin[0] = x0;
    origin[1] = y0;
    kernels->block(tr->block, size, img, w, h, origin);
    tr->copied = img;
    tr->copied_x = x0;
    tr->copied_y = y0;
    tr->copied_size = size;
    return tr->block;
}

/* Bilinear weights of the fraction (fx, fy) in fixed point, summing to exactly 1 << MVO_W_BITS. */
static void bilinear_weights(float fx, float fy, int16_t weights[4])
{
    const float one = (float)(1 << MVO_W_BITS);
    weights[0] = (int16_t)((1.0f - fx) * (1.0f - fy) * one + 0.5f);
    weights[1] = (int16_t)(fx * (1.0f - fy) * one + 0.5f);
    weights[2] = (int16_t)((1.0f - fx) * fy * one + 0.5f);
    weights[3] = (int16_t)((1 << MVO_W_BITS) - weights[0] - weights[1] - weights[2]);
}

/* The 2x2 pixels from r interpolated in Q7, each product shifted on its own (see mvo_kernels_t): at most
 * 255 << 7, since the weights sum to 1 << 14. */
static int32_t interpolate(const uint8_t *r, int stride, const int16_t w[4])
{
    return ((r[0] * w[0]) >> 7) + ((r[1] * w[1]) >> 7) + ((r[stride] * w[2]) >> 7) + ((r[stride + 1] * w[3]) >> 7);
}

void mvo_lk_template_c(mvo_window_t *window, const uint8_t *src, int stride, const int16_t weights[4], int64_t g[3])
{
    int64_t g11 = 0, g12 = 0, g22 = 0;
    int i, j;
    for (j = 0; j < MVO_SIDE; ++j) {
        for (i = 0; i < MVO_LANES; ++i) {
            const int at = j * MVO_LANES + i;
            int32_t value = 0, dx = 0, dy = 0;
            if (i < MVO_SIDE) {
                const uint8_t *r = src + (j + 1) * stride + (i + 1); /* the window's pixel (i, j) */
                value = interpolate(r, stride, weights);
                dx = interpolate(r + 1, stride, weights) - interpolate(r - 1, stride, weights);
                dy = interpolate(r + stride, stride, weights) - interpolate(r - stride, stride, weights);
                g11 += dx * dx; /* each product below 2^31 */
                g12 += dx * dy;
                g22 += dy * dy;
            }
            window->patch[at] = (int16_t)value;
            window->gx[at] = (int16_t)dx;
            window->gy[at] = (int16_t)dy;
        }
    }
    g[0] = g11;
    g[1] = g12;
    g[2] = g22;
}

void mvo_lk_sums_c(const mvo_window_t *window, const uint8_t *src, int stride, const int16_t weights[4],
                   int64_t sums[2])
{
    int64_t s1 = 0, s2 = 0;
    int i, j;
    for (j = 0; j < MVO_SIDE; ++j) {
        for (i = 0; i < MVO_SIDE; ++i) {
            const int at = j * MVO_LANES + i;
            const int32_t diff = window->patch[at] - interpolate(src + j * stride + i, stride, weights);
            s1 += diff * window->gx[at];
            s2 += diff * window->gy[at];
        }
    }
    sums[0] = s1;
    sums[1] = s2;
}

static const mvo_kernels_t kPortable = {mvo_lk_template_c, mvo_lk_sums_c, mvo_block_c};

/* v as the nearest float, for |v| < 2^43: two halves that convert exactly, then one rounding (no 64-bit conversion
 * routine whose rounding could differ between platforms). */
static float wide_float(int64_t v)
{
    const int32_t high = (int32_t)(v >> 20);
    const int32_t low = (int32_t)(v - (int64_t)high * 1048576);
    return (float)high * 1048576.0f + (float)low;
}

/* Track (x, y) from pyramid a to pyramid b; 1 and the position in b when it converges inside the frame. Fixed point
 * much as in OpenCV's tracker: the template and its central differences are interpolated once per level, the target
 * on every iteration, both in Q7 by the kernels (s->kernels or the portable ones); their sums are exact integers and
 * the solve runs in float. */
static int lk_point(const mvo_t *s, mvo_tracker_t *tr, const uint8_t *pa, const uint8_t *pb, float x, float y,
                    float *ox, float *oy)
{
    const mvo_kernels_t *kernels = s->kernels ? s->kernels : &kPortable;
    float dx = 0.0f, dy = 0.0f;
    int level;
    for (level = MVO_LEVELS - 1; level >= 0; --level) {
        const uint8_t *a = pa + kOffset[level], *b = pb + kOffset[level], *src;
        const int w = kWidth[level], h = kHeight[level];
        const float px = x * kScale[level], py = y * kScale[level]; /* exact: powers of two */
        const int x0 = mvo_floor(px), y0 = mvo_floor(py);
        int16_t weights[4];
        int64_t g[3], sums[2];
        float g11, g12, g22, det, min_eig, inverse_det, previous_x = 0.0f, previous_y = 0.0f;
        int iteration, stride;
        if (level < MVO_LEVELS - 1) {
            dx *= 2.0f;
            dy *= 2.0f;
        }
        bilinear_weights(px - (float)x0, py - (float)y0, weights);
        src = block_at(tr, kernels, a, w, h, x0 - MVO_WINDOW - 1, y0 - MVO_WINDOW - 1, MVO_RAW, &stride);
        kernels->lk_template(&tr->window, src, stride, weights, g);
        g11 = wide_float(g[0]) * (1.0f / 65536.0f); /* Q7 squared, and full differences: half differences in pixels */
        g12 = wide_float(g[1]) * (1.0f / 65536.0f);
        g22 = wide_float(g[2]) * (1.0f / 65536.0f);
        det = g11 * g22 - g12 * g12;
        min_eig = 0.5f * ((g11 + g22) - mvo_root((g11 - g22) * (g11 - g22) + 4.0f * g12 * g12)) / (float)(MVO_SIDE * MVO_SIDE);
        if (min_eig < MVO_MIN_EIGEN || det < 1e-12f) {
            return 0;
        }
        inverse_det = 1.0f / det;
        for (iteration = 0; iteration < MVO_LK_ITERATIONS; ++iteration) {
            const float qx = px + dx, qy = py + dy;
            const int bx = mvo_floor(qx), by = mvo_floor(qy);
            float b1, b2, step_x, step_y;
            if (qx < -(float)MVO_WINDOW || qy < -(float)MVO_WINDOW || qx > (float)(w + MVO_WINDOW) || qy > (float)(h + MVO_WINDOW)) {
                return 0;
            }
            bilinear_weights(qx - (float)bx, qy - (float)by, weights);
            src = block_at(tr, kernels, b, w, h, bx - MVO_WINDOW, by - MVO_WINDOW, MVO_SIDE + 1, &stride);
            kernels->lk_sums(&tr->window, src, stride, weights, sums);
            b1 = wide_float(sums[0]) * (1.0f / 32768.0f); /* Q7 times Q7 full differences: pixels */
            b2 = wide_float(sums[1]) * (1.0f / 32768.0f);
            step_x = (g22 * b1 - g12 * b2) * inverse_det;
            step_y = (g11 * b2 - g12 * b1) * inverse_det;
            dx += step_x;
            dy += step_y;
            if (step_x * step_x + step_y * step_y < (level ? MVO_LK_COARSE_EPSILON * MVO_LK_COARSE_EPSILON : MVO_LK_EPSILON * MVO_LK_EPSILON)) {
                break;
            }
            if (iteration > 0 && mvo_abs(step_x + previous_x) < MVO_LK_SWING
                && mvo_abs(step_y + previous_y) < MVO_LK_SWING) {
                dx -= 0.5f * step_x; /* swinging back and forth: the middle, as OpenCV's tracker takes */
                dy -= 0.5f * step_y;
                break;
            }
            previous_x = step_x;
            previous_y = step_y;
        }
    }
    *ox = x + dx;
    *oy = y + dy;
    return *ox >= 0.0f && *oy >= 0.0f && *ox < (float)MVO_WIDTH && *oy < (float)MVO_HEIGHT;
}

typedef struct {
    const mvo_t *s;
    const uint8_t *previous, *current;
} mvo_track_job_t;

/* Tracks half `part` of the features into the newest frame, forward and back, into s_went and s_tracked. */
static void track_half(void *context, int part)
{
    const mvo_track_job_t *job = context;
    const int begin = part * job->s->count / 2, end = (part + 1) * job->s->count / 2;
    int k;
    for (k = begin; k < end; ++k) {
        const mvo_feature_t *f = &job->s->feature[k];
        float back_x, back_y, ex, ey;
        s_tracked[k] = 0;
        if (!lk_point(job->s, &s_tracker[part], job->previous, job->current, f->x, f->y, &s_went[k][0], &s_went[k][1])
            || !lk_point(job->s, &s_tracker[part], job->current, job->previous, s_went[k][0], s_went[k][1], &back_x,
                         &back_y)) {
            continue;
        }
        ex = back_x - f->x;
        ey = back_y - f->y;
        s_tracked[k] = (uint8_t)(ex * ex + ey * ey < MVO_FB_MAX * MVO_FB_MAX);
    }
}

/* Track every feature into the newest frame with the forward-backward check, in two halves (at the same time when
 * the caller gave s->parallel); keeps the survivors in order and their previous positions in mvo_pa. Returns the
 * survivors' count. */
int mvo_track_features(mvo_t *s, const uint8_t *previous, const uint8_t *current)
{
    mvo_track_job_t job;
    int k, kept = 0;
    job.s = s;
    job.previous = previous;
    job.current = current;
    s_tracker[0].copied = s_tracker[1].copied = 0; /* new frames in the pyramids: no copied block is current */
    if (s->parallel) {
        s->parallel(track_half, &job);
    } else {
        track_half(&job, 0);
        track_half(&job, 1);
    }
    for (k = 0; k < s->count; ++k) {
        mvo_feature_t f = s->feature[k];
        if (!s_tracked[k]) {
            continue;
        }
        mvo_pa[kept][0] = f.x;
        mvo_pa[kept][1] = f.y;
        f.x = s_went[k][0];
        f.y = s_went[k][1];
        s->feature[kept++] = f;
    }
    s->count = kept;
    return kept;
}
