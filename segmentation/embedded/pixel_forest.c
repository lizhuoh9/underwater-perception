#include "pixel_forest.h"

#include <string.h>

#include "pixel_forest_model.h"

#if PF_TREE_COUNT < 1 || PF_TREE_COUNT > 65535
#error "PF_TREE_COUNT must be in 1..65535"
#endif
#if PF_NODE_COUNT < 1 || PF_NODE_COUNT > 65536
#error "PF_NODE_COUNT must be in 1..65536"
#endif
#if PF_THRESHOLD_Q < 0 || PF_THRESHOLD_Q > 256
#error "PF_THRESHOLD_Q must be in 0..256"
#endif
#if PF_MEDIAN_SIZE != 1 && PF_MEDIAN_SIZE != 3
#error "PF_MEDIAN_SIZE must be 1 or 3"
#endif

/* A compile-time assertion also protects the generated model's byte layout. */
typedef char pf_node_must_be_eight_bytes[sizeof(PixelForestNode) == 8 ? 1 : -1];

typedef struct PFPlanes {
    uint8_t *luma;
    uint8_t *rfrac;
    uint8_t *gfrac;
    uint8_t *mean_luma3;
    uint8_t *mean_luma9;
    uint8_t *mean_rfrac9;
    uint8_t *mean_gfrac9;
    uint32_t *integral;
} PFPlanes;

static int valid_dimensions(int width, int height)
{
    return width >= 1 && width <= PF_MAX_WIDTH &&
           height >= 1 && height <= PF_MAX_HEIGHT;
}

size_t pf_workspace_bytes(int width, int height)
{
    size_t planes;
    if (!valid_dimensions(width, height)) {
        return 0;
    }
    planes = 7u * (size_t)width * (size_t)height;
    planes = (planes + 3u) & ~(size_t)3u;
    return planes + 4u * (size_t)(width + 9) * (size_t)(height + 9);
}

static int valid_span(const void *ptr, size_t bytes)
{
    return ptr != NULL &&
           (bytes == 0 || (uintptr_t)ptr <= UINTPTR_MAX - (bytes - 1u));
}

static int overlaps(const void *a, size_t a_bytes,
                    const void *b, size_t b_bytes)
{
    uintptr_t ap = (uintptr_t)a;
    uintptr_t bp = (uintptr_t)b;
    if (a_bytes == 0 || b_bytes == 0) {
        return 0;
    }
    return ap <= bp ? bp - ap < a_bytes : ap - bp < b_bytes;
}

static int check_buffers(const uint16_t *rgb565, int width, int height,
                         uint8_t *output, size_t output_bytes,
                         void *workspace, size_t workspace_bytes)
{
    size_t required = pf_workspace_bytes(width, height);
    size_t input_bytes;
    if (required == 0) {
        return PF_ERR_INVALID;
    }
    input_bytes = 2u * (size_t)width * (size_t)height;
    if (!valid_span(rgb565, input_bytes) ||
        !valid_span(output, output_bytes) ||
        !valid_span(workspace, required)) {
        return PF_ERR_INVALID;
    }
    if (((uintptr_t)workspace & 3u) != 0 ||
        ((uintptr_t)rgb565 & 1u) != 0) {
        return PF_ERR_ALIGNMENT;
    }
    if (workspace_bytes < required) {
        return PF_ERR_WORKSPACE;
    }
    if (overlaps(rgb565, input_bytes, output, output_bytes) ||
        overlaps(rgb565, input_bytes, workspace, required) ||
        overlaps(output, output_bytes, workspace, required)) {
        return PF_ERR_OVERLAP;
    }
    return PF_OK;
}

static int clamp_coordinate(int coordinate, int length)
{
    if (coordinate < 0) {
        return 0;
    }
    return coordinate >= length ? length - 1 : coordinate;
}

static void unpack_rgb565(uint16_t packed, unsigned *r,
                          unsigned *g, unsigned *b)
{
    *r = ((unsigned)(packed >> 11) * 255u) / 31u;
    *g = ((unsigned)((packed >> 5) & 63u) * 255u) / 63u;
    *b = ((unsigned)(packed & 31u) * 255u) / 31u;
}

/* Radius-four replicated padding lets one integral image serve 3x3 and 9x9. */
static void build_integral(const uint8_t *plane, int width, int height,
                           uint32_t *integral)
{
    int px;
    int py;
    int stride = width + 9;
    memset(integral, 0, (size_t)stride * sizeof(*integral));
    for (py = 0; py < height + 8; ++py) {
        int sy = clamp_coordinate(py - 4, height);
        uint32_t row_sum = 0;
        size_t row = (size_t)(py + 1) * (size_t)stride;
        size_t previous = (size_t)py * (size_t)stride;
        integral[row] = 0;
        for (px = 0; px < width + 8; ++px) {
            int sx = clamp_coordinate(px - 4, width);
            row_sum += plane[(size_t)sy * (size_t)width + (size_t)sx];
            integral[row + (size_t)px + 1u] =
                integral[previous + (size_t)px + 1u] + row_sum;
        }
    }
}

static void box_mean(const uint32_t *integral, int width, int height,
                     int radius, uint8_t *output)
{
    int x;
    int y;
    int stride = width + 9;
    unsigned side = (unsigned)(2 * radius + 1);
    unsigned area = side * side;
    for (y = 0; y < height; ++y) {
        int top = y + 4 - radius;
        int bottom = y + 4 + radius + 1;
        for (x = 0; x < width; ++x) {
            int left = x + 4 - radius;
            int right = x + 4 + radius + 1;
            uint32_t sum =
                integral[(size_t)bottom * (size_t)stride + (size_t)right] +
                integral[(size_t)top * (size_t)stride + (size_t)left] -
                integral[(size_t)top * (size_t)stride + (size_t)right] -
                integral[(size_t)bottom * (size_t)stride + (size_t)left];
            output[(size_t)y * (size_t)width + (size_t)x] =
                (uint8_t)(sum / area);
        }
    }
}

static PFPlanes prepare_planes(const uint16_t *rgb565, int width, int height,
                                void *workspace)
{
    PFPlanes p;
    size_t n = (size_t)width * (size_t)height;
    size_t i;
    uint8_t *base = (uint8_t *)workspace;
    p.luma = base;
    p.rfrac = base + n;
    p.gfrac = base + 2u * n;
    p.mean_luma3 = base + 3u * n;
    p.mean_luma9 = base + 4u * n;
    p.mean_rfrac9 = base + 5u * n;
    p.mean_gfrac9 = base + 6u * n;
    p.integral = (uint32_t *)(void *)(base + ((7u * n + 3u) & ~(size_t)3u));
    for (i = 0; i < n; ++i) {
        unsigned r;
        unsigned g;
        unsigned b;
        unsigned total;
        unpack_rgb565(rgb565[i], &r, &g, &b);
        total = r + g + b;
        if (total == 0) {
            total = 1;
        }
        p.luma[i] = (uint8_t)((r + 2u * g + b) / 4u);
        p.rfrac[i] = (uint8_t)((255u * r) / total);
        p.gfrac[i] = (uint8_t)((255u * g) / total);
    }
    build_integral(p.luma, width, height, p.integral);
    box_mean(p.integral, width, height, 1, p.mean_luma3);
    box_mean(p.integral, width, height, 4, p.mean_luma9);
    build_integral(p.rfrac, width, height, p.integral);
    box_mean(p.integral, width, height, 4, p.mean_rfrac9);
    build_integral(p.gfrac, width, height, p.integral);
    box_mean(p.integral, width, height, 4, p.mean_gfrac9);
    return p;
}

static unsigned absolute_difference(unsigned a, unsigned b)
{
    return a >= b ? a - b : b - a;
}

static void pixel_features(const uint16_t *rgb565, const PFPlanes *p,
                           int width, int height, int x, int y,
                           uint8_t features[PF_FEATURE_COUNT])
{
    size_t i = (size_t)y * (size_t)width + (size_t)x;
    unsigned r;
    unsigned g;
    unsigned b;
    unsigned maximum;
    unsigned minimum;
    unsigned local_min = 255;
    unsigned local_max = 0;
    unsigned horizontal;
    unsigned vertical;
    int dx;
    int dy;
    unpack_rgb565(rgb565[i], &r, &g, &b);
    maximum = r > g ? r : g;
    maximum = maximum > b ? maximum : b;
    minimum = r < g ? r : g;
    minimum = minimum < b ? minimum : b;
    horizontal = absolute_difference(
        p->luma[(size_t)y * (size_t)width +
                (size_t)clamp_coordinate(x + 1, width)],
        p->luma[(size_t)y * (size_t)width +
                (size_t)clamp_coordinate(x - 1, width)]);
    vertical = absolute_difference(
        p->luma[(size_t)clamp_coordinate(y + 1, height) *
                (size_t)width + (size_t)x],
        p->luma[(size_t)clamp_coordinate(y - 1, height) *
                (size_t)width + (size_t)x]);
    for (dy = -2; dy <= 2; ++dy) {
        size_t row = (size_t)clamp_coordinate(y + dy, height) * (size_t)width;
        for (dx = -2; dx <= 2; ++dx) {
            unsigned value = p->luma[row +
                (size_t)clamp_coordinate(x + dx, width)];
            if (value < local_min) {
                local_min = value;
            }
            if (value > local_max) {
                local_max = value;
            }
        }
    }
    features[0] = (uint8_t)r;
    features[1] = (uint8_t)g;
    features[2] = (uint8_t)b;
    features[3] = p->rfrac[i];
    features[4] = p->gfrac[i];
    features[5] = p->luma[i];
    features[6] = (uint8_t)((255u * (maximum - minimum)) /
                           (maximum == 0 ? 1u : maximum));
    features[7] = (uint8_t)(((int)r - (int)g + 255) / 2);
    features[8] = (uint8_t)(((int)g - (int)b + 255) / 2);
    features[9] = p->mean_luma3[i];
    features[10] = p->mean_luma9[i];
    features[11] = (uint8_t)((horizontal + vertical) / 2u);
    features[12] = (uint8_t)absolute_difference(p->luma[i], p->mean_luma3[i]);
    features[13] = (uint8_t)(local_max - local_min);
    features[14] = p->mean_rfrac9[i];
    features[15] = p->mean_gfrac9[i];
}

/* The generated model is trusted: indices are global and leaf feature is 255. */
static uint8_t predict_one(const uint8_t features[PF_FEATURE_COUNT])
{
    uint32_t sum = 0;
    unsigned tree;
    for (tree = 0; tree < PF_TREE_COUNT; ++tree) {
        uint16_t index = PF_ROOTS[tree];
        const PixelForestNode *node = &PF_NODES[index];
        while (node->feature != 255u) {
            index = features[node->feature] <= node->threshold ?
                node->left : node->right;
            node = &PF_NODES[index];
        }
        sum += node->probability;
    }
    return (uint8_t)((sum + PF_TREE_COUNT / 2u) / PF_TREE_COUNT);
}

int pf_extract_features(const uint16_t *rgb565, int width, int height,
                        uint8_t *features, void *workspace,
                        size_t workspace_bytes)
{
    PFPlanes p;
    int x;
    int y;
    int status;
    if (!valid_dimensions(width, height)) {
        return PF_ERR_INVALID;
    }
    status = check_buffers(rgb565, width, height, features,
                          PF_FEATURE_COUNT * (size_t)width * (size_t)height,
                          workspace, workspace_bytes);
    if (status != PF_OK) {
        return status;
    }
    p = prepare_planes(rgb565, width, height, workspace);
    for (y = 0; y < height; ++y) {
        for (x = 0; x < width; ++x) {
            size_t i = (size_t)y * (size_t)width + (size_t)x;
            pixel_features(rgb565, &p, width, height, x, y,
                           features + i * PF_FEATURE_COUNT);
        }
    }
    return PF_OK;
}

int pf_predict_features(const uint8_t *rows, size_t count, uint8_t *scores)
{
    size_t i;
    size_t row_bytes;
    if (count > SIZE_MAX / PF_FEATURE_COUNT) {
        return PF_ERR_INVALID;
    }
    row_bytes = count * PF_FEATURE_COUNT;
    if (!valid_span(rows, row_bytes) || !valid_span(scores, count)) {
        return PF_ERR_INVALID;
    }
    if (overlaps(rows, row_bytes, scores, count)) {
        return PF_ERR_OVERLAP;
    }
    for (i = 0; i < count; ++i) {
        scores[i] = predict_one(rows + i * PF_FEATURE_COUNT);
    }
    return PF_OK;
}

#if PF_MEDIAN_SIZE == 3
static void median3(const uint8_t *input, int width, int height,
                    uint8_t *output)
{
    int x;
    int y;
    for (y = 0; y < height; ++y) {
        for (x = 0; x < width; ++x) {
            uint8_t values[9];
            int count = 0;
            int dx;
            int dy;
            int k;
            for (dy = -1; dy <= 1; ++dy) {
                size_t row = (size_t)clamp_coordinate(y + dy, height) *
                             (size_t)width;
                for (dx = -1; dx <= 1; ++dx) {
                    uint8_t value = input[row +
                        (size_t)clamp_coordinate(x + dx, width)];
                    k = count++;
                    while (k > 0 && values[k - 1] > value) {
                        values[k] = values[k - 1];
                        --k;
                    }
                    values[k] = value;
                }
            }
            output[(size_t)y * (size_t)width + (size_t)x] = values[4];
        }
    }
}
#endif

int pf_predict_rgb565(const uint16_t *rgb565, int width, int height,
                      uint8_t *scores, uint8_t *masks, void *workspace,
                      size_t workspace_bytes)
{
    PFPlanes p;
    size_t n;
    size_t required;
    size_t i;
    int x;
    int y;
    int status;
    uint8_t features[PF_FEATURE_COUNT];
    if (!valid_dimensions(width, height)) {
        return PF_ERR_INVALID;
    }
    n = (size_t)width * (size_t)height;
    required = pf_workspace_bytes(width, height);
    status = check_buffers(rgb565, width, height, scores, n,
                          workspace, workspace_bytes);
    if (status != PF_OK) {
        return status;
    }
    if (!valid_span(masks, n)) {
        return PF_ERR_INVALID;
    }
    if (overlaps(masks, n, rgb565, 2u * n) ||
        overlaps(masks, n, scores, n) ||
        overlaps(masks, n, workspace, required)) {
        return PF_ERR_OVERLAP;
    }
    p = prepare_planes(rgb565, width, height, workspace);
    for (y = 0; y < height; ++y) {
        for (x = 0; x < width; ++x) {
            i = (size_t)y * (size_t)width + (size_t)x;
            pixel_features(rgb565, &p, width, height, x, y, features);
            scores[i] = predict_one(features);
        }
    }
#if PF_MEDIAN_SIZE == 3
    /* Feature planes are dead now, so the first one is safe median scratch. */
    median3(scores, width, height, p.luma);
    memcpy(scores, p.luma, n);
#endif
#if PF_THRESHOLD_Q == 0
    memset(masks, 1, n);
#elif PF_THRESHOLD_Q == 256
    memset(masks, 0, n);
#else
    for (i = 0; i < n; ++i) {
        masks[i] = (uint8_t)(scores[i] >= PF_THRESHOLD_Q);
    }
#endif
    return PF_OK;
}
