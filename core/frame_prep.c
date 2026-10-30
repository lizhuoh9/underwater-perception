/* See frame_prep.h. */
#include "frame_prep.h"

#include <stddef.h>
#include <string.h>

/* Positions along a row are in 1/FP_WIDTH source pixels: source pixel i spans [i FP_WIDTH, (i + 1) FP_WIDTH) and
 * output column c spans [c w, (c + 1) w), so the overlaps weigh the source pixels and add up to w for each column.
 * Rows alike with FP_HEIGHT and h. An output pixel's weights then add up to w h, and its weighted sum is at most
 * 255 w h < 2^32 for sides up to FP_MAX_SIDE. */

/* One source row onto the 320 columns: sums[c] = the sum over the source pixels that column c covers of overlap times
 * grey value. Since w >= FP_WIDTH, a source pixel holds at most one column's right edge, which splits it between that
 * column and the next. */
static void row_sums(const uint8_t *row, uint32_t width, uint32_t step, uint32_t sums[FP_WIDTH])
{
    size_t at = 0; /* the next source pixel's byte, an index so that nothing points past the row */
    uint32_t c, right = 0, next = FP_WIDTH, carry = 0; /* next: where that pixel ends */
    for (c = 0; c < FP_WIDTH; ++c) {
        uint32_t whole = 0, sum = carry;
        right += width;
        for (; next <= right; next += FP_WIDTH, at += step) {
            whole += row[at];
        }
        sum += FP_WIDTH * whole;
        carry = 0;
        if (next - FP_WIDTH < right) { /* that pixel starts inside this column and ends in the next */
            const uint32_t grey = row[at];
            sum += (right + FP_WIDTH - next) * grey;
            carry = (next - right) * grey;
            next += FP_WIDTH;
            at += step;
        }
        sums[c] = sum;
    }
}

int fp_area_resize(const uint8_t *src, int w, int h, int stride, int step, uint8_t *out, int row_begin, int row_end)
{
    uint32_t acc[FP_WIDTH], sums[FP_WIDTH];
    const uint32_t width = (uint32_t)w, height = (uint32_t)h, total = width * height;
    uint32_t summed = UINT32_MAX; /* the source row whose sums are in sums */
    int r, c;
    if (src == NULL || out == NULL || w < FP_WIDTH || h < FP_HEIGHT || w > FP_MAX_SIDE || h > FP_MAX_SIDE || step < 1
        || step > 4 || stride < (w - 1) * step + 1 || row_begin < 0 || row_begin > row_end || row_end > FP_HEIGHT) {
        return 0;
    }
    for (r = row_begin; r < row_end; ++r) {
        const uint32_t top = (uint32_t)r * height, bottom = top + height;
        uint32_t j;
        memset(acc, 0, sizeof acc);
        for (j = top / FP_HEIGHT; j * FP_HEIGHT < bottom; ++j) {
            const uint32_t row_top = j * FP_HEIGHT, row_bottom = row_top + FP_HEIGHT;
            const uint32_t wy = (bottom < row_bottom ? bottom : row_bottom) - (top > row_top ? top : row_top);
            if (j != summed) { /* a row split between two output rows is summed once */
                row_sums(src + (size_t)j * (size_t)stride, width, (uint32_t)step, sums);
                summed = j;
            }
            for (c = 0; c < FP_WIDTH; ++c) {
                acc[c] += wy * sums[c];
            }
        }
        for (c = 0; c < FP_WIDTH; ++c) {
            out[r * FP_WIDTH + c] = (uint8_t)((acc[c] + total / 2) / total);
        }
    }
    return 1;
}
