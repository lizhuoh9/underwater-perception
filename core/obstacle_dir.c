/* See obstacle_dir.h. */
#include "obstacle_dir.h"

#include <float.h>
#include <stddef.h>
#include <string.h>

/* No byte of the state is padding: the board hashes it whole. */
typedef char od_state_has_no_padding[(offsetof(od_state_t, choice) == 3 * OD_DIRECTIONS + 3
                                      && sizeof(od_state_t) == offsetof(od_state_t, frames) + 4) ? 1 : -1];

uint32_t od_state_size(void) { return (uint32_t)sizeof(od_state_t); }

uint32_t od_layout_size(void) { return (uint32_t)sizeof(od_layout_t); }

static int finite_float(float value) { return value >= -FLT_MAX && value <= FLT_MAX; }

/* The azimuth sector of a ray's x / z, left to right. */
static uint8_t sector_of(float x)
{
    if (x < -OD_OUTER) {
        return 0;
    }
    if (x < -OD_INNER) {
        return 1;
    }
    if (x <= OD_INNER) {
        return 2;
    }
    return x <= OD_OUTER ? 3 : 4;
}

/* The elevation band of a ray's y / z (y down), up to down. */
static uint8_t band_of(float y)
{
    if (y < -OD_INNER) {
        return 0;
    }
    return y <= OD_INNER ? 1 : 2;
}

int od_direction_of(float x_over_z, float y_over_z) { return band_of(y_over_z) * OD_COLUMNS + sector_of(x_over_z); }

int od_layout(od_layout_t *layout, const mvo_camera_t *camera, int mask_w, int mask_h)
{
    uint16_t columns[OD_COLUMNS] = {0}, rows[OD_ROWS] = {0};
    int i, j;
    if (!layout) {
        return 0;
    }
    memset(layout, 0, sizeof *layout);
    if (!camera || !(camera->fx > 0.0f) || !(camera->fy > 0.0f) || !finite_float(camera->fx)
            || !finite_float(camera->fy) || !finite_float(camera->cx) || !finite_float(camera->cy)
            || mask_w < 1 || mask_h < 1 || mask_w > OD_MAX_SIDE || mask_h > OD_MAX_SIDE) {
        return 0;
    }
    for (i = 0; i < mask_w; ++i) { /* the ray through the cell's centre on the 320x240 image, on the principal row */
        float p[2];
        p[0] = ((float)i + 0.5f) * (float)MVO_WIDTH / (float)mask_w;
        p[1] = camera->cy;
        if (!mvo_undistort(camera, p, p)) {
            memset(layout, 0, sizeof *layout);
            return 0;
        }
        layout->sector[i] = sector_of((p[0] - camera->cx) / camera->fx);
        ++columns[layout->sector[i]];
    }
    for (i = 0; i < mask_h; ++i) { /* on the principal column */
        float p[2];
        p[0] = camera->cx;
        p[1] = ((float)i + 0.5f) * (float)MVO_HEIGHT / (float)mask_h;
        if (!mvo_undistort(camera, p, p)) {
            memset(layout, 0, sizeof *layout);
            return 0;
        }
        layout->band[i] = band_of((p[1] - camera->cy) / camera->fy);
        ++rows[layout->band[i]];
    }
    for (i = 0; i < OD_ROWS; ++i) {
        for (j = 0; j < OD_COLUMNS; ++j) {
            layout->cells[i * OD_COLUMNS + j] = (uint16_t)(rows[i] * columns[j]);
        }
    }
    layout->width = mask_w;
    layout->height = mask_h;
    return 1;
}

void od_init(od_state_t *state)
{
    if (state) {
        memset(state, 0, sizeof *state);
        state->choice = -1;
    }
}

static int distance(int a, int b) { return a < b ? b - a : a - b; }

/* Whether free direction d is a better choice than best (see obstacle_dir.h; best < 0: none yet). */
static int better(const od_state_t *state, int d, int best)
{
    const int band = d / OD_COLUMNS, sector = d % OD_COLUMNS;
    const int steps = distance(sector, OD_COLUMNS / 2) + distance(band, OD_ROWS / 2);
    int best_steps;
    if (best < 0) {
        return 1;
    }
    best_steps = distance(best % OD_COLUMNS, OD_COLUMNS / 2) + distance(best / OD_COLUMNS, OD_ROWS / 2);
    if (steps != best_steps) {
        return steps < best_steps;
    }
    if (distance(band, OD_ROWS / 2) != distance(best / OD_COLUMNS, OD_ROWS / 2)) { /* turning before rising or diving */
        return distance(band, OD_ROWS / 2) < distance(best / OD_COLUMNS, OD_ROWS / 2);
    }
    return state->share[d] < state->share[best];
}

int od_update(od_state_t *state, const od_layout_t *layout, const uint8_t *mask)
{
    uint16_t hits[OD_DIRECTIONS] = {0};
    int row, column, d, best = -1;
    if (!state || !layout || !mask || layout->width < 1 || layout->height < 1 || layout->width > OD_MAX_SIDE
            || layout->height > OD_MAX_SIDE) {
        return -1;
    }
    for (column = 0; column < layout->width; ++column) {
        if (layout->sector[column] >= OD_COLUMNS) {
            return -1;
        }
    }
    for (row = 0; row < layout->height; ++row) {
        if (layout->band[row] >= OD_ROWS) {
            return -1;
        }
    }
    for (row = 0; row < layout->height; ++row) {
        const uint8_t *cells = mask + row * layout->width;
        uint16_t *line = hits + layout->band[row] * OD_COLUMNS;
        for (column = 0; column < layout->width; ++column) {
            if (cells[column]) {
                ++line[layout->sector[column]];
            }
        }
    }
    for (d = 0; d < OD_DIRECTIONS; ++d) {
        const uint32_t total = layout->cells[d];
        if (!total) {
            state->share[d] = state->blocked[d] = state->clearing[d] = 0;
            continue;
        }
        state->share[d] = (uint8_t)(((uint32_t)hits[d] * 255u + total / 2u) / total);
        if (state->share[d] >= OD_BLOCK) {
            state->blocked[d] = 1;
            state->clearing[d] = 0;
        } else if (state->blocked[d] && state->share[d] < OD_CLEAR) {
            if (++state->clearing[d] >= OD_RELEASE) {
                state->blocked[d] = 0;
                state->clearing[d] = 0;
            }
        } else {
            state->clearing[d] = 0;
        }
    }
    for (d = 0; d < OD_DIRECTIONS; ++d) {
        if (layout->cells[d] && !state->blocked[d] && better(state, d, best)) {
            best = d;
        }
    }
    state->choice = best;
    ++state->frames;
    return best;
}
