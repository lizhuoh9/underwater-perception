/* See mono_map.h. */
#include "mono_map.h"

#include <string.h>

static float s_depths[MVO_MAX_FEATURES]; /* scratch for the median depth that sets the cell size */

static int mm_floor(float v)
{
    int i = (int)v;
    return v < (float)i ? i - 1 : i;
}

void mm_init(mm_map_t *m) { memset(m, 0, sizeof *m); }

uint32_t mm_map_size(void) { return (uint32_t)sizeof(mm_map_t); }

int mm_observe(const mvo_t *vo, const uint8_t *mask, int mask_w, int mask_h, float (*out)[3], int capacity)
{
    int k, count = 0;
    if (!vo || !mask || !out || mask_w <= 0 || mask_h <= 0 || capacity <= 0) {
        return 0;
    }
    for (k = 0; k < vo->count && count < capacity; ++k) {
        const mvo_feature_t *f = &vo->feature[k];
        int column, row, i;
        if (!f->has_point || f->spread < MM_MIN_SPREAD) {
            continue;
        }
        column = (int)(f->x * (float)mask_w / (float)MVO_WIDTH);
        row = (int)(f->y * (float)mask_h / (float)MVO_HEIGHT);
        column = column < 0 ? 0 : (column >= mask_w ? mask_w - 1 : column);
        row = row < 0 ? 0 : (row >= mask_h ? mask_h - 1 : row);
        if (!mask[row * mask_w + column]) {
            continue;
        }
        for (i = 0; i < 3; ++i) {
            out[count][i] = vo->world_rotation[3 * i] * f->point[0] + vo->world_rotation[3 * i + 1] * f->point[1]
                            + vo->world_rotation[3 * i + 2] * f->point[2] + vo->world_centre[i];
        }
        ++count;
    }
    return count;
}

int mm_cell_of(const mm_map_t *m, float x, float z, int *column, int *row)
{
    if (m->cell <= 0.0f) {
        return 0;
    }
    *column = mm_floor(x / m->cell) - m->origin[0];
    *row = mm_floor(z / m->cell) - m->origin[1];
    return *column >= 0 && *row >= 0 && *column < MM_SIZE && *row < MM_SIZE;
}

static void add_odds(mm_map_t *m, int column, int row, int delta)
{
    int8_t *cell = &m->odds[row * MM_SIZE + column];
    const int value = *cell + delta;
    *cell = (int8_t)(value > MM_LIMIT ? MM_LIMIT : (value < -MM_LIMIT ? -MM_LIMIT : value));
}

/* Sets the cell size from the median depth (along the camera's optical axis) of the observations, and centres the
 * grid on the camera. */
static int set_scale(mm_map_t *m, const mvo_t *vo, const float (*points)[3], int count)
{
    int k, i, j;
    if (count < MM_MIN_POINTS) {
        return 0;
    }
    for (k = 0; k < count; ++k) { /* depth = third camera axis (third column of the camera-to-world rotation) . (p - c) */
        float key = 0.0f;
        for (i = 0; i < 3; ++i) {
            key += vo->world_rotation[3 * i + 2] * (points[k][i] - vo->world_centre[i]);
        }
        for (j = k - 1; j >= 0 && s_depths[j] > key; --j) { /* insertion sort */
            s_depths[j + 1] = s_depths[j];
        }
        s_depths[j + 1] = key;
    }
    m->cell = (count % 2 ? s_depths[count / 2] : 0.5f * (s_depths[count / 2 - 1] + s_depths[count / 2])) / MM_CELLS_PER_DEPTH;
    if (m->cell <= 0.0f) {
        m->cell = 0.0f;
        return 0;
    }
    m->origin[0] = mm_floor(vo->world_centre[0] / m->cell) - MM_SIZE / 2;
    m->origin[1] = mm_floor(vo->world_centre[2] / m->cell) - MM_SIZE / 2;
    m->slice_y = vo->world_centre[1];
    return 1;
}

/* Recentres the grid on world cell (x, z) when it lies MM_SIZE / 4 or more from the centre: the cells move, those
 * that fall off are forgotten and the new ones start unknown. */
static void follow(mm_map_t *m, int x, int z)
{
    const int dx = x - MM_SIZE / 2 - m->origin[0], dz = z - MM_SIZE / 2 - m->origin[1];
    int row, first, last, step;
    if (dx < MM_SIZE / 4 && dx > -MM_SIZE / 4 && dz < MM_SIZE / 4 && dz > -MM_SIZE / 4) {
        return;
    }
    first = dz >= 0 ? 0 : MM_SIZE - 1; /* the order that never overwrites a row still to be read */
    last = dz >= 0 ? MM_SIZE : -1;
    step = dz >= 0 ? 1 : -1;
    for (row = first; row != last; row += step) {
        int8_t *target = m->odds + row * MM_SIZE;
        const int source = row + dz;
        if (source < 0 || source >= MM_SIZE || dx >= MM_SIZE || dx <= -MM_SIZE) {
            memset(target, 0, MM_SIZE);
        } else if (dx >= 0) {
            memmove(target, m->odds + source * MM_SIZE + dx, (size_t)(MM_SIZE - dx));
            memset(target + MM_SIZE - dx, 0, (size_t)dx);
        } else {
            memmove(target - dx, m->odds + source * MM_SIZE, (size_t)(MM_SIZE + dx));
            memset(target, 0, (size_t)-dx);
        }
    }
    m->origin[0] += dx;
    m->origin[1] += dz;
}

int mm_add(mm_map_t *m, const mvo_t *vo, const float (*points)[3], int count)
{
    int k, taken = 0;
    float band;
    if (!m || !vo || count < 0 || (count > 0 && !points)) {
        return 0;
    }
    if (count > MVO_MAX_FEATURES) {
        count = MVO_MAX_FEATURES;
    }
    if (m->cell <= 0.0f && !set_scale(m, vo, points, count)) {
        return 0;
    }
    if (vo->world_centre[1] - m->slice_y >= m->cell || m->slice_y - vo->world_centre[1] >= m->cell) {
        memset(m->odds, 0, sizeof m->odds);
        m->slice_y = vo->world_centre[1];
        ++m->height_resets;
    }
    follow(m, mm_floor(vo->world_centre[0] / m->cell), mm_floor(vo->world_centre[2] / m->cell));
    band = MM_BAND * MM_CELLS_PER_DEPTH * m->cell;
    for (k = 0; k < count; ++k) {
        const float height = points[k][1] - m->slice_y;
        int column, row;
        if (height > band || height < -band || !mm_cell_of(m, points[k][0], points[k][2], &column, &row)) {
            continue;
        }
        add_odds(m, column, row, MM_HIT);
        ++taken;
    }
    m->observations += (uint32_t)taken;
    m->frames += taken > 0;
    return taken;
}
