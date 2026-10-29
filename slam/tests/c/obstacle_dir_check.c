/* The obstacle directions' contract (core/obstacle_dir.h): the layout, the hysteresis and the choice. */
#include "obstacle_dir.h"

#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <string.h>

#define W 80
#define H 60

static uint8_t mask[W * H];
static od_layout_t layout;
static od_state_t state, saved;

/* The lake clip's camera: 110 degrees across a 960x544 frame stretched to 320x240. */
static const mvo_camera_t lake = {112.0302963f, 148.2792358f, 160.0f, 120.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

/* Marks the cells of direction d as not water in a share of its columns (share of 16 columns, all its rows). */
static void fill(int d, int columns)
{
    int row, column, seen = 0;
    for (column = 0; column < W; ++column) {
        if (layout.sector[column] != d % OD_COLUMNS) {
            continue;
        }
        if (seen++ >= columns) {
            break;
        }
        for (row = 0; row < H; ++row) {
            if (layout.band[row] == d / OD_COLUMNS) {
                mask[row * W + column] = 1;
            }
        }
    }
}

static void layout_follows_the_angles(void)
{
    int d, i, previous = 0;
    mvo_camera_t bad = lake;
    assert(od_layout(&layout, &lake, W, H) == 1);
    assert(layout.width == W && layout.height == H);
    for (d = 0; d < OD_DIRECTIONS; ++d) { /* 110 degrees: sectors of 16 columns, bands of 20 rows */
        assert(layout.cells[d] == 16 * 20);
    }
    for (i = 0; i < W; ++i) { /* left to right, never back */
        assert(layout.sector[i] >= previous && layout.sector[i] < OD_COLUMNS);
        previous = layout.sector[i];
    }
    assert(layout.sector[15] == 0 && layout.sector[16] == 1 && layout.sector[31] == 1 && layout.sector[32] == 2);
    assert(layout.sector[47] == 2 && layout.sector[48] == 3 && layout.sector[63] == 3 && layout.sector[64] == 4);
    assert(layout.band[19] == 0 && layout.band[20] == 1 && layout.band[39] == 1 && layout.band[40] == 2);
    /* a narrow camera (about 64 degrees across): nothing beyond 40 degrees, so the outer sectors go unseen */
    bad.fx = 256.0f;
    assert(od_layout(&layout, &bad, W, H) == 1);
    for (d = 0; d < OD_DIRECTIONS; ++d) {
        assert((layout.cells[d] == 0) == (d % OD_COLUMNS == 0 || d % OD_COLUMNS == OD_COLUMNS - 1));
    }
    /* refused: no focal length, a non-finite centre, masks of no or too many cells */
    bad = lake;
    bad.fx = 0.0f;
    assert(od_layout(&layout, &bad, W, H) == 0 && layout.width == 0);
    bad = lake;
    bad.cx = FLT_MAX * 2.0f;
    assert(od_layout(&layout, &bad, W, H) == 0);
    assert(od_layout(&layout, &lake, 0, H) == 0 && od_layout(&layout, &lake, W, OD_MAX_SIDE + 1) == 0);
    assert(od_layout(NULL, &lake, W, H) == 0 && od_layout(&layout, NULL, W, H) == 0);
    od_init(&state);
    saved = state;
    assert(od_update(&state, &layout, mask) == -1 && memcmp(&state, &saved, sizeof state) == 0);
    assert(od_update(NULL, &layout, mask) == -1 && od_update(&state, NULL, mask) == -1);
    assert(od_update(&state, &layout, NULL) == -1 && memcmp(&state, &saved, sizeof state) == 0);
}

static void blocking_is_immediate_and_clearing_slow(void)
{
    const int d = OD_CENTRE;
    assert(od_layout(&layout, &lake, W, H) == 1);
    od_init(&state);
    assert(state.choice == -1 && state.frames == 0);
    memset(mask, 0, sizeof mask);
    assert(od_update(&state, &layout, mask) == OD_CENTRE && state.frames == 1);
    fill(d, 8); /* half of it: blocked at once */
    assert(od_update(&state, &layout, mask) != OD_CENTRE && state.blocked[d] && state.share[d] == 128);
    memset(mask, 0, sizeof mask);
    fill(d, 7); /* 44%: neither blocking nor clearing */
    od_update(&state, &layout, mask);
    assert(state.blocked[d] && state.clearing[d] == 0);
    memset(mask, 0, sizeof mask);
    fill(d, 4); /* 25%: clearing, one frame of OD_RELEASE */
    od_update(&state, &layout, mask);
    assert(state.blocked[d] && state.clearing[d] == 1);
    memset(mask, 0, sizeof mask);
    fill(d, 5); /* 31%: not clearing, the run starts again */
    od_update(&state, &layout, mask);
    assert(state.blocked[d] && state.clearing[d] == 0);
    memset(mask, 0, sizeof mask);
    od_update(&state, &layout, mask);
    assert(state.blocked[d] && state.clearing[d] == 1);
    assert(od_update(&state, &layout, mask) == OD_CENTRE && !state.blocked[d] && state.clearing[d] == 0);
}

static void choice_is_the_nearest_free_direction(void)
{
    int d;
    assert(od_layout(&layout, &lake, W, H) == 1);
    od_init(&state);
    memset(mask, 0, sizeof mask);
    fill(OD_CENTRE, 16);
    /* one step away: up, left, right and down are free and all water; turning goes first, then the lower index */
    assert(od_update(&state, &layout, mask) == OD_CENTRE - 1);
    fill(OD_CENTRE - 1, 2); /* left now holds a little: right is the most open */
    assert(od_update(&state, &layout, mask) == OD_CENTRE + 1);
    fill(OD_CENTRE + 1, 4); /* right holds more than left: left again */
    assert(od_update(&state, &layout, mask) == OD_CENTRE - 1);
    fill(OD_CENTRE - 1, 16);
    fill(OD_CENTRE + 1, 16); /* the level sides blocked: up (lower index) or down, both water */
    assert(od_update(&state, &layout, mask) == OD_CENTRE - OD_COLUMNS);
    fill(OD_CENTRE - OD_COLUMNS, 16);
    assert(od_update(&state, &layout, mask) == OD_CENTRE + OD_COLUMNS);
    memset(mask, 1, sizeof mask);
    assert(od_update(&state, &layout, mask) == -1 && state.choice == -1);
    for (d = 0; d < OD_DIRECTIONS; ++d) {
        assert(state.blocked[d] && state.share[d] == 255);
    }
    /* unseen directions are never blocked nor chosen */
    {
        mvo_camera_t narrow = lake;
        narrow.fx = 256.0f;
        assert(od_layout(&layout, &narrow, W, H) == 1);
        od_init(&state);
        memset(mask, 1, sizeof mask);
        for (d = 0; d < W * H; ++d) { /* water only where the outer sectors would be */
            if (layout.sector[d % W] == 0 || layout.sector[d % W] == OD_COLUMNS - 1) {
                mask[d] = 0;
            }
        }
        assert(od_update(&state, &layout, mask) == -1);
        for (d = 0; d < OD_DIRECTIONS; ++d) {
            if (!layout.cells[d]) {
                assert(!state.blocked[d] && state.share[d] == 0);
            }
        }
    }
}

int main(void)
{
    assert(od_state_size() == sizeof(od_state_t) && od_layout_size() == sizeof(od_layout_t));
    layout_follows_the_angles();
    blocking_is_immediate_and_clearing_slow();
    choice_is_the_nearest_free_direction();
    printf("obstacle_dir_check ok state_bytes=%u layout_bytes=%u\n", (unsigned)od_state_size(), (unsigned)od_layout_size());
    return 0;
}
