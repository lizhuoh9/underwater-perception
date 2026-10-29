/*
 * obstacle_dir.h - which directions of the camera's view are blocked, from the segmentation's water mask alone: the
 * obstacle-direction half of the robot's avoidance (the odometry's map, voxel_map.h, gives the ranges).
 *
 * The mask of a frame (mask_h rows of mask_w cells spread evenly over the 320x240 image, nonzero where it is not water)
 * is split into OD_ROWS x OD_COLUMNS directions by the angle of each cell's ray from the optical axis: azimuth sectors
 * with edges at +-OD_INNER and +-OD_OUTER (the tangents of 15 and 40 degrees), left to right, and elevation bands with
 * edges at +-OD_INNER, up to down. A direction's share is the part of its cells that is not water, 0 to 255. A direction
 * is blocked as soon as its share reaches OD_BLOCK and stays blocked until its share has stayed below OD_CLEAR for
 * OD_RELEASE frames in a row: a free direction that is not free is the costly error, so blocking is immediate and
 * clearing is not. A direction the camera does not see (no cells) is never blocked nor chosen. The choice is the free
 * direction nearest straight ahead: the fewest sector and band steps from the centre, then a level direction before one
 * up or down (turning before rising or diving), then the lower share, then the lower index; -1 when every seen
 * direction is blocked. The layout stays a sector per column and a band per row: with a lens (mvo_camera_t's
 * distortion) a column takes the sector of its cell on the principal point's row, a row the band of its cell on the
 * principal point's column, both undistorted (mvo_undistort); under strong distortion a corner cell may sit one
 * direction off.
 *
 * The mask gives direction, not range: everything that is not water within the visibility counts, the lake bed below
 * and the water surface above included (the surface bounds the robot too). The directions are the camera's: without an
 * IMU "up" is the image's up, not gravity's.
 *
 * C99, no heap, no libm. The layout is computed once in float, the updates in integers: a PC and an ESP32-P4 give the
 * same results. Not thread-safe for the same state.
 */
#ifndef OBSTACLE_DIR_H
#define OBSTACLE_DIR_H

#include <stdint.h>

#include "mono_vo.h"

#define OD_COLUMNS 5 /* azimuth sectors, left to right */
#define OD_ROWS 3    /* elevation bands, up to down */
#define OD_DIRECTIONS (OD_ROWS * OD_COLUMNS) /* index band * OD_COLUMNS + sector */
#define OD_CENTRE (OD_DIRECTIONS / 2)       /* straight ahead */
#define OD_MAX_SIDE 128                     /* mask rows and columns at most */
#define OD_INNER 0.26794919f /* tan 15 degrees */
#define OD_OUTER 0.83909963f /* tan 40 degrees */
#define OD_BLOCK 128   /* share (of 255) from which a direction is blocked: half of it not water */
#define OD_CLEAR 77    /* share (of 255) below which a blocked direction counts as clearing: under 30% */
#define OD_RELEASE 2   /* clearing frames in a row that free a blocked direction */

/* Where each mask cell looks, for one camera and mask size. */
typedef struct {
    int32_t width, height;            /* the mask's cells; 0 when the layout was refused */
    uint8_t sector[OD_MAX_SIDE];      /* of each mask column */
    uint8_t band[OD_MAX_SIDE];        /* of each mask row */
    uint16_t cells[OD_DIRECTIONS];    /* mask cells in each direction */
} od_layout_t;

typedef struct {
    uint8_t share[OD_DIRECTIONS];   /* the last frame's share of cells that are not water, 0 to 255 */
    uint8_t blocked[OD_DIRECTIONS]; /* 1 blocked, 0 free (or unseen) */
    uint8_t clearing[OD_DIRECTIONS]; /* clearing frames in a row while blocked */
    uint8_t pad[3];                 /* zero: the whole state is hashed (the board checks it against the PC) */
    int32_t choice;                 /* the free direction nearest straight ahead, -1 when none */
    uint32_t frames;                /* masks taken */
} od_state_t;

/* Returns 1, or 0 (and a layout that refuses every mask) for a camera without positive focal lengths, a lens model
 * that folds over at a cell's centre (mvo_undistort) or a mask size outside 1..OD_MAX_SIDE. */
int od_layout(od_layout_t *layout, const mvo_camera_t *camera, int mask_w, int mask_h);
void od_init(od_state_t *state);
/* Takes one frame's mask (layout->height rows of layout->width cells); returns the choice, or -1 also when the layout
 * or the arguments are invalid (the state then stays as it was). */
int od_update(od_state_t *state, const od_layout_t *layout, const uint8_t *mask);
/* The direction (band * OD_COLUMNS + sector) of a ray in front of the camera, from its x / z and y / z, by the same
 * edges as the layout's cells. */
int od_direction_of(float x_over_z, float y_over_z);
uint32_t od_state_size(void);
uint32_t od_layout_size(void);

#endif
