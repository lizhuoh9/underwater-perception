/*
 * mono_map.h - a top-view obstacle map from the monocular visual odometry (mono_vo.h) and the segmentation's water
 * mask, for the same 320x240 camera.
 *
 * After each odometry step, the tracked features that carry a triangulated point, were triangulated from rays at least
 * MM_MIN_SPREAD apart and sit on a mask cell that is not water are obstacle observations: their points go to the world
 * frame (the odometry's first camera: x right, y down, z forward). The map is a horizontal slice through the robot's
 * surroundings, seen from above: a square grid of obstacle evidence over (x, z) that takes observations within MM_BAND
 * scene depths above or below a fixed slice height. Unobserved cells remain unknown: a ray at one height cannot prove
 * the whole height band free. When the camera moves one cell away from the slice height, the map is cleared and the
 * slice is centred on the camera again. A single camera fixes no scale, so the cell size follows
 * the scene: the median depth of the first frame's MM_MIN_POINTS or more observations, divided by MM_CELLS_PER_DEPTH.
 * The grid rolls with the robot: when the camera strays MM_SIZE / 4 cells from its centre, it is recentred on the
 * camera and what falls off is forgotten.
 *
 * C99, no heap, no libm, integer grid updates: a PC and an ESP32-S3 build the same map. One mm_map_t is about 16 KB;
 * it is not thread-safe.
 */
#ifndef MONO_MAP_H
#define MONO_MAP_H

#include <stdint.h>

#include "mono_vo.h"

#define MM_SIZE 128           /* cells per side */
#define MM_HIT 6              /* log-odds added where an obstacle is seen */
#define MM_LIMIT 100          /* obstacle evidence is capped at MM_LIMIT */
#define MM_MIN_POINTS 20      /* observations needed to fix the cell size */
#define MM_CELLS_PER_DEPTH 8.0f
#define MM_BAND 0.5f          /* scene depths above or below the camera that the slice takes */
#define MM_MIN_SPREAD MVO_MAP_MIN_SPREAD /* the build configuration's (mono_vo.h): 1 degree, 1.5 on a P4 */

typedef struct {
    int8_t odds[MM_SIZE * MM_SIZE]; /* row-major: row z, column x */
    float cell;                     /* cell size in odometry units; 0 until set */
    int32_t origin[2];              /* world cell (x, z) of the grid's first column and row */
    uint32_t observations;          /* observations filed in the grid */
    uint32_t frames;                /* updates that filed at least one */
    float slice_y;                  /* fixed world height of the current slice */
    uint32_t height_resets;         /* clears after moving one cell away from the slice */
} mm_map_t;

void mm_init(mm_map_t *m);
uint32_t mm_map_size(void);

/* World points of the current frame's obstacle observations (see above, before the height band) into out, at most
 * capacity; returns their count. mask: mask_h rows of mask_w cells spread evenly over the 320x240 frame, nonzero where
 * it is not water. */
int mm_observe(const mvo_t *vo, const uint8_t *mask, int mask_w, int mask_h, float (*out)[3], int capacity);

/* Files count observations made from the odometry's current camera; returns how many the slice took. */
int mm_add(mm_map_t *m, const mvo_t *vo, const float (*points)[3], int count);

/* The grid cell (column, row) of world point (x, z), or 0 when it lies outside (or the cell size is not set yet). */
int mm_cell_of(const mm_map_t *m, float x, float z, int *column, int *row);

#endif
