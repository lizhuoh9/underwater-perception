/*
 * voxel_map.h - the robot's continuing three-dimensional map: the monocular odometry's (mono_vo.h) triangulated points,
 * accumulated in voxels for as long as the odometry runs.
 *
 * After each ESSENTIAL step, every tracked feature with a point triangulated from rays at least VM_MIN_SPREAD apart is an
 * observation (when a water mask comes with the frame, only on a cell that is not water): its point goes to the world
 * frame (the odometry's first camera: x right, y down, z forward) and adds to the voxel holding it. ROTATION steps add
 * nothing, as their points were only turned, not seen again from a new place. LOST steps add nothing but do not stop the
 * map: the odometry keeps its world frame through them (a lost step keeps the position and takes the turn it can
 * measure), so the map goes on in the same frame; it counts them.
 *
 * A single camera fixes no scale, and the odometry's unit drifts (on the lake clip by a factor of ten over half a
 * minute), so the voxel side follows the scene: the median depth of a frame's observations (at least VM_MIN_POINTS of
 * them) divided by VM_CELLS_PER_DEPTH. It is set by the first such frame and reset whenever a frame's value is more than
 * VM_RESCALE times larger or smaller; the voxels held are then re-keyed to the new side, those meeting in one voxel
 * merged (observations added, the larger frame count and the latest frame kept).
 *
 * A voxel keeps its observations, the frames that observed it and the last of them; one seen in VM_CONFIRM frames is an
 * obstacle (vm_occupied). Unobserved space is unknown, not free. The voxels are a dense array in the order they were
 * first seen, found through an open-addressing index of VM_SLOTS slots. Before an insertion beyond VM_CAPACITY voxels,
 * every voxel farther than VM_RADIUS voxels from the camera (along any axis) is dropped and, when they are fewer than
 * VM_EVICT, others until VM_EVICT are gone: those seen in fewer than VM_CONFIRM frames before obstacles, each the longest
 * unseen first (ages count to 127 frames; beyond that, and within one age, the first seen goes first); voxels seen in
 * the current frame stay. Dropping single sightings first keeps about half as many obstacles again, and twice the
 * history, for the same memory (on the lake clip at 5 Hz, half of the obstacles held were last seen within 4 s, nine in
 * ten within 8 s; half again as many voxels added little there and did not fit the board's last free block). So the map
 * holds the robot's surroundings and its recent past, in the odometry's frame, which drifts: it is not a global
 * reconstruction and closes no loops.
 *
 * The voxels are fine, 16 to a scene depth, and 4096 of them are held (the ESP32-P4 has the memory): against voxels
 * twice as large and four times fewer, on the lake clip at 5 Hz with 15x15 windows, the median distance of the
 * obstacles held to the reference's surface came out 0.064 m instead of 0.067 near the camera and 0.074 m instead of
 * 0.106 over the whole path (16 seeds). Single sightings land in different fine voxels more often, so the planner reads
 * them two to a side (PP_GROUP in path_plan.h).
 *
 * C99, no heap, no libm: a PC and an ESP32-P4 build the same map bit for bit. One vm_map_t is about 56 KB; it is not
 * thread-safe.
 */
#ifndef VOXEL_MAP_H
#define VOXEL_MAP_H

#include <stdint.h>

#include "mono_vo.h"

#define VM_CAPACITY 4096 /* voxels held at most */
#define VM_SLOTS 8192    /* index slots, a power of two above VM_CAPACITY */
#define VM_EVICT 512     /* voxels dropped at once when full (more when more of them are far) */
#define VM_RADIUS 96     /* voxels from the camera, along any axis, beyond which voxels go first (6 scene depths) */
#define VM_CELLS_PER_DEPTH 16.0f
#define VM_MIN_POINTS 20 /* observations that measure the scene's depth */
#define VM_RESCALE 2.0f  /* the side is reset when the scene says it should be this many times larger or smaller */
#define VM_MIN_SPREAD 3.0e-4f /* squared sine of 1 degree, as in mono_map.h */
#define VM_CONFIRM 2     /* frames that make a voxel an obstacle */
#define VM_MAX_KEY 32767 /* voxel indices stay within int16_t */

typedef struct {
    int16_t key[3];  /* voxel indices: floor(world / side) */
    uint8_t hits;    /* observations, saturating at 255 */
    uint8_t frames;  /* frames that observed it, saturating at 255 */
    uint16_t seen;   /* the map's frame count (its low 16 bits) when last observed */
} vm_voxel_t;

typedef struct {
    vm_voxel_t voxel[VM_CAPACITY]; /* the first `count` are held, oldest first */
    uint16_t index[VM_SLOTS];      /* 1 + a voxel's place in voxel[], 0 where free */
    float side;                    /* voxel side in odometry units; 0 until set */
    uint32_t count;                /* voxels held */
    uint32_t frames;               /* steps taken, of any kind */
    uint32_t observations;         /* observations filed */
    uint32_t evicted;              /* voxels dropped to make room */
    uint32_t merged;               /* voxels merged into another by a new side */
    uint32_t rescales;             /* times the side was reset after the first */
    uint32_t lost;                 /* LOST steps the map went through */
} vm_map_t;

void vm_init(vm_map_t *m);
uint32_t vm_map_size(void);

/* After each odometry step, with the state and the step it produced: files the step's observations (see above).
 * mask: mask_h rows of mask_w cells spread evenly over the 320x240 frame, nonzero where it is not water, or NULL for
 * none. Returns the observations filed (0 also for invalid arguments, which leave the map as it was). */
int vm_update(vm_map_t *m, const mvo_t *vo, const mvo_step_t *step, const uint8_t *mask, int mask_w, int mask_h);

/* The voxel indices of world point p; 0 when the side is not set yet or they would leave +-VM_MAX_KEY. */
int vm_key(const vm_map_t *m, const float p[3], int16_t key[3]);

/* The voxel at those indices, or NULL when the map holds none there. */
const vm_voxel_t *vm_find(const vm_map_t *m, const int16_t key[3]);

/* 1 when world point p lies in a voxel observed in at least VM_CONFIRM frames. */
int vm_occupied(const vm_map_t *m, const float p[3]);

#endif
