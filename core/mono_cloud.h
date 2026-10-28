/*
 * A bounded sparse three-dimensional map of the monocular VO's feature tracks.
 *
 * This map uses only triangulated features from the grey camera frames; it does not depend on a segmentation mask.
 * Coordinates are in the first camera's axes (x right, y down, z forward), in the VO's arbitrary, drifting scale.
 * Each feature id has one entry, updated when a later triangulation has a wider ray separation. Feature tracks can be
 * lost and re-detected with a different id, so this does not merge revisits or eliminate all duplicate landmarks.
 *
 * The first MVO_LOST step freezes the map until mc_init; the new VO and new map must be started together. ROTATION
 * steps do not add points. A continuous map has merely avoided an explicit LOST; it is not a guarantee of accuracy or
 * free space. Points of moving objects can also enter it. When full, a new id replaces an old id not tracked in this
 * frame, choosing the oldest ID by the VO's unsigned allocation counter. The map therefore represents a bounded local
 * history, not a complete global reconstruction. `dropped` counts old points evicted; every tracked id is protected
 * even when it has no depth. ID age is ambiguous if one session spans a full 2^32 feature allocations.
 *
 * C99, no heap, no libm; caller owns mc_map_t. The implementation is not thread-safe for the same map.
 */
#ifndef MONO_CLOUD_H
#define MONO_CLOUD_H

#include <stdint.h>
#include "mono_vo.h"

#define MC_MAX_POINTS 512
#define MC_MIN_SPREAD MVO_MAP_MIN_SPREAD /* the build configuration's (mono_vo.h): 1 degree, 1.5 on a P4 */

typedef struct {
    float xyz[3];          /* world coordinates in odometry units */
    uint32_t id;           /* VO feature id, unique only within this session */
    float spread;          /* squared sine of the widest ray angle stored for this feature */
    uint32_t observations; /* accepted observations, saturating at UINT32_MAX */
} mc_point_t;

typedef struct {
    mc_point_t points[MC_MAX_POINTS];
    uint32_t count;
    uint32_t dropped;       /* old points evicted to make room for new ids, saturating at UINT32_MAX */
    int32_t continuous;     /* 0 after LOST, 1 since init without LOST */
} mc_map_t;

/* The VO must also be reinitialized when starting a new cloud. */
void mc_init(mc_map_t *map);
uint32_t mc_map_size(void);

/* Consume the VO state and step from the same completed frame. Return the number of accepted feature observations
 * (new or existing ids), excluding refused/invalid ones. For an ESSENTIAL step, invalid pointers, feature counts or
 * duplicate ids return 0 without changing the map. All tracked ids are protected before insertion, independent of
 * feature-array order; with a valid map and MC_MAX_POINTS > MVO_MAX_FEATURES a free eviction candidate always exists.
 * LOST latches even when vo is NULL, so callers can stop writes immediately on the reported loss. */
int mc_update(mc_map_t *map, const mvo_t *vo, const mvo_step_t *step);

#endif
