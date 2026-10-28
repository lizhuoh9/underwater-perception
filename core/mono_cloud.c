/* See mono_cloud.h. */
#include "mono_cloud.h"

#include <float.h>
#include <string.h>

static int finite_float(float value)
{
    return value >= -FLT_MAX && value <= FLT_MAX;
}

void mc_init(mc_map_t *map)
{
    if (map) {
        memset(map, 0, sizeof *map);
        map->continuous = 1;
    }
}

uint32_t mc_map_size(void) { return (uint32_t)sizeof(mc_map_t); }

int mc_update(mc_map_t *map, const mvo_t *vo, const mvo_step_t *step)
{
    uint32_t active[(MC_MAX_POINTS + 31) / 32] = {0}, index;
    uint16_t order[MVO_MAX_FEATURES];
    int k, accepted = 0;
    if (!map || !step) {
        return 0;
    }
    if (step->kind == MVO_LOST) {
        map->continuous = 0;
        return 0;
    }
    if (!map->continuous || step->kind != MVO_ESSENTIAL || !vo || vo->count < 0
            || vo->count > MVO_MAX_FEATURES || map->count > MC_MAX_POINTS) {
        return 0;
    }
    /* Process ids in a stable order, regardless of which feature the tracker stored first. */
    for (k = 0; k < vo->count; ++k) {
        uint16_t value = (uint16_t)k;
        int j = k;
        while (j > 0 && vo->feature[order[j - 1]].id > vo->feature[value].id) {
            order[j] = order[j - 1];
            --j;
        }
        order[j] = value;
    }
    for (k = 1; k < vo->count; ++k) {
        if (vo->feature[order[k - 1]].id == vo->feature[order[k]].id) return 0;
    }
    /* Mark all currently tracked ids before any replacement, including features without a usable 3D point. */
    for (index = 0; index < map->count; ++index) {
        int lo = 0, hi = vo->count;
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if (vo->feature[order[mid]].id < map->points[index].id) lo = mid + 1;
            else hi = mid;
        }
        if (lo < vo->count && vo->feature[order[lo]].id == map->points[index].id) {
            active[index / 32] |= (uint32_t)1 << (index % 32);
        }
    }
    for (k = 0; k < vo->count; ++k) {
        const mvo_feature_t *feature = &vo->feature[order[k]];
        mc_point_t *point;
        float world[3];
        int i;
        if (!feature->has_point || feature->spread < MC_MIN_SPREAD || !finite_float(feature->spread)
                || !(feature->point[2] > 0.0f) || !finite_float(feature->point[0])
                || !finite_float(feature->point[1]) || !finite_float(feature->point[2])) {
            continue;
        }
        for (i = 0; i < 3; ++i) {
            world[i] = vo->world_rotation[3 * i] * feature->point[0]
                     + vo->world_rotation[3 * i + 1] * feature->point[1]
                     + vo->world_rotation[3 * i + 2] * feature->point[2] + vo->world_centre[i];
        }
        if (!finite_float(world[0]) || !finite_float(world[1]) || !finite_float(world[2])) {
            continue;
        }
        for (index = 0; index < map->count && map->points[index].id != feature->id; ++index) {
        }
        if (index == map->count) {
            if (map->count == MC_MAX_POINTS) {
                uint32_t candidate, oldest_age = 0, oldest = MC_MAX_POINTS;
                for (candidate = 0; candidate < MC_MAX_POINTS; ++candidate) {
                    if (!(active[candidate / 32] & ((uint32_t)1 << (candidate % 32)))) {
                        uint32_t age = vo->next_id - map->points[candidate].id;
                        if (oldest == MC_MAX_POINTS || age > oldest_age) {
                            oldest = candidate;
                            oldest_age = age;
                        }
                    }
                }
                if (oldest == MC_MAX_POINTS) continue;
                index = oldest;
                if (map->dropped < UINT32_MAX) ++map->dropped;
            } else {
                index = map->count++;
            }
            active[index / 32] |= (uint32_t)1 << (index % 32);
            point = &map->points[index];
            point->id = feature->id;
            point->spread = feature->spread;
            point->observations = 1;
            memcpy(point->xyz, world, sizeof point->xyz);
        } else {
            point = &map->points[index];
            if (point->observations < UINT32_MAX) ++point->observations;
            if (feature->spread > point->spread) {
                point->spread = feature->spread;
                memcpy(point->xyz, world, sizeof point->xyz);
            }
        }
        ++accepted;
    }
    return accepted;
}
