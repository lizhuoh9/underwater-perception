/* The build's constants (core/mono_vo.h and the headers after it): prints its name and feature count, the map
 * admissions and the evidence penalty's constant as their float bits, the tracking window and the voxel map's and the
 * plan's sizes (which the Python side repeats). */
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "mono_cloud.h"
#include "mono_map.h"
#include "mono_vo_internal.h"
#include "path_plan.h"
#include "voxel_map.h"

static uint32_t bits(float value)
{
    uint32_t out;
    memcpy(&out, &value, sizeof out);
    return out;
}

int main(void)
{
    printf("config=%s max_features=%d mm=%" PRIu32 " vm=%" PRIu32 " mc=%" PRIu32 " evidence_log=%" PRIu32 "\n",
           MVO_CONFIG_NAME, MVO_MAX_FEATURES, bits(MM_MIN_SPREAD), bits(VM_MIN_SPREAD), bits(MC_MIN_SPREAD),
           bits(MVO_EVIDENCE_LOG));
    printf("window=%d lanes=%d voxel_capacity=%d voxel_slots=%d cells_per_depth=%" PRIu32 " plan_group=%d\n", MVO_WINDOW,
           MVO_LANES, VM_CAPACITY, VM_SLOTS, bits(VM_CELLS_PER_DEPTH), PP_GROUP);
    return 0;
}
