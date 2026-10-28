/* The build configurations of core/mono_vo.h, compiled with and without MVO_CONFIG_P4: prints what each sets, the map
 * admissions and the evidence penalty's constant as their float bits. */
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "mono_cloud.h"
#include "mono_map.h"
#include "mono_vo_internal.h"
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
    return 0;
}
