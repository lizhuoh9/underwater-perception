/*
 * path_plan.h - local avoidance and 3D path choice: which way the robot should head next, from the voxel map's ranges
 * (voxel_map.h) and the mask's blocked directions (obstacle_dir.h), towards a goal direction.
 *
 * Every step it weighs the 3 x 5 directions of the camera's view that obstacle_dir.h defines (bands up, level and down,
 * sectors from hard left to hard right). A single camera fixes no scale, so ranges count in steps: the odometry's recent
 * speed, the median of its last step lengths, is one step, and a range of n steps is reached in n frames at that speed.
 * Each direction's range is the distance from the camera to the second nearest obstacle voxel (seen in VM_CONFIRM
 * frames) whose centre lies in it, within PP_HORIZON_STEPS; with fewer than two such voxels the map leaves its range
 * unknown (one voxel alone is too easily an outlier). A direction is passable when the camera sees it, the mask (once one
 * has come) shows it free, and the map puts no obstacle within PP_SAFE_STEPS there. The map adds vetoes but lifts
 * none: it holds only textured points, and where it holds none there may still be a surface the mask sees. On the lake
 * clip, letting a far map range reopen a direction the mask blocked sent the plan into corridors blocked within 0.25 m
 * in 2-4% of its steps, against none without.
 *
 * The plan heads for the passable direction nearest the goal's (PP_GO): the fewest sector and band steps from it, then a
 * level direction before one up or down (turning before rising or diving), then the longer range (unknown counting as
 * beyond the horizon), then the direction already planned, then the lower index. With none passable it creeps towards
 * the seen direction with the longest known range (on a tie the one nearer the goal's, then the lower index) when that
 * reaches PP_CREEP_STEPS (PP_CREEP), else turns in place (PP_TURN) towards the side where the map holds fewer obstacles
 * within PP_SAFE_STEPS (all around, behind included), then where the mask's level band holds less that is not water,
 * then the goal's side, then the side the previous plan turned to if it turned, then left; with neither a known speed
 * nor a mask yet it stops (PP_STOP). The goal is a direction in the odometry's world (the first camera's axes: x right,
 * y down, z forward), or none for straight ahead. One behind the camera counts as the level hard sector on its side, or
 * as the centre of the band above or below when it points more up or down than sideways; one of zero length or not
 * finite counts as straight ahead.
 *
 * The map knows no range where the mask sees water: open water has no texture to triangulate, and the voxel map files no
 * point on a water cell. The plan bounds it instead. Anything nearer than the visibility along a water ray would itself
 * be seen, and every surface in view is nearer than the visibility, so the water directions are clear at least to the
 * surfaces the camera sees now: `clear` is PP_CLEAR_SHARE of the median range of the voxels filed in this frame (in
 * front, within the horizon), when there are at least PP_MIN_VIEW of them. On the lake clip no water direction held a
 * surface within 8 steps, so the map had nothing to veto there (the few vetoes it cast came from stray points); over 16
 * seeds the reference was nearer than `clear` for at most 1% of the water directions at 5 Hz and for none at 10 Hz,
 * and in a probe every such one still had 14 steps or more of free water. At the full median it was 1.4%.
 *
 * The plan is local: it remembers nothing but its last choice, and the map it reads holds only the last seconds. It
 * says where to head and how many steps of free water lie there (the map's range, or `clear` for a water direction the
 * map knows nothing of); the robot's controller turns that into thrust.
 *
 * C99, no heap, no libm (mvo_sqrt for the ranges reported): a PC and an ESP32-S3 give the same plan. Not thread-safe
 * for the same plan.
 */
#ifndef PATH_PLAN_H
#define PATH_PLAN_H

#include <stdint.h>

#include "mono_vo.h"
#include "obstacle_dir.h"
#include "voxel_map.h"

#define PP_HORIZON_STEPS 40.0f /* obstacles farther away do not count */
#define PP_SAFE_STEPS 8.0f     /* free water a passable direction needs (1.6 s at 5 Hz) */
#define PP_CREEP_STEPS 3.0f    /* known free water worth creeping into when nothing is passable */
#define PP_CLEAR_SHARE 0.75f   /* of the median range of the surfaces in view: the water directions' bound */
#define PP_MIN_VIEW 10         /* surfaces in view the bound needs */

enum { PP_STOP = 0, PP_GO = 1, PP_CREEP = 2, PP_TURN = 3 };

typedef struct {
    int32_t action;              /* PP_GO, PP_CREEP, PP_TURN or PP_STOP */
    int32_t direction;           /* the direction to head for (PP_GO, PP_CREEP), else -1 */
    int32_t turn;                /* PP_TURN: -1 left, 1 right; else 0 */
    int32_t goal;                /* the goal's direction */
    float speed;                 /* one step, in odometry units: the recent speed; 0 when unknown */
    float range[OD_DIRECTIONS];  /* each direction's range in odometry units, -1 when unknown */
    uint16_t passable;           /* bit d: direction d passable */
    uint16_t known;              /* bit d: the map knows direction d's range */
    float nearest;               /* the nearest obstacle voxel in front of the camera, odometry units; -1 none */
    float clear;                 /* the water directions are free at least this far, odometry units; -1 unknown */
    uint32_t frames;             /* plans made */
} pp_plan_t;

void pp_init(pp_plan_t *plan);
uint32_t pp_plan_size(void);

/* Plans after an odometry step: map updated with that step (vm_update), dirs the obstacle directions after the latest
 * mask (NULL for none) on layout (NULL: every direction counts as seen), goal a direction in world axes (NULL: straight
 * ahead). Returns the action, or -1 for invalid arguments, which leave the plan as it was. */
int pp_update(pp_plan_t *plan, const vm_map_t *map, const mvo_t *vo, const od_layout_t *layout,
              const od_state_t *dirs, const float goal[3]);

#endif
