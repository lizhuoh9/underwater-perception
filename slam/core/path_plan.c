/* See path_plan.h. */
#include "path_plan.h"

#include <float.h>
#include <stddef.h>
#include <string.h>

/* No byte of the plan is padding: the board hashes it whole. */
typedef char pp_plan_has_no_padding[(offsetof(pp_plan_t, range) == 20
                                     && offsetof(pp_plan_t, passable) == 20 + 4 * OD_DIRECTIONS
                                     && sizeof(pp_plan_t) == offsetof(pp_plan_t, frames) + 4) ? 1 : -1];

static int finite_float(float value) { return value >= -FLT_MAX && value <= FLT_MAX; }

static int distance(int a, int b) { return a < b ? b - a : a - b; }

void pp_init(pp_plan_t *plan)
{
    int d;
    if (!plan) {
        return;
    }
    memset(plan, 0, sizeof *plan);
    plan->direction = -1;
    plan->goal = OD_CENTRE;
    for (d = 0; d < OD_DIRECTIONS; ++d) {
        plan->range[d] = -1.0f;
    }
    plan->nearest = -1.0f;
}

uint32_t pp_plan_size(void) { return (uint32_t)sizeof(pp_plan_t); }

/* The voxels read PP_GROUP to a side (see path_plan.h): cells in the order their first voxel was seen, found through an
 * open-addressing index like the map's own. */
typedef struct {
    int16_t key[3];  /* the voxels' indices divided by PP_GROUP, rounded down */
    uint16_t frames; /* the frames that observed its voxels, added (at most PP_GROUP^3 voxels of 255 each) */
    uint8_t now;     /* one of its voxels was filed in the map's latest frame */
} pp_cell_t;

static pp_cell_t s_cells[VM_CAPACITY];
static uint16_t s_slots[VM_SLOTS]; /* 1 + a cell's place in s_cells, 0 where free */

static int16_t group_of(int16_t k) { return (int16_t)(k >= 0 ? k / PP_GROUP : -((-k + PP_GROUP - 1) / PP_GROUP)); }

/* Groups the map's voxels into s_cells; returns how many cells. */
static uint32_t group_voxels(const vm_map_t *map)
{
    uint32_t k, n = 0;
    memset(s_slots, 0, sizeof s_slots);
    for (k = 0; k < map->count; ++k) {
        const vm_voxel_t *v = &map->voxel[k];
        int16_t key[3];
        uint32_t i, frames;
        key[0] = group_of(v->key[0]);
        key[1] = group_of(v->key[1]);
        key[2] = group_of(v->key[2]);
        i = ((uint32_t)(uint16_t)key[0] * 73856093u ^ (uint32_t)(uint16_t)key[1] * 19349663u
             ^ (uint32_t)(uint16_t)key[2] * 83492791u) & (VM_SLOTS - 1u);
        while (s_slots[i] && memcmp(s_cells[s_slots[i] - 1].key, key, sizeof key) != 0) {
            i = (i + 1u) & (VM_SLOTS - 1u);
        }
        if (!s_slots[i]) {
            memcpy(s_cells[n].key, key, sizeof key);
            s_cells[n].frames = 0;
            s_cells[n].now = 0;
            s_slots[i] = (uint16_t)(++n);
        }
        frames = (uint32_t)s_cells[s_slots[i] - 1].frames + v->frames;
        s_cells[s_slots[i] - 1].frames = (uint16_t)(frames > 65535u ? 65535u : frames);
        s_cells[s_slots[i] - 1].now |= (uint8_t)(v->seen == (uint16_t)map->frames);
    }
    return n;
}

/* One step: the median of the odometry's last step lengths, as mvo_step_length takes it; 0 before the first step or
 * when one is not a positive finite length. */
static float recent_speed(const mvo_t *vo)
{
    float lengths[MVO_SPEED_STEPS];
    const int n = vo->recent_count;
    int i, j;
    if (n < 1 || n > MVO_SPEED_STEPS) {
        return 0.0f;
    }
    for (i = 0; i < n; ++i) { /* insertion sort */
        const float v = vo->recent_length[i];
        if (!(v > 0.0f) || !finite_float(v)) {
            return 0.0f;
        }
        for (j = i - 1; j >= 0 && lengths[j] > v; --j) {
            lengths[j + 1] = lengths[j];
        }
        lengths[j + 1] = v;
    }
    return n % 2 ? lengths[n / 2] : 0.5f * (lengths[n / 2 - 1] + lengths[n / 2]);
}

/* The median of the ranges whose squares are view[0..n-1] (n > 0; reordered): the middle one, or the mean of the two
 * middle ones. */
static float median_range(float *view, int n)
{
    int i, j;
    for (i = 1; i < n; ++i) { /* insertion sort */
        const float v = view[i];
        for (j = i - 1; j >= 0 && view[j] > v; --j) {
            view[j + 1] = view[j];
        }
        view[j + 1] = v;
    }
    return n % 2 ? mvo_sqrt(view[n / 2]) : 0.5f * (mvo_sqrt(view[n / 2 - 1]) + mvo_sqrt(view[n / 2]));
}

/* World vector w in the camera's axes: the transpose of the camera-to-world rotation times w. */
static void to_camera(const mvo_t *vo, const float w[3], float c[3])
{
    int i;
    for (i = 0; i < 3; ++i) {
        c[i] = vo->world_rotation[i] * w[0] + vo->world_rotation[3 + i] * w[1] + vo->world_rotation[6 + i] * w[2];
    }
}

/* The goal's direction (see path_plan.h). */
static int goal_direction(const mvo_t *vo, const float goal[3])
{
    float g[3], across, up;
    if (!goal) {
        return OD_CENTRE;
    }
    to_camera(vo, goal, g);
    if (!finite_float(g[0]) || !finite_float(g[1]) || !finite_float(g[2])
            || (g[0] == 0.0f && g[1] == 0.0f && g[2] == 0.0f)) {
        return OD_CENTRE;
    }
    if (g[2] > 0.0f) {
        return od_direction_of(g[0] / g[2], g[1] / g[2]);
    }
    across = g[0] < 0.0f ? -g[0] : g[0];
    up = g[1] < 0.0f ? -g[1] : g[1];
    if (across >= up) {
        return OD_COLUMNS + (g[0] > 0.0f ? OD_COLUMNS - 1 : 0); /* level, the hard sector on its side */
    }
    return (g[1] < 0.0f ? 0 : 2 * OD_COLUMNS) + OD_COLUMNS / 2; /* the centre of the band above or below */
}

static int steps_between(int a, int b)
{
    return distance(a % OD_COLUMNS, b % OD_COLUMNS) + distance(a / OD_COLUMNS, b / OD_COLUMNS);
}

/* Whether passable direction d beats best (-1: none yet) towards target; second[] the squared ranges, -1 unknown. */
static int better(int d, int best, int target, const float second[OD_DIRECTIONS], int previous)
{
    int a, b;
    if (best < 0) {
        return 1;
    }
    a = steps_between(d, target);
    b = steps_between(best, target);
    if (a != b) {
        return a < b;
    }
    a = distance(d / OD_COLUMNS, OD_ROWS / 2);
    b = distance(best / OD_COLUMNS, OD_ROWS / 2);
    if (a != b) {
        return a < b; /* turning before rising or diving */
    }
    if ((second[d] < 0.0f) != (second[best] < 0.0f)) {
        return second[d] < 0.0f; /* unknown counts as beyond the horizon */
    }
    if (second[d] != second[best]) {
        return second[d] > second[best];
    }
    return d == previous;
}

/* The side to turn to when nothing is passable: -1 left, 1 right (see path_plan.h). */
static int turn_side(const pp_plan_t *plan, uint32_t left, uint32_t right, const od_state_t *dirs, int target)
{
    if (left != right) {
        return left < right ? -1 : 1;
    }
    if (dirs) {
        const int mask_left = dirs->share[OD_COLUMNS] + dirs->share[OD_COLUMNS + 1];
        const int mask_right = dirs->share[2 * OD_COLUMNS - 2] + dirs->share[2 * OD_COLUMNS - 1];
        if (mask_left != mask_right) {
            return mask_left < mask_right ? -1 : 1;
        }
    }
    if (target % OD_COLUMNS != OD_COLUMNS / 2) {
        return target % OD_COLUMNS < OD_COLUMNS / 2 ? -1 : 1;
    }
    return plan->turn ? plan->turn : -1;
}

int pp_update(pp_plan_t *plan, const vm_map_t *map, const mvo_t *vo, const od_layout_t *layout,
              const od_state_t *dirs, const float goal[3])
{
    float first[OD_DIRECTIONS], second[OD_DIRECTIONS]; /* squared distances of the two nearest obstacles, -1 none */
    float view[MVO_MAX_FEATURES];                      /* squared distances of the surfaces in view */
    float speed, horizon2 = 0.0f, safe2 = 0.0f, creep2 = 0.0f, nearest2 = -1.0f;
    uint32_t k, left = 0, right = 0;
    int in_view = 0;
    uint16_t passable = 0, known = 0;
    int d, i, best = -1, target, action, turn = 0;
    const int masked = dirs && dirs->frames > 0; /* a mask has come */
    if (!plan || !map || !vo || map->count > VM_CAPACITY) {
        return -1;
    }
    for (d = 0; d < OD_DIRECTIONS; ++d) {
        first[d] = second[d] = -1.0f;
    }
    speed = recent_speed(vo);
    if (speed > 0.0f && map->side > 0.0f) {
        horizon2 = PP_HORIZON_STEPS * speed * (PP_HORIZON_STEPS * speed);
        safe2 = PP_SAFE_STEPS * speed * (PP_SAFE_STEPS * speed);
        creep2 = PP_CREEP_STEPS * speed * (PP_CREEP_STEPS * speed);
        const uint32_t count = group_voxels(map);
        const float side = PP_GROUP * map->side;
        for (k = 0; k < count; ++k) {
            const pp_cell_t *v = &s_cells[k];
            const int now = v->now;
            float p[3], q[3], d2;
            for (i = 0; i < 3; ++i) {
                p[i] = ((float)v->key[i] + 0.5f) * side - vo->world_centre[i];
            }
            to_camera(vo, p, q);
            d2 = q[0] * q[0] + q[1] * q[1] + q[2] * q[2];
            if (!(d2 <= horizon2)) { /* NaN too */
                continue;
            }
            if (now && q[2] > 0.0f && in_view < MVO_MAX_FEATURES) {
                view[in_view++] = d2; /* filed in this frame: a surface in view */
            }
            if (v->frames < VM_CONFIRM) {
                continue;
            }
            if (d2 < safe2) { /* the turn's tally, all around */
                left += q[0] < 0.0f;
                right += q[0] > 0.0f;
            }
            if (!(q[2] > 0.0f)) {
                continue;
            }
            if (nearest2 < 0.0f || d2 < nearest2) {
                nearest2 = d2;
            }
            d = od_direction_of(q[0] / q[2], q[1] / q[2]);
            if (first[d] < 0.0f || d2 < first[d]) {
                second[d] = first[d];
                first[d] = d2;
            } else if (second[d] < 0.0f || d2 < second[d]) {
                second[d] = d2;
            }
        }
    }
    for (d = 0; d < OD_DIRECTIONS; ++d) {
        const int seen = !layout || layout->cells[d] > 0;
        const int has = second[d] >= 0.0f, near = has && second[d] < safe2;
        if (has) {
            known |= (uint16_t)(1u << d);
        }
        if (seen && !near && !(masked && dirs->blocked[d])) {
            passable |= (uint16_t)(1u << d);
        }
    }
    target = goal_direction(vo, goal);
    for (d = 0; d < OD_DIRECTIONS; ++d) {
        if ((passable >> d) & 1u && better(d, best, target, second, plan->direction)) {
            best = d;
        }
    }
    action = PP_GO;
    if (best < 0) { /* creep where the map knows the most free water */
        for (d = 0; d < OD_DIRECTIONS; ++d) {
            if ((!layout || layout->cells[d] > 0) && second[d] >= creep2 && second[d] >= 0.0f
                    && (best < 0 || second[d] > second[best]
                        || (second[d] == second[best] && steps_between(d, target) < steps_between(best, target)))) {
                best = d;
            }
        }
        action = PP_CREEP;
    }
    if (best < 0) {
        action = speed > 0.0f || masked ? PP_TURN : PP_STOP;
        turn = action == PP_TURN ? turn_side(plan, left, right, masked ? dirs : NULL, target) : 0;
    }
    plan->action = action;
    plan->direction = best;
    plan->turn = turn;
    plan->goal = target;
    plan->speed = speed;
    for (d = 0; d < OD_DIRECTIONS; ++d) {
        plan->range[d] = second[d] >= 0.0f ? mvo_sqrt(second[d]) : -1.0f;
    }
    plan->passable = passable;
    plan->known = known;
    plan->nearest = nearest2 >= 0.0f ? mvo_sqrt(nearest2) : -1.0f;
    plan->clear = in_view >= PP_MIN_VIEW ? PP_CLEAR_SHARE * median_range(view, in_view) : -1.0f;
    if (plan->frames < UINT32_MAX) {
        ++plan->frames;
    }
    return action;
}
