/* See voxel_map.h. */
#include "voxel_map.h"

#include <float.h>
#include <stddef.h>
#include <string.h>

#define VM_FAR 256 /* the eviction class of voxels beyond VM_RADIUS; ages run 0..255 */

/* The index needs a free slot to end every probe, its size a power of two, and its entries 1 + a voxel's place. */
typedef char vm_slots_fit_the_capacity[(VM_SLOTS > VM_CAPACITY && (VM_SLOTS & (VM_SLOTS - 1)) == 0
                                        && VM_CAPACITY < 65535) ? 1 : -1];
/* No byte of the map is padding: the board hashes it whole. */
typedef char vm_map_has_no_padding[(sizeof(vm_voxel_t) == 10
                                    && offsetof(vm_map_t, side) == VM_CAPACITY * 10 + VM_SLOTS * 2
                                    && sizeof(vm_map_t) == offsetof(vm_map_t, lost) + 4) ? 1 : -1];

void vm_init(vm_map_t *m)
{
    if (m) {
        memset(m, 0, sizeof *m);
    }
}

uint32_t vm_map_size(void) { return (uint32_t)sizeof(vm_map_t); }

static int finite_float(float value) { return value >= -FLT_MAX && value <= FLT_MAX; }

static int floor_int(float v) /* v within +-VM_MAX_KEY */
{
    const int i = (int)v;
    return v < (float)i ? i - 1 : i;
}

static uint32_t home(const int16_t key[3])
{
    uint32_t h = (uint32_t)(int32_t)key[0] * 73856093u ^ (uint32_t)(int32_t)key[1] * 19349663u
                 ^ (uint32_t)(int32_t)key[2] * 83492791u;
    h ^= h >> 16;
    return h & (VM_SLOTS - 1u);
}

static int same(const int16_t a[3], const int16_t b[3]) { return a[0] == b[0] && a[1] == b[1] && a[2] == b[2]; }

/* The index slot of key: the one naming its voxel, or the free one where it would go. */
static uint32_t slot_of(const vm_map_t *m, const int16_t key[3])
{
    uint32_t i = home(key);
    while (m->index[i] && !same(m->voxel[m->index[i] - 1].key, key)) {
        i = (i + 1u) & (VM_SLOTS - 1u);
    }
    return i;
}

int vm_key(const vm_map_t *m, const float p[3], int16_t key[3])
{
    int16_t out[3];
    int i;
    if (!m || !p || !key || !(m->side > 0.0f)) {
        return 0;
    }
    for (i = 0; i < 3; ++i) {
        const float q = p[i] / m->side;
        if (!(q > -(float)VM_MAX_KEY && q < (float)VM_MAX_KEY)) { /* NaN too */
            return 0;
        }
        out[i] = (int16_t)floor_int(q);
    }
    memcpy(key, out, sizeof out);
    return 1;
}

const vm_voxel_t *vm_find(const vm_map_t *m, const int16_t key[3])
{
    uint32_t i;
    if (!m || !key || m->count > VM_CAPACITY) {
        return NULL;
    }
    i = slot_of(m, key);
    return m->index[i] ? &m->voxel[m->index[i] - 1] : NULL;
}

int vm_occupied(const vm_map_t *m, const float p[3])
{
    int16_t key[3];
    const vm_voxel_t *voxel;
    if (!vm_key(m, p, key)) {
        return 0;
    }
    voxel = vm_find(m, key);
    return voxel && voxel->frames >= VM_CONFIRM;
}

/* Whether feature f is an observation: a point well triangulated in front, on a cell that is not water when a mask
 * is given. */
static int usable(const mvo_feature_t *f, const uint8_t *mask, int mask_w, int mask_h)
{
    float x, y;
    int column, row;
    if (!f->has_point || !(f->spread >= VM_MIN_SPREAD) || !finite_float(f->spread) || !(f->point[2] > 0.0f)
            || !finite_float(f->point[0]) || !finite_float(f->point[1]) || !finite_float(f->point[2])) {
        return 0;
    }
    if (!mask) {
        return 1;
    }
    x = f->x * (float)mask_w / (float)MVO_WIDTH;
    y = f->y * (float)mask_h / (float)MVO_HEIGHT;
    column = !(x >= 0.0f) ? 0 : (x >= (float)mask_w ? mask_w - 1 : (int)x);
    row = !(y >= 0.0f) ? 0 : (y >= (float)mask_h ? mask_h - 1 : (int)y);
    return mask[row * mask_w + column] != 0;
}

/* The side this frame's observations ask for: their median depth (along the optical axis, the third coordinate of a
 * feature's point) over VM_CELLS_PER_DEPTH, or 0 with fewer than VM_MIN_POINTS of them. */
static float scene_side(const mvo_t *vo, const uint8_t *mask, int mask_w, int mask_h)
{
    float depths[MVO_MAX_FEATURES], side;
    int k, j, count = 0;
    for (k = 0; k < vo->count; ++k) {
        const mvo_feature_t *f = &vo->feature[k];
        if (!usable(f, mask, mask_w, mask_h)) {
            continue;
        }
        for (j = count - 1; j >= 0 && depths[j] > f->point[2]; --j) { /* insertion sort */
            depths[j + 1] = depths[j];
        }
        depths[j + 1] = f->point[2];
        ++count;
    }
    if (count < VM_MIN_POINTS) {
        return 0.0f;
    }
    side = (count % 2 ? depths[count / 2] : 0.5f * (depths[count / 2 - 1] + depths[count / 2])) / VM_CELLS_PER_DEPTH;
    return side > 0.0f && finite_float(side) ? side : 0.0f;
}

/* Rebuilds the index from the voxels, dropping those marked with no observations and merging each voxel into an
 * earlier one with the same key; the order stays. */
static void reindex(vm_map_t *m)
{
    uint32_t k, kept = 0;
    memset(m->index, 0, sizeof m->index);
    for (k = 0; k < m->count; ++k) {
        const vm_voxel_t voxel = m->voxel[k];
        uint32_t i;
        if (!voxel.hits) {
            continue;
        }
        i = slot_of(m, voxel.key);
        if (m->index[i]) {
            vm_voxel_t *into = &m->voxel[m->index[i] - 1];
            const uint16_t now = (uint16_t)m->frames;
            into->hits = (uint8_t)(into->hits + voxel.hits > 255 ? 255 : into->hits + voxel.hits);
            into->frames = voxel.frames > into->frames ? voxel.frames : into->frames;
            if ((uint16_t)(now - voxel.seen) < (uint16_t)(now - into->seen)) {
                into->seen = voxel.seen;
            }
            if (m->merged < UINT32_MAX) {
                ++m->merged;
            }
            continue;
        }
        m->voxel[kept] = voxel; /* kept <= k: only voxels already read are overwritten */
        m->index[i] = (uint16_t)(kept + 1u);
        ++kept;
    }
    m->count = kept;
}

/* Re-keys the voxels to a new side, each by its centre; one that would leave the keys is dropped. */
static void rescale(vm_map_t *m, float side)
{
    const float ratio = m->side / side;
    uint32_t k;
    int i;
    for (k = 0; k < m->count; ++k) {
        vm_voxel_t *voxel = &m->voxel[k];
        for (i = 0; i < 3; ++i) {
            const float q = ((float)voxel->key[i] + 0.5f) * ratio;
            if (!(q > -(float)VM_MAX_KEY && q < (float)VM_MAX_KEY)) {
                voxel->hits = 0;
                if (m->evicted < UINT32_MAX) {
                    ++m->evicted;
                }
                break;
            }
            voxel->key[i] = (int16_t)floor_int(q);
        }
    }
    m->side = side;
    if (m->rescales < UINT32_MAX) {
        ++m->rescales;
    }
    reindex(m);
}

/* The eviction class of a voxel, the highest dropped first: 0 when seen in this frame (never dropped), VM_FAR beyond
 * VM_RADIUS of the camera's voxel, else its age in frames, from 128 up for a voxel seen in fewer than VM_CONFIRM frames
 * (an outlier or a passing glimpse more often than a surface) and below 128 for an obstacle. */
static int class_of(const vm_map_t *m, const vm_voxel_t *v, const int32_t camera[3])
{
    const uint16_t age = (uint16_t)((uint16_t)m->frames - v->seen);
    int i;
    if (age == 0) {
        return 0;
    }
    for (i = 0; i < 3; ++i) {
        const int32_t d = (int32_t)v->key[i] - camera[i];
        if (d > VM_RADIUS || d < -VM_RADIUS) {
            return VM_FAR;
        }
    }
    return (v->frames < VM_CONFIRM ? 128 : 0) + (age > 127 ? 127 : (int)age);
}

/* Drops every far voxel and, when they are fewer than VM_EVICT, the longest unseen until VM_EVICT are gone: the classes
 * above a cutoff whole and the cutoff's first-seen voxels (array order, the same on every target) up to the count.
 * Ages stop at 127 frames, so after a long time without new voxels many share one class; a whole class could be the
 * whole map. Never one seen in this frame. */
static void evict(vm_map_t *m, const int32_t camera[3])
{
    uint16_t classes[VM_FAR + 1] = {0};
    uint32_t k, need, above = 0, quota = 0;
    int cutoff = VM_FAR;
    for (k = 0; k < m->count; ++k) {
        ++classes[class_of(m, &m->voxel[k], camera)];
    }
    need = classes[VM_FAR] < VM_EVICT ? VM_EVICT - classes[VM_FAR] : 0;
    if (need) {
        for (cutoff = VM_FAR - 1; cutoff > 1 && above + classes[cutoff] < need; --cutoff) {
            above += classes[cutoff];
        }
        quota = need - above; /* more than the cutoff class holds only at cutoff 1: then all of it */
    }
    for (k = 0; k < m->count; ++k) {
        const int c = class_of(m, &m->voxel[k], camera);
        int drop = c == VM_FAR || (c > cutoff && c < VM_FAR);
        if (!drop && c == cutoff && quota > 0) {
            --quota;
            drop = 1;
        }
        if (drop) {
            m->voxel[k].hits = 0;
            if (m->evicted < UINT32_MAX) {
                ++m->evicted;
            }
        }
    }
    reindex(m);
}

/* The voxel at key, appended with no observations yet when absent; NULL when the map is full of voxels it may not
 * drop. */
static vm_voxel_t *find_or_insert(vm_map_t *m, const int16_t key[3], const int32_t camera[3])
{
    uint32_t i = slot_of(m, key);
    vm_voxel_t *voxel;
    if (m->index[i]) {
        return &m->voxel[m->index[i] - 1];
    }
    if (m->count >= VM_CAPACITY) {
        evict(m, camera);
        if (m->count >= VM_CAPACITY) {
            return NULL;
        }
        i = slot_of(m, key);
    }
    voxel = &m->voxel[m->count];
    memcpy(voxel->key, key, sizeof voxel->key);
    voxel->hits = voxel->frames = 0;
    voxel->seen = (uint16_t)(m->frames - 1u); /* not this frame yet */
    m->index[i] = (uint16_t)(m->count + 1u);
    ++m->count;
    return voxel;
}

int vm_update(vm_map_t *m, const mvo_t *vo, const mvo_step_t *step, const uint8_t *mask, int mask_w, int mask_h)
{
    int32_t camera[3];
    float side;
    int k, i, accepted = 0;
    if (!m || !step || m->count > VM_CAPACITY || (mask && (mask_w < 1 || mask_h < 1))) {
        return 0;
    }
    if (step->kind == MVO_ESSENTIAL && (!vo || vo->count < 0 || vo->count > MVO_MAX_FEATURES)) {
        return 0;
    }
    ++m->frames;
    if (step->kind == MVO_LOST && m->lost < UINT32_MAX) {
        ++m->lost;
    }
    if (step->kind != MVO_ESSENTIAL) {
        return 0;
    }
    side = scene_side(vo, mask, mask_w, mask_h);
    if (side > 0.0f && !(m->side > 0.0f)) {
        m->side = side;
    } else if (side > 0.0f && (side > VM_RESCALE * m->side || side * VM_RESCALE < m->side)) {
        rescale(m, side);
    }
    if (!(m->side > 0.0f)) {
        return 0;
    }
    for (i = 0; i < 3; ++i) { /* the camera's voxel, clamped: a camera beyond the keys makes every voxel far */
        const float q = vo->world_centre[i] / m->side;
        camera[i] = !(q > -(float)VM_MAX_KEY) ? -VM_MAX_KEY - VM_RADIUS - 1
                    : (!(q < (float)VM_MAX_KEY) ? VM_MAX_KEY + VM_RADIUS + 1 : (int32_t)floor_int(q));
    }
    for (k = 0; k < vo->count; ++k) {
        const mvo_feature_t *f = &vo->feature[k];
        float world[3];
        int16_t key[3];
        vm_voxel_t *voxel;
        if (!usable(f, mask, mask_w, mask_h)) {
            continue;
        }
        for (i = 0; i < 3; ++i) {
            world[i] = vo->world_rotation[3 * i] * f->point[0] + vo->world_rotation[3 * i + 1] * f->point[1]
                       + vo->world_rotation[3 * i + 2] * f->point[2] + vo->world_centre[i];
        }
        if (!vm_key(m, world, key) || !(voxel = find_or_insert(m, key, camera))) {
            continue;
        }
        if (voxel->hits < 255) {
            ++voxel->hits;
        }
        if (voxel->seen != (uint16_t)m->frames) {
            voxel->seen = (uint16_t)m->frames;
            if (voxel->frames < 255) {
                ++voxel->frames;
            }
        }
        ++accepted;
    }
    m->observations += (uint32_t)accepted;
    return accepted;
}
