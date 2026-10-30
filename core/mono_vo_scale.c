/* See mono_vo.h: the run's metric scale from a depth sensor and a forward ranger, and the world centre in metres. */
#include "mono_vo_internal.h"

#include <string.h>

int mvo_depth_reading(mvo_t *s, float depth)
{
    if (!mvo_finite(depth)) {
        return 0;
    }
    s->scale.depth = depth;
    s->scale.has_depth = 1;
    return 1;
}

int mvo_range_setup(mvo_t *s, const float axis[3], float half_angle_cos)
{
    const float size2 = dot3(axis, axis);
    float unit;
    if (!mvo_finite(size2) || !(size2 > 1.0e-12f) || !(half_angle_cos > 0.0f && half_angle_cos < 1.0f)) {
        return 0;
    }
    unit = mvo_rsqrt(size2);
    s->scale.beam[0] = axis[0] * unit;
    s->scale.beam[1] = axis[1] * unit;
    s->scale.beam[2] = axis[2] * unit;
    s->scale.beam_cos = half_angle_cos;
    s->scale.has_beam = 1;
    return 1;
}

int mvo_range_reading(mvo_t *s, float range)
{
    if (!s->scale.has_beam || !mvo_finite(range) || !(range > 0.0f)) {
        return 0;
    }
    s->scale.range = range;
    s->scale.has_range = 1;
    return 1;
}

int mvo_metric(const mvo_t *s, float centre[3], float *scale, float *variance)
{
    const int known = s->scale.scale > 0.0f;
    centre[0] = known ? s->scale.metric[0] : 0.0f;
    centre[1] = known ? s->scale.metric[1] : 0.0f;
    centre[2] = known ? s->scale.metric[2] : 0.0f;
    if (scale) {
        *scale = s->scale.scale;
    }
    if (variance) {
        *variance = known ? s->scale.variance : 0.0f;
    }
    return known;
}

void mvo_scale_start(mvo_t *s)
{
    mvo_scale_t *m = &s->scale;
    const float depth = m->depth, beam_cos = m->beam_cos, beam[3] = {m->beam[0], m->beam[1], m->beam[2]};
    const int has_depth = m->has_depth, has_beam = m->has_beam;
    memset(m, 0, sizeof *m);
    if (has_beam) {
        memcpy(m->beam, beam, sizeof m->beam);
        m->beam_cos = beam_cos;
        m->has_beam = 1;
    }
    if (has_depth) { /* the first frame's depth: the vertical's zero */
        m->first_depth = depth;
        m->has_first_depth = 1;
    }
}

/* One observation z of the scale with variance r. The first sets it, since it scales the path so far too, only when
 * `doubt`, its variance with the odometry's own error counted, is at most MVO_SCALE_FIRST_SPREAD of it, squared; the
 * others weigh in by the variances. */
static void update(mvo_scale_t *m, float z, float r, float doubt)
{
    float gain;
    if (!(z > 0.0f && z < 3.0e38f) || !(r > 0.0f && r < 3.0e38f)) {
        return;
    }
    if (!(m->scale > 0.0f)) {
        if (doubt <= (MVO_SCALE_FIRST_SPREAD * z) * (MVO_SCALE_FIRST_SPREAD * z)) {
            m->scale = z;
            m->variance = r;
        }
        return;
    }
    gain = m->variance / (m->variance + r);
    m->scale += gain * (z - m->scale);
    m->variance -= gain * m->variance;
}

/* The range against the tracked points in the beam: the distance a share MVO_RANGE_SHARE of them lie within (the
 * nearer ones, where the ranger's echo comes from), interpolated as a quantile; at least MVO_RANGE_POINTS of them. */
static void observe_range(mvo_t *s)
{
    mvo_scale_t *m = &s->scale;
    float near[MVO_MAX_FEATURES], position, d;
    int n = 0, i, j, low;
    for (i = 0; i < s->count; ++i) {
        const float *p = s->feature[i].point;
        float distance;
        if (!s->feature[i].has_point) {
            continue;
        }
        distance = mvo_root(dot3(p, p));
        if (distance > 0.0f && dot3(p, m->beam) > m->beam_cos * distance) {
            near[n++] = distance;
        }
    }
    if (n < MVO_RANGE_POINTS) {
        return;
    }
    for (i = 1; i < n; ++i) { /* insertion sort: a beam holds few points */
        const float v = near[i];
        for (j = i; j > 0 && near[j - 1] > v; --j) {
            near[j] = near[j - 1];
        }
        near[j] = v;
    }
    position = MVO_RANGE_SHARE * (float)(n - 1);
    low = (int)position;
    d = near[low] + (position - (float)low) * (near[low + 1 < n ? low + 1 : low] - near[low]);
    if (d > 0.0f) {
        const float z = m->range / d, r = (MVO_RANGE_NOISE * z) * (MVO_RANGE_NOISE * z);
        update(m, z, r, r);
    }
}

/* The depth's change over the last MVO_DEPTH_HORIZON seconds against the odometry's along the world's down over the
 * same steps: their ratio when the depth changed by MVO_DEPTH_MIN_CHANGE both ways alike. Its variance is the sensor's
 * noise on both ends over the square of the odometry's change; for a first observation the odometry's own vertical
 * error, MVO_DEPTH_TRAVEL of the distance it moved meanwhile, counts too. */
static void observe_depth(mvo_t *s, const float down[3])
{
    mvo_scale_t *m = &s->scale;
    const float odometry = dot3(s->world_centre, down);
    int k, back = -1;
    for (k = 0; k < m->ring_count; ++k) { /* the oldest reading within the horizon */
        const int at = (m->ring_next - m->ring_count + k + MVO_DEPTH_RING) % MVO_DEPTH_RING;
        if (m->ring_time[at] >= m->time - MVO_DEPTH_HORIZON) {
            back = at;
            break;
        }
    }
    if (back >= 0) {
        const float dz = m->depth - m->ring_depth[back], dv = odometry - m->ring_down[back];
        if (mvo_abs(dz) >= MVO_DEPTH_MIN_CHANGE && dz * dv > 0.0f) {
            const float z = dz / dv, noise = 2.0f * MVO_DEPTH_NOISE * MVO_DEPTH_NOISE;
            const float travel = MVO_DEPTH_TRAVEL * z * (m->path - m->ring_path[back]);
            update(m, z, noise / (dv * dv), (noise + travel * travel) / (dv * dv));
        }
    }
    m->ring_depth[m->ring_next] = m->depth;
    m->ring_down[m->ring_next] = odometry;
    m->ring_time[m->ring_next] = m->time;
    m->ring_path[m->ring_next] = m->path;
    m->ring_next = (m->ring_next + 1) % MVO_DEPTH_RING;
    if (m->ring_count < MVO_DEPTH_RING) {
        ++m->ring_count;
    }
}

void mvo_scale_step(mvo_t *s, const float moved[3])
{
    mvo_scale_t *m = &s->scale;
    const float before = m->scale;
    float up[3], down[3];
    int has_down;
    mvo_imu_up(s, up);
    has_down = dot3(up, up) > 0.0f;
    down[0] = -up[0];
    down[1] = -up[1];
    down[2] = -up[2];
    m->time += m->step_time;
    m->path += mvo_root(dot3(moved, moved));
    if (m->scale > 0.0f) {
        m->variance += MVO_SCALE_DRIFT * m->scale * m->scale * m->step_time;
    }
    if (m->has_range) {
        observe_range(s);
    }
    if (m->has_depth && has_down) {
        observe_depth(s, down);
    }
    if (m->scale > 0.0f) {
        int i;
        for (i = 0; i < 3; ++i) { /* the first scale takes the path so far too */
            m->metric[i] = before > 0.0f ? m->metric[i] + m->scale * moved[i] : m->scale * s->world_centre[i];
        }
        if (m->has_depth && has_down && m->has_first_depth) {
            const float offset = (m->depth - m->first_depth) - dot3(m->metric, down);
            for (i = 0; i < 3; ++i) {
                m->metric[i] += offset * down[i];
            }
        }
    }
    m->has_depth = 0;
    m->has_range = 0;
}
