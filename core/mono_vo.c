/* See mono_vo.h. The odometry is split over the units that mono_vo_internal.h lists; this one holds the public
 * functions and the step. */
#include "mono_vo_internal.h"

#include <string.h>

/* The scratch the units share (see mono_vo_internal.h). */
float mvo_pa[MVO_MAX_FEATURES][2], mvo_pb[MVO_MAX_FEATURES][2];
float mvo_ra[MVO_MAX_FEATURES][2], mvo_rb[MVO_MAX_FEATURES][2];
uint8_t mvo_inlier[MVO_MAX_FEATURES], mvo_best[MVO_MAX_FEATURES], mvo_epipolar[MVO_MAX_FEATURES];
float mvo_values[MVO_MAX_FEATURES];

uint32_t mvo_state_size(void) { return (uint32_t)sizeof(mvo_t); }

uint32_t mvo_pyramid_bytes(void) { return (uint32_t)MVO_PYRAMID_BYTES; }

const char *mvo_config_name(void) { return MVO_CONFIG_NAME; }

uint32_t mvo_max_features(void) { return (uint32_t)MVO_MAX_FEATURES; }

float mvo_sqrt(float v) { return mvo_root(v); }

int mvo_tracks(const mvo_t *s, float *xy, uint32_t *ids, int capacity)
{
    int k;
    for (k = 0; k < s->count && k < capacity; ++k) {
        xy[2 * k] = s->feature[k].x;
        xy[2 * k + 1] = s->feature[k].y;
        ids[k] = s->feature[k].id;
    }
    return k;
}

void mvo_init(mvo_t *s, const mvo_camera_t *camera, uint32_t seed, uint8_t *pyramid_a, uint8_t *pyramid_b)
{
    memset(s, 0, sizeof *s);
    s->camera = *camera;
    s->pyramid[0] = pyramid_a;
    s->pyramid[1] = pyramid_b;
    s->current = 1; /* the first frame goes into pyramid 0 */
    s->rng = seed ? seed : 1u;
    s->world_rotation[0] = s->world_rotation[4] = s->world_rotation[8] = 1.0f;
    s->gyro.mount[0] = s->gyro.mount[4] = s->gyro.mount[8] = 1.0f;
    s->gyro.pending[0] = 1.0f;
}

uint8_t *mvo_frame(mvo_t *s) { return s->pyramid[s->current ^ 1]; }

static int64_t mvo_now(const mvo_t *s) { return s->timing.clock ? s->timing.clock() : 0; }

int mvo_gyro_mount(mvo_t *s, const float mount[9])
{
    int i;
    for (i = 0; i < 9; ++i) {
        if (!mvo_finite(mount[i])) {
            return 0;
        }
    }
    memcpy(s->gyro.mount, mount, sizeof s->gyro.mount);
    return 1;
}

/* mvo_gyro_rate's work; returns 0 when it drops the reading. */
static int integrate(mvo_t *s, const float rate[3], float dt)
{
    mvo_gyro_t *g = &s->gyro;
    const float half = 0.5f * dt;
    float w[3], h[3], h2, sine, d[4], q[4], scale;
    if (!(dt > 0.0f)) {
        return 0;
    }
    w[0] = rate[0] - g->bias[0];
    w[1] = rate[1] - g->bias[1];
    w[2] = rate[2] - g->bias[2];
    mat3_vec(g->mount, w, h);
    h[0] *= half; /* half the reading's turn, in the camera's axes */
    h[1] *= half;
    h[2] *= half;
    h2 = dot3(h, h);
    if (!(h2 < 1.0f)) { /* not finite, or a reading turning 2 rad or more: far outside the series */
        return 0;
    }
    sine = 1.0f - h2 * (1.0f / 6.0f - h2 * (1.0f / 120.0f)); /* sin |h| / |h| */
    d[0] = 1.0f - h2 * (0.5f - h2 * (1.0f / 24.0f));         /* cos |h| */
    d[1] = sine * h[0];
    d[2] = sine * h[1];
    d[3] = sine * h[2];
    memcpy(q, g->pending, sizeof q); /* pending = pending * d: a reading turns the camera about its own axes */
    g->pending[0] = q[0] * d[0] - q[1] * d[1] - q[2] * d[2] - q[3] * d[3];
    g->pending[1] = q[0] * d[1] + q[1] * d[0] + q[2] * d[3] - q[3] * d[2];
    g->pending[2] = q[0] * d[2] - q[1] * d[3] + q[2] * d[0] + q[3] * d[1];
    g->pending[3] = q[0] * d[3] + q[1] * d[2] - q[2] * d[1] + q[3] * d[0];
    scale = mvo_rsqrt(g->pending[0] * g->pending[0] + g->pending[1] * g->pending[1] + g->pending[2] * g->pending[2]
                      + g->pending[3] * g->pending[3]);
    g->pending[0] *= scale;
    g->pending[1] *= scale;
    g->pending[2] *= scale;
    g->pending[3] *= scale;
    g->pending_time += dt;
    return 1;
}

void mvo_gyro_rate(mvo_t *s, const float rate[3], float dt) { (void)integrate(s, rate, dt); }

/* v turned by the unit quaternion q (w, x, y, z): v + 2 w (q x v) + 2 q x (q x v). */
static void quat_turn(const float q[4], const float v[3], float out[3])
{
    float a[3], b[3];
    a[0] = q[2] * v[2] - q[3] * v[1];
    a[1] = q[3] * v[0] - q[1] * v[2];
    a[2] = q[1] * v[1] - q[2] * v[0];
    b[0] = q[2] * a[2] - q[3] * a[1];
    b[1] = q[3] * a[0] - q[1] * a[2];
    b[2] = q[1] * a[1] - q[2] * a[0];
    out[0] = v[0] + 2.0f * (q[0] * a[0] + b[0]);
    out[1] = v[1] + 2.0f * (q[0] * a[1] + b[1]);
    out[2] = v[2] + 2.0f * (q[0] * a[2] + b[2]);
}

int mvo_imu_reading(mvo_t *s, const float rate[3], const float accel[3], float dt)
{
    mvo_gyro_t *g = &s->gyro;
    float before[4], f[3], c[3], a[3], b[3];
    int i;
    memcpy(before, g->pending, sizeof before);
    if (!integrate(s, rate, dt)) {
        return 0;
    }
    for (i = 0; i < 3; ++i) {
        f[i] = accel[i] - g->accel_bias[i];
    }
    if (!(dot3(f, f) < MVO_IMU_FORCE_MAX * MVO_IMU_FORCE_MAX)) { /* not finite, or a knock: no reading of gravity */
        return 0;
    }
    mat3_vec(g->mount, f, c);
    quat_turn(before, c, a); /* into A at the reading's two ends: their mean is its middle to second order */
    quat_turn(g->pending, c, b);
    for (i = 0; i < 3; ++i) {
        g->force[i] += 0.5f * dt * (a[i] + b[i]);
    }
    g->force_time += dt;
    return 1;
}

int mvo_imu_set_accel_bias(mvo_t *s, const float bias[3])
{
    int i;
    for (i = 0; i < 3; ++i) {
        if (!mvo_finite(bias[i])) {
            return 0;
        }
    }
    memcpy(s->gyro.accel_bias, bias, sizeof s->gyro.accel_bias);
    return 1;
}

/* The direction of the agreeing ups' sum, settled or not; zero for none. */
static void mean_up(const mvo_gyro_t *g, float up[3])
{
    const float size2 = dot3(g->up_sum, g->up_sum);
    const float scale = size2 > 0.0f ? mvo_rsqrt(size2) : 0.0f;
    up[0] = g->up_sum[0] * scale;
    up[1] = g->up_sum[1] * scale;
    up[2] = g->up_sum[2] * scale;
}

void mvo_imu_up(const mvo_t *s, float up[3])
{
    if (s->gyro.up_steps < MVO_IMU_UP_STEPS) {
        up[0] = up[1] = up[2] = 0.0f;
        return;
    }
    mean_up(&s->gyro, up);
}

void mvo_gyro_bias(const mvo_t *s, float bias[3]) { memcpy(bias, s->gyro.bias, sizeof s->gyro.bias); }

int mvo_gyro_set_bias(mvo_t *s, const float bias[3])
{
    int i;
    for (i = 0; i < 3; ++i) {
        if (!(bias[i] >= -MVO_GYRO_BIAS_LIMIT && bias[i] <= MVO_GYRO_BIAS_LIMIT)) {
            return 0;
        }
    }
    memcpy(s->gyro.bias, bias, sizeof s->gyro.bias);
    s->gyro.bias_steps = MVO_GYRO_BIAS_STEPS;
    return 1;
}

int mvo_gyro_used(const mvo_t *s) { return s->gyro.used; }

/* The readings since the last frame as one rotation R_AB (row-major) and the seconds they cover, then dropped; returns
 * 0 when there were none. */
static int take_readings(mvo_t *s, float rotation[9], float *time)
{
    mvo_gyro_t *g = &s->gyro;
    const float w = g->pending[0], x = g->pending[1], y = g->pending[2], z = g->pending[3];
    const int any = g->pending_time > 0.0f;
    rotation[0] = 1.0f - 2.0f * (y * y + z * z);
    rotation[1] = 2.0f * (x * y - w * z);
    rotation[2] = 2.0f * (x * z + w * y);
    rotation[3] = 2.0f * (x * y + w * z);
    rotation[4] = 1.0f - 2.0f * (x * x + z * z);
    rotation[5] = 2.0f * (y * z - w * x);
    rotation[6] = 2.0f * (x * z - w * y);
    rotation[7] = 2.0f * (y * z + w * x);
    rotation[8] = 1.0f - 2.0f * (x * x + y * y);
    *time = g->pending_time;
    if (any) { /* the usual time a step's readings cover: an average that one step moves by a sixteenth at most */
        float t = g->pending_time;
        if (g->usual_time <= 0.0f) {
            g->usual_time = t;
        } else {
            if (t > 2.0f * g->usual_time) t = 2.0f * g->usual_time;
            if (t < 0.5f * g->usual_time) t = 0.5f * g->usual_time;
            g->usual_time += (t - g->usual_time) * 0.0625f;
        }
    }
    g->pending[0] = 1.0f;
    g->pending[1] = g->pending[2] = g->pending[3] = 0.0f;
    g->pending_time = 0.0f;
    g->measured[0] = g->measured[1] = g->measured[2] = 0.0f;
    g->level_time = 0.0f;
    if (g->force_time > 0.0f) { /* the up the forces measured, into the world through A's rotation (the last frame) */
        float up[3], size2;
        mat3_vec(s->world_rotation, g->force, up);
        size2 = dot3(up, up);
        if (size2 > 0.0f && size2 < 3.0e38f) {
            const float scale = mvo_rsqrt(size2);
            g->measured[0] = up[0] * scale;
            g->measured[1] = up[1] * scale;
            g->measured[2] = up[2] * scale;
            g->level_time = g->force_time;
        }
    }
    g->force[0] = g->force[1] = g->force[2] = 0.0f;
    g->force_time = 0.0f;
    return any;
}

/* The bias from a calm step whose rotation the camera found itself (vision, R_AB) against the gyro's (gyro, R_AB, over
 * time seconds). To first order the gyro's miss, the axis-angle of gyro^T vision, is minus the bias error (camera
 * axes) times the time; capped at MVO_GYRO_MISS, so that a pose the pairs got wrong moves the estimate little. Not
 * when the readings cover a time far from their usual one: then some are missing, or extra, and their miss over that
 * time is no bias. */
static void learn_bias(mvo_t *s, const float gyro[9], const float vision[9], float time)
{
    mvo_gyro_t *g = &s->gyro;
    float m[9], miss[3], back[3], size2, weight;
    int i;
    if (!(gyro[0] + gyro[4] + gyro[8] >= 3.0f - 2.0f * MVO_GYRO_CALM)) { /* the trace is 1 + 2 cos of the angle */
        return;
    }
    if (time < 0.75f * g->usual_time || time > 1.3333334f * g->usual_time) {
        return;
    }
    mat3_transpose(gyro, m);
    mat3_mul(m, vision, m);
    miss[0] = 0.5f * (m[7] - m[5]);
    miss[1] = 0.5f * (m[2] - m[6]);
    miss[2] = 0.5f * (m[3] - m[1]);
    size2 = dot3(miss, miss);
    if (!(size2 < 1.0f)) { /* not finite */
        return;
    }
    if (size2 > MVO_GYRO_MISS * MVO_GYRO_MISS) {
        const float scale = MVO_GYRO_MISS * mvo_rsqrt(size2);
        miss[0] *= scale;
        miss[1] *= scale;
        miss[2] *= scale;
    }
    mat3_transpose(g->mount, m); /* back into the gyro's axes */
    mat3_vec(m, miss, back);
    if (g->bias_steps < MVO_GYRO_BIAS_STEPS) {
        ++g->bias_steps;
    }
    weight = 1.0f / ((float)g->bias_steps * time);
    for (i = 0; i < 3; ++i) {
        g->bias[i] -= weight * back[i];
        if (g->bias[i] > MVO_GYRO_BIAS_LIMIT) g->bias[i] = MVO_GYRO_BIAS_LIMIT;
        if (g->bias[i] < -MVO_GYRO_BIAS_LIMIT) g->bias[i] = -MVO_GYRO_BIAS_LIMIT;
    }
}

static void refill(mvo_t *s)
{
    int64_t start = mvo_now(s);
    if ((float)s->count < MVO_REFILL * (float)MVO_MAX_FEATURES) {
        mvo_detect(s);
    }
    s->timing.detect_us += mvo_now(s) - start;
}

void mvo_start(mvo_t *s)
{
    float unused[9], time;
    take_readings(s, unused, &time); /* readings before the first frame belong to no step */
    mvo_scale_start(s);
    s->current ^= 1;
    mvo_build_pyramid(s, s->pyramid[s->current]);
    s->count = 0;
    s->timing.detect_us = s->timing.track_us = s->timing.motion_us = 0;
    refill(s);
}

/* Until the world's up has settled, the step's measured up joins the agreeing ones, or starts them again when it
 * lies outside MVO_IMU_UP_AGREE of their mean, and nothing turns. From the step that settles it on, the world
 * rotation after the step turns so that the step's measured up moves towards the world's up: about their cross
 * product in the world's axes, which tilts and leaves the heading, by the sine of their angle times
 * MVO_IMU_LEVEL_RATE times the readings' seconds (at most 1), and at most MVO_IMU_LEVEL_MAX. A measured up more
 * than a right angle from the world's turns nothing. */
static void level(mvo_t *s)
{
    mvo_gyro_t *g = &s->gyro;
    float up[3], v[3], gain, size2, h2, sine, q[4], turn[9];
    if (!(g->level_time > 0.0f)) {
        return;
    }
    if (g->up_steps < MVO_IMU_UP_STEPS) {
        mean_up(g, up);
        if (g->up_steps > 0 && !(dot3(g->measured, up) >= MVO_IMU_UP_AGREE)) { /* not still: start again */
            g->up_sum[0] = g->up_sum[1] = g->up_sum[2] = 0.0f;
            g->up_steps = 0;
        }
        g->up_sum[0] += g->measured[0];
        g->up_sum[1] += g->measured[1];
        g->up_sum[2] += g->measured[2];
        if (++g->up_steps < MVO_IMU_UP_STEPS) {
            return;
        }
    }
    mean_up(g, up);
    if (!(dot3(g->measured, up) > 0.0f)) {
        return;
    }
    gain = g->level_time * MVO_IMU_LEVEL_RATE;
    if (gain > 1.0f) gain = 1.0f;
    v[0] = gain * (g->measured[1] * up[2] - g->measured[2] * up[1]);
    v[1] = gain * (g->measured[2] * up[0] - g->measured[0] * up[2]);
    v[2] = gain * (g->measured[0] * up[1] - g->measured[1] * up[0]);
    size2 = dot3(v, v);
    if (size2 > MVO_IMU_LEVEL_MAX * MVO_IMU_LEVEL_MAX) {
        const float scale = MVO_IMU_LEVEL_MAX * mvo_rsqrt(size2);
        v[0] *= scale;
        v[1] *= scale;
        v[2] *= scale;
    }
    h2 = 0.25f * dot3(v, v); /* the turn's half angle, squared: at most a degree */
    sine = 0.5f * (1.0f - h2 * (1.0f / 6.0f - h2 * (1.0f / 120.0f)));
    q[0] = 1.0f - h2 * (0.5f - h2 * (1.0f / 24.0f));
    q[1] = sine * v[0];
    q[2] = sine * v[1];
    q[3] = sine * v[2];
    turn[0] = 1.0f - 2.0f * (q[2] * q[2] + q[3] * q[3]);
    turn[1] = 2.0f * (q[1] * q[2] - q[0] * q[3]);
    turn[2] = 2.0f * (q[1] * q[3] + q[0] * q[2]);
    turn[3] = 2.0f * (q[1] * q[2] + q[0] * q[3]);
    turn[4] = 1.0f - 2.0f * (q[1] * q[1] + q[3] * q[3]);
    turn[5] = 2.0f * (q[2] * q[3] - q[0] * q[1]);
    turn[6] = 2.0f * (q[1] * q[3] - q[0] * q[2]);
    turn[7] = 2.0f * (q[2] * q[3] + q[0] * q[1]);
    turn[8] = 1.0f - 2.0f * (q[1] * q[1] + q[2] * q[2]);
    mat3_mul(turn, s->world_rotation, s->world_rotation);
}

static void finish(mvo_t *s, mvo_step_t *out, const float rotation_ab[9], const float centre[3], int kind, int tracks,
                   int inliers, int scale_points)
{
    float moved[3];
    int i;
    memcpy(out->rotation, rotation_ab, sizeof out->rotation);
    memcpy(out->centre, centre, sizeof out->centre);
    out->kind = kind;
    out->tracks = tracks;
    out->inliers = inliers;
    out->scale_points = scale_points;
    mat3_vec(s->world_rotation, centre, moved);
    for (i = 0; i < 3; ++i) {
        s->world_centre[i] += moved[i];
    }
    mat3_mul(s->world_rotation, rotation_ab, s->world_rotation);
    level(s);
    mvo_scale_step(s, moved);
    memcpy(out->world_rotation, s->world_rotation, sizeof out->world_rotation);
    memcpy(out->world_centre, s->world_centre, sizeof out->world_centre);
    ++s->steps;
}

/* The depth ratios of the points the previous step left in this frame pair's A, into mvo_values; returns their count.
 * A point counts when it is an inlier of the pose (r, t) and this pair puts it in front of both cameras nearer than
 * MVO_RATIO_DEPTH baselines (see mono_vo.h for why that limit is not MVO_MAX_DEPTH). */
static int shared_ratios(const mvo_t *s, int n, const float r[9], const float t[3])
{
    int k, shared = 0;
    for (k = 0; k < n; ++k) {
        float p[3], q[3];
        if (mvo_epipolar[k] && s->feature[k].has_point && s->feature[k].point[2] > 0.0f
            && mvo_triangulate(r, t, mvo_ra[k], mvo_rb[k], p) && p[2] > 0.0f && p[2] < MVO_RATIO_DEPTH) {
            mat3_vec(r, p, q);
            if (q[2] + t[2] > 0.0f && q[2] + t[2] < MVO_RATIO_DEPTH) {
                mvo_values[shared++] = s->feature[k].point[2] / p[2];
            }
        }
    }
    return shared;
}

float mvo_step_length(mvo_t *s, float *ratios, int count)
{
    float recent[MVO_SPEED_STEPS], speed = 1.0f, length;
    if (s->recent_count > 0) {
        memcpy(recent, s->recent_length, (size_t)s->recent_count * sizeof recent[0]); /* the median reorders */
        speed = mvo_median(recent, s->recent_count);
    }
    if (count < MVO_MIN_SCALE_POINTS) {
        length = speed;
    } else {
        length = mvo_median(ratios, count);
        if (s->recent_count > 0 && length > MVO_SPEED_LIMIT * speed) {
            length = MVO_SPEED_LIMIT * speed;
        } else if (s->recent_count > 0 && length * MVO_SPEED_LIMIT < speed) {
            length = speed / MVO_SPEED_LIMIT;
        }
    }
    s->recent_length[s->recent_next] = length;
    s->recent_next = (s->recent_next + 1) % MVO_SPEED_STEPS;
    if (s->recent_count < MVO_SPEED_STEPS) {
        ++s->recent_count;
    }
    return length;
}

/* Carries the chain of shared depths through a LOST step with the essential pose (r, t) its tracks fit, whose
 * translation the step does not take: on the lake clip at 5 Hz such poses' directions of travel lie a median 87-126
 * degrees from the reference's on fast turns, where the accepted ones lie 10-11 degrees from it. Their depths still
 * carry the scale. When this was adopted (before MVO_RATIO_DEPTH and the 300 features), dropping every point at each
 * of the 11 lost steps a run restarted the scale from the recent speed each time, and the scale jumped by 1.4 times in
 * the median across them; carried on, the path after one alignment came out 0.33 m off in the median of 16 seeds
 * instead of 0.40 m. The points take the length the shared depth ratios give,
 * as on an ESSENTIAL step (without adding it to the recent speeds), only when at least MVO_MIN_SCALE_POINTS of them
 * agree on it: their interquartile range at most MVO_CARRY_SPREAD of their median. Else it returns 0 and changes no
 * point. A still camera's noise fits poses whose ratios scatter (the range 0.64-1.71 of the median, 10th to 90th
 * percentile, against 0.08-0.80 on the lake's lost steps), and their points would set the scale when the camera moves
 * again. Without shared points there is no scale to carry: a test of the points' parallax alone let a still camera's
 * noise through in a third of its poses at 2-3 px, and after two noisy hovers the next move's length was off by more
 * than 23% in a quarter to two fifths of the trials. */
int mvo_carry_depths(mvo_t *s, int n, const float r[9], const float t[3])
{
    float recent[MVO_SPEED_STEPS], speed, length;
    int k, shared = shared_ratios(s, n, r, t);
    if (shared < MVO_MIN_SCALE_POINTS) {
        return 0;
    }
    length = mvo_median(mvo_values, shared); /* sorts, so the quartiles are at hand */
    if (mvo_values[(3 * shared) / 4] - mvo_values[shared / 4] > MVO_CARRY_SPREAD * length) {
        return 0;
    }
    if (s->recent_count > 0) {
        memcpy(recent, s->recent_length, (size_t)s->recent_count * sizeof recent[0]); /* the median reorders */
        speed = mvo_median(recent, s->recent_count);
    } else {
        speed = length;
    }
    if (s->recent_count > 0 && length > MVO_SPEED_LIMIT * speed) {
        length = MVO_SPEED_LIMIT * speed;
    } else if (s->recent_count > 0 && length * MVO_SPEED_LIMIT < speed) {
        length = speed / MVO_SPEED_LIMIT;
    }
    for (k = 0; k < n; ++k) { /* as on an ESSENTIAL step: points in B, in front of both cameras */
        float p[3], q[3];
        s->feature[k].has_point = 0;
        if (mvo_epipolar[k] && mvo_best[k] && mvo_triangulate(r, t, mvo_ra[k], mvo_rb[k], p)) {
            mat3_vec(r, p, q);
            q[0] += t[0];
            q[1] += t[1];
            q[2] += t[2];
            if (p[2] > 0.0f && q[2] > 0.0f) {
                s->feature[k].point[0] = length * q[0];
                s->feature[k].point[1] = length * q[1];
                s->feature[k].point[2] = length * q[2];
                s->feature[k].spread = mvo_ray_spread(r, k);
                s->feature[k].has_point = 1;
            }
        }
    }
    return 1;
}

/* An ESSENTIAL step with the pose (r, t) and its count of inliers in front: the length from the shared depth ratios,
 * the points at the new scale, the centre. */
static void essential_step(mvo_t *s, mvo_step_t *out, int n, const float r[9], const float t[3], int count,
                           int64_t start)
{
    float length, rt[9], centre[3];
    int k, kept = 0, shared;
    shared = shared_ratios(s, n, r, t);
    length = mvo_step_length(s, mvo_values, shared);
    /* Every epipolar inlier stays tracked, far ones too: dropping the points beyond MVO_MAX_DEPTH baselines threw away
     * most features on short steps and broke the chain of depths that carries the scale. Only those in front within
     * that range get a point, in B at the new scale. */
    for (k = 0; k < n; ++k) {
        mvo_feature_t f = s->feature[k];
        float p[3], q[3];
        if (!mvo_epipolar[k]) {
            continue;
        }
        f.has_point = 0;
        if (mvo_best[k] && mvo_triangulate(r, t, mvo_ra[k], mvo_rb[k], p)) {
            mat3_vec(r, p, q);
            q[0] += t[0];
            q[1] += t[1];
            q[2] += t[2];
            if (p[2] > 0.0f && q[2] > 0.0f) {
                f.point[0] = length * q[0];
                f.point[1] = length * q[1];
                f.point[2] = length * q[2];
                f.spread = mvo_ray_spread(r, k);
                f.has_point = 1;
            }
        }
        s->feature[kept++] = f;
    }
    s->count = kept;
    mat3_transpose(r, rt);
    mat3_vec(rt, t, centre);
    centre[0] *= -length;
    centre[1] *= -length;
    centre[2] *= -length;
    s->timing.motion_us = mvo_now(s) - start;
    finish(s, out, rt, centre, MVO_ESSENTIAL, n, count, shared);
}

/* The step with the gyro's rotation (R_AB) where the camera alone would end it in ROTATION or LOST: the translation
 * alone for that rotation (mvo_translation_pose), taken as ESSENTIAL when the translation evidence favours it over the
 * rotation alone; else the rotation without translation, as ROTATION when MVO_MIN_INLIERS and half of the tracks agree
 * with it (their points turned with it), else as LOST, the chain of depths carried on as the camera's own lost steps
 * carry it. */
static void gyro_step(mvo_t *s, mvo_step_t *out, int n, const float rotation_ab[9], int64_t start)
{
    static const float zero[3] = {0, 0, 0};
    float r[9], t[3] = {0.0f, 0.0f, 0.0f};
    int k, count, support = 0;
    mat3_transpose(rotation_ab, r); /* points: X_B = R X_A */
    count = n >= MVO_MIN_INLIERS ? mvo_translation_pose(s, n, r, t) : 0;
    if (count >= MVO_MIN_INLIERS && mvo_translation_evidence(s, n, rotation_ab, r, t)) {
        s->gyro.used = MVO_GYRO_TRANSLATION;
        essential_step(s, out, n, r, t, count, start);
        return;
    }
    for (k = 0; k < n; ++k) {
        support += mvo_rotation_inlier(s, r, k);
    }
    if (support >= MVO_MIN_INLIERS && 2 * support >= n) {
        for (k = 0; k < n; ++k) {
            if (s->feature[k].has_point) {
                if (mvo_rotation_inlier(s, r, k)) mat3_vec(r, s->feature[k].point, s->feature[k].point);
                else s->feature[k].has_point = 0;
            }
        }
        s->gyro.used = MVO_GYRO_ROTATION;
        s->timing.motion_us = mvo_now(s) - start;
        finish(s, out, rotation_ab, zero, MVO_ROTATION, n, support, 0);
        return;
    }
    if (count < MVO_MIN_INLIERS || !mvo_carry_depths(s, n, r, t)) {
        for (k = 0; k < n; ++k) {
            s->feature[k].has_point = 0;
        }
    }
    s->gyro.used = MVO_GYRO_LOST;
    s->timing.motion_us = mvo_now(s) - start;
    finish(s, out, rotation_ab, zero, MVO_LOST, n, support, 0);
}

void mvo_step(mvo_t *s, mvo_step_t *out)
{
    static const float identity[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1}, zero[3] = {0, 0, 0};
    const int previous = s->current;
    float rotation_ab[9], rotation_fit[9], turn[9], r[9], t[3], gyro[9], gyro_time;
    int64_t start;
    uint32_t rng_before_essential;
    int n, k, homography_count, rotation_count, count;
    const int with_gyro = take_readings(s, gyro, &gyro_time);
    s->scale.step_time = gyro_time > 0.0f ? gyro_time : MVO_SCALE_STEP_TIME;
    s->timing.detect_us = s->timing.track_us = s->timing.motion_us = 0;
    s->gyro.used = MVO_GYRO_NONE;
    s->current ^= 1;
    start = mvo_now(s);
    mvo_build_pyramid(s, s->pyramid[s->current]);
    n = mvo_track_features(s, s->pyramid[previous], s->pyramid[s->current]);
    s->timing.track_us = mvo_now(s) - start;
    start = mvo_now(s);
    for (k = 0; k < n; ++k) {
        mvo_pb[k][0] = s->feature[k].x;
        mvo_pb[k][1] = s->feature[k].y;
        mvo_undistort(&s->camera, mvo_pa[k], mvo_pa[k]); /* where the models work: a pinhole's pixels */
        mvo_undistort(&s->camera, mvo_pb[k], mvo_pb[k]);
        to_ray(&s->camera, mvo_pa[k], mvo_ra[k]);
        to_ray(&s->camera, mvo_pb[k], mvo_rb[k]);
    }
    homography_count = n >= MVO_MIN_TURN_INLIERS ? mvo_pure_rotation(s, n, rotation_ab) : 0;
    if (homography_count >= MVO_MIN_INLIERS && mvo_parallax(s, n, rotation_ab) < MVO_MIN_PARALLAX) {
        float rotation[9];
        if (with_gyro) {
            gyro_step(s, out, n, gyro, start);
            refill(s);
            return;
        }
        mat3_transpose(rotation_ab, rotation); /* points: X_B = R X_A */
        for (k = 0; k < n; ++k) {
            if (s->feature[k].has_point) {
                if (mvo_rotation_inlier(s, rotation, k)) mat3_vec(rotation, s->feature[k].point, s->feature[k].point);
                else s->feature[k].has_point = 0;
            }
        }
        s->timing.motion_us = mvo_now(s) - start;
        finish(s, out, rotation_ab, zero, MVO_ROTATION, n, homography_count, 0);
        refill(s);
        return;
    }
    rotation_count = mvo_rotation_model(s, n, rotation_fit);
    rng_before_essential = s->rng;
    if (homography_count >= MVO_MIN_INLIERS) mat3_transpose(rotation_ab, turn);
    count = n >= MVO_MIN_INLIERS && rotation_count >= 0
                ? mvo_essential_pose(s, n, r, t, homography_count >= MVO_MIN_INLIERS ? turn : NULL) : 0;
    if (count < MVO_MIN_INLIERS || !mvo_translation_evidence(s, n, rotation_fit, r, t)) {
        /* A rotation fit is evidence only of motion indistinguishable from rotation at this image accuracy. */
        s->rng = rng_before_essential;
        if (with_gyro) {
            gyro_step(s, out, n, gyro, start);
            refill(s);
            return;
        }
        if (rotation_count >= MVO_MIN_INLIERS && 2 * rotation_count >= n) {
            float rotation[9];
            mat3_transpose(rotation_fit, rotation);
            for (k = 0; k < n; ++k) {
                if (s->feature[k].has_point) {
                    if (mvo_rotation_inlier(s, rotation, k)) mat3_vec(rotation, s->feature[k].point, s->feature[k].point);
                    else s->feature[k].has_point = 0;
                }
            }
            s->timing.motion_us = mvo_now(s) - start;
            finish(s, out, rotation_fit, zero, MVO_ROTATION, n, rotation_count, 0);
            refill(s);
            return;
        }
        if (count < MVO_MIN_INLIERS || !mvo_carry_depths(s, n, r, t)) {
            for (k = 0; k < n; ++k) {
                s->feature[k].has_point = 0;
            }
        }
        s->timing.motion_us = mvo_now(s) - start;
        if (rotation_count >= MVO_MIN_TURN_INLIERS && 2 * rotation_count >= n) {
            finish(s, out, rotation_fit, zero, MVO_LOST, n, rotation_count, 0);
        } else if (homography_count >= MVO_MIN_TURN_INLIERS && 2 * homography_count >= n) {
            if (count >= MVO_MIN_INLIERS) {
                /* The rotation fit lacks the support the homography has: mostly the camera moved and the homography
                 * read the parallax as a turn. Its pitch is biased, not its size: on such lost steps of the lake clip
                 * at 5 Hz, +1.5 degrees of pitch a step against -0.2 for the essential pose, which models the travel,
                 * for errors of 6.0 and 6.2 degrees in all. A still camera with noisy tracks lands here too, rarely.
                 * The pose's rotation stands, its translation does not. */
                float rt[9];
                mat3_transpose(r, rt);
                finish(s, out, rt, zero, MVO_LOST, n, count, 0);
            } else {
                finish(s, out, rotation_ab, zero, MVO_LOST, n, homography_count, 0);
            }
        } else {
            finish(s, out, identity, zero, MVO_LOST, n, 0, 0);
        }
        refill(s);
        return;
    }
    essential_step(s, out, n, r, t, count, start);
    if (with_gyro) { /* the camera's own pose, against which the gyro learns its bias */
        learn_bias(s, gyro, out->rotation, gyro_time);
    }
    refill(s);
}
