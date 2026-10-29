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
}

uint8_t *mvo_frame(mvo_t *s) { return s->pyramid[s->current ^ 1]; }

static int64_t mvo_now(const mvo_t *s) { return s->timing.clock ? s->timing.clock() : 0; }

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
    s->current ^= 1;
    mvo_build_pyramid(s, s->pyramid[s->current]);
    s->count = 0;
    s->timing.detect_us = s->timing.track_us = s->timing.motion_us = 0;
    refill(s);
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
 * carry the scale. Dropping every point at each of the 11 lost steps a run restarted the scale from the recent speed
 * each time, and the scale jumped by 1.4 times in the median across them; carried on, the path after one alignment came
 * out 0.33 m off in the median of 16 seeds instead of 0.40 m. The points take the length the shared depth ratios give,
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

void mvo_step(mvo_t *s, mvo_step_t *out)
{
    static const float identity[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1}, zero[3] = {0, 0, 0};
    const int previous = s->current;
    float rotation_ab[9], rotation_fit[9], turn[9], r[9], t[3], centre[3];
    int64_t start;
    uint32_t rng_before_essential;
    int n, k, homography_count, rotation_count, count, shared = 0;
    s->timing.detect_us = s->timing.track_us = s->timing.motion_us = 0;
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
    {
        float length, rt[9];
        int kept = 0;
        shared = shared_ratios(s, n, r, t);
        length = mvo_step_length(s, mvo_values, shared);
        /* Every epipolar inlier stays tracked, far ones too: dropping the points beyond MVO_MAX_DEPTH baselines threw
         * away most features on short steps and broke the chain of depths that carries the scale. Only those in front
         * within that range get a point, in B at the new scale. */
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
    refill(s);
}
