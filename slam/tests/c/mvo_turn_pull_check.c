/*
 * A fast turn's pull (core/mono_vo_essential.c, included for its static functions): the refinement's normal-equation
 * terms for the pull against central differences of pull_cost along the rotation steps, its ceiling far from the pure
 * turn along its axis, no cost across the axis, and the gate in mvo_essential_pose on pairs from a slow and a fast
 * turn. Prints key=value pairs for tests/test_mvo.py.
 */
#include "mono_vo_essential.c"

#include <stdio.h>

#define CHECK_PAIRS 120

static float absolute(float v)
{
    return v < 0.0f ? -v : v;
}

static float uniform(uint32_t *state) /* in [0, 1) */
{
    *state = *state * 1664525u + 1013904223u;
    return (float)(*state >> 8) / 16777216.0f;
}

/* Pairs of points 1 to 9 in front of A seen from B = turn A + 0.05 t, a quarter pixel or so off. */
static void make_pairs(const mvo_t *s, const float turn[9], const float t[3], uint32_t *state)
{
    int k, j;
    for (k = 0; k < CHECK_PAIRS; ++k) {
        const float x = uniform(state), y = uniform(state), z = 1.0f + 8.0f * uniform(state);
        const float pa[3] = {(x - 0.5f) * 2.0f * z, (y - 0.5f) * 1.2f * z, z};
        float pb[3];
        mvo_pa[k][0] = s->camera.fx * pa[0] / pa[2] + s->camera.cx;
        mvo_pa[k][1] = s->camera.fy * pa[1] / pa[2] + s->camera.cy;
        mat3_vec(turn, pa, pb);
        for (j = 0; j < 3; ++j) {
            pb[j] += 0.05f * t[j];
        }
        mvo_pb[k][0] = s->camera.fx * pb[0] / pb[2] + s->camera.cx + 0.5f * (uniform(state) - 0.5f);
        mvo_pb[k][1] = s->camera.fy * pb[1] / pb[2] + s->camera.cy + 0.5f * (uniform(state) - 0.5f);
        to_ray(&s->camera, mvo_pa[k], mvo_ra[k]);
        to_ray(&s->camera, mvo_pb[k], mvo_rb[k]);
    }
}

int main(void)
{
    static mvo_t s;
    const float slow_turn[3] = {0.0f, 0.02f, 0.0f}, fast_turn[3] = {0.0f, 0.13f, 0.01f}; /* Cayley: 2.3 and 15 deg */
    const float near_turn[3] = {0.01f, -0.015f, 0.005f}, far_turn[3] = {0.0f, 0.3f, 0.0f}, tilt[3] = {0.1f, 0.0f, 0.0f};
    float turn[9], r[9], r0[9], t[3] = {0.3f, 0.0f, 0.95f}, found_t[3], basis[6], d[3], w, u, along, norm, gap = 0.0f;
    float largest = 0.0f;
    float slow_weight, fast_weight, ceiling, across;
    uint32_t state = 777u;
    int i;
    norm = mvo_sqrt(dot3(t, t));
    for (i = 0; i < 3; ++i) {
        t[i] /= norm;
    }
    s.camera.fx = 112.0f;
    s.camera.fy = 148.0f;
    s.camera.cx = 160.0f;
    s.camera.cy = 120.0f;
    s.rng = 12345u;

    /* the normal-equation terms: 2 w (axis . d) axis is half the cost's gradient in the rotation steps of perturb */
    mvo_cayley(fast_turn, s_turn);
    s_turn_axis[0] = s_turn[7] - s_turn[5];
    s_turn_axis[1] = s_turn[2] - s_turn[6];
    s_turn_axis[2] = s_turn[3] - s_turn[1];
    norm = mvo_sqrt(dot3(s_turn_axis, s_turn_axis));
    for (i = 0; i < 3; ++i) {
        s_turn_axis[i] /= norm;
    }
    s_turn_weight = MVO_TURN_PULL;
    mvo_cayley(near_turn, r0);
    mat3_mul(r0, s_turn, r);
    tangent_basis(t, basis);
    turn_offset(r, d);
    along = dot3(d, s_turn_axis);
    u = along * along / MVO_TURN_SOFT;
    w = s_turn_weight / ((1.0f + u) * (1.0f + u));
    for (i = 0; i < 3; ++i) {
        const float h = 1e-3f;
        float p[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, moved_r[9], moved_t[3], plus, minus, numeric, expected;
        p[i] = h;
        perturb(r, t, basis, p, moved_r, moved_t);
        plus = pull_cost(moved_r);
        p[i] = -h;
        perturb(r, t, basis, p, moved_r, moved_t);
        minus = pull_cost(moved_r);
        numeric = (plus - minus) / (2.0f * h);
        expected = 4.0f * w * along * s_turn_axis[i];
        gap = absolute(numeric - expected) > gap ? absolute(numeric - expected) : gap;
        largest = absolute(numeric) > largest ? absolute(numeric) : largest;
    }

    /* far from the pure turn along its axis the pull's cost stays under its ceiling, weight MVO_TURN_SOFT, and across
     * the axis (a tilt) it costs nothing */
    mvo_cayley(far_turn, r0);
    mat3_mul(r0, s_turn, r);
    ceiling = pull_cost(r) / (s_turn_weight * MVO_TURN_SOFT);
    mvo_cayley(tilt, r0);
    mat3_mul(r0, s_turn, r);
    across = pull_cost(r) / (s_turn_weight * MVO_TURN_SOFT);

    /* the gate: pairs from a slow turn leave no pull, from a fast one a pull */
    mvo_cayley(slow_turn, turn);
    make_pairs(&s, turn, t, &state);
    mvo_essential_pose(&s, CHECK_PAIRS, r, found_t, turn);
    slow_weight = s_turn_weight;
    mvo_cayley(fast_turn, turn);
    make_pairs(&s, turn, t, &state);
    mvo_essential_pose(&s, CHECK_PAIRS, r, found_t, turn);
    fast_weight = s_turn_weight;

    printf("gradient_gap=%.6g gradient_largest=%.6g ceiling=%.6g across=%.6g slow_weight=%.6g fast_weight=%.6g\n", gap,
           largest, ceiling, across, slow_weight, fast_weight);
    return 0;
}
