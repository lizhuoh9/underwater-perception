/*
 * The refinement's analytic Jacobian (core/mono_vo_essential.c, included for its static functions) against central
 * differences of its residuals, on 200 pairs near a known pose with up to a pixel of error. Prints the largest gap
 * between the two for the difference steps h and h / 2, and the largest derivative: a right Jacobian's gap shrinks as
 * h^2 (to a quarter when h halves), a wrong one's does not. Run by tests/test_mvo.py.
 */
#include "mono_vo_essential.c"

#include <stdio.h>

#define CHECK_PAIRS 200

static float absolute(float v)
{
    return v < 0.0f ? -v : v;
}

static float uniform(uint32_t *state) /* in [0, 1) */
{
    *state = *state * 1664525u + 1013904223u;
    return (float)(*state >> 8) / 16777216.0f;
}

/* The largest |analytic - central difference| over the rows and the five columns of s_refiner[0]'s Jacobian. */
static float gap(const mvo_t *s, const float r[9], const float t[3], const float basis[6], const uint8_t *mark,
                 int used, float h)
{
    static float plus[MVO_MAX_FEATURES], minus[MVO_MAX_FEATURES];
    float worst = 0.0f;
    int j, k;
    for (j = 0; j < 5; ++j) {
        float p[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, moved_r[9], moved_t[3];
        p[j] = h;
        perturb(r, t, basis, p, moved_r, moved_t);
        pose_residuals(s, moved_r, moved_t, CHECK_PAIRS, mark, plus);
        p[j] = -h;
        perturb(r, t, basis, p, moved_r, moved_t);
        pose_residuals(s, moved_r, moved_t, CHECK_PAIRS, mark, minus);
        for (k = 0; k < used; ++k) {
            const float d = absolute((plus[k] - minus[k]) / (2.0f * h) - s_refiner[0].jacobian[k][j]);
            worst = d > worst ? d : worst;
        }
    }
    return worst;
}

int main(void)
{
    static mvo_t s;
    static uint8_t mark[MVO_MAX_FEATURES];
    const float turn[3] = {0.02f, -0.03f, 0.01f};
    float r[9], t[3] = {0.1f, -0.2f, 0.97f}, basis[6], e[9], norm, largest = 0.0f;
    uint32_t state = 12345u;
    mvo_jacobian_job_t job;
    int k, j, used = 0;
    norm = mvo_sqrt(dot3(t, t));
    for (j = 0; j < 3; ++j) {
        t[j] /= norm;
    }
    s.camera.fx = 112.0f;
    s.camera.fy = 148.0f;
    s.camera.cx = 160.0f;
    s.camera.cy = 120.0f;
    mvo_cayley(turn, r);
    for (k = 0; k < CHECK_PAIRS; ++k) { /* a point in front of A, seen from B = R A + 0.05 t, then a pixel or so off */
        const float x = uniform(&state), y = uniform(&state), z = 1.0f + 8.0f * uniform(&state);
        const float pa[3] = {(x - 0.5f) * 2.5f * z, (y - 0.5f) * 1.6f * z, z};
        float pb[3];
        mvo_pa[k][0] = s.camera.fx * pa[0] / pa[2] + s.camera.cx;
        mvo_pa[k][1] = s.camera.fy * pa[1] / pa[2] + s.camera.cy;
        mat3_vec(r, pa, pb);
        for (j = 0; j < 3; ++j) {
            pb[j] += 0.05f * t[j];
        }
        mvo_pb[k][0] = s.camera.fx * pb[0] / pb[2] + s.camera.cx + 1.8f * (uniform(&state) - 0.5f);
        mvo_pb[k][1] = s.camera.fy * pb[1] / pb[2] + s.camera.cy + 1.2f * (uniform(&state) - 0.5f);
        mark[k] = (uint8_t)(k % 9 != 0);
        used += mark[k];
    }
    tangent_basis(t, basis);
    job.n = CHECK_PAIRS;
    job.used = used;
    job.mark = mark;
    job.scratch = &s_refiner[0];
    essential_of(r, t, e);
    fundamental(&s.camera, e, job.f);
    pose_derivatives(&s.camera, r, t, basis, job.df);
    jacobian_rows(&job, 0);
    jacobian_rows(&job, 1);
    for (k = 0; k < used; ++k) {
        for (j = 0; j < 5; ++j) {
            largest = absolute(s_refiner[0].jacobian[k][j]) > largest ? absolute(s_refiner[0].jacobian[k][j]) : largest;
        }
    }
    printf("rows=%d largest=%g gap_h=%g gap_half=%g\n", used, (double)largest,
           (double)gap(&s, r, t, basis, mark, used, 1e-3f), (double)gap(&s, r, t, basis, mark, used, 5e-4f));
    return 0;
}
