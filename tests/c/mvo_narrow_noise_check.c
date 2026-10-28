/* Noisy stationary correspondences on a narrow, locally textured field. */
#include "mono_vo_internal.h"
#include <stdio.h>
#include <stdlib.h>

#define COUNT 36
static float before[COUNT][2], after[COUNT][2];
static uint32_t rng;
static float uniform(void) { rng = rng * 1664525u + 1013904223u; return (float)(rng >> 8) / 16777216.0f; }
static float normal(void) { float x = -6.0f; int k; for (k=0;k<12;++k) x += uniform(); return x; }
void mvo_build_pyramid(const mvo_t *s, uint8_t *p) { (void)s; (void)p; }
void mvo_detect(mvo_t *s) { (void)s; }
int mvo_track_features(mvo_t *s, const uint8_t *a, const uint8_t *b)
{
    int k; (void)a; (void)b;
    for (k=0;k<COUNT;++k) {
        mvo_pa[k][0]=before[k][0]; mvo_pa[k][1]=before[k][1];
        s->feature[k].x=after[k][0]; s->feature[k].y=after[k][1]; s->feature[k].has_point=0;
    }
    return s->count=COUNT;
}
int main(int argc, char **argv)
{
    static mvo_t state;
    const mvo_camera_t camera={4000,4000,160,120};
    int trial,k, false_count=0, lost=0, rotations=0;
    float sigma, outliers;
    int start, trials;
    if (argc!=5) return 2;
    sigma=(float)atof(argv[1]); outliers=(float)atof(argv[2]); start=atoi(argv[3]); trials=atoi(argv[4]);
    for (trial=0;trial<trials;++trial) {
        mvo_step_t step;
        rng=(uint32_t)(start+trial);
        for (k=0;k<COUNT;++k) {
            float jx=.4f*uniform()-.2f, jy=.4f*uniform()-.2f, nx=normal(), ny=normal();
            before[k][0]=230.0f+8.6f*((float)(k%6)-2.5f)+jx;
            before[k][1]=120.0f+8.6f*((float)(k/6)-2.5f)+jy;
            if ((float)k < outliers*COUNT) {
                after[k][0]=208.5f+43.0f*uniform(); after[k][1]=98.5f+43.0f*uniform();
            } else {
                after[k][0]=before[k][0]+sigma*nx; after[k][1]=before[k][1]+sigma*ny;
            }
        }
        mvo_init(&state,&camera,12345u,0,0); mvo_step(&state,&step);
        false_count += step.kind==MVO_ESSENTIAL;
        lost += step.kind==MVO_LOST; rotations += step.kind==MVO_ROTATION;
        if (step.kind==MVO_ESSENTIAL && false_count<=10) printf("false_seed=%d\n",start+trial);
    }
    printf("trials=%d false=%d lost=%d rotations=%d\n",trials,false_count,lost,rotations);
    return 0;
}
