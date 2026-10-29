/*
 * mono_vo.h - frame-to-frame monocular visual odometry for a 320x240 grey camera (the C port of the "pairwise" method
 * of uwslam.experiments.mono_vo).
 *
 * Each frame: Shi-Tomasi corners (refilled when fewer than MVO_REFILL of MVO_MAX_FEATURES survive), pyramidal
 * Lucas-Kanade from the previous frame in fixed point with a forward-backward check (each level stopping after
 * MVO_LK_ITERATIONS, on a step shorter than the epsilon, or, as OpenCV's tracker does, in the middle of a step that
 * swings back), then the motion. A RANSAC
 * homography gives the rotation a pure turn would need; when the tracks stray from it by less than MVO_MIN_PARALLAX
 * pixels (median) the step is that rotation. Otherwise a separately fitted rotation is compared with an essential
 * matrix, including when the homography is weak. Translation is accepted only when capped residuals, a heuristic noise
 * scale, and a model-complexity penalty favor it; ambiguous cases retain no translation. The essential solver uses
 * eight-point MSAC on calibrated rays, each new best sample refitted on its inliers by least squares, then a
 * search over travel directions and Levenberg-Marquardt with a robust loss from both the best sample and the best
 * direction, the lower cost winning with pairs behind a camera counted as outliers. Short steps leave a wide
 * rotation-translation ambiguity, which these guard against. Fast turns leave another: a turn trades against sideways
 * travel for a fraction of a pixel, and the essential matrix took up to a quarter of each fast step as travel (on the
 * lake clip at 5 Hz, 14 degrees over a two-second swing that one long-baseline estimate across it does not lose). So
 * when the homography's rotation and the decomposed essential pose both turn by more than MVO_TURN_GATE, the refinement
 * and the choice between the two poses add a pull towards the former's angle about its own axis, the amount of turn the
 * valley trades: MVO_TURN_PULL at MVO_TURN_FULL, growing with the angle squared up to MVO_TURN_CAP times that, and
 * Geman-McClure in the angle from it (half as strong at MVO_TURN_SOFT), so that where the pairs decide (pitch, roll,
 * real parallax) it moves little, along the flat valley it decides, and a rotation far off stops pulling. Only along
 * that axis: a pull on the whole rotation also dragged in the homography's tilt, which the floor's parallax biases (10
 * degrees more pitch over the clip). Slower steps get none: on slow travel the homography's rotation carries the
 * parallax, often turning five degrees where the camera turns one or two. Every inlier of the essential matrix stays
 * tracked; those triangulated in front of both cameras within MVO_MAX_DEPTH baselines carry a point to the next step.
 * The step length comes from the median depth ratio of points the previous step also triangulated, kept within
 * MVO_SPEED_LIMIT times the recent speed (the median of the last MVO_SPEED_STEPS lengths): on a fast turn a few shared
 * points can give a median far off, and the chain of shared depths would carry that scale on. A shared point counts
 * when it is the pose's inlier in front of both cameras nearer than MVO_RATIO_DEPTH baselines, not MVO_MAX_DEPTH: a
 * ratio grows with the new pair's parallax, so cutting at the pair's own noisy depths dropped the points whose noise
 * put them far and, on short steps, where many sit near that limit, biased the median. On the lake clip the path
 * after one alignment came out 0.190 m off in the median of 16 seeds instead of 0.214 m at 5 Hz, and 0.327 instead of
 * 0.394 m at 10 Hz; with no limit at all, points too noisy to help made 10 Hz worse. With fewer than
 * MVO_MIN_SCALE_POINTS shared points the length is the recent speed (1 for the first). A step with fewer than
 * MVO_MIN_INLIERS counts as lost and keeps the position; a lost step still takes a rotation supported by at least
 * MVO_MIN_TURN_INLIERS and half of the tracks. Where only the homography has that support and the essential pose had
 * MVO_MIN_INLIERS inliers, the pose's rotation stands instead: a homography the rotation fit does not share has mostly
 * read the travel's parallax as a turn, over a floor as pitch. Any lost step whose essential pose had MVO_MIN_INLIERS
 * inliers carries the chain of shared depths on with that pose, though not its translation: its points go on to the
 * next step at the length the shared depth ratios give, when at least MVO_MIN_SCALE_POINTS of them agree on it (their
 * interquartile range at most MVO_CARRY_SPREAD of their median); else they are dropped and the next step takes the
 * recent speed. A still camera's noise fits poses whose ratios scatter, so it nearly always drops them. ROTATION means
 * translation is indistinguishable under the heuristic, not proof of physically zero translation. Poses are
 * camera-to-world, world = the first camera (x right, y down, z forward). A lens with distortion (mvo_camera_t's k1, k2,
 * p1, p2, k3: OpenCV's model, as its calibration gives them) is taken out of each step's pairs before any model sees
 * them, which then work on the pixels a pinhole with the same fx, fy, cx, cy would have seen (mvo_undistort); the
 * corners, the tracking and the features' pixels stay in the raw image, where the masks are. With all five zero the
 * pairs reach the models untouched.
 *
 * C99, no heap, no libm (a local square root, by Newton steps without division), xorshift32 for RANSAC, so a PC build
 * and an ESP32-P4 build give identical results with -ffp-contract=off and without fast-math. It follows the OpenCV
 * prototype's design, not its numbers: the prototype uses OpenCV's five-point solver and its own Lucas-Kanade, stops
 * tracking the pairs recoverPose drops as too far, and takes every median depth ratio as it comes. One mvo_t is about
 * 9.8 KB; the caller also provides two image pyramids of MVO_PYRAMID_BYTES (about 100 KB each, 16-byte aligned for
 * vector kernels), separate blocks so that they fit a fragmented heap. Most of a step splits in two halves that can run at once (mvo_t.parallel): pyramid level 1
 * (half the rows each), the tracking (half the features), each RANSAC batch (half the samples), the corner search (half
 * the rows), the direction search (half the directions), the two pose refinements, and a lone refinement's Jacobian
 * (its rows); the results are the same either way. The tracker's kernels (two fixed-point ones and a block copy)
 * can be replaced by bit-exact platform versions (mvo_t.kernels). Otherwise it is not thread-safe.
 */
#ifndef MONO_VO_H
#define MONO_VO_H

#include <stdint.h>

#define MVO_WIDTH 320
#define MVO_HEIGHT 240
#define MVO_LEVELS 4 /* pyramid levels: 320x240, 160x120, 80x60, 40x30 */

/* The build, for an ESP32-P4 at 5 Hz, named in the board's reports (mvo_config_name). It tracks 300 features: on the
 * lake clip at 5 Hz, 300 instead of 250 brought the path after one alignment from 0.33 to 0.26 m in the median of 16
 * seeds and from 0.079 to 0.067 m on 10 s windows (lower in 68% of 192 paired runs), the heading unchanged; at 10 Hz it
 * was worse. The extra features triangulate less well: the 5 Hz voxel map's local error is 7.1 cm instead of 6.1.
 * Filing only points from rays 1.5 degrees apart took that back, but forward motion leaves the least parallax straight
 * ahead, and the range there was then known in 34% of the 5 Hz steps instead of 71%, so the maps file from 1 degree (MM_,
 * VM_ and MC_MIN_SPREAD). It tracks 19x19 windows: with the lake clip's lens, the heading over 10 s windows at 5 Hz came
 * out 6.2 degrees in the median instead of 8.1 with 15x15 windows (lower in 59% of 192 paired runs), the position about
 * the same (0.066 against 0.067 m); at 3 and 10 Hz the heading was worse, and 350 or 400 features with them were worse
 * at 5 Hz. */
#define MVO_CONFIG_NAME "p4"
#define MVO_MAX_FEATURES 300
#define MVO_WINDOW 9 /* Lucas-Kanade half window: 19x19 */
#define MVO_LANES 24 /* a window row as stored: MVO_SIDE values, then zeros (three vectors of eight 16-bit lanes) */

#define MVO_REFILL 0.6f
#define MVO_MIN_DISTANCE 8.0f
#define MVO_QUALITY 0.01f
#define MVO_LK_ITERATIONS 10 /* at 5 Hz most windows still moving after 10 end up failing the forward-backward check */
#define MVO_LK_EPSILON 0.01f
#define MVO_LK_COARSE_EPSILON 0.05f /* above full resolution: the finer levels refine it anyway */
#define MVO_LK_SWING 0.01f /* a step undoing the previous one to within this (pixels): settled between the two */
#define MVO_FB_MAX 1.0f
#define MVO_RANSAC_PIXELS 1.0f
#define MVO_HOMOGRAPHY_PIXELS 2.0f
#define MVO_MIN_PARALLAX 1.0f
#define MVO_MIN_INLIERS 15
#define MVO_MIN_TURN_INLIERS 8 /* a lost step's rotation: four more than a homography needs (five gave wild turns) */
#define MVO_MIN_SCALE_POINTS 8
#define MVO_SPEED_STEPS 10   /* the recent speed: the median of this many step lengths */
#define MVO_SPEED_LIMIT 3.0f /* a step length stays within this factor of the recent speed */
#define MVO_CARRY_SPREAD 0.5f /* a lost step's shared depth ratios agree: interquartile range at most this x median */
#define MVO_MAX_DEPTH 50.0f /* cheirality: points farther than this many baselines get no point (but stay tracked) */
#define MVO_RATIO_DEPTH 60.0f /* shared points this pair puts farther than this many baselines give no depth ratio */
#define MVO_UNDISTORT_STEPS 5 /* Newton steps inverting the lens model: within 1e-3 px of OpenCV's in the tests */
#define MVO_TURN_GATE 0.0038053019f /* 1 - cos 5 degrees: a step either rotation turns less does not pull */
#define MVO_TURN_FULL 0.0151922470f /* 1 - cos 10 degrees: where the pull is MVO_TURN_PULL */
#define MVO_TURN_PULL 3000.0f /* per squared radian, against a robust cost in which a pair costs at most 1 */
#define MVO_TURN_CAP 4.0f /* the pull grows no further than this many times MVO_TURN_PULL (about 20 degrees) */
#define MVO_TURN_SOFT 0.0048738787f /* (4 degrees)^2 in radians: a rotation this far from the pose pulls half as hard */
#define MVO_PYRAMID_BYTES (MVO_WIDTH * MVO_HEIGHT * 4 / 3 + 64) /* one frame's pyramid, level 0 first */
#define MVO_SIDE (2 * MVO_WINDOW + 1) /* the tracking window's side */

#if defined(__GNUC__)
#define MVO_ALIGN16 __attribute__((aligned(16))) /* where vector kernels load whole 16-byte blocks */
#else
#define MVO_ALIGN16
#endif

enum { MVO_LOST = 0, MVO_ROTATION = 1, MVO_ESSENTIAL = 2 };

typedef struct {
    float fx, fy, cx, cy;     /* pinhole intrinsics of the 320x240 frame, pixels */
    float k1, k2, p1, p2, k3; /* that frame's lens distortion in OpenCV's model and order (radial k1, k2, k3 and
                               * tangential p1, p2 on normalised coordinates); all zero for a pinhole */
} mvo_camera_t;

typedef struct {
    float rotation[9]; /* R_AB, row-major: a ray d_B appears in the previous frame A as R_AB d_B */
    float centre[3];   /* this frame's centre in the previous frame's axes */
    int32_t kind;      /* MVO_LOST, MVO_ROTATION or MVO_ESSENTIAL */
    int32_t tracks;    /* features that survived tracking */
    int32_t inliers;   /* inliers of the model used */
    int32_t scale_points; /* points shared with the previous step that set the length */
    float world_rotation[9]; /* camera-to-world after this step, row-major */
    float world_centre[3];
} mvo_step_t;

typedef struct {
    int64_t (*clock)(void); /* microseconds; NULL: no timing */
    int64_t detect_us, track_us, motion_us;
} mvo_timing_t;

typedef struct {
    float x, y;       /* pixel position in the current frame */
    float point[3];   /* triangulated position in the current frame's axes, at the current scale */
    float spread;     /* squared sine of the angle between the two rays that triangulated it: depth is good when large */
    int32_t has_point;
    uint32_t id;      /* lasting identity, given when detected: the same id in two frames is the same track */
} mvo_feature_t;

/* A feature's tracking window at one pyramid level, row-major, MVO_SIDE rows of MVO_LANES values, those from MVO_SIDE
 * on 0: the previous frame interpolated at the window's pixels and its central differences across x and y, all Q7 (see
 * mvo_kernels_t). */
typedef struct {
    int16_t patch[MVO_SIDE * MVO_LANES] MVO_ALIGN16;
    int16_t gx[MVO_SIDE * MVO_LANES];
    int16_t gy[MVO_SIDE * MVO_LANES];
} mvo_window_t;

/* The kernels of the Lucas-Kanade tracker: two fixed-point ones and the copy of a window at the image border. A
 * platform may replace them (mvo_t.kernels) with its own, vector ones for instance, as long as they give exactly the
 * same results as mvo_lk_template_c, mvo_lk_sums_c and mvo_block_c. Images are 8-bit with a row stride in bytes;
 * weights are the bilinear weights of a fraction in Q14, summing to 1 << 14. Interpolating at pixel r is the sum over
 * the 2x2 pixels from r (r, r + 1, r + stride, r + stride + 1) of (pixel * weight) >> 7, a Q7 value: each product is
 * shifted on its own, as 16-bit vector lanes do. A kernel may read up to 16 bytes before and 48 after the pixels it
 * uses. */
typedef struct {
    /* src: the pixel one up and one left of the window's top-left pixel. Fills the window: patch = the interpolation
     * at each window pixel, gx and gy = the interpolation one pixel further on minus the one a pixel before, across x
     * and across y; g = the sums of gx * gx, gx * gy and gy * gy. */
    void (*lk_template)(mvo_window_t *window, const uint8_t *src, int stride, const int16_t weights[4], int64_t g[3]);
    /* src: the window's top-left pixel in the other image. sums = the sums of (patch - the interpolation) * gx and
     * * gy over the window. */
    void (*lk_sums)(const mvo_window_t *window, const uint8_t *src, int stride, const int16_t weights[4],
                    int64_t sums[2]);
    /* The size x size block (size MVO_SIDE + 1 or MVO_SIDE + 3) of the w x h image img whose top-left pixel is
     * (origin[0], origin[1]), each coordinate clamped to the image, into out (16-byte aligned) with rows
     * (size + 15) & ~15 bytes apart; the bytes after a row's size are left as they come. */
    void (*block)(uint8_t *out, int size, const uint8_t *img, int w, int h, const int origin[2]);
} mvo_kernels_t;

typedef struct {
    mvo_camera_t camera;
    uint8_t *pyramid[2];      /* both frames' pyramids (MVO_PYRAMID_BYTES each, the caller's) */
    int32_t current;          /* which pyramid holds the newest frame */
    mvo_feature_t feature[MVO_MAX_FEATURES];
    int32_t count;
    float recent_length[MVO_SPEED_STEPS]; /* the last step lengths, a ring */
    int32_t recent_count, recent_next;
    float world_rotation[9], world_centre[3];
    uint32_t rng;
    uint32_t steps;
    uint32_t next_id;         /* id of the next detected feature */
    mvo_timing_t timing;
    /* Optional, set after mvo_init: runs work(context, 0) and work(context, 1) at the same time (one per core) and
     * returns when both are done. NULL runs them one after the other; the results are the same either way. */
    void (*parallel)(void (*work)(void *context, int part), void *context);
    const mvo_kernels_t *kernels; /* optional, set after mvo_init; NULL: the portable ones below */
} mvo_t;

/* pyramid_a and pyramid_b: MVO_PYRAMID_BYTES each, kept for the life of the state. */
void mvo_init(mvo_t *s, const mvo_camera_t *camera, uint32_t seed, uint8_t *pyramid_a, uint8_t *pyramid_b);
/* Where the next 320x240 grey frame (row-major bytes) goes before mvo_start or mvo_step. */
uint8_t *mvo_frame(mvo_t *s);
void mvo_start(mvo_t *s);
void mvo_step(mvo_t *s, mvo_step_t *out);
/* A step's length from the depth ratios of its shared points (count of them, reordered), as described above, and
 * remembered for the steps after it: mvo_step's own, public for tests. */
float mvo_step_length(mvo_t *s, float *ratios, int count);
uint32_t mvo_state_size(void);
uint32_t mvo_pyramid_bytes(void);
/* The build's name ("p4", see above) and MVO_MAX_FEATURES. */
const char *mvo_config_name(void);
uint32_t mvo_max_features(void);
float mvo_sqrt(float v);
/* Where a pinhole with the camera's fx, fy, cx, cy would see the ray its lens shows at pixel `in` of the 320x240 frame:
 * OpenCV's distortion model inverted by MVO_UNDISTORT_STEPS Newton steps from `in` (in and out may be the same).
 * Returns 1, or 0 where an estimate reaches a fold of the model (its radial factor or its Jacobian's determinant not
 * positive), out then holding that estimate. With all five coefficients zero out = in exactly. */
int mvo_undistort(const mvo_camera_t *camera, const float in[2], float out[2]);
/* The features tracked into the newest frame: pixels (x, y pairs) and ids, at most capacity; returns the count. */
int mvo_tracks(const mvo_t *s, float *xy, uint32_t *ids, int capacity);
/* The portable tracker kernels (see mvo_kernels_t). */
void mvo_lk_template_c(mvo_window_t *window, const uint8_t *src, int stride, const int16_t weights[4], int64_t g[3]);
void mvo_lk_sums_c(const mvo_window_t *window, const uint8_t *src, int stride, const int16_t weights[4],
                   int64_t sums[2]);
void mvo_block_c(uint8_t *out, int size, const uint8_t *img, int w, int h, const int origin[2]);

#endif
