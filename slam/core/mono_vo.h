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
 * A gyro is optional (mvo_gyro_mount, mvo_gyro_rate). Its readings over a step, less the estimated bias, add up to the
 * step's rotation in the camera's axes. Where the camera alone would end that step in ROTATION or LOST (travel over a
 * floor that the homography reads as a turn, or a swing whose travel the pairs cannot tell from turning), the step
 * takes the gyro's rotation instead and solves for the translation alone: two-pair MSAC on the epipolar constraint,
 * reweighted least squares and the direction search. It is ESSENTIAL when the translation evidence favours that
 * translation over the gyro's rotation alone, else ROTATION with the gyro's rotation when MVO_MIN_INLIERS and half of
 * the tracks agree with it, else LOST with it. The camera's own ESSENTIAL steps that the gyro turns by less than
 * MVO_GYRO_CALM teach the bias: the gyro's miss against the pose's rotation (at most MVO_GYRO_MISS) over the step's
 * time, the mean over the first MVO_GYRO_BIAS_STEPS such steps and an exponential average with that weight after. A
 * step whose readings cover less than three quarters or more than four thirds of their usual time (a running average)
 * teaches none, so that readings lost on the way cannot set the bias, and the estimate stays within
 * MVO_GYRO_BIAS_LIMIT. A step without readings is the camera's alone, bit for bit. On AQUALOC's harbour sequences 01
 * and 07 (a camera looking down at the seabed, the dataset's own 200 Hz gyro on its clock), at 5 Hz over 16 seeds, the
 * path after one alignment over 10 s windows came out 0.060 m off in the median instead of 0.233 (lower in 93% of 1040
 * paired runs): the camera alone took much of the travel over that floor for turns. Over the whole sequences it came
 * out 2.03 m off instead of 3.25, and the yaw 8.3 degrees instead of 66; without the bias learnt, sequence 01's 1.3
 * degrees per second left a yaw error of 121 degrees. The camera's own rotations take a little travel across the view
 * for a turn that tilts its axis, so the learnt bias stays a few tenths of a degree per second off on each axis. The
 * fast swings of a forward-looking camera, where the lake clip loses its travel, are untested: that clip has no gyro.
 *
 * An accelerometer on the gyro's mounting is optional too (mvo_imu_reading: a gyro reading with the specific force over
 * the same seconds, less a calibrated bias, mvo_imu_set_accel_bias; a force beyond MVO_IMU_FORCE_MAX is dropped as a
 * knock or a fault). The forces since the last frame, turned into its axes with the gyro, measure that frame's up,
 * which its world rotation carries into the world's axes. MVO_IMU_UP_STEPS such ups in a row that agree (each within
 * MVO_IMU_UP_AGREE of their mean; one outside starts the count again, so that thrust at the start, which reads as tilt,
 * does not set it) settle the world's up for good, and nothing is levelled before. After every step from then on the
 * world rotation turns towards it about the level axis of the two (a tilt, never a turn of the heading) by the sine of
 * their angle times MVO_IMU_LEVEL_RATE times the readings' seconds, at most MVO_IMU_LEVEL_MAX, so the tilt the steps
 * gather fades with a time constant of about a second. Only the world poses change: a step's own motion, the bias
 * learnt, the points and the scale stay as they were, and a step without forces is levelled by none, bit for bit. On
 * the same harbour sequences, with the accelerometer's bias calibrated on the other sequence (about 0.6 m/s^2 across
 * the vertical, a 4-degree tilt, alike on both), the tilt over the whole sequences came out 1.6 degrees off in the
 * median of both and 16 seeds instead of 19 with the gyro alone, the whole orientation 7.7 instead of 21, and the
 * path's height, after an alignment that turns it only about the vertical, 0.32 m off instead of 0.76; over 10 s
 * windows the tilt 0.68 degrees instead of 2.0 (lower in 97% of 1040 paired runs). The path after a full alignment
 * stayed where it was (2.05 against 2.03 m): almost all of its error is horizontal, which gravity does not show.
 * Uncalibrated, the tilt came out 5.3 degrees off. The motion's own accelerations count as tilt; a robot's thrust is
 * untested.
 *
 * A metric scale is optional (mvo_metric: the world centre in metres, once the scale is known). The odometry's unit
 * is the camera's and drifts; a depth sensor (mvo_depth_reading: a frame's depth below the surface in metres, given
 * before mvo_start for the first frame and before mvo_step for the others) and a forward ranger (mvo_range_setup: its
 * beam's axis in the camera's axes and the cosine of its half angle; mvo_range_reading: a frame's range in metres)
 * observe it. A Kalman filter on the scale, metres per unit, as a random walk whose variance grows by MVO_SCALE_DRIFT
 * times the scale's square per second (the readings' seconds, else MVO_SCALE_STEP_TIME a step), weighs the
 * observations. Once the accelerometer's up has settled, the depth's change over the last MVO_DEPTH_HORIZON seconds
 * against the odometry's change along the world's down observes it, when the depth changed by MVO_DEPTH_MIN_CHANGE,
 * both the same way, with MVO_DEPTH_NOISE on both ends. With the first frame's depth, the metric centre's vertical is
 * the depth's change since then. A range observes it against the tracked points in the beam, at least
 * MVO_RANGE_POINTS of them: the distance within which a share MVO_RANGE_SHARE of them lie, as the echo comes from the
 * nearest surface, with a relative error of MVO_RANGE_NOISE. The first observation scales the path so far as well,
 * so it must be good to MVO_SCALE_FIRST_SPREAD of itself, a depth's counting the odometry's own vertical error too,
 * MVO_DEPTH_TRAVEL of the distance it moved meanwhile: a small climb or dive over a long swim gives no first scale.
 * After it each step adds its move times the scale. Later observations are not gated: on AQUALOC an innovation
 * gate, that vertical error in every observation's variance, or leaving out windows over lost steps each lost
 * accuracy, as the scale drifts and single observations scatter widely (their variances weigh them). The odometry's
 * own steps, world poses and points stay as they were, bit for bit, and without readings nothing changes. On
 * AQUALOC's harbour sequences (their depth sensor) and on rangers simulated from the lake's reconstruction it was
 * measured against the reference in metres: results/aqualoc and results/lake_clip/ranger.
 *
 * C99, no heap, no libm (a local square root, by Newton steps without division), xorshift32 for RANSAC, so a PC build
 * and an ESP32-P4 build give identical results with -ffp-contract=off and without fast-math. It follows the OpenCV
 * prototype's design, not its numbers: the prototype uses OpenCV's five-point solver and its own Lucas-Kanade, stops
 * tracking the pairs recoverPose drops as too far, and takes every median depth ratio as it comes. One mvo_t is about
 * 10 KB; the caller also provides two image pyramids of MVO_PYRAMID_BYTES (about 100 KB each, 16-byte aligned for
 * vector kernels), separate blocks so that they fit a fragmented heap. Most of a step splits in two halves that can run at once (mvo_t.parallel): pyramid level 1
 * (half the rows each), the tracking (half the features), each RANSAC batch (half the samples), the corner search (half
 * the rows), the direction search (half the directions), the two pose refinements, a lone refinement's Jacobian (its
 * rows) and, with a gyro, each batch of two-pair samples and the search over directions for its rotation; the results
 * are the same either way. The tracker's kernels (two fixed-point ones and a block copy)
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
#define MVO_RATIO_DEPTH 60.0f /* shared points this pair puts farther than this many baselines give no depth ratio; not
                               * below MVO_MAX_DEPTH, where cutting on the pair's own depths biased the scale (above) */
#define MVO_UNDISTORT_STEPS 5 /* Newton steps inverting the lens model: within 1e-3 px of OpenCV's in the tests */
#define MVO_TURN_GATE 0.0038053019f /* 1 - cos 5 degrees: a step either rotation turns less does not pull */
#define MVO_TURN_FULL 0.0151922470f /* 1 - cos 10 degrees: where the pull is MVO_TURN_PULL */
#define MVO_TURN_PULL 3000.0f /* per squared radian, against a robust cost in which a pair costs at most 1 */
#define MVO_TURN_CAP 4.0f /* the pull grows no further than this many times MVO_TURN_PULL (about 20 degrees) */
#define MVO_TURN_SOFT 0.0048738787f /* (4 degrees)^2 in radians: a rotation this far from the pose pulls half as hard */
#define MVO_GYRO_BIAS_STEPS 50 /* the bias estimate's memory, in calm ESSENTIAL steps */
#define MVO_GYRO_CALM 0.0013704652f /* 1 - cos 3 degrees: a step the gyro turns further teaches no bias */
#define MVO_GYRO_MISS 0.0174524064f /* sin 1 degree: the most a step's miss counts */
#define MVO_GYRO_BIAS_LIMIT 0.17453293f /* 10 degrees per second: the most bias the estimate takes, per axis */
#define MVO_IMU_UP_STEPS 10 /* the world's up: the mean of this many agreeing steps' measured ups */
#define MVO_IMU_UP_AGREE 0.9961947f /* cos 5 degrees: of AQUALOC's 1072 re-chained runs, 3 degrees left 3 unsettled */
#define MVO_IMU_LEVEL_RATE 1.0f /* per second of readings: the share of a step's tilt that the step takes back */
#define MVO_IMU_LEVEL_MAX 0.0348995f /* sin 2 degrees: the most one step's levelling turns */
#define MVO_IMU_FORCE_MAX 49.0f /* m/s^2, 5 g: a reading's force beyond is a knock or a fault, not gravity */
#define MVO_SCALE_DRIFT 0.03f /* per second: the scale's variance grows by this times the scale's square */
#define MVO_SCALE_STEP_TIME 0.2f /* s: a step's time when no IMU readings measure it (the board's 5 Hz) */
#define MVO_SCALE_FIRST_SPREAD 0.3f /* the first observation's standard deviation at most: this share of it */
#define MVO_DEPTH_HORIZON 2.0f /* s: the depth's change over this time observes the scale */
#define MVO_DEPTH_MIN_CHANGE 0.1f /* m: a smaller change observes nothing */
#define MVO_DEPTH_NOISE 0.02f /* m: the depth sensor's noise, one standard deviation */
#define MVO_DEPTH_TRAVEL 0.07f /* the odometry's vertical error, levelled by the IMU: this share of its travel */
#define MVO_DEPTH_RING 32 /* depth readings kept: MVO_DEPTH_HORIZON at up to 15 frames a second */
#define MVO_RANGE_NOISE 0.2f /* a range's relative error against the points in its beam */
#define MVO_RANGE_POINTS 5 /* points in the beam a range needs */
#define MVO_RANGE_SHARE 0.1f /* the points' quantile a range is compared with */
#define MVO_PYRAMID_BYTES (MVO_WIDTH * MVO_HEIGHT * 4 / 3 + 64) /* one frame's pyramid, level 0 first */
#define MVO_SIDE (2 * MVO_WINDOW + 1) /* the tracking window's side */

#if defined(__GNUC__)
#define MVO_ALIGN16 __attribute__((aligned(16))) /* where vector kernels load whole 16-byte blocks */
#else
#define MVO_ALIGN16
#endif

enum { MVO_LOST = 0, MVO_ROTATION = 1, MVO_ESSENTIAL = 2 };
/* How a step used the gyro (mvo_gyro_used): not at all, or for one of these kinds of step. */
enum { MVO_GYRO_NONE = 0, MVO_GYRO_TRANSLATION = 1, MVO_GYRO_ROTATION = 2, MVO_GYRO_LOST = 3 };

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

/* The gyro's part of the state (see mvo_gyro_mount and mvo_gyro_rate), and an accelerometer's on the same mounting
 * (mvo_imu_reading). */
typedef struct {
    float mount[9];     /* row-major: a rate in the gyro's axes, turned into the camera's */
    float bias[3];      /* the estimated bias, rad/s in the gyro's axes, taken off every reading */
    float pending[4];   /* the readings since the last frame as one rotation R_AB, a unit quaternion (w, x, y, z) */
    float pending_time; /* the seconds they cover; 0: no readings, the next step goes without the gyro */
    float usual_time;   /* a running average of the seconds a step's readings cover; 0 before the first */
    int32_t bias_steps; /* the steps the estimate learnt from, at most MVO_GYRO_BIAS_STEPS */
    int32_t used;       /* how the last step used the gyro (MVO_GYRO_*) */
    float accel_bias[3]; /* the accelerometer's calibrated bias, m/s^2 in its axes, taken off every reading */
    float force[3];      /* the specific force since the last frame in A's axes, summed over the readings' seconds */
    float force_time;    /* the seconds it covers; 0: the next step is not levelled */
    float measured[3];   /* the step's measured up in the world's axes, unit; zero when it has none */
    float level_time;    /* the seconds behind measured */
    float up_sum[3];     /* the sum of the agreeing measured ups: the world's up is its direction */
    int32_t up_steps;    /* how many it holds; MVO_IMU_UP_STEPS: settled */
} mvo_gyro_t;

/* The metric scale's part of the state (mvo_depth_reading, mvo_range_setup, mvo_range_reading, mvo_metric). */
typedef struct {
    float depth, range;      /* the next frame's readings, m */
    int32_t has_depth, has_range;
    float beam[3], beam_cos; /* the ranger's axis (unit, the camera's axes) and the cosine of its half angle */
    int32_t has_beam;
    float first_depth;       /* the run's first frame's depth: the metric vertical's zero */
    int32_t has_first_depth;
    /* recent depths, with the odometry's position along the down, the time and the odometry's path length then */
    float ring_depth[MVO_DEPTH_RING], ring_down[MVO_DEPTH_RING], ring_time[MVO_DEPTH_RING], ring_path[MVO_DEPTH_RING];
    int32_t ring_count, ring_next;
    float time, step_time;   /* seconds since the run's start, and this step's */
    float path;              /* the odometry's path length since the run's start, its unit */
    float scale, variance;   /* metres per odometry unit (0: not observed yet) and its variance */
    float metric[3];         /* the world centre in metres */
} mvo_scale_t;

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
    mvo_gyro_t gyro;
    mvo_scale_t scale;
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
/* The gyro's mounting, row-major: mount turns a rate in the gyro's axes into the camera's (mvo_init sets the
 * identity). Returns 0, keeping the mounting it had, when an entry is not finite. */
int mvo_gyro_mount(mvo_t *s, const float mount[9]);
/* One gyro reading for the step to the next frame: the angular rate (rad/s in the gyro's axes, bias and all) held for
 * dt seconds. Readings add up until mvo_step uses them; mvo_start drops them. The integration is a fourth-order series
 * in each reading's half angle, exact to float precision while that stays under 0.1 rad (e.g. 100 Hz readings below
 * 1100 degrees per second). A reading without a positive dt, or whose half angle is not finite or reaches 1 rad, is
 * dropped. */
void mvo_gyro_rate(mvo_t *s, const float rate[3], float dt);
/* The bias estimate, rad/s in the gyro's axes. */
void mvo_gyro_bias(const mvo_t *s, float bias[3]);
/* Sets the estimate (a calibration, or an estimate carried over from a longer run) as one learnt from
 * MVO_GYRO_BIAS_STEPS steps. Returns 0, changing nothing, when an entry is not finite or beyond MVO_GYRO_BIAS_LIMIT. */
int mvo_gyro_set_bias(mvo_t *s, const float bias[3]);
/* How the last step used the gyro (MVO_GYRO_*). */
int mvo_gyro_used(const mvo_t *s);
/* One reading of an IMU, a gyro and an accelerometer on the mounting mvo_gyro_mount sets: the rate as mvo_gyro_rate
 * takes it, and the accelerometer's specific force (m/s^2, the IMU's axes, the calibrated bias still in it) over the
 * same dt seconds. Whatever mvo_gyro_rate drops is dropped whole; a force, less the bias, that is not finite or beyond
 * MVO_IMU_FORCE_MAX drops the accelerometer's part only. Returns 1 when the force was taken, else 0. The forces since
 * the last frame, turned into its axes with the gyro, level the next step (see above); mvo_start drops them. */
int mvo_imu_reading(mvo_t *s, const float rate[3], const float accel[3], float dt);
/* The accelerometer's bias (a calibration, m/s^2 in the IMU's axes; mvo_init sets zero), taken off every reading
 * after it. Returns 0, changing nothing, when an entry is not finite. */
int mvo_imu_set_accel_bias(mvo_t *s, const float bias[3]);
/* The world's up as the accelerometer measured it (unit, the world's axes: the first camera's); zero until it has
 * settled (see above). */
void mvo_imu_up(const mvo_t *s, float up[3]);
/* A depth sensor's reading for the next frame: its depth below the surface, m (see above). Returns 0, keeping
 * nothing, when it is not finite. */
int mvo_depth_reading(mvo_t *s, float depth);
/* A forward ranger's beam: its axis in the camera's axes (any nonzero length) and the cosine of its half angle, given
 * after mvo_init, which clears it. Returns 0, changing nothing, for a zero or not finite axis or a cosine not between
 * 0 and 1. mvo_start keeps it. */
int mvo_range_setup(mvo_t *s, const float axis[3], float half_angle_cos);
/* The ranger's reading for the next frame: the range along the beam, m. Returns 0, keeping nothing, without a beam
 * or when it is not finite and positive. One given before mvo_start is dropped: there are no points to compare yet. */
int mvo_range_reading(mvo_t *s, float range);
/* The world centre in metres, the scale (metres per odometry unit) and its variance (either pointer may be NULL);
 * returns 0, all zero, until the scale is known. */
int mvo_metric(const mvo_t *s, float centre[3], float *scale, float *variance);
/* The portable tracker kernels (see mvo_kernels_t). */
void mvo_lk_template_c(mvo_window_t *window, const uint8_t *src, int stride, const int16_t weights[4], int64_t g[3]);
void mvo_lk_sums_c(const mvo_window_t *window, const uint8_t *src, int stride, const int16_t weights[4],
                   int64_t sums[2]);
void mvo_block_c(uint8_t *out, int size, const uint8_t *img, int w, int h, const int origin[2]);

#endif
