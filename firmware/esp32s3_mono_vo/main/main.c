/*
 * Runs monocular visual odometry (core/mono_vo.c), a mask-independent sparse 3D cloud (core/mono_cloud.c), the
 * continuing voxel map (core/voxel_map.c, with the masks when they come), the planner (core/path_plan.c) and, when
 * masks come with the frames, the obstacle map (core/mono_map.c) and the obstacle directions (core/obstacle_dir.c) on
 * frames streamed over the USB-Serial-JTAG port, so a whole clip can be checked against the PC build of the same C code and timed. The host (python -m uwslam.mvo_stream) sends the camera and the seed, then one
 * 320x240 grey frame at a time, each optionally with the segmentation's 60x80 water mask; the board answers every frame
 * with one line. While idle it repeats a `ready` line, so the host can open the port at any time.
 *
 *   host:  selftest <cases> <seed>                vector kernels and safe edge dispatch against portable C
 *   board: selftest cases=<n> template_mismatches=<n> sums_mismatches=<n> block_mismatches=<n> template_us_c=<t>
 *               template_us_simd=<t> sums_us_c=<t> sums_us_simd=<t> block_us_c=<t> block_us_simd=<t>
 *                                                 (microseconds per call, averaged)
 *   host:  start <fx> <fy> <cx> <cy> <seed> [<cores> [<simd>]]   the four floats as their IEEE-754 bits, unsigned
 *                                                       decimal; cores 2 (default) runs half of each split part on the
 *                                                       second core; simd 1 (default) uses the vector kernels
 *   board: started state_bytes=<n> cores=<1 or 2> simd=<0 or 1>
 *   host:  frame <index> <fnv> [<mask fnv>]       then the frame's 76800 bytes, and with a mask fnv its 600 bytes:
 *                                                 bit i of the mask (byte i / 8, least significant bit first) is cell
 *                                                 i in row-major order, set where it is not water
 *   board: first index=0 tracks=<n> detect_us=<t> dir_choice=<d> dir_blocked=<bits> dir_us=<t>
 *                                                 (the frame that starts the run)
 *          step index=<i> kind=<k> tracks=<n> inliers=<n> scale_points=<n> hash=<fnv of the mvo_step_t>
 *               detect_us=<t> track_us=<t> motion_us=<t> total_us=<t> observed=<n> mapped=<n> map_us=<t>
 *               map_continuous=<0 or 1> cloud_count=<n> cloud_accepted=<n> cloud_dropped=<n>
 *               cloud_continuous=<0 or 1> cloud_us=<t> voxels=<n> voxel_filed=<n> voxel_us=<t>
 *               dir_choice=<d> dir_blocked=<bits> dir_us=<t> plan=<action> plan_dir=<d> plan_turn=<-1, 0 or 1>
 *               plan_hash=<fnv of the pp_plan_t> plan_us=<t>
 *                                                 dir_*: the obstacle directions after this frame's mask (unchanged
 *                                                 without one): the choice (-1: none free) and bit d set where
 *                                                 direction d is blocked; plan*: the plan after this step
 *   host:  end
 *   board: map cell_bits=<float bits> origin_x=<n> origin_z=<n> observations=<n> frames=<n>
 *              slice_y_bits=<float bits> height_resets=<n> hash=<fnv of the mm_map_t> map_continuous=<0 or 1>
 *          voxels side_bits=<float bits> count=<n> frames=<n> observations=<n> evicted=<n> merged=<n> rescales=<n>
 *              lost=<n> hash=<fnv of the vm_map_t>
 *          dirs frames=<n> choice=<d> blocked=<bits> hash=<fnv of the od_state_t>
 *          plan frames=<n> action=<a> direction=<d> turn=<t> hash=<fnv of the pp_plan_t>
 *          end heap_internal_free=<bytes> largest_block=<bytes> stack_free_words=<n> map_continuous=<0 or 1>
 *              cloud_count=<n> cloud_dropped=<n> cloud_continuous=<0 or 1> cloud_hash=<fnv of the mc_map_t>
 *   host:  cloud                                  read the current 3D cloud without changing it
 *   board: cloud count=<n> dropped=<n> continuous=<0 or 1> hash=<fnv of the mc_map_t>
 *          point index=<i> id=<id> x_bits=<bits> y_bits=<bits> z_bits=<bits> spread_bits=<bits> observations=<n>
 *          cloud_end count=<n>                    floats are sent as IEEE-754 uint32 bits
 *   board: ready ... map_policy=1 cloud_policy=1 freeze the obstacle map and the cloud after the first LOST until
 *                    the next start; the voxel map goes on through LOST steps
 *
 * Errors are reported on an `error` line and the board waits for the next command.
 */
#include <inttypes.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "driver/usb_serial_jtag.h"
#include "driver/usb_serial_jtag_vfs.h"
#include "esp_cpu.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "sdkconfig.h"

#include "mono_cloud.h"
#include "mono_map.h"
#include "mono_vo.h"
#include "obstacle_dir.h"
#include "path_plan.h"
#include "voxel_map.h"

#define FRAME_BYTES (MVO_WIDTH * MVO_HEIGHT)
#define MASK_WIDTH 80
#define MASK_HEIGHT 60
#define MASK_BYTES (MASK_WIDTH * MASK_HEIGHT / 8)
#define LINE_BYTES 160
#define IDLE_MS 1000       /* a `ready` line this often while no command comes */
#define CHUNK_WAIT_MS 5000 /* longest pause allowed inside a frame's bytes */

/* cloud_hash covers the entire struct, whose representation the host reconstructs from the exported point fields. */
typedef char cloud_point_wire_layout[(sizeof(float) == 4 && sizeof(mc_point_t) == 24 &&
                                     offsetof(mc_point_t, id) == 12 && offsetof(mc_point_t, spread) == 16 &&
                                     offsetof(mc_point_t, observations) == 20) ? 1 : -1];
typedef char cloud_map_wire_layout[(offsetof(mc_map_t, count) == MC_MAX_POINTS * 24 &&
                                   offsetof(mc_map_t, dropped) == MC_MAX_POINTS * 24 + 4 &&
                                   offsetof(mc_map_t, continuous) == MC_MAX_POINTS * 24 + 8 &&
                                   sizeof(mc_map_t) == MC_MAX_POINTS * 24 + 12) ? 1 : -1];

/* Internal RAM from the heap: the two pyramids (about 100 KB each) first, then the state (about 8 KB), the obstacle
 * map (about 16 KB) and the fixed-capacity cloud (about 12 KB), in separate blocks; the voxel map (about 14 KB) last,
 * once the driver and the second core's task have their memory (see app_main). The obstacle directions are small enough
 * to be static. */
static uint8_t *s_pyramid[2];
static mvo_t *s_state;
static mm_map_t *s_map;
static mc_map_t *s_cloud;
static vm_map_t *s_voxels;
static od_layout_t s_layout;
static od_state_t s_dirs;
static pp_plan_t s_plan;
static uint8_t s_packed[MASK_BYTES], s_mask[MASK_WIDTH * MASK_HEIGHT];
static float s_points[MVO_MAX_FEATURES][3];
static int s_session_ready, s_streaming, s_started, s_map_continuous;
static int64_t s_next_frame;
static SemaphoreHandle_t s_go, s_done; /* hand half of a split part to the second core and wait for it */
static void (*s_work)(void *context, int part);
static void *s_context;

/* mvo_pie.S: the tracker kernels with the S3's vector instructions, giving the portable ones' integers. */
void mvo_pie_lk_template(mvo_window_t *window, const uint8_t *src, int stride, const int16_t weights[4], int64_t g[3]);
void mvo_pie_lk_sums(const mvo_window_t *window, const uint8_t *src, int stride, const int16_t weights[4],
                     int64_t sums[2]);
void mvo_pie_block(uint8_t *out, int size, const uint8_t *img, int w, int h, const int origin[2]);

static void mvo_pie_block_safe(uint8_t *out, int size, const uint8_t *img, int w, int h, const int origin[2])
{
    /* The vector loads include an aligned block before and another after the requested pixels. */
    if (origin[0] < 16 || origin[0] + ((size + 15) & ~15) + 16 > w) {
        mvo_block_c(out, size, img, w, h, origin);
    } else {
        mvo_pie_block(out, size, img, w, h, origin);
    }
}

static const mvo_kernels_t s_simd = {mvo_pie_lk_template, mvo_pie_lk_sums, mvo_pie_block_safe};

static void helper_task(void *unused)
{
    (void)unused;
    for (;;) {
        xSemaphoreTake(s_go, portMAX_DELAY);
        s_work(s_context, 1);
        xSemaphoreGive(s_done);
    }
}

/* mvo_t.parallel: part 1 on the second core, part 0 on this one. */
static void run_both(void (*work)(void *context, int part), void *context)
{
    s_work = work;
    s_context = context;
    xSemaphoreGive(s_go);
    work(context, 0);
    xSemaphoreTake(s_done, portMAX_DELAY);
}

static uint32_t fnv(uint32_t hash, const void *data, size_t size)
{
    const uint8_t *bytes = data;
    size_t i;
    for (i = 0; i < size; ++i) {
        hash = (hash ^ bytes[i]) * 16777619u;
    }
    return hash;
}

static int64_t clock_us(void) { return esp_timer_get_time(); }

/* One line without its end, or 0 when nothing arrives within wait_ms. */
static int read_line(char *line, int size, int wait_ms)
{
    int length = 0;
    for (;;) {
        char c;
        if (usb_serial_jtag_read_bytes(&c, 1, pdMS_TO_TICKS(length ? CHUNK_WAIT_MS : wait_ms)) != 1) {
            return 0;
        }
        if (c == '\n') {
            line[length] = '\0';
            return 1;
        }
        if (c != '\r' && length < size - 1) {
            line[length++] = c;
        }
    }
}

static int read_exact(uint8_t *out, int size)
{
    int got = 0;
    while (got < size) {
        const int part = usb_serial_jtag_read_bytes(out + got, (uint32_t)(size - got), pdMS_TO_TICKS(CHUNK_WAIT_MS));
        if (part <= 0) {
            return 0;
        }
        got += part;
    }
    return 1;
}

static uint32_t next_random(uint32_t *state)
{
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return *state = x;
}

/* Random windows of random images through both kernel sets; before a run, pyramid 0 serves as the image and
 * pyramid 1 holds the windows and the copied blocks. */
static void selftest(const char *line)
{
    static const int strides[4] = {40, 80, 160, 320}, heights[4] = {30, 60, 120, 240};
    uint32_t state, cycles[6] = {0, 0, 0, 0, 0, 0};
    int cases, k, template_bad = 0, sums_bad = 0, block_bad = 0;
    uint8_t *image = s_pyramid[0];
    mvo_window_t *windows = (mvo_window_t *)s_pyramid[1]; /* 16-byte aligned */
    uint8_t *blocks[2] = {s_pyramid[1] + 2 * sizeof(mvo_window_t), s_pyramid[1] + 2 * sizeof(mvo_window_t) + 1024};
    if (s_streaming && s_started) {
        printf("error selftest requires a new or ended session\n");
        return;
    }
    if (sscanf(line, "selftest %d %" SCNu32, &cases, &state) != 2 || cases < 1 || state == 0) {
        printf("error bad selftest line\n");
        return;
    }
    for (k = 0; k < MVO_PYRAMID_BYTES; ++k) {
        image[k] = (uint8_t)next_random(&state);
    }
    for (k = 0; k < cases; ++k) {
        const int stride = strides[next_random(&state) % 4];
        const int x = 1 + (int)(next_random(&state) % (uint32_t)(stride - 21)), y = 1 + (int)(next_random(&state) % 40u);
        const uint8_t *src = image + y * stride + x;
        int16_t weights[4];
        int64_t g[2][3], sums[2][2];
        uint32_t begin;
        weights[0] = (int16_t)(next_random(&state) % 16385u);
        weights[1] = (int16_t)(next_random(&state) % (uint32_t)(16385 - weights[0]));
        weights[2] = (int16_t)(next_random(&state) % (uint32_t)(16385 - weights[0] - weights[1]));
        weights[3] = (int16_t)(16384 - weights[0] - weights[1] - weights[2]);
        begin = esp_cpu_get_cycle_count();
        mvo_lk_template_c(&windows[0], src, stride, weights, g[0]);
        cycles[0] += esp_cpu_get_cycle_count() - begin;
        begin = esp_cpu_get_cycle_count();
        mvo_pie_lk_template(&windows[1], src, stride, weights, g[1]);
        cycles[1] += esp_cpu_get_cycle_count() - begin;
        template_bad += memcmp(&windows[0], &windows[1], sizeof windows[0]) != 0
                        || memcmp(g[0], g[1], sizeof g[0]) != 0;
        src += (int)(next_random(&state) % 5u) * stride + (int)(next_random(&state) % 5u) - 2 * stride - 2;
        begin = esp_cpu_get_cycle_count();
        mvo_lk_sums_c(&windows[0], src, stride, weights, sums[0]);
        cycles[2] += esp_cpu_get_cycle_count() - begin;
        begin = esp_cpu_get_cycle_count();
        mvo_pie_lk_sums(&windows[0], src, stride, weights, sums[1]);
        cycles[3] += esp_cpu_get_cycle_count() - begin;
        sums_bad += memcmp(sums[0], sums[1], sizeof sums[0]) != 0;
        { /* a block anywhere near a random image level, including the start of the allocation */
            const int level = (int)(next_random(&state) % 4u), w = strides[level], h = heights[level];
            const int size = next_random(&state) % 2u ? MVO_SIDE + 1 : MVO_SIDE + 3, stride = (size + 15) & ~15;
            int origin[2], row;
            origin[0] = (int)(next_random(&state) % (uint32_t)(w + 40)) - 20;
            origin[1] = (int)(next_random(&state) % (uint32_t)(h + 40)) - 20;
            begin = esp_cpu_get_cycle_count();
            mvo_block_c(blocks[0], size, image, w, h, origin);
            cycles[4] += esp_cpu_get_cycle_count() - begin;
            begin = esp_cpu_get_cycle_count();
            mvo_pie_block_safe(blocks[1], size, image, w, h, origin);
            cycles[5] += esp_cpu_get_cycle_count() - begin;
            for (row = 0; row < size; ++row) { /* only the bytes that are the block's */
                block_bad += memcmp(blocks[0] + row * stride, blocks[1] + row * stride, (size_t)size) != 0;
            }
        }
    }
    printf("selftest cases=%d template_mismatches=%d sums_mismatches=%d block_mismatches=%d template_us_c=%.2f "
           "template_us_simd=%.2f sums_us_c=%.2f sums_us_simd=%.2f block_us_c=%.2f block_us_simd=%.2f\n",
           cases, template_bad, sums_bad, block_bad, (double)cycles[0] / CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ / cases,
           (double)cycles[1] / CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ / cases, (double)cycles[2] / CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ / cases,
           (double)cycles[3] / CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ / cases, (double)cycles[4] / CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ / cases,
           (double)cycles[5] / CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ / cases);
}

static void start(const char *line)
{
    uint32_t bits[4], seed;
    int cores = 2, simd = 1, fields;
    mvo_camera_t camera;
    fields = sscanf(line, "start %" SCNu32 " %" SCNu32 " %" SCNu32 " %" SCNu32 " %" SCNu32 " %d %d", &bits[0], &bits[1],
                    &bits[2], &bits[3], &seed, &cores, &simd);
    if (fields < 5 || cores < 1 || cores > 2 || simd < 0 || simd > 1) {
        printf("error bad start line\n");
        return;
    }
    memcpy(&camera.fx, &bits[0], sizeof camera.fx);
    memcpy(&camera.fy, &bits[1], sizeof camera.fy);
    memcpy(&camera.cx, &bits[2], sizeof camera.cx);
    memcpy(&camera.cy, &bits[3], sizeof camera.cy);
    mvo_init(s_state, &camera, seed, s_pyramid[0], s_pyramid[1]);
    s_state->timing.clock = clock_us;
    s_state->parallel = cores == 2 ? run_both : NULL;
    s_state->kernels = simd ? &s_simd : NULL;
    mm_init(s_map);
    mc_init(s_cloud);
    vm_init(s_voxels);
    od_layout(&s_layout, &camera, MASK_WIDTH, MASK_HEIGHT); /* a refused camera leaves a layout that takes no mask */
    od_init(&s_dirs);
    pp_init(&s_plan);
    s_started = 0;
    s_next_frame = 0;
    s_session_ready = 1;
    s_streaming = 1;
    s_map_continuous = 1;
    printf("started state_bytes=%" PRIu32 " cores=%d simd=%d\n", mvo_state_size(), cores, simd);
}

static void frame(const char *line)
{
    uint32_t expected, mask_expected = 0, blocked = 0;
    int index, fields, with_mask, i;
    int64_t dir_us = 0;
    uint8_t *pixels;
    fields = sscanf(line, "frame %d %" SCNu32 " %" SCNu32, &index, &expected, &mask_expected);
    if (fields < 2) {
        printf("error bad frame line\n");
        return;
    }
    with_mask = fields == 3;
    /* A sender transmits the header and payload together: consume it even before start to keep command framing. */
    pixels = s_streaming ? mvo_frame(s_state) : s_pyramid[0];
    if (!read_exact(pixels, FRAME_BYTES) || (with_mask && !read_exact(s_packed, MASK_BYTES))) {
        printf("error frame %d arrived incomplete\n", index);
        return;
    }
    if (!s_streaming) {
        printf("error start required\n");
        return;
    }
    if (fnv(2166136261u, pixels, FRAME_BYTES) != expected || (with_mask && fnv(2166136261u, s_packed, MASK_BYTES) != mask_expected)) {
        printf("error frame %d hash mismatch\n", index);
        return;
    }
    if (index < 0 || (int64_t)index != s_next_frame) {
        printf("error frame %d out of sequence, expected %" PRId64 "\n", index, s_next_frame);
        return;
    }
    ++s_next_frame;
    for (i = 0; with_mask && i < MASK_WIDTH * MASK_HEIGHT; ++i) {
        s_mask[i] = (uint8_t)((s_packed[i >> 3] >> (i & 7)) & 1u);
    }
    if (with_mask) {
        const int64_t begin = esp_timer_get_time();
        od_update(&s_dirs, &s_layout, s_mask);
        dir_us = esp_timer_get_time() - begin;
    }
    for (i = 0; i < OD_DIRECTIONS; ++i) {
        blocked |= (uint32_t)(s_dirs.blocked[i] != 0) << i;
    }
    if (!s_started) {
        mvo_start(s_state);
        s_started = 1;
        printf("first index=%d tracks=%" PRId32 " detect_us=%" PRId64 " dir_choice=%" PRId32 " dir_blocked=%" PRIu32
               " dir_us=%" PRId64 "\n", index, s_state->count, s_state->timing.detect_us, s_dirs.choice, blocked, dir_us);
    } else {
        mvo_step_t step;
        int observed = 0, mapped = 0, cloud_accepted, voxel_filed;
        int64_t begin = esp_timer_get_time(), total, map_us = 0, cloud_us, voxel_us, plan_us;
        mvo_step(s_state, &step);
        total = esp_timer_get_time() - begin;
        begin = esp_timer_get_time();
        cloud_accepted = mc_update(s_cloud, s_state, &step);
        cloud_us = esp_timer_get_time() - begin;
        begin = esp_timer_get_time();
        voxel_filed = vm_update(s_voxels, s_state, &step, with_mask ? s_mask : NULL, MASK_WIDTH, MASK_HEIGHT);
        voxel_us = esp_timer_get_time() - begin;
        begin = esp_timer_get_time();
        pp_update(&s_plan, s_voxels, s_state, &s_layout, &s_dirs, NULL); /* straight ahead */
        plan_us = esp_timer_get_time() - begin;
        begin = esp_timer_get_time();
        if (step.kind == MVO_LOST) {
            s_map_continuous = 0;
        }
        if (s_map_continuous) {
            if (with_mask) {
                observed = mm_observe(s_state, s_mask, MASK_WIDTH, MASK_HEIGHT, s_points, MVO_MAX_FEATURES);
            }
            mapped = mm_add(s_map, s_state, (const float (*)[3])s_points, observed);
        }
        map_us = esp_timer_get_time() - begin;
        printf("step index=%d kind=%" PRId32 " tracks=%" PRId32 " inliers=%" PRId32 " scale_points=%" PRId32
               " hash=%" PRIu32 " detect_us=%" PRId64 " track_us=%" PRId64 " motion_us=%" PRId64 " total_us=%" PRId64
               " observed=%d mapped=%d map_us=%" PRId64 " map_continuous=%d"
               " cloud_count=%" PRIu32 " cloud_accepted=%d cloud_dropped=%" PRIu32
               " cloud_continuous=%" PRId32 " cloud_us=%" PRId64 " voxels=%" PRIu32 " voxel_filed=%d voxel_us=%" PRId64
               " dir_choice=%" PRId32 " dir_blocked=%" PRIu32 " dir_us=%" PRId64 " plan=%" PRId32 " plan_dir=%" PRId32
               " plan_turn=%" PRId32 " plan_hash=%" PRIu32 " plan_us=%" PRId64 "\n",
               index, step.kind, step.tracks, step.inliers, step.scale_points, fnv(2166136261u, &step, sizeof step),
               s_state->timing.detect_us, s_state->timing.track_us, s_state->timing.motion_us, total, observed, mapped,
               map_us, s_map_continuous, s_cloud->count, cloud_accepted, s_cloud->dropped,
               s_cloud->continuous, cloud_us, s_voxels->count, voxel_filed, voxel_us, s_dirs.choice, blocked, dir_us,
               s_plan.action, s_plan.direction, s_plan.turn, fnv(2166136261u, &s_plan, sizeof s_plan), plan_us);
    }
}

static void end(void)
{
    uint32_t cell_bits, slice_y_bits, side_bits, blocked = 0;
    int i;
    if (!s_streaming) {
        printf("error start required\n");
        return;
    }
    memcpy(&cell_bits, &s_map->cell, sizeof cell_bits);
    memcpy(&slice_y_bits, &s_map->slice_y, sizeof slice_y_bits);
    printf("map cell_bits=%" PRIu32 " origin_x=%" PRId32 " origin_z=%" PRId32 " observations=%" PRIu32 " frames=%" PRIu32
           " slice_y_bits=%" PRIu32 " height_resets=%" PRIu32 " hash=%" PRIu32 " map_continuous=%d\n",
           cell_bits, s_map->origin[0], s_map->origin[1], s_map->observations, s_map->frames,
           slice_y_bits, s_map->height_resets, fnv(2166136261u, s_map, sizeof *s_map), s_map_continuous);
    memcpy(&side_bits, &s_voxels->side, sizeof side_bits);
    printf("voxels side_bits=%" PRIu32 " count=%" PRIu32 " frames=%" PRIu32 " observations=%" PRIu32 " evicted=%" PRIu32
           " merged=%" PRIu32 " rescales=%" PRIu32 " lost=%" PRIu32 " hash=%" PRIu32 "\n", side_bits, s_voxels->count,
           s_voxels->frames, s_voxels->observations, s_voxels->evicted, s_voxels->merged, s_voxels->rescales,
           s_voxels->lost, fnv(2166136261u, s_voxels, sizeof *s_voxels));
    for (i = 0; i < OD_DIRECTIONS; ++i) {
        blocked |= (uint32_t)(s_dirs.blocked[i] != 0) << i;
    }
    printf("dirs frames=%" PRIu32 " choice=%" PRId32 " blocked=%" PRIu32 " hash=%" PRIu32 "\n", s_dirs.frames,
           s_dirs.choice, blocked, fnv(2166136261u, &s_dirs, sizeof s_dirs));
    printf("plan frames=%" PRIu32 " action=%" PRId32 " direction=%" PRId32 " turn=%" PRId32 " hash=%" PRIu32 "\n",
           s_plan.frames, s_plan.action, s_plan.direction, s_plan.turn, fnv(2166136261u, &s_plan, sizeof s_plan));
    printf("end heap_internal_free=%u largest_block=%u stack_free_words=%u map_continuous=%d"
           " cloud_count=%" PRIu32 " cloud_dropped=%" PRIu32 " cloud_continuous=%" PRId32 " cloud_hash=%" PRIu32 "\n",
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
           (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL), (unsigned)uxTaskGetStackHighWaterMark(NULL),
           s_map_continuous, s_cloud->count, s_cloud->dropped, s_cloud->continuous,
           fnv(2166136261u, s_cloud, sizeof *s_cloud));
    s_streaming = 0;
}

static void cloud(void)
{
    uint32_t i;
    if (!s_session_ready) {
        printf("error start required\n");
        return;
    }
    printf("cloud count=%" PRIu32 " dropped=%" PRIu32 " continuous=%" PRId32 " hash=%" PRIu32 "\n",
           s_cloud->count, s_cloud->dropped, s_cloud->continuous, fnv(2166136261u, s_cloud, sizeof *s_cloud));
    for (i = 0; i < s_cloud->count; ++i) {
        const mc_point_t *point = &s_cloud->points[i];
        uint32_t x, y, z, spread;
        memcpy(&x, &point->xyz[0], sizeof x);
        memcpy(&y, &point->xyz[1], sizeof y);
        memcpy(&z, &point->xyz[2], sizeof z);
        memcpy(&spread, &point->spread, sizeof spread);
        printf("point index=%" PRIu32 " id=%" PRIu32 " x_bits=%" PRIu32 " y_bits=%" PRIu32
               " z_bits=%" PRIu32 " spread_bits=%" PRIu32 " observations=%" PRIu32 "\n",
               i, point->id, x, y, z, spread, point->observations);
    }
    printf("cloud_end count=%" PRIu32 "\n", s_cloud->count);
}

void app_main(void)
{
    usb_serial_jtag_driver_config_t config = {.tx_buffer_size = 2048, .rx_buffer_size = 8192};
    char line[LINE_BYTES];
    int helper_ready = 0;
    /* The pyramids first, while the largest free region is still whole; the driver's buffers fit elsewhere. Aligned to
     * 16 bytes, so that the vector kernels' block loads never reach before them. */
    s_pyramid[0] = heap_caps_aligned_alloc(16, MVO_PYRAMID_BYTES, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    s_pyramid[1] = heap_caps_aligned_alloc(16, MVO_PYRAMID_BYTES, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    s_state = heap_caps_malloc(sizeof *s_state, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    s_map = heap_caps_malloc(sizeof *s_map, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    s_cloud = heap_caps_malloc(sizeof *s_cloud, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    ESP_ERROR_CHECK(usb_serial_jtag_driver_install(&config));
    usb_serial_jtag_vfs_use_driver(); /* printf goes through the driver too */
    s_go = xSemaphoreCreateBinary();
    s_done = xSemaphoreCreateBinary();
    if (s_go && s_done) {
        helper_ready = xTaskCreatePinnedToCore(helper_task, "mvo_helper", 4096, NULL, 5, NULL, 1) == pdPASS;
    }
    /* Last, after the driver and the second core's task: allocated before them, it left so little internal RAM that two
     * cores ran no faster than one. */
    s_voxels = heap_caps_malloc(sizeof *s_voxels, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    while (!s_state || !s_pyramid[0] || !s_pyramid[1] || !s_map || !s_cloud || !s_voxels || !s_go || !s_done
           || !helper_ready) {
        printf("error startup memory=%u + 2 x %u + %u + %u + %u bytes (free %u, largest block %u) got state=%d "
               "pyramids=%d/%d map=%d cloud=%d voxels=%d semaphores=%d/%d helper=%d\n", (unsigned)sizeof(mvo_t),
               (unsigned)MVO_PYRAMID_BYTES, (unsigned)sizeof(mm_map_t), (unsigned)sizeof(mc_map_t),
               (unsigned)sizeof(vm_map_t), (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
               (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT), s_state != NULL,
               s_pyramid[0] != NULL, s_pyramid[1] != NULL, s_map != NULL, s_cloud != NULL, s_voxels != NULL,
               s_go != NULL, s_done != NULL, helper_ready);
        fflush(stdout);
        vTaskDelay(pdMS_TO_TICKS(IDLE_MS));
    }
    for (;;) {
        if (!read_line(line, sizeof line, IDLE_MS)) {
            printf("ready mono_vo cpu_mhz=%d state_bytes=%" PRIu32 " core_sha256=%s header_sha256=%s map_sha256=%s "
                   "map_header_sha256=%s map_policy=1 cloud_sha256=%s cloud_header_sha256=%s cloud_policy=1"
                   " cloud_bytes=%" PRIu32 " cloud_capacity=%d voxel_sha256=%s voxel_header_sha256=%s"
                   " voxel_bytes=%" PRIu32 " dir_sha256=%s dir_header_sha256=%s plan_sha256=%s plan_header_sha256=%s\n",
                   CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ, mvo_state_size(), MVO_CORE_SHA256, MVO_HEADER_SHA256, MM_CORE_SHA256,
                   MM_HEADER_SHA256, MC_CORE_SHA256, MC_HEADER_SHA256, mc_map_size(), MC_MAX_POINTS, VM_CORE_SHA256,
                   VM_HEADER_SHA256, vm_map_size(), OD_CORE_SHA256, OD_HEADER_SHA256, PP_CORE_SHA256, PP_HEADER_SHA256);
        } else if (strncmp(line, "selftest ", 9) == 0) {
            selftest(line);
        } else if (strncmp(line, "start ", 6) == 0) {
            start(line);
        } else if (strncmp(line, "frame ", 6) == 0) {
            frame(line);
        } else if (strcmp(line, "end") == 0) {
            end();
        } else if (strcmp(line, "cloud") == 0) {
            cloud();
        } else {
            printf("error unknown command\n");
        }
        fflush(stdout);
    }
}
