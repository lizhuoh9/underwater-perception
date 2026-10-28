/* Exercise the actual firmware frame/start code with host stand-ins for ESP-IDF, VO and map calls. */
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef void *SemaphoreHandle_t;
typedef struct { int tx_buffer_size, rx_buffer_size; } usb_serial_jtag_driver_config_t;
#define portMAX_DELAY 0
#define pdMS_TO_TICKS(n) (n)
#define pdPASS 1
#define MALLOC_CAP_INTERNAL 1
#define MALLOC_CAP_8BIT 2
#define CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ 240
#define ESP_ERROR_CHECK(x) (x)
#define MVO_CORE_SHA256 "core"
#define MVO_HEADER_SHA256 "header"
#define MM_CORE_SHA256 "map"
#define MM_HEADER_SHA256 "mapheader"
#define MC_CORE_SHA256 "cloud"
#define MC_HEADER_SHA256 "cloudheader"
#define VM_CORE_SHA256 "voxels"
#define VM_HEADER_SHA256 "voxelheader"
#define OD_CORE_SHA256 "dirs"
#define OD_HEADER_SHA256 "dirheader"
#define PP_CORE_SHA256 "plan"
#define PP_HEADER_SHA256 "planheader"
#define CONFIG_IDF_TARGET_ESP32S3 1 /* the S3's build: vector kernels (stubbed below) and the USB-Serial-JTAG link */
#define CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG 1

static uint8_t input[320 * 240 + 600], pixels[320 * 240];
static int input_pos, fake_kind, observations, additions, initializations;

int usb_serial_jtag_read_bytes(void *out, uint32_t size, int timeout)
{
    (void)timeout;
    memcpy(out, input + input_pos, size);
    input_pos += (int)size;
    return (int)size;
}
void usb_serial_jtag_vfs_use_driver(void) {}
int usb_serial_jtag_driver_install(const usb_serial_jtag_driver_config_t *config) { (void)config; return 0; }
uint32_t esp_cpu_get_cycle_count(void) { return 0; }
int64_t esp_timer_get_time(void) { return 0; }
void *heap_caps_aligned_alloc(unsigned a, unsigned b, unsigned c) { (void)a; (void)b; (void)c; return NULL; }
void *heap_caps_malloc(unsigned a, unsigned b) { (void)a; (void)b; return NULL; }
unsigned heap_caps_get_free_size(unsigned caps) { (void)caps; return 0; }
unsigned heap_caps_get_largest_free_block(unsigned caps) { (void)caps; return 0; }
SemaphoreHandle_t xSemaphoreCreateBinary(void) { return NULL; }
void xSemaphoreTake(SemaphoreHandle_t sem, int ticks) { (void)sem; (void)ticks; }
void xSemaphoreGive(SemaphoreHandle_t sem) { (void)sem; }
int xTaskCreatePinnedToCore(void (*fn)(void *), const char *name, int stack, void *arg, int priority,
                            void *handle, int core)
{ (void)fn; (void)name; (void)stack; (void)arg; (void)priority; (void)handle; (void)core; return pdPASS; }
unsigned uxTaskGetStackHighWaterMark(void *task) { (void)task; return 0; }
void vTaskDelay(int ticks) { (void)ticks; }

#include "../../firmware/esp32s3_mono_vo/main/main.c"

static mvo_camera_t last_camera;
void mvo_init(mvo_t *s, const mvo_camera_t *camera, uint32_t seed, uint8_t *a, uint8_t *b)
{
    (void)seed; (void)a; (void)b;
    last_camera = *camera;
    memset(s, 0, sizeof *s);
    s->world_rotation[0] = s->world_rotation[4] = s->world_rotation[8] = 1.0f;
    ++initializations;
}
uint8_t *mvo_frame(mvo_t *s) { (void)s; return pixels; }
int mvo_undistort(const mvo_camera_t *c, const float in[2], float out[2])
{
    (void)c;
    out[0] = in[0];
    out[1] = in[1];
    return 1;
}
void mvo_start(mvo_t *s)
{
    s->count = 1;
    s->feature[0].id = 17;
    s->feature[0].has_point = 1;
    s->feature[0].point[0] = 1.0f; s->feature[0].point[1] = 2.0f; s->feature[0].point[2] = 3.0f;
    s->feature[0].spread = 0.5f;
}
void mvo_step(mvo_t *s, mvo_step_t *out)
{ (void)s; memset(out, 0, sizeof *out); out->kind = fake_kind; }
uint32_t mvo_state_size(void) { return sizeof(mvo_t); }
float mvo_sqrt(float v) { return v > 0.0f ? v : 0.0f; } /* the planner's ranges are not checked here */
void mm_init(mm_map_t *map) { memset(map, 0, sizeof *map); }
int mm_observe(const mvo_t *vo, const uint8_t *mask, int w, int h, float (*out)[3], int capacity)
{ (void)vo; (void)mask; (void)w; (void)h; (void)out; (void)capacity; ++observations; return 1; }
int mm_add(mm_map_t *map, const mvo_t *vo, const float (*points)[3], int count)
{
    (void)vo; (void)points;
    ++additions;
    ++map->odds[0]; map->cell = 1.25f; ++map->origin[0]; --map->origin[1];
    ++map->observations; ++map->frames; map->slice_y = 3.0f; ++map->height_resets;
    return count;
}
void mvo_pie_lk_template(mvo_window_t *w, const uint8_t *s, int stride, const int16_t q[4], int64_t g[3])
{ (void)w; (void)s; (void)stride; (void)q; memset(g, 0, 3 * sizeof *g); }
void mvo_pie_lk_sums(const mvo_window_t *w, const uint8_t *s, int stride, const int16_t q[4], int64_t sums[2])
{ (void)w; (void)s; (void)stride; (void)q; memset(sums, 0, 2 * sizeof *sums); }
void mvo_lk_template_c(mvo_window_t *w, const uint8_t *s, int stride, const int16_t q[4], int64_t g[3])
{ (void)w; (void)s; (void)stride; (void)q; memset(g, 0, 3 * sizeof *g); }
void mvo_lk_sums_c(const mvo_window_t *w, const uint8_t *s, int stride, const int16_t q[4], int64_t sums[2])
{ (void)w; (void)s; (void)stride; (void)q; memset(sums, 0, 2 * sizeof *sums); }
void mvo_pie_block(uint8_t *out, int size, const uint8_t *img, int w, int h, const int origin[2])
{ (void)out; (void)size; (void)img; (void)w; (void)h; (void)origin; }
void mvo_block_c(uint8_t *out, int size, const uint8_t *img, int w, int h, const int origin[2])
{ (void)out; (void)size; (void)img; (void)w; (void)h; (void)origin; }

static void submit(int index, int kind, int with_mask)
{
    char line[100];
    uint32_t image_hash, mask_hash;
    fake_kind = kind;
    input_pos = 0;
    memset(input, 0x5a, sizeof input);
    image_hash = fnv(2166136261u, input, FRAME_BYTES);
    mask_hash = fnv(2166136261u, input + FRAME_BYTES, MASK_BYTES);
    if (with_mask) snprintf(line, sizeof line, "frame %d %" PRIu32 " %" PRIu32, index, image_hash, mask_hash);
    else snprintf(line, sizeof line, "frame %d %" PRIu32, index, image_hash);
    frame(line);
    assert(input_pos == FRAME_BYTES + (with_mask ? MASK_BYTES : 0));
}

static void submit_bad_hash(int index)
{
    char line[100];
    uint32_t image_hash;
    input_pos = 0;
    memset(input, 0x5a, sizeof input);
    image_hash = fnv(2166136261u, input, FRAME_BYTES);
    snprintf(line, sizeof line, "frame %d %" PRIu32, index, image_hash + 1u);
    frame(line);
    assert(input_pos == FRAME_BYTES);
}

int main(void)
{
    static uint8_t pyramid[2][MVO_PYRAMID_BYTES];
    static mvo_t state;
    static mm_map_t map, frozen;
    static mc_map_t cloud_map, frozen_cloud;
    static vm_map_t voxel_map;
    uint32_t pyramid_hash[2];
    s_pyramid[0] = pyramid[0]; s_pyramid[1] = pyramid[1]; s_state = &state; s_map = &map; s_cloud = &cloud_map;
    s_voxels = &voxel_map;

    /* A cloud request or frame before start must not read heap contents. */
    memset(s_cloud, 0xa5, sizeof *s_cloud);
    memcpy(&frozen_cloud, s_cloud, sizeof frozen_cloud);
    input_pos = 0;
    frame("frame 0 0");
    assert(input_pos == FRAME_BYTES);
    input_pos = 0;
    frame("frame 0 0 0");
    assert(input_pos == FRAME_BYTES + MASK_BYTES);
    end();
    cloud();
    assert(!s_session_ready && memcmp(s_cloud, &frozen_cloud, sizeof frozen_cloud) == 0);

    start("start 1 2 3 4 5 1 0");
    assert(s_map_continuous && s_cloud->continuous && s_cloud->count == 0 && initializations == 1);
    pyramid_hash[0] = fnv(2166136261u, pyramid[0], sizeof pyramid[0]);
    selftest("selftest 1 1"); /* allowed after a fresh start and before the first frame */
    assert(pyramid_hash[0] != fnv(2166136261u, pyramid[0], sizeof pyramid[0]));
    submit(1, MVO_ESSENTIAL, 1); /* rejected, but its full payload must be consumed */
    assert(!s_started && observations == 0 && additions == 0 && s_cloud->count == 0);
    submit(0, MVO_ROTATION, 0);
    submit(0, MVO_ESSENTIAL, 1); /* duplicate */
    submit(2, MVO_ESSENTIAL, 0); /* gap */
    submit_bad_hash(1);          /* same index is valid for a retry */
    assert(observations == 0 && additions == 0 && s_cloud->count == 0);
    submit(1, MVO_ROTATION, 1);
    pyramid_hash[0] = fnv(2166136261u, pyramid[0], sizeof pyramid[0]);
    pyramid_hash[1] = fnv(2166136261u, pyramid[1], sizeof pyramid[1]);
    selftest("selftest 1 1"); /* the test uses pyramids as scratch, so it must not run mid-session */
    assert(pyramid_hash[0] == fnv(2166136261u, pyramid[0], sizeof pyramid[0]));
    assert(pyramid_hash[1] == fnv(2166136261u, pyramid[1], sizeof pyramid[1]));
    submit(2, MVO_ESSENTIAL, 0);
    assert(observations == 1 && additions == 2);
    assert(s_cloud->count == 1 && s_cloud->points[0].id == 17 && s_cloud->points[0].observations == 1);
    assert(s_cloud->points[0].xyz[0] == 1.0f && s_cloud->points[0].xyz[1] == 2.0f && s_cloud->points[0].xyz[2] == 3.0f);
    memcpy(&frozen, &map, sizeof map);
    submit(3, MVO_LOST, 0);
    assert(!s_cloud->continuous && s_cloud->count == 1);
    memcpy(&frozen_cloud, s_cloud, sizeof frozen_cloud);
    submit(4, MVO_LOST, 1);
    submit(5, MVO_ESSENTIAL, 1);
    assert(!s_map_continuous && observations == 1 && additions == 2);
    /* the voxel map goes on through the LOST steps: it counts them and keeps taking steps */
    assert(s_voxels->lost == 2 && s_voxels->frames == 5);
    assert(memcmp(&map, &frozen, sizeof map) == 0);
    assert(memcmp(s_cloud, &frozen_cloud, sizeof frozen_cloud) == 0);
    end();
    cloud();
    submit(6, MVO_ESSENTIAL, 1); /* end closes frame input while leaving cloud readable */
    assert(memcmp(s_cloud, &frozen_cloud, sizeof frozen_cloud) == 0);
    selftest("selftest 1 1"); /* allowed after end; next start reinitializes VO */

    start("start 1 2 3 4 5 1 0");
    assert(s_map_continuous && s_cloud->continuous && s_cloud->count == 0 && initializations == 2 && map.cell == 0.0f);
    memcpy(&frozen, &map, sizeof map);
    submit(0, MVO_ROTATION, 0);
    submit(1, MVO_LOST, 0);
    submit(2, MVO_ROTATION, 1);
    assert(!s_map_continuous && observations == 1 && additions == 2);
    assert(!s_cloud->continuous && s_cloud->count == 0);
    assert(memcmp(&map, &frozen, sizeof map) == 0);
    end();

    start("start 1 2 3 4 5 1 0");
    assert(s_map_continuous && s_cloud->continuous && s_cloud->count == 0 && initializations == 3);
    submit(0, MVO_ROTATION, 0);
    submit(1, MVO_ROTATION, 1);
    assert(s_map_continuous && observations == 2 && additions == 3 && map.cell == 1.25f);
    end();

    /* The lens rides after cores and simd: all five coefficients (as float bits) or none. */
    {
        const float lens[5] = {-0.25f, 0.06f, 0.001f, -0.0008f, -0.005f};
        uint32_t bits[5];
        char line[160];
        memcpy(bits, lens, sizeof bits);
        snprintf(line, sizeof line, "start 1 2 3 4 5 1 0 %lu %lu %lu %lu %lu", (unsigned long)bits[0],
                 (unsigned long)bits[1], (unsigned long)bits[2], (unsigned long)bits[3], (unsigned long)bits[4]);
        start(line);
        assert(initializations == 4 && last_camera.k1 == lens[0] && last_camera.k2 == lens[1]
               && last_camera.p1 == lens[2] && last_camera.p2 == lens[3] && last_camera.k3 == lens[4]);
        start("start 1 2 3 4 5 1 0 6 7"); /* a partial lens is refused */
        assert(initializations == 4);
        start("start 1 2 3 4 5 1 0");
        assert(initializations == 5 && last_camera.k1 == 0.0f && last_camera.k2 == 0.0f && last_camera.p1 == 0.0f
               && last_camera.p2 == 0.0f && last_camera.k3 == 0.0f);
    } /* no end: the host test counts the map and end lines above */
    return 0;
}
