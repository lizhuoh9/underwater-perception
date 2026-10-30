/*
 * live.c - the live build's camera input (CONFIG_SPIRAM, sdkconfig.live): a camera's compressed frames, each decoded and
 * averaged here, on the board. The command loop takes a frame in while the worker task works on an earlier one; the
 * worker always takes the newest, and a frame still waiting when a newer one comes in is dropped, as a camera's would
 * be. The slots and the decoded image live in PSRAM. See main.c for the commands.
 */
#include "stream.h"

#include "sdkconfig.h"

#if CONFIG_SPIRAM
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#if CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG
#include "driver/usb_serial_jtag.h"
#else
#include "driver/uart.h"
#endif
#include "driver/jpeg_decode.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"

#include "frame_prep.h"
#include "jpeg_check.h"

#define JPEG_MAX_BYTES (512 * 1024) /* the most bytes one compressed frame may have */
#define JPEG_SLOTS 3                /* one coming in, one waiting, one being worked on */
#define DRAIN_BYTES 4096            /* the bytes of a refused frame are read through this many at a time */
#define WORKER_STACK 16384          /* the odometry's, as the replay build's command loop (CONFIG_ESP_MAIN_TASK_STACK_SIZE) */
#define WORKER_PRIORITY 3           /* the step's task, on the first core */
/* The command loop, on the first core too, above the worker: the link's driver drops what its receive buffer cannot
 * hold, so a frame's bytes must be taken in as they come. (On the second core above the helper, the step took as
 * long: tracking, split across both cores, then waited instead.) */
#define LOOP_PRIORITY (WORKER_PRIORITY + 1)
#define YUV444_BYTES 3
#define YUV444_Y 2         /* the luminance's byte in each pixel the JPEG engine writes */
#define DECODED_ALIGN 128u /* the decoded size given to the driver: whole cache lines (64 bytes on the P4) */

typedef struct {
    uint8_t *data;
    uint32_t bytes;
    int index;
    jpeg_shape_t shape;
    int64_t arrived; /* esp_timer, when its last byte came in */
} jpeg_slot_t;

typedef struct {
    const uint8_t *luma; /* the first pixel's Y */
    int width, height, stride;
    uint8_t *frame;
    int done[2]; /* each half's fp_area_resize result */
} prep_job_t;

static jpeg_slot_t s_slots[JPEG_SLOTS];
static uint8_t *s_decoded; /* the JPEG engine's YUV444, rows and columns padded to whole MCUs */
static size_t s_decoded_bytes;
static uint8_t *s_drain;
static jpeg_decoder_handle_t s_jpeg;
static SemaphoreHandle_t s_live_lock;
static TaskHandle_t s_worker;
static int s_waiting = -1, s_working = -1; /* slots: the newest frame not yet taken, the one being worked on */
static int64_t s_next_jpeg;                /* the lowest index the next frame may have */
static uint32_t s_received, s_dropped;

#if CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG
#define LINK_WRITE_BYTES 1024 /* the driver refuses a write larger than its transmit buffer (main.c, link_open) */

static void link_write(const void *data, size_t size)
{
    const uint8_t *bytes = data;
    while (size > 0) {
        const size_t part = size < LINK_WRITE_BYTES ? size : LINK_WRITE_BYTES;
        if (usb_serial_jtag_write_bytes(bytes, part, pdMS_TO_TICKS(CHUNK_WAIT_MS)) <= 0) {
            return;
        }
        bytes += part;
        size -= part;
    }
}
#else
static void link_write(const void *data, size_t size)
{
    uart_write_bytes(CONFIG_ESP_CONSOLE_UART_NUM, data, size);
}
#endif

/* Half of the frame's rows on each core. */
static void prep_part(void *context, int part)
{
    prep_job_t *job = context;
    job->done[part] = fp_area_resize(job->luma, job->width, job->height, job->stride, YUV444_BYTES, job->frame,
                                     part ? FP_HEIGHT / 2 : 0, part ? FP_HEIGHT : FP_HEIGHT / 2);
}

/* The frame in the slot: YUV444 from the JPEG engine, its luminance averaged onto mvo_frame(), then the step. YUV444
 * keeps the JPEG's own luminance: the engine's RGB takes it as limited-range (BT.601's 298/256 gain), and asked for
 * grey from a colour JPEG it trips its 2D-DMA's receive check (IDF 5.3.2). */
static void work_on(const jpeg_slot_t *slot)
{
    const jpeg_decode_cfg_t config = {.output_format = JPEG_DECODE_OUT_FORMAT_YUV444};
    const jpeg_shape_t *shape = &slot->shape;
    const uint32_t stride = (shape->width + shape->mcu_width - 1u) / shape->mcu_width * shape->mcu_width * YUV444_BYTES;
    const uint32_t rows = (shape->height + shape->mcu_height - 1u) / shape->mcu_height * shape->mcu_height;
    /* The decoded image's own bytes only: the driver invalidates the cache over all the size it is given. */
    const uint32_t decoded_bytes = (stride * rows + DECODED_ALIGN - 1u) / DECODED_ALIGN * DECODED_ALIGN;
    const int64_t begin = esp_timer_get_time();
    mvo_t *state = stream_state();
    prep_job_t job = {s_decoded + YUV444_Y, (int)shape->width, (int)shape->height, (int)stride, mvo_frame(state), {0, 0}};
    uint32_t out_bytes = 0, dropped;
    int64_t decoded, prepared;
    char extra[200];
    if (jpeg_decoder_process(s_jpeg, &config, slot->data, slot->bytes, s_decoded, decoded_bytes, &out_bytes)
        != ESP_OK) {
        printf("error jpeg %d did not decode\n", slot->index);
        return;
    }
    decoded = esp_timer_get_time();
    if (state->parallel) {
        run_both(prep_part, &job);
    } else {
        prep_part(&job, 0);
        prep_part(&job, 1);
    }
    prepared = esp_timer_get_time();
    if (!job.done[0] || !job.done[1]) {
        printf("error jpeg %d was not averaged\n", slot->index);
        return;
    }
    xSemaphoreTake(s_live_lock, portMAX_DELAY);
    dropped = s_dropped;
    xSemaphoreGive(s_live_lock);
    snprintf(extra, sizeof extra,
             " source=%" PRIu32 "x%" PRIu32 " jpeg_bytes=%" PRIu32 " wait_us=%" PRId64 " decode_us=%" PRId64
             " prep_us=%" PRId64 " grey_hash=%" PRIu32 " dropped=%" PRIu32,
             shape->width, shape->height, slot->bytes, begin - slot->arrived, decoded - begin, prepared - decoded,
             fnv(2166136261u, job.frame, FRAME_BYTES), dropped);
    process_frame(slot->index, extra);
    fflush(stdout);
}

static void worker_task(void *unused)
{
    (void)unused;
    for (;;) {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        for (;;) {
            int slot;
            xSemaphoreTake(s_live_lock, portMAX_DELAY);
            slot = s_waiting;
            s_waiting = -1;
            s_working = slot;
            xSemaphoreGive(s_live_lock);
            if (slot < 0) {
                break;
            }
            work_on(&s_slots[slot]);
            xSemaphoreTake(s_live_lock, portMAX_DELAY);
            s_working = -1;
            xSemaphoreGive(s_live_lock);
        }
    }
}

/* Returns once no frame waits and none is being worked on. */
void live_wait_idle(void)
{
    for (;;) {
        int busy;
        xSemaphoreTake(s_live_lock, portMAX_DELAY);
        busy = s_waiting >= 0 || s_working >= 0;
        xSemaphoreGive(s_live_lock);
        if (!busy) {
            return;
        }
        vTaskDelay(1); /* a tick: no delay would yield only to this loop's priority, never to the worker below it */
    }
}

/* A refused frame's bytes, read through a buffer the worker never touches, to keep the link's framing. */
static void drain(uint32_t bytes)
{
    while (bytes > 0) {
        const uint32_t part = bytes < DRAIN_BYTES ? bytes : DRAIN_BYTES;
        if (!read_exact(s_drain, (int)part)) {
            return;
        }
        bytes -= part;
    }
}

void live_jpeg(const char *line)
{
    jpeg_shape_t shape;
    uint32_t bytes, expected;
    int index, slot, replaced = -1;
    if (sscanf(line, "jpeg %d %" SCNu32 " %" SCNu32, &index, &bytes, &expected) != 3) {
        printf("error bad jpeg line\n"); /* its bytes cannot be told apart; the next header starts a fresh line */
        return;
    }
    if (index < 0 || bytes == 0 || bytes > JPEG_MAX_BYTES) {
        drain(bytes);
        printf("error jpeg %d refused: index below 0 or not 1 to %d bytes\n", index, JPEG_MAX_BYTES);
        return;
    }
    xSemaphoreTake(s_live_lock, portMAX_DELAY);
    for (slot = 0; slot == s_waiting || slot == s_working; ++slot) {
    }
    xSemaphoreGive(s_live_lock);
    /* The worker only ever takes the waiting slot, so this one stays the loop's while its bytes come in. */
    if (!read_exact(s_slots[slot].data, (int)bytes)) {
        printf("error jpeg %d arrived incomplete\n", index);
        return;
    }
    if (!stream_open()) {
        printf("error start required\n");
        return;
    }
    if (fnv(2166136261u, s_slots[slot].data, bytes) != expected) {
        printf("error jpeg %d hash mismatch\n", index);
        return;
    }
    if ((int64_t)index < s_next_jpeg) {
        printf("error jpeg %d out of sequence, expected %" PRId64 " or later\n", index, s_next_jpeg);
        return;
    }
    s_next_jpeg = (int64_t)index + 1; /* a frame refused or never sent leaves a gap, as a camera's skipped frame does */
    if (!jpeg_check(s_slots[slot].data, bytes, FP_WIDTH, FP_HEIGHT, FP_MAX_SIDE, &shape)) {
        printf("error jpeg %d is not a baseline colour JPEG of %dx%d to %dx%d\n", index, FP_WIDTH, FP_HEIGHT,
               FP_MAX_SIDE, FP_MAX_SIDE);
        return;
    }
    ++s_received;
    s_slots[slot].bytes = bytes;
    s_slots[slot].index = index;
    s_slots[slot].shape = shape;
    s_slots[slot].arrived = esp_timer_get_time();
    xSemaphoreTake(s_live_lock, portMAX_DELAY);
    if (s_waiting >= 0) {
        replaced = s_slots[s_waiting].index;
        ++s_dropped;
    }
    s_waiting = slot;
    xSemaphoreGive(s_live_lock);
    if (replaced >= 0) {
        printf("dropped index=%d\n", replaced);
    }
    xTaskNotifyGive(s_worker);
}

/* The last frame the odometry saw. */
void live_grey(void)
{
    const uint8_t *pixels;
    if (!stream_open() || !stream_started()) {
        printf("error grey needs a started run\n");
        return;
    }
    pixels = stream_last_frame();
    printf("grey bytes=%d hash=%" PRIu32 "\n", FRAME_BYTES, fnv(2166136261u, pixels, FRAME_BYTES));
    fflush(stdout);
    link_write(pixels, FRAME_BYTES);
}

void live_start(void)
{
    s_next_jpeg = 0;
    s_received = 0;
    s_dropped = 0;
}

void live_report(void)
{
    printf("live received=%" PRIu32 " dropped=%" PRIu32 " psram_free=%u worker_stack_free_bytes=%u\n", s_received,
           s_dropped, (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
           (unsigned)uxTaskGetStackHighWaterMark(s_worker));
}

int live_init(void)
{
    const jpeg_decode_memory_alloc_cfg_t input = {.buffer_direction = JPEG_DEC_ALLOC_INPUT_BUFFER};
    const jpeg_decode_memory_alloc_cfg_t output = {.buffer_direction = JPEG_DEC_ALLOC_OUTPUT_BUFFER};
    const jpeg_decode_engine_cfg_t engine = {.intr_priority = 0, .timeout_ms = 1000};
    size_t size;
    int k;
    for (k = 0; k < JPEG_SLOTS; ++k) {
        s_slots[k].data = jpeg_alloc_decoder_mem(JPEG_MAX_BYTES, &input, &size);
        if (!s_slots[k].data) {
            return 0;
        }
    }
    s_decoded = jpeg_alloc_decoder_mem((size_t)FP_MAX_SIDE * FP_MAX_SIDE * YUV444_BYTES, &output, &s_decoded_bytes);
    s_drain = heap_caps_malloc(DRAIN_BYTES, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    s_live_lock = xSemaphoreCreateMutex();
    if (!s_decoded || !s_drain || !s_live_lock || jpeg_new_decoder_engine(&engine, &s_jpeg) != ESP_OK) {
        return 0;
    }
    vTaskPrioritySet(NULL, LOOP_PRIORITY);
    return xTaskCreatePinnedToCore(worker_task, "slam_worker", WORKER_STACK, NULL, WORKER_PRIORITY, &s_worker, 0)
           == pdPASS;
}
#endif
