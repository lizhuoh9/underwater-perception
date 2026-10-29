/**********************************************************************
 * water_seg_demo main.cpp
 *
 * Board check and benchmark for the ESP-DL water segmenter without a camera:
 * runs ESP-DL's embedded self-test, then for each runtime configuration
 * (feature maps in PSRAM or internal RAM, one or two cores) segments one
 * stored calibration frame in both byte orders, compares the masks with the
 * Python int8 reference, and times preprocess + inference + threshold, also
 * separately, so the CPU left for other tasks can be planned.
 **********************************************************************/

#include <cinttypes>
#include <cstddef>

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "sdkconfig.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "water_seg_model.hpp"
#include "water_seg_test_frame.h"

namespace {
constexpr const char *TAG = "water_seg_demo";
constexpr int kWarmupRuns = 2;
constexpr int kTimedRuns = 20;
constexpr size_t kMaskCells = WaterSegModel::kMaskWidth * WaterSegModel::kMaskHeight;
constexpr size_t kFramePixels = water_seg_config::kInputWidth * water_seg_config::kInputHeight;
static_assert(sizeof(kTestFrameExpectedMask) == kMaskCells, "Test frame header belongs to another export");

struct Config {
    const char *name;
    WaterSegOptions options;
};

// Internal-RAM budgets are caps: esp-dl places what fits there and the rest in PSRAM.
constexpr Config kConfigs[] = {
    {"PSRAM maps, 1 core", {0, false}},
    {"PSRAM maps, 2 cores", {0, true}},
    {"<=128 KB internal, 1 core", {128 * 1024, false}},
    {"<=128 KB internal, 2 cores", {128 * 1024, true}},
    {"<=256 KB internal, 1 core", {256 * 1024, false}},
    {"<=256 KB internal, 2 cores", {256 * 1024, true}},
};
constexpr size_t kConfigCount = sizeof(kConfigs) / sizeof(kConfigs[0]);

uint8_t g_mask[kMaskCells];
uint16_t g_swapped[kFramePixels];

int count_differences(const uint8_t *mask) {
    int differ = 0;
    for (size_t i = 0; i < kMaskCells; ++i) {
        differ += mask[i] != kTestFrameExpectedMask[i];
    }
    return differ;
}

bool segment(WaterSegModel &model, const uint16_t *frame, bool big_endian) {
    return model.segment(frame, water_seg_config::kInputWidth, water_seg_config::kInputHeight, big_endian, g_mask);
}

struct Result {
    bool loaded = false;
    double ms = 0;
    double preprocess_ms = 0;
    double inference_ms = 0;
    double threshold_ms = 0;
    int differ_le = -1;
    int differ_be = -1;
    size_t internal_kb = 0;
    size_t psram_kb = 0;
};

Result measure(const Config &config, bool profile) {
    Result result;
    const size_t internal_before = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    const size_t psram_before = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    WaterSegModel model;
    if (!model.begin(config.options)) {
        return result;
    }
    result.internal_kb = (internal_before - heap_caps_get_free_size(MALLOC_CAP_INTERNAL)) / 1024;
    result.psram_kb = (psram_before - heap_caps_get_free_size(MALLOC_CAP_SPIRAM)) / 1024;
    // The timed runs repeat these arguments, so they cannot fail once these succeed.
    if (!segment(model, kTestFrameRgb565, false)) {
        return result;
    }
    result.differ_le = count_differences(g_mask);
    if (!segment(model, g_swapped, true)) {
        return result;
    }
    result.differ_be = count_differences(g_mask);
    result.loaded = true;
    for (int run = 0; run < kWarmupRuns; ++run) {
        segment(model, kTestFrameRgb565, false);
    }
    WaterSegTiming parts;
    const int64_t start = esp_timer_get_time();
    for (int run = 0; run < kTimedRuns; ++run) {
        segment(model, kTestFrameRgb565, false);
        const WaterSegTiming &last = model.last_timing();
        parts.preprocess_us += last.preprocess_us;
        parts.inference_us += last.inference_us;
        parts.threshold_us += last.threshold_us;
    }
    result.ms = (esp_timer_get_time() - start) / 1000.0 / kTimedRuns;
    result.preprocess_ms = parts.preprocess_us / 1000.0 / kTimedRuns;
    result.inference_ms = parts.inference_us / 1000.0 / kTimedRuns;
    result.threshold_ms = parts.threshold_us / 1000.0 / kTimedRuns;
    if (profile) {
        model.profile();
    }
    return result;
}
}  // namespace

static void run_once(void) {
    ESP_LOGI(TAG, "CPU %d MHz, PSRAM %u KB, internal heap free %u KB", CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ,
             static_cast<unsigned>(heap_caps_get_total_size(MALLOC_CAP_SPIRAM) / 1024),
             static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024));
    // Tells builds of different exports apart in a saved log.
    ESP_LOGI(TAG, "Model config: threshold code %d, input/output exponent %d/%d, self-test frame %s",
             water_seg_config::kThresholdCode, water_seg_config::kInputExponent, water_seg_config::kOutputExponent,
             kTestFrameId);
    {
        WaterSegModel model;
        if (!model.begin()) {
            ESP_LOGE(TAG, "Model load failed");
            return;
        }
        ESP_LOGI(TAG, "ESP-DL self-test against embedded ESP-PPQ values: %s", model.self_test() ? "pass" : "FAIL");
    }
    for (size_t i = 0; i < kFramePixels; ++i) {
        g_swapped[i] = __builtin_bswap16(kTestFrameRgb565[i]);  // camera (big-endian) byte order
    }

    Result results[kConfigCount];
    size_t fastest = 0;
    for (size_t i = 0; i < kConfigCount; ++i) {
        results[i] = measure(kConfigs[i], false);
        if (results[i].loaded && (!results[fastest].loaded || results[i].ms < results[fastest].ms)) {
            fastest = i;
        }
    }
    ESP_LOGI(TAG, "Frame %s, %d timed runs each (preprocess + inference + threshold):", kTestFrameId, kTimedRuns);
    for (size_t i = 0; i < kConfigCount; ++i) {
        const Result &r = results[i];
        if (!r.loaded) {
            ESP_LOGW(TAG, "  %-28s  could not allocate its feature maps (or segment the frame)", kConfigs[i].name);
            continue;
        }
        ESP_LOGI(TAG,
                 "  %-28s %7.1f ms %5.1f FPS (pre %.1f + infer %.1f + mask %.1f ms) | mask diff LE %d BE %d"
                 " | model RAM internal %u KB, PSRAM %u KB",
                 kConfigs[i].name, r.ms, 1000.0 / r.ms, r.preprocess_ms, r.inference_ms, r.threshold_ms, r.differ_le,
                 r.differ_be, static_cast<unsigned>(r.internal_kb), static_cast<unsigned>(r.psram_kb));
    }
    if (results[fastest].loaded) {
        ESP_LOGI(TAG, "Per-layer profile of the fastest configuration (%s):", kConfigs[fastest].name);
        measure(kConfigs[fastest], true);
    }
}

// The USB-Serial-JTAG port re-enumerates on reset: wait 3 s so a monitor can reopen it, then repeat the
// whole run every 5 s.
extern "C" void app_main(void) {
    vTaskDelay(pdMS_TO_TICKS(3000));
    for (int round = 1;; ++round) {
        ESP_LOGI(TAG, "=== round %d ===", round);
        run_once();
        vTaskDelay(pdMS_TO_TICKS(5000));
    }
}
