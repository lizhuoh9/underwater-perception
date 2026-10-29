#include "water_seg_model.hpp"

#include <new>

#include "dl_tensor_base.hpp"
#include "esp_log.h"
#include "esp_timer.h"

extern const uint8_t water_seg_espdl[] asm("_binary_water_seg_espdl_start");

namespace {
namespace config = water_seg_config;
constexpr const char *TAG = "water_seg";
static_assert(config::kOutputWidth * 2 == config::kInputWidth && config::kOutputHeight * 2 == config::kInputHeight,
              "The exported network predicts at half the input resolution");
static_assert(config::kThresholdCode >= -128 && config::kThresholdCode <= 128, "Threshold is an int8 output code");

bool shape_is(const dl::TensorBase *tensor, int height, int width, int channels) {
    const auto &shape = tensor->shape;
    return shape.size() == 4 && shape[0] == 1 && shape[1] == height && shape[2] == width && shape[3] == channels;
}

// The exported activations are per-tensor; a per-channel exponent means another model.
bool exponent_is(const dl::ExponentInfo &exponent, int expected) {
    return !exponent.is_per_channel() && exponent.get() == expected;
}
}  // namespace

WaterSegModel::~WaterSegModel() {
    delete _model;
}

bool WaterSegModel::begin(const WaterSegOptions &options) {
    if (ready()) return true;
    _mode = options.multi_core ? dl::RUNTIME_MODE_MULTI_CORE : dl::RUNTIME_MODE_SINGLE_CORE;
    _model = new (std::nothrow) dl::Model(reinterpret_cast<const char *>(water_seg_espdl),
                                          fbs::MODEL_LOCATION_IN_FLASH_RODATA, options.max_internal_bytes);
    if (_model == nullptr) return false;
    _input = _model->get_input();
    _output = _model->get_output();
    // esp-dl only logs "root_alloc failed" when the planned feature-map pools do not fit.
    if (_input == nullptr || _output == nullptr || _input->data == nullptr || _output->data == nullptr ||
        _input->dtype != dl::DATA_TYPE_INT8 ||
        _output->dtype != dl::DATA_TYPE_INT8 || !exponent_is(_input->exponent, config::kInputExponent) ||
        !exponent_is(_output->exponent, config::kOutputExponent) ||
        !shape_is(_input, config::kInputHeight, config::kInputWidth, 3) ||
        !shape_is(_output, config::kOutputHeight, config::kOutputWidth, 1)) {
        ESP_LOGE(TAG, "Model tensors unallocated or not matching water_seg_model_config.h");
        delete _model;
        _model = nullptr;
        _input = _output = nullptr;
        return false;
    }
    // Same tables esp-dl's ImagePreprocessor builds (NormQuant), without pulling in its
    // ~800 KB of conversions for every other pixel format.
    for (int channel = 0; channel < 3; ++channel) {
        const float inv_std = 1.f / config::kStd[channel];
        for (int value = 0; value < 256; ++value) {
            _lut[channel][value] = dl::quantize<int8_t>((value - config::kMean[channel]) * inv_std,
                                                        DL_RESCALE(config::kInputExponent));
        }
    }
    return true;
}

bool WaterSegModel::ready() const {
    return _model != nullptr;
}

bool WaterSegModel::segment(const uint16_t *rgb565, int width, int height, bool big_endian, uint8_t *mask) {
    _timing = {};
    if (!ready() || rgb565 == nullptr || mask == nullptr || width != config::kInputWidth ||
        height != config::kInputHeight) {
        return false;
    }
    const int64_t start = esp_timer_get_time();
    // RGB565 widened by shift exactly as esp-dl does (R<<3, G<<2, B<<3), NHWC input.
    auto *pixels = static_cast<int8_t *>(_input->data);
    for (int i = 0; i < width * height; ++i) {
        const uint16_t value = big_endian ? __builtin_bswap16(rgb565[i]) : rgb565[i];
        pixels[3 * i] = _lut[0][(value & 0xF800) >> 8];
        pixels[3 * i + 1] = _lut[1][(value & 0x07E0) >> 3];
        pixels[3 * i + 2] = _lut[2][(value & 0x001F) << 3];
    }
    const int64_t preprocessed = esp_timer_get_time();
    _model->run(_mode);
    const int64_t inferred = esp_timer_get_time();
    const auto *codes = static_cast<const int8_t *>(_output->data);
    for (int i = 0; i < kMaskWidth * kMaskHeight; ++i) {
        mask[i] = codes[i] >= config::kThresholdCode;
    }
    _timing = {preprocessed - start, inferred - preprocessed, esp_timer_get_time() - inferred};
    return true;
}

bool WaterSegModel::self_test() {
    return ready() && _model->test() == ESP_OK;
}

void WaterSegModel::profile() {
    if (ready()) _model->profile();
}
