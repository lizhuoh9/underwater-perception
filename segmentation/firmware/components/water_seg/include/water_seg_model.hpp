/**********************************************************************
 * water_seg_model.hpp
 *
 * ESP-DL water/non-water segmentation of one 160x120 RGB565 camera frame.
 * The model, its normalisation and its threshold come from one export
 * (uwseg.export_p4, or a training run of uwseg.train_cnn_lite or
 * train_cnn_espdl). begin() checks the model's tensor types, shapes and
 * exponents against the header, which cannot tell apart models sharing those
 * values (the dense and the lite CNN do): keep one export's files together;
 * the build prints the embedded model's SHA-256. Output cells are 1 for
 * non-water (sea floor, rocks, fish, divers, wrecks, ...) and 0 for water;
 * they are not collision distances.
 **********************************************************************/

#ifndef WATER_SEG_MODEL_HPP
#define WATER_SEG_MODEL_HPP

#include <cstdint>

#include "dl_model_base.hpp"
#include "water_seg_model_config.h"

struct WaterSegOptions {
    // Internal RAM, in bytes, esp-dl may use for feature maps before PSRAM. 0 keeps esp-dl's
    // default, which places every feature map in PSRAM whenever PSRAM is enabled.
    int max_internal_bytes = 0;
    // Split each layer across both cores (esp-dl RUNTIME_MODE_MULTI_CORE).
    bool multi_core = false;
};

// Wall-clock split of the last segment() call, in microseconds (all zero if it failed).
// Not synchronised: read it from the task that calls segment().
struct WaterSegTiming {
    int64_t preprocess_us = 0;  // RGB565 -> int8 input codes
    int64_t inference_us = 0;   // esp-dl Model::run
    int64_t threshold_us = 0;   // int8 output codes -> mask
};

class WaterSegModel {
public:
    static constexpr int kMaskWidth = water_seg_config::kOutputWidth;
    static constexpr int kMaskHeight = water_seg_config::kOutputHeight;

    WaterSegModel() = default;
    ~WaterSegModel();
    WaterSegModel(const WaterSegModel &) = delete;
    WaterSegModel &operator=(const WaterSegModel &) = delete;

    // Loads the embedded model and rejects it unless tensor types, shapes and
    // exponents match water_seg_model_config.h and its feature maps were allocated.
    bool begin(const WaterSegOptions &options = WaterSegOptions());
    bool ready() const;

    // One frame of exactly kInputWidth x kInputHeight RGB565 pixels. esp32-camera
    // delivers big-endian byte order; check with a colour target before relying on it.
    // mask receives kMaskWidth * kMaskHeight bytes, row-major.
    bool segment(const uint16_t *rgb565, int width, int height, bool big_endian, uint8_t *mask);
    const WaterSegTiming &last_timing() const { return _timing; }

    // Runs ESP-DL's comparison against the ESP-PPQ test values embedded in the model.
    bool self_test();
    // Per-module latency and memory report from ESP-DL.
    void profile();

private:
    dl::Model *_model = nullptr;
    dl::TensorBase *_input = nullptr;
    dl::TensorBase *_output = nullptr;
    dl::runtime_mode_t _mode = dl::RUNTIME_MODE_SINGLE_CORE;
    // esp-dl NormQuant tables: device-unit channel value -> int8 input code.
    int8_t _lut[3][256] = {};
    WaterSegTiming _timing;
};

#endif  // WATER_SEG_MODEL_HPP
