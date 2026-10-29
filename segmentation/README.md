# uwseg

Water / non-water pixel segmentation for a small underwater robot's OV2640 camera. A small int8 CNN turns one
160×120 RGB565 frame into an 80×60 mask: 1 where the camera sees something that is not water (sea floor, rocks, reef,
fish, divers, wrecks, robots), 0 where it sees water. ESP-DL runs it on an ESP32-P4. The model was chosen and scored on
SUIM underwater photographs against two baselines kept here as well: an underwater dark-channel prior (UDCP) and an
integer pixel forest. The mask says what is water, not how far away an obstacle is.

[中文说明](README.zh-CN.md)

```text
camera frame, 160×120 RGB565
└── firmware/components/water_seg   ESP-DL int8 CNN on the ESP32-P4 (model files: results/esp32p4_export)
    └── 80×60 mask, row-major: 1 = non-water, 0 = water
```

## Key results

All accuracy numbers are pixel rates on SUIM TEST photographs, scored at the photographs' native resolution and
computed on the PC. Non-water is SUIM classes 1–7, water class 0. The miss rate (FNR) is the share of non-water pixels
called water, the false-alarm rate (FPR) the share of water pixels called non-water, mIoU the mean of the two classes'
IoU over all pixels. The 110 TEST images were reused across the research rounds; a near-duplicate audit found 12 of them
to be copies or adjacent frames of calibration or training images and 24 more to be other shots of the same scene, so
the strictest subset keeps 74 images.

**The deployed model** is the lite CNN of [results/cnn_espdl_run_02](results/cnn_espdl_run_02): 22.3 MMAC per frame,
39,569 parameters and 16 convolution layers, the dense CNN of `cnn_espdl_run_01` (69.7 MMAC) with its stride-1 3×3
convolutions made depthwise-separable. Its architecture was chosen on the 80 calibration images only. The rule for its
deployment threshold (miss rate at most 5% on calibration) was registered before training; applied after training it
gave int8 code +3, locked before TEST was opened, and TEST was scored once. On the strictest subset
([report](results/cnn_espdl_run_02/evaluation/RESULTS.md), in Chinese):

| Method | Miss (FNR) | False alarm (FPR) | mIoU |
|---|---:|---:|---:|
| Lite CNN int8, deployment threshold +3 | 6.70% | 8.34% | 0.8588 |
| Lite CNN float reference, deployment threshold | 7.84% | 7.56% | 0.8530 |
| Lite CNN int8, comparison threshold +10 (the forest's miss cap, 7.88%) | 11.87% | 5.05% | 0.8308 |
| Dense CNN int8, threshold +5 (chosen after its TEST score was seen) | 6.92% | 7.29% | 0.8644 |
| Dense CNN int8, locked threshold +11 | 11.12% | 4.56% | 0.8417 |
| Pixel forest, threshold 71 (the recommended forest) | 7.57% | 24.84% | 0.7332 |
| Pixel forest, threshold 57 | 5.92% | 31.43% | 0.7025 |
| UDCP with a per-frame Otsu threshold, RGB565 input | 31.23% | 33.41% | 0.5086 |

- Against the dense CNN at the same 5% cap the lite CNN misses 0.22 percentage points fewer pixels (95% bootstrap
  interval −0.83 to +0.45) and raises 1.06 points more false alarms (+0.03 to +2.21). Against the recommended forest
  it raises 16.50 points fewer false alarms (−21.72 to −12.10). On all 110 images it misses 5.83% and false-alarms
  7.58% (mIoU 0.8739).
- The int8 rows are ESP-PPQ's simulation of the graph as it was quantized for the ESP32-S3, the target at the time.
  [CAVEATS.md](results/cnn_espdl_run_02/CAVEATS.md) limits how far these numbers go: the 5% cap was set after the
  dense CNN's TEST score was seen, the locked lite weights miss the architecture rule's margin on calibration, and the
  per-class rates rest on few images (robots appear in 5).
- On the ESP32-P4 the same float model is quantized again ([results/esp32p4_export](results/esp32p4_export)) and keeps
  the deployment threshold code +3 at output exponent −3, which was chosen on the S3 graph's calibration curve. **The
  P4 int8 graph's accuracy on TEST has not been measured**; the board has been checked on one self-test frame only.

**The baselines.** On all 110 images the integer pixel forest (12 trees of depth 6, 16 RGB565 colour and texture
features per pixel) misses 5.93% instead of UDCP-Otsu's 29.37% on the same RGB565 input, and false-alarms 27.44%
instead of 31.76% ([report](results/forest_comparison/evaluation/RESULTS.md)). Raising its threshold from 57 to 71, a control frozen on
calibration before TEST was scored, lowers false alarms to 21.52% and raises misses to 7.74%
([report](results/forest_lowfp/evaluation/RESULTS.md); [leak-free re-score](results/evaluation_dedup/RESULTS.md)).
On the 56 calibration images without training neighbours the forest's threshold rules select 57, 71 and 72 again
([recheck](results/calibration_recheck/RESULTS.md)).

**On the board.** The firmware's demo ([firmware/water_seg_demo](firmware/water_seg_demo)) runs ESP-DL's self-test,
then segments one stored calibration frame 20 times in each runtime configuration. Measured on an ESP32-P4 at 360 MHz
with the deployed model ([log](results/esp32p4_export/board_timing.log)), per frame (preprocess + inference +
threshold):

| Feature maps | 1 core | 2 cores | Model RAM (internal + PSRAM) |
|---|---:|---:|---:|
| PSRAM | 47.0 ms | 43.8 ms | 8 KB + 312 KB |
| Up to 128 KB internal RAM, rest PSRAM | 20.4 ms | 15.1 ms | 136 KB + 125 KB |
| Up to 256 KB internal RAM, rest PSRAM | 15.5 ms | 10.5 ms | 264 KB + 50 KB |

ESP-DL's self-test passed, and in every configuration the board's mask for the stored frame equals the PC's int8
reference in both byte orders. In the fastest configuration the 10.5 ms are 1.4 ms of preprocessing, 9.0 ms of
inference and 0.1 ms of thresholding (95 frames per second).

Historically, on an ESP32-S3 at 240 MHz (feature maps up to 128 KB in internal RAM, two cores), the same model as
quantized for the S3 segmented all 110 TEST images streamed from the PC in 27.46 ms per frame on average, and every
board mask equalled the PC's int8 reference ([report](results/board_eval_run02/RESULTS.md), in Chinese). The S3
firmware and its streaming tools are not part of this repository.

## Repository layout

```text
.
├── src/uwseg/                  Python package (python -m uwseg.<module>)
│   ├── paths.py                where the data, sources and results live
│   ├── prepare_data.py, prepare_forest_training.py, prepare_lowfp_training.py
│   │                           the pinned SUIM archive, its splits and their manifests
│   ├── near_duplicates.py, prepare_dedup_split.py
│   │                           the decoded-pixel near-duplicate audit and the leak-free TEST subsets
│   ├── udcp.py, validate.py    UDCP baseline, metrics, calibration and its evaluation
│   ├── pixel_forest.py, forest_native.py, train_pixel_forest.py, optimize_pixel_forest.py
│   │                           integer pixel forest: features, training, low false-positive round, C bridge
│   ├── compare_pixel_forest.py, compare_lowfp.py, verify_lowfp_control.py, compare_dedup.py, recheck_calibration.py
│   │                           the forest's locked evaluations
│   ├── cnn_seg.py, cnn_lite.py the ESP-DL CNNs: input path, dense and lite architectures
│   ├── train_cnn_espdl.py, train_cnn_lite.py
│   │                           training, ESP-PPQ quantization for the ESP32-P4, calibration lock
│   ├── compare_cnn_espdl.py, compare_cnn_lite.py, export_cnn_threshold_variant.py
│   │                           the CNNs' locked TEST evaluations
│   └── export_p4.py            a locked run quantized for the ESP32-P4: the files the firmware embeds
├── tests/                      unittest suite
├── embedded/                   the pixel forest in C99 (the baseline's MCU implementation; see its README)
├── firmware/
│   ├── components/water_seg/   ESP-IDF component: the model embedded in flash, RGB565 in, mask out
│   └── water_seg_demo/         ESP32-P4 app: self-test and timing, no camera
├── data/                       the dataset goes here (not committed) beside the committed provenance notes,
│                               visual review record and frozen near-duplicate audit
└── results/                    committed evidence, one folder per result set (see Results below)
```

## Setup

Python 3.10 or newer (3.10–3.12 for the CNN tools, which ESP-PPQ limits). Run everything from this folder, the
project root that holds this README:

```bash
pip install -e .                 # numpy, Pillow: data preparation, UDCP, the forest's inference
pip install -e ".[cnn]"          # torch, esp-ppq, onnx: the CNN's training, evaluation and export
pip install -e ".[forest]"       # scikit-learn: training the pixel forest
python -m unittest discover -s tests -t .
```

The `cnn` extra pins the versions the committed runs used. Those runs used the CUDA 12.4 build of torch 2.6.0 (from
`https://download.pytorch.org/whl/cu124`); training uses CUDA when it is available. The forest tools also look for a
local package folder `.deps/` in the project root.

With the `cnn` extra installed, six tests skip in a fresh checkout: three need the forest's host library (see
[embedded/README.md](embedded/README.md)) and three need the prepared data. Without torch the CNN tests skip too.

## Data

The data are SUIM underwater photographs with semantic masks, from a pinned third-party mirror (provenance, split
rules and the mask encoding: [data/SOURCES.md](data/SOURCES.md)). The photographs are not part of this repository.

```bash
python -m uwseg.prepare_data --download     # 174 MB archive, SHA-256 checked; 80 calibration and 110 TEST pairs
python -m uwseg.prepare_forest_training     # the first 512 training pairs
python -m uwseg.prepare_lowfp_training      # all 1,395 training pairs
```

Each step checks the archive and the earlier manifests by SHA-256 and refuses to overwrite a file whose content
differs. The CNN runs, the leak-free evaluation and the calibration recheck use the near-duplicate audit
`data/dedup_manifest.json`, committed with the visual review it was built from (`data/near_duplicate_review.json`).
The audit pins the three manifests above by SHA-256, and every committed CNN run pins the audit. It also records the
SHA-256 of the screening code that made it: the research scripts, before they became this package. To check that this
package's screening reproduces it, write the recomputed audit to a new file:

```bash
python -m uwseg.prepare_dedup_split --output data/dedup_manifest_recomputed.json
```

It should differ from the committed audit only in `code_sha256`. `compare_dedup` and `recheck_calibration` re-verify
that field against the package and refuse the committed audit; give them the recomputed one.

## Training

```bash
python -m uwseg.train_cnn_lite --architecture lite --exploration results/cnn_espdl_run_02/exploration --output <fresh folder>
python -m uwseg.train_cnn_espdl --output <fresh folder>                 # the dense CNN
python -m uwseg.train_pixel_forest --output <fresh folder>              # the first forest (needs .[forest])
python -m uwseg.optimize_pixel_forest --output <fresh folder>           # the low false-positive round
```

A CNN run writes `experiment_plan.json` before training, then the PyTorch and ONNX models, the ESP-PPQ export for the
ESP32-P4 (`water_seg.espdl`, the native graph `water_seg_native`), the firmware header, `water_seg.compile_req.yml`,
a self-test frame, the calibration curves and `LOCKED.json`; no TEST image is opened. The lite run selects two
thresholds on calibration: the deployment one (miss cap 5%) and the comparison one (the forest's cap). The committed
runs were quantized for the ESP32-S3; a new run quantizes for the ESP32-P4 and selects its thresholds on the P4
graph, so it does not reproduce the committed int8 numbers. Its input quantization rounds half to even, as esp-dl does
on the P4 (half up on the S3). Training is seeded, but the committed weights came from one GPU and may not be
reproduced bit for bit on other hardware.

## Evaluation

```bash
python -m uwseg.compare_cnn_lite --run <run folder> --output <run folder>/evaluation
python -m uwseg.compare_cnn_espdl --run <run folder> --output <run folder>/evaluation    # a dense run
python -m uwseg.validate --output <fresh folder>                                          # UDCP
```

`compare_cnn_lite` first verifies the run's lock (artifacts, sources, inputs, exploration record, both thresholds
against the curves, the header), replays the calibration images through the reloaded int8 graph, and checks the
baselines' per-image counts against their summaries. It scores the run beside the dense CNN's recorded results
(`results/cnn_espdl_run_01`) and the forest and UDCP results of `results/evaluation_dedup`. Every tool refuses to
overwrite its output.

The committed runs were made by the research scripts before they became this package, and their locks record those
scripts' SHA-256, so verifying them with the package reports changed sources. The CNN runs' ESP32-S3 exports are not
committed either. Re-running a committed evaluation therefore means a new run made with the package. The forest's
evaluations also record the SHA-256 of the compiled C library they ran; `compare_dedup` reproduces only with those
binaries. Scored again with the package from the same data, the UDCP evaluation and the calibration recheck gave the
committed results, and the recomputed audit equalled the committed one apart from its code hashes.

## Exporting for the ESP32-P4

```bash
python -m uwseg.export_p4 --kernels firmware/water_seg_demo/managed_components/espressif__esp-dl/spec/kernels.yml \
    --output <fresh folder> [--run results/cnn_espdl_run_02]
```

The export quantizes the run's ONNX model with ESP-PPQ for target `esp32p4` (the 128 calibration inputs training
uses), and writes `water_seg.espdl`, `water_seg_model_config.h`, `water_seg.compile_req.yml` (the graph's operators
and every int8 P4 convolution kernel in esp-dl's `spec/kernels.yml`), `water_seg_test_frame.h` (the first calibration
frame and the mask the P4 graph gives it), `export.json`, and ESP-PPQ's dumps `water_seg.info` and `water_seg.json`.
It needs the prepared data and the esp-dl spec file, which ESP-IDF downloads into `managed_components/` on the demo's
first build. The threshold stays the run's locked code; it is not selected again on the P4 graph.

Repeated from the same inputs, the export gave the same quantized weights and exponents, header and self-test frame as
`results/esp32p4_export`; the `.espdl` file itself differs in ESP-PPQ's test values, which it draws at random, and in
the order of its tensors. The committed export was written
by the export script's research version: its `export.json` calls the threshold `s3_threshold` and its
`compile_req.yml` comment names `run_02`, where the module writes `locked_threshold` and the run folder's name.

## Firmware

`firmware/components/water_seg` embeds `water_seg.espdl` in flash and registers its `compile_req.yml`, so esp-dl keeps
the operators and kernels the model needs. `WaterSegModel::begin()` checks the model's tensor types, shapes and
exponents against the header; `segment()` converts RGB565 to int8 input codes with three 256-entry tables (esp-dl's
NormQuant, without the conversions for every other pixel format that its `ImagePreprocessor` would link) and
thresholds the output. esp-dl 3.3.12 is pinned by the component's and the app's `idf_component.yml`.

Build and flash the demo in an ESP-IDF 5.3 or newer shell (the demo's `CMakeLists.txt` defines `MALLOC_CAP_SIMD`
for IDF 5.3, which lacks it; firmware for real use should run IDF 5.4 or newer):

```bash
idf.py -C firmware/water_seg_demo -B firmware/water_seg_demo/build -p <PORT> flash monitor
idf.py -C firmware/water_seg_demo -B <build folder> -D WATER_SEG_MODEL_DIR=<export folder> build   # another export
```

`sdkconfig.defaults` selects the ESP32-P4 at 360 MHz, PSRAM, 16 MB flash and the console on the chip's own
USB-Serial-JTAG port. The component's model folder defaults to `results/esp32p4_export`; the build prints the
embedded model's path and SHA-256, because `WATER_SEG_MODEL_DIR` stays cached in a build folder.

To use the component in another firmware, add `firmware/components` to `EXTRA_COMPONENT_DIRS` before
`include(project.cmake)`, add `water_seg` to `main`'s `REQUIRES`, and keep PSRAM enabled:

```cpp
#include "water_seg_model.hpp"

static WaterSegModel water;
static uint8_t mask[WaterSegModel::kMaskWidth * WaterSegModel::kMaskHeight];

water.begin(WaterSegOptions{256 * 1024, true});  // the fastest configuration measured on the P4
water.segment(frame_rgb565, 160, 120, /*big_endian=*/true, mask);  // one 160x120 RGB565 frame per call
```

Camera drivers often deliver RGB565 big-endian; check the order with a colour target, since the wrong one still runs
but segments badly. Camera capture on the P4 is not part of this repository and has not been tested.

## Board evaluation

Flash the demo and read its console. Each round prints the CPU clock and PSRAM size, the model's threshold code and
exponents, ESP-DL's self-test result (`pass` expected), then for each of the six runtime configurations the time per
frame split into preprocessing, inference and threshold, the number of mask cells that differ from the PC's int8
reference in little- and big-endian byte order (0 expected), and the internal RAM and PSRAM the model took; last, a
per-layer profile of the fastest configuration. The demo repeats this every 5 s. Compare with
`results/esp32p4_export/board_timing.log`. This checks one stored frame; the P4 has not been run over the TEST set.

## Results

| Folder | What it holds | Made by |
|---|---|---|
| [validation_results](results/validation_results) | UDCP on all 110 TEST images: window and threshold locked on calibration, per-image metrics | `validate` |
| [forest_comparison](results/forest_comparison) | the first pixel forest (12 trees, depth 6, threshold 57) against UDCP; its candidates and lock | `train_pixel_forest`, `compare_pixel_forest` |
| [forest_lowfp](results/forest_lowfp) | the low false-positive round: a 16×8 forest (threshold 72; missed its goal) and the threshold-only control (the 12 trees at 71, recommended) | `optimize_pixel_forest`, `compare_lowfp`, `verify_lowfp_control` |
| [evaluation_dedup](results/evaluation_dedup) | the locked forest and UDCP results re-scored on the leak-free TEST subsets (110, 98, 74 images); the CNN comparisons' baseline | `compare_dedup` |
| [calibration_recheck](results/calibration_recheck) | the forest thresholds re-selected on the 56 calibration images without training neighbours | `recheck_calibration` |
| [cnn_espdl_run_01](results/cnn_espdl_run_01) | the dense CNN: lock, calibration curves, and its evaluations at the locked threshold and the post-hoc +5 variant | `train_cnn_espdl`, `compare_cnn_espdl`, `export_cnn_threshold_variant` |
| [cnn_espdl_run_02](results/cnn_espdl_run_02) | the deployed lite CNN: lock, float model (`model.pt`, `water_seg.onnx`), header, calibration curves, architecture exploration, evaluation, caveats | `train_cnn_lite`, `compare_cnn_lite` |
| [exploration_v3](results/exploration_v3) | a calibration-only attempt to beat the lite CNN: nothing qualified (report and result logs) | not included |
| [board_eval_run02](results/board_eval_run02) | historical: the lite CNN on an ESP32-S3 over all 110 TEST images | not included |
| [esp32p4_export](results/esp32p4_export) | the lite CNN quantized for the ESP32-P4: the files the firmware embeds, and the board's timing log | `export_p4` (its research version), `water_seg_demo` |

Most reports are in Chinese, as the research wrote them. They link some files that are not in this repository:
figures and galleries made of SUIM photographs (re-running a tool on the prepared data draws them again), the S3
board run's per-image output, and the exploration's scripts and weights. Two reports link the near-duplicate
audit as `../data/dedup_manifest.json`; it is `data/dedup_manifest.json` in the project root. The CNN runs'
ESP32-S3 model files, the compiled forest libraries and the S3 build records are not included either; the locks
still list their hashes.

## Limitations

- SUIM photographs are not the robot's camera. Water colour, exposure, white balance and the housing window differ;
  the thresholds need checking on the robot's own OV2640 footage, split by recording session.
- The TEST set was looked at in earlier rounds, and SUIM has no session IDs: the leak-free subsets drop the leaks the
  audit found, which does not prove scene-level independence.
- The P4's int8 graph differs from the S3's whose accuracy was measured (ESP-PPQ quantizes the P4's convolution
  weights per channel and the S3's per tensor, [exploration_v3](results/exploration_v3/README.md)). Its accuracy on
  TEST, and whether its threshold should be selected again on its own calibration curve, are open.
- The calibration-only exploration found an ImageNet-pretrained MobileNetV3-small encoder clearly better in float,
  but most of the gain vanished under the S3's per-tensor quantization (its best int8 version missed the selection
  margin); on the P4 it has not been tried.
- The firmware's self-test frame, `water_seg_test_frame.h`, is a 160×120 RGB565 copy of a SUIM calibration image;
  check the SUIM terms before redistributing it (see [data/SOURCES.md](data/SOURCES.md)).
