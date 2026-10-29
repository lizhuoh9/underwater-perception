# Selection rule for the next locked run (cnn_espdl_run_03)

Written 2026-10-23T10:42:11+09:00, while round 1 was still training and before any round-1 or round-2 result was read.

1. Data: the 80 calibration images only. No TEST image or label is loaded by any exploration or selection step.
2. Candidates: lite (= run_02 architecture, the baseline), lite_gc, lite_dice, lite_ema (round 1);
   mnv3s_nose (round 2, ImageNet-pretrained MobileNetV3-small encoder without SE blocks).
   mnv3s (with SE) and mnv3s_nose_scratch are float references only: ESP-PPQ flags the SE fc1 requantization
   on the S3, and the scratch run only separates the effect of pretraining.
3. Seeds 20261022, 20261023, 20261024 for every candidate; same data, augmentation, epochs and schedule as run_02.
4. Screening: mean float water FPR at 5% miss (interpolated) over the three seeds. sigma = standard deviation of
   that metric over the three lite seeds. Finalists beat lite's mean by more than max(1.0 pp, 2 sigma).
5. Finalists are exported to int8 with ESP-PPQ per seed. A finalist qualifies when its mean int8 FPR at 5% miss
   beats lite's mean int8 value by the same margin and its mean int8-minus-float gap is at most 1.5 pp.
6. Speed: the qualifier must run at most 60 ms per frame single-core (<=128 KB internal RAM config) in
   water_seg_demo on the ESP32-S3, measured on the board before locking.
7. Choice: among qualifiers within the speed budget, the lowest mean int8 FPR at 5% miss. The locked run uses
   that candidate's seed with the lowest int8 calibration FPR at 5% miss as trained (no retraining); thresholds
   follow run_02's rule (deployment cap 5%, comparison cap = forest cap 0.07879003161275763).
8. If nothing qualifies, run_02 stays the deployed model and that is the result.

## Amendment, 2026-10-23T11:30:04+09:00

Read so far: round 1 (lite sigma = 1.15 pp; no tweak passes) and mnv3s_nose seeds 20261022/20261023 in float
(8.00% / 9.44% FPR at 5% miss). Its int8 export fails on the S3: ESP-PPQ quantizes S3 conv weights per tensor
with power-of-two scales, and the pretrained, BN-folded weights lose small channels (int8 FPR at 5% miss
45-74% under default, KL/min-max and equalization settings; layer-wise error 119% in the stem). mnv3s_nose is
therefore not deployable and drops out at step 5. Two candidates are added under the unchanged rule:

- lite_kd: run_02's architecture trained with 0.5 * BCE(labels) + 0.5 * BCE(teacher probability), the teacher
  being the float mnv3s_nose seed-20261022 model (calibration images are never used in training).
- mnv3s_relu: mnv3s_nose with ReLU in place of Hardswish and without the global gate, quantized with ESP-PPQ
  layer-wise equalization (opt level 1), since equalization needs ReLU between layers.

Same seeds, screening margin, int8 check and speed budget as above.
