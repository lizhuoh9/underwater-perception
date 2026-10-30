# Live input on the ESP32-P4: a camera's JPEG frames, prepared on the board

The live build of [firmware/mono_vo_stream](../../../firmware/mono_vo_stream) (`sdkconfig.live` on top of the
USB-Serial-JTAG console; `live.c`) takes frames the way a camera delivers them, compressed and at the camera's own size,
and prepares them itself. A frame's structure is checked when it arrives (`jpeg_check.c`); the P4's JPEG engine decodes
it to YUV444; [core/frame_prep](../../../core/frame_prep.h) averages its luminance onto the odometry's 320×240 frame,
each output pixel the mean of the source rectangle it covers, in integers, half the rows on each core; a worker task
runs the odometry, the voxel map, the older maps and the plan on that frame while the next one comes in. A frame still
waiting when a newer one arrives is dropped, as a camera's would be. No masks reach the live build, so it gives no
obstacle directions.

Until the robot's camera is wired to the P4, the PC plays that camera ([uwslam.live_feed](../../../src/uwslam/live_feed.py)):
every frame the 5 Hz board stream takes (each 6th video frame, 170 up to the reference's last keyframe) as it is in the
video, 960×544 in colour, JPEG at quality 80 with 4:2:0 chroma (89,485 bytes on average, at most 106,440), sent over
the chip's USB-Serial-JTAG port ([feed.json](feed.json)). The dev board runs at 360 MHz with its PSRAM at 200 MHz.

## Two runs

`check` sends a frame only after the last one's step, then reads back the 320×240 frame the odometry saw (`grey`);
`live` sends a frame every 200 ms whatever the board is doing. Records and raw serial lines are in [board/](board).

| | `check` | `live` |
|---|---:|---:|
| Frames: stepped / dropped (after the first) | 169 / 0 | 167 / 2 (frames 140, 158) |
| JPEG decode, median / max | 6.1 / 6.1 ms | 6.1 / 6.5 ms |
| Averaging onto 320×240, median / max | 21.6 / 22.7 ms | 21.7 / 36.2 ms |
| Odometry step, median / max | 134 / 164 ms | 162 / 194 ms |
| A frame's work (decode, averaging, step, maps, plan), mean / median / max | 158 / 164 / 193 ms | 192 / 198 / 226 ms |
| Waiting for the worker, median / max | | 29 / 192 ms |
| From a frame's last byte sent to its answer, mean / median / max | | 251 / 234 / 407 ms |
| Free at the end: internal RAM / PSRAM / the worker's stack | 66,452 / 19,387,740 / 11,716 B | 66,452 / 19,387,740 / 11,384 B |

In `live` the step is slower than in `check` (162 against 134 ms in the median) because the next frame comes in during
it: the USB-Serial-JTAG driver wakes the command loop for every 64-byte packet and drops what its 8 KB receive buffer
cannot hold, so the loop runs above the worker. A camera on the P4's MIPI-CSI port would hand its frames over by
DMA instead (ESP-IDF's CSI driver moves them with the DW-GDMA); that is not measured here.

## The board's frames against the PC

[summary.json](summary.json), from `uwslam.live_feed check` on the `check` run:

- The PC core on the board's own 320×240 frames gives all 169 steps bit for bit.
- Those frames differ from the PC's decoding of the same JPEGs (libjpeg's luminance, averaged by the same
  core/frame_prep.c) by at most one grey level, in 2.9% of the pixels (mean difference −0.017).
- Over the 16 seeds of `uwslam.mvo.SEEDS` they score like the frames the PC tools make from the video, median
  [smallest–largest]:

| Frames | Position RMSE after alignment | Heading RMSE | Scale ratio spread over 5 s windows | Lost steps |
|---|---:|---:|---:|---:|
| The board's (JPEG decoded and averaged on the P4) | 0.19 m [0.13–0.28] | 18.7° [6–30] | 2.0× [1.4–3.9] | 12 [10–14] |
| The PC's decoding of the same JPEGs | 0.21 m [0.14–0.32] | 14.6° [9–27] | 2.2× [1.5–5.2] | 12.5 [9–15] |
| The PC tools' frames from the video ([mono_vo_c](../mono_vo_c/RESULTS.md)) | 0.19 m [0.16–0.26] | 19.4° [5–26] | 1.8× [1.4–2.6] | 13 [11–15] |

The seed alone can change a score twofold on this clip, so the rows do not differ beyond it: decoding JPEG at quality
80 on the board costs the odometry nothing measurable.

## Chroma subsampling and other sizes

The same frames as 4:2:2 and 4:4:4 JPEGs (`prepare --sampling`, in [sampling_422](sampling_422) and
[sampling_444](sampling_444)), streamed with `--check`: the board's 320×240 frames and all 169 steps come out with the
same hashes as from the 4:2:0 JPEGs, frame for frame, since only the chroma changes. Decoding takes longer with more
chroma: 7.8 ms (4:2:2) and 11.2 ms (4:4:4) in the median, against 6.1 ms.

The clip's first frames resized to eight sizes ([sizes](sizes)), one frame each: every frame is within one grey level
of the PC's (`decode_difference_by_frame` in summary.json) and all 7 steps equal the PC's. Widths that are no multiple
of 32, of 16 or even of 8 pixels decode as well.

| Size | 640×480 | 970×544 | 1000×750 | 1280×720 | 2048×1536 | 336×248 | 330×240 | 960×544 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Decode, ms | 3.9 | 6.2 | 8.4 | 10.0 | 31.3 | 1.7 | 1.6 | 6.1 |
| Averaging, ms | 14.7 | 22.3 | 31.7 | 33.8 | 101.4 | 7.3 | 5.9 | 22.0 |

At 5 Hz a camera's frames should stay near 960×544: at 2048×1536 decoding and averaging alone take 133 ms.

## Refused frames and recovery

[error_paths_serial.log](error_paths_serial.log): a session fed, in turn, a good frame; a header naming 600,000 bytes
(more than a frame may have); a frame whose hash does not match; a good frame after that gap; an index going back; a
grey, a progressive and a 4:1:1 JPEG; a negative index; a JPEG whose scan was cut in half (EOI kept); a good frame;
a frame with its last 500 bytes missing, so that it swallows the next frame's header; and a good frame. The board
refuses each bad one with an error line naming it (the cut scan: the driver finds a different number of data units
than the image's size gives, and the frame does not decode) and keeps reading at the right place. The frame whose
header was swallowed is lost, its bytes read as 155 lines of garbage, the last one ended by the next header's leading
newline, and the good frame after it steps: the host starts every header on a fresh line. Every good frame stepped;
13 of 13 checks passed, and the session's end reports 5 frames received.

## Why YUV444, and the PSRAM's clock

From ESP-IDF 5.3.2's JPEG driver and 2D-DMA:

- Asked for grey, the engine passes a colour JPEG's decoded data through unconverted, and the 2D-DMA's receive check
  fails on the first frame: [grey_output_serial.log](grey_output_serial.log), the live firmware of the time with only
  its output format set to grey.
- Its RGB output converts with a limited-range BT.601 matrix (298/256 on the luminance), while a JPEG's YCbCr is full
  range: the frames would come out with 16% more contrast, clipped at both ends.
- YUV444 needs no matrix: the third of each pixel's three bytes is the JPEG's own luminance (read from there, the
  board's frames match libjpeg's luminance, above). The engine writes whole MCUs, so a decoded row is the width
  rounded up to 16 pixels (4:2:0, 4:2:2) or 8 (4:4:4), times 3 bytes.
- The driver's header parser reads without bounds, and jpeg_decoder_get_info does not free its buffer when a frame's
  sampling is unknown: the live build checks each frame's structure itself and never calls it.

The PSRAM runs at 200 MHz, an experimental option in ESP-IDF 5.3 that its own ESP32-P4 JPEG and camera examples set.

## Re-run

```bash
idf.py -C firmware/mono_vo_stream -B firmware/mono_vo_stream/build_live -D SDKCONFIG=build_live/sdkconfig \
    -D "SDKCONFIG_DEFAULTS=sdkconfig.defaults;sdkconfig.usb_serial_jtag;sdkconfig.live" -p <PORT> flash
python -m uwslam.live_feed prepare <video> --reconstruction <folder> --output <fresh folder>   # [--sampling 422]
python -m uwslam.live_feed stream <prepared folder> --port <PORT> --label check --check
python -m uwslam.live_feed stream <prepared folder> --port <PORT> --label live
python -m uwslam.live_feed check <prepared folder> --label check --reconstruction <folder> --output <fresh folder>
```

`prepare` gives the same frames when `frames_sha256` in its feed.json equals this one's. `stream` runs in ESP-IDF's
Python (it needs only pyserial), `prepare` and `check` in the project's. The sizes feed and the refused frames were made
by scripts outside the repository from the prepared feed's first frames.
