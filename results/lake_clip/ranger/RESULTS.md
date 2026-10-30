# Underwater rangers on the lake clip, simulated

Three rangers simulated from the reconstruction along the reference poses at the 170 frames of 5 Hz: a 15 degree beam, the nearest surface within 8 m, each run with its own noise of 1 cm plus 0.5% of the range and 5% of its readings lost (see uwslam.ranger). Echoes: forward 92%, left 3%, right 98% of the frames; the reconstruction holds only what the camera filmed. Its unit is taken as the metre, as in every lake score. The forward ranger's readings give the odometry its metric scale (mvo_range_reading); the core's own steps stay as they were.

Over 16 seeds, median [smallest–largest]; windows are 10 s cut every 5 s from the whole runs:

| Score | Whole clip | Windows |
|---|---:|---:|
| Metric ATE (m): the path in metres, rotation and shift only | 0.194 [0.162–0.291] | 0.092 [0.049–0.248] |
| ATE after a similarity alignment (m): the scale from the reference | 0.189 [0.156–0.257] | 0.069 [0.044–0.165] |

The scale a similarity alignment would give the metric path (1: in metres): 1.031 [0.917–1.085]. The scale was first known at step 1 [1–1] and then in a share 1.00 [1.00–1.00] of the steps; runs with a scale: 16 of 16.

Re-run (the output folder must not exist or be empty), and check the board against the run with seed uwslam.mvo.SEED (its steps, and every step's metric centre and scale):

```bash
python -m uwslam.ranger <video> --reconstruction <folder> --output <fresh folder> --board-frames <frames file>
python -m uwslam.mvo_stream --port <port> --frames <frames file> --sense <folder>/sense_5hz.bin --expected <folder>/board_expected_ranger_5hz.json --label <name>
```

## ESP32 boards

The host streams grayscale frames and optional masks over the board's serial link (`python -m uwslam.mvo_stream`) of an ESP32-P4: its USB-UART bridge, or the chip's own USB-Serial-JTAG port. The board reports each step's hash and stage times. These timings exclude capture and transfer. Each record provides its own clock, build (core/mono_vo.h), rate, core count and kernel choice (the vector kernels are mvo_vector.S). Before streaming, the board checks its vector dispatch against C on random windows. Records and raw serial lines are in `board/`.

| Record | Clock | Config | Rate | Cores | Kernels | Frames | Steps equal to the PC | Map | Per step, mean / median / max | Tracking | Motion | Detection | Map update | Voxel map / directions / plan / gyro (update) |
|---|---:|---|---:|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| `p4_360mhz_usj_5hz` | 360 MHz | p4 | 5 Hz | 2 | vector | 170 | 169 of 169 | — | 127 / 132 / 155 ms | 86 ms | 25 ms | 16 ms | — | metric path and scale equal to the PC's |

Records named `*_usj*` were streamed over the ESP32-P4's own USB-Serial-JTAG port, not its USB-UART bridge: the firmware was built as in the second line below, which changes only the console.

Re-run (after `uwslam.mvo compare --board-frames` and `uwslam.mono_map --board-masks`):

```bash
idf.py -C firmware/mono_vo_stream -B firmware/mono_vo_stream/build -D SDKCONFIG=build/sdkconfig -p <port> flash   # the console on a USB-UART bridge
idf.py -C firmware/mono_vo_stream -B firmware/mono_vo_stream/build_usj -D SDKCONFIG=build_usj/sdkconfig -D "SDKCONFIG_DEFAULTS=sdkconfig.defaults;sdkconfig.usb_serial_jtag" -p <port> flash   # on USB-Serial-JTAG
python -m uwslam.mvo_stream --port <port> --frames <frames file> --masks <masks file> --label <name>
python -m uwslam.mvo_stream --port <port> --frames <frames file> [--imu <IMU readings>] --sense <sense file> --expected <expectation with a metric entry> --label <name>   # depths or ranges
```
