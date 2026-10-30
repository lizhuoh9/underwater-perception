# AQUALOC harbour sequence 01 on the ESP32-P4

`checks.json` and `camera.json` are what `python -m uwslam.aqualoc prepare` reported for this sequence (the gyro and the accelerometer against the reference, the resampled camera against the core's own undistortion); `board_expected_imu.json`, `board_expected_gyro.json` and `board_expected_camera.json` are the PC's runs of the sequence at 5 Hz with seed 12345 with the IMU (the accelerometer calibrated with the bias fitted to seq07), with the gyro and with the camera alone (`python -m uwslam.aqualoc board --calibration`), which the board's records in `board/` match. The scores over every seed are in [../RESULTS.md](../RESULTS.md).

## ESP32 boards

The host streams grayscale frames and optional masks over the board's serial link (`python -m uwslam.mvo_stream`) of an ESP32-P4: its USB-UART bridge, or the chip's own USB-Serial-JTAG port. The board reports each step's hash and stage times. These timings exclude capture and transfer. Each record provides its own clock, build (core/mono_vo.h), rate, core count and kernel choice (the vector kernels are mvo_vector.S). Before streaming, the board checks its vector dispatch against C on random windows. Records and raw serial lines are in `board/`.

| Record | Clock | Config | Rate | Cores | Kernels | Frames | Steps equal to the PC | Map | Per step, mean / median / max | Tracking | Motion | Detection | Map update | Voxel map / directions / plan / gyro (update) |
|---|---:|---|---:|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| `p4_360mhz_usj_camera` | 360 MHz | p4 | 5 Hz | 2 | vector | 1146 | 1145 of 1145 | — | 75 / 76 / 127 ms | 43 ms | 26 ms | 6 ms | — | — |
| `p4_360mhz_usj_gyro` | 360 MHz | p4 | 5 Hz | 2 | vector | 1146 | 1145 of 1145 | — | 87 / 87 / 139 ms | 42 ms | 38 ms | 6 ms | — | gyro's use and final bias equal to the PC's (45 µs to integrate a step's readings) |
| `p4_360mhz_usj_imu` | 360 MHz | p4 | 5 Hz | 2 | vector | 1146 | 1145 of 1145 | — | 87 / 87 / 139 ms | 42 ms | 38 ms | 6 ms | — | IMU: the gyro's use, its final bias and the world's up equal to the PC's (92 µs to integrate a step's readings) |

Records named `*_usj*` were streamed over the ESP32-P4's own USB-Serial-JTAG port, not its USB-UART bridge: the firmware was built as in the second line below, which changes only the console.

Re-run (after `uwslam.mvo compare --board-frames` and `uwslam.mono_map --board-masks`, or for a gyro or IMU sequence `uwslam.aqualoc board`):

```bash
idf.py -C firmware/mono_vo_stream -B firmware/mono_vo_stream/build -D SDKCONFIG=build/sdkconfig -p <port> flash   # the console on a USB-UART bridge
idf.py -C firmware/mono_vo_stream -B firmware/mono_vo_stream/build_usj -D SDKCONFIG=build_usj/sdkconfig -D "SDKCONFIG_DEFAULTS=sdkconfig.defaults;sdkconfig.usb_serial_jtag" -p <port> flash   # on USB-Serial-JTAG
python -m uwslam.mvo_stream --port <port> --frames <frames file> --masks <masks file> --label <name>
python -m uwslam.mvo_stream --port <port> --frames <frames file> --gyro <gyro readings> --expected <board_expected_gyro.json> --label <name>   # with the gyro
python -m uwslam.mvo_stream --port <port> --frames <frames file> --imu <IMU readings> --expected <board_expected_imu.json> --label <name>   # with the IMU
```
