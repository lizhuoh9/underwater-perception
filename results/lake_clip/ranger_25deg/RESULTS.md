# Underwater rangers on the lake clip, simulated

Three rangers simulated from the reconstruction along the reference poses at the 170 frames of 5 Hz: a 25 degree beam, the nearest surface within 8 m, each run with its own noise of 1 cm plus 0.5% of the range and 5% of its readings lost (see uwslam.ranger). Echoes: forward 96%, left 8%, right 98% of the frames; the reconstruction holds only what the camera filmed. Its unit is taken as the metre, as in every lake score. The forward ranger's readings give the odometry its metric scale (mvo_range_reading); the core's own steps stay as they were.

Over 16 seeds, median [smallest–largest]; windows are 10 s cut every 5 s from the whole runs:

| Score | Whole clip | Windows |
|---|---:|---:|
| Metric ATE (m): the path in metres, rotation and shift only | 0.194 [0.153–0.264] | 0.116 [0.082–0.200] |
| ATE after a similarity alignment (m): the scale from the reference | 0.189 [0.156–0.257] | 0.069 [0.044–0.165] |

The scale a similarity alignment would give the metric path (1: in metres): 1.003 [0.970–1.029]. The scale was first known at step 1 [1–1] and then in a share 1.00 [1.00–1.00] of the steps; runs with a scale: 16 of 16.

Re-run (the output folder must not exist or be empty), and check the board against the run with seed uwslam.mvo.SEED (its steps, and every step's metric centre and scale):

```bash
python -m uwslam.ranger <video> --reconstruction <folder> --output <fresh folder> --board-frames <frames file>
python -m uwslam.mvo_stream --port <port> --frames <frames file> --sense <folder>/sense_5hz.bin --expected <folder>/board_expected_ranger_5hz.json --label <name>
```
