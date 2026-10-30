/*
 * sensors.c - the commands that give the odometry the robot's other sensors (the protocol is in main.c): an IMU's
 * mounting, readings and accelerometer bias (`mount`, `gyro`, `imu`, `accel_bias`), a depth sensor's readings
 * (`depth`) and a forward ranger's beam and readings (`beam`, `range`).
 */
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "esp_timer.h"

#include "stream.h"

#define GYRO_READINGS 256 /* the most gyro or IMU readings one step takes (uwslam.mvo_stream.GYRO_READINGS) */
#define GYRO_READING_BYTES 16
#define IMU_READING_BYTES 28

static uint8_t s_readings[GYRO_READINGS * IMU_READING_BYTES];

static void mount(const char *line)
{
    uint32_t bits[9];
    float matrix[9];
    if (!stream_open()) {
        printf("error start required\n");
        return;
    }
    if (sscanf(line, "mount %" SCNu32 " %" SCNu32 " %" SCNu32 " %" SCNu32 " %" SCNu32 " %" SCNu32 " %" SCNu32
               " %" SCNu32 " %" SCNu32, &bits[0], &bits[1], &bits[2], &bits[3], &bits[4], &bits[5], &bits[6],
               &bits[7], &bits[8]) != 9) {
        printf("error bad mount line\n");
        return;
    }
    memcpy(matrix, bits, sizeof matrix);
    if (!mvo_gyro_mount(stream_state(), matrix)) {
        printf("error mount not finite\n");
        return;
    }
    printf("mounted\n");
}

static void accel_bias(const char *line)
{
    uint32_t bits[3];
    float bias[3];
    if (!stream_open()) {
        printf("error start required\n");
        return;
    }
    if (sscanf(line, "accel_bias %" SCNu32 " %" SCNu32 " %" SCNu32, &bits[0], &bits[1], &bits[2]) != 3) {
        printf("error bad accel_bias line\n");
        return;
    }
    memcpy(bias, bits, sizeof bias);
    if (!mvo_imu_set_accel_bias(stream_state(), bias)) {
        printf("error accel_bias not finite\n");
        return;
    }
    printf("accel_bias set\n");
}

/* A step's gyro readings (`gyro`, with_force 0) or IMU readings (`imu`, with_force 1), integrated by the core into
 * the rotation, and the forces into the up, that the next frame's step may use. */
static void readings(const char *line, int with_force)
{
    const char *name = with_force ? "imu" : "gyro";
    const int size = with_force ? IMU_READING_BYTES : GYRO_READING_BYTES;
    uint32_t expected;
    int count, i, dropped = 0;
    int64_t begin;
    if (sscanf(line + strlen(name), " %d %" SCNu32, &count, &expected) != 2 || count < 0) {
        printf("error bad %s line\n", name);
        return;
    }
    if (count > GYRO_READINGS) { /* consume what was sent, a buffer at a time, to keep the command framing */
        int left = (count < 65536 ? count : 65536) * size;
        while (left > 0) {
            const int part = left < (int)sizeof s_readings ? left : (int)sizeof s_readings;
            if (!read_exact(s_readings, part)) {
                break;
            }
            left -= part;
        }
        printf("error %s readings over %d\n", name, GYRO_READINGS);
        return;
    }
    /* As with a frame: consume the payload even when it cannot be used, to keep the command framing. */
    if (!read_exact(s_readings, count * size)) {
        printf("error %s readings arrived incomplete\n", name);
        return;
    }
    if (!stream_open() || !stream_started()) {
        printf("error %s readings need a started run\n", name);
        return;
    }
    if (fnv(2166136261u, s_readings, (size_t)count * size) != expected) {
        printf("error %s hash mismatch\n", name);
        return;
    }
    begin = esp_timer_get_time();
    for (i = 0; i < count; ++i) {
        float reading[7]; /* dt, the rate, then the force */
        memcpy(reading, s_readings + i * size, (size_t)size);
        if (with_force) {
            dropped += !mvo_imu_reading(stream_state(), reading + 1, reading + 4, reading[0]);
        } else {
            mvo_gyro_rate(stream_state(), reading + 1, reading[0]);
        }
    }
    if (with_force) {
        printf("imu readings=%d dropped=%d us=%" PRId64 "\n", count, dropped, esp_timer_get_time() - begin);
    } else {
        printf("gyro readings=%d us=%" PRId64 "\n", count, esp_timer_get_time() - begin);
    }
}

/* The forward ranger's beam: its axis in the camera's axes and the cosine of its half angle, as float bits. */
static void beam(const char *line)
{
    uint32_t bits[4];
    float values[4];
    if (!stream_open()) {
        printf("error start required\n");
        return;
    }
    if (sscanf(line, "beam %" SCNu32 " %" SCNu32 " %" SCNu32 " %" SCNu32, &bits[0], &bits[1], &bits[2], &bits[3])
        != 4) {
        printf("error bad beam line\n");
        return;
    }
    memcpy(values, bits, sizeof values);
    if (!mvo_range_setup(stream_state(), values, values[3])) {
        printf("error beam refused: a finite axis and a cosine between 0 and 1\n");
        return;
    }
    printf("beam set\n");
}

/* A depth (`depth`) or range (`range`) reading for the next frame, as float bits; before the first frame too. */
static void reading(const char *line, int is_range)
{
    const char *name = is_range ? "range" : "depth";
    uint32_t bits;
    float value;
    if (!stream_open()) {
        printf("error start required\n");
        return;
    }
    if (sscanf(line + strlen(name), " %" SCNu32, &bits) != 1) {
        printf("error bad %s line\n", name);
        return;
    }
    memcpy(&value, &bits, sizeof value);
    if (is_range ? !mvo_range_reading(stream_state(), value) : !mvo_depth_reading(stream_state(), value)) {
        printf(is_range ? "error range refused: a beam first, and a positive finite range\n"
                        : "error depth not finite\n");
        return;
    }
    printf("%s set\n", name);
}

int sensor_command(const char *line)
{
    if (strncmp(line, "gyro ", 5) == 0) {
        readings(line, 0);
    } else if (strncmp(line, "imu ", 4) == 0) {
        readings(line, 1);
    } else if (strncmp(line, "accel_bias ", 11) == 0) {
        accel_bias(line);
    } else if (strncmp(line, "mount ", 6) == 0) {
        mount(line);
    } else if (strncmp(line, "beam ", 5) == 0) {
        beam(line);
    } else if (strncmp(line, "depth ", 6) == 0) {
        reading(line, 0);
    } else if (strncmp(line, "range ", 6) == 0) {
        reading(line, 1);
    } else {
        return 0;
    }
    return 1;
}
