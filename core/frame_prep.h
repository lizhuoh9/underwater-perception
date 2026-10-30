/*
 * frame_prep.h - the odometry's 320x240 grey frame from a camera's larger image. The image is stretched onto 320x240
 * whatever its aspect, as uwslam.mvo.camera_for describes it (the camera's intrinsics carry the aspect): each output
 * pixel is the mean of the source rectangle it covers, source pixels that the rectangle's edges cut counting by the
 * part inside. The grey value is the image's luminance byte, every step-th byte along a row: a grey image's every
 * byte, a YUV444 image's Y (what the ESP32-P4's JPEG engine gives from a colour JPEG), a YUYV image's every other byte.
 * All in integers, rounded half up, so that a PC and an ESP32-P4 give the same bytes from the same image.
 *
 * C99, no heap, no libm.
 */
#ifndef FRAME_PREP_H
#define FRAME_PREP_H

#include <stdint.h>

#define FP_WIDTH 320
#define FP_HEIGHT 240
#define FP_MAX_SIDE 2048 /* the largest source side: the weighted sums then stay within 32 bits */

/* Rows [row_begin, row_end) of the 320x240 frame (out: the whole frame, row-major) averaged from the w x h image src,
 * whose pixel (x, y) has its grey value at src[y stride + x step]; no byte outside those is read. Calls for
 * complementary row ranges make the whole frame, so cores can share it. Returns 1, or 0 with out untouched when src or
 * out is NULL, w or h is below the frame's or above FP_MAX_SIDE, step is not 1 to 4, stride is below (w - 1) step + 1,
 * or [row_begin, row_end) is not a range within 0 to 240. */
int fp_area_resize(const uint8_t *src, int w, int h, int stride, int step, uint8_t *out, int row_begin, int row_end);

#endif
