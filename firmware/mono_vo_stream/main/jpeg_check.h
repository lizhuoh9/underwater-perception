/*
 * jpeg_check.h - the live build's check of a camera's JPEG frame before ESP-IDF's JPEG driver parses it. The driver
 * (ESP-IDF 5.3.2) reads its headers without bounds, past the end of a truncated JPEG, and jpeg_decoder_get_info does not
 * free its header buffer when a frame's sampling is one it does not know; so a frame reaches the driver only when this
 * check finds the structure the driver decodes, and the live build takes the frame's size from here, not from the
 * driver. Portable C99, also built on the PC (tests/test_jpeg_check.py).
 */
#ifndef JPEG_CHECK_H
#define JPEG_CHECK_H

#include <stdint.h>

typedef struct {
    uint32_t width, height;
    uint32_t mcu_width, mcu_height; /* the luminance's sampling times 8: the decoder writes whole MCUs */
} jpeg_shape_t;

/* 1 when data[0, size) is a JPEG the live build decodes: SOI first and EOI last; marker segments up to the scan, each
 * inside the data; one baseline frame header (SOF0) before the scan: 8-bit, three components, the first sampled 1x1,
 * 2x1 or 2x2 and the other two 1x1, from min_width x min_height to max_side x max_side, with a pixel count that is a
 * multiple of 8 (the driver's rule). Fills shape; else returns 0 and leaves shape untouched. */
int jpeg_check(const uint8_t *data, uint32_t size, uint32_t min_width, uint32_t min_height, uint32_t max_side,
               jpeg_shape_t *shape);

#endif
