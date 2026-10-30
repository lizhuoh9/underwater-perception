/* See jpeg_check.h. */
#include "jpeg_check.h"

#include <stddef.h>

#define MARKER_SOF0 0xC0u
#define MARKER_DHT 0xC4u
#define MARKER_JPG 0xC8u
#define MARKER_DAC 0xCCu
#define MARKER_SOS 0xDAu
#define FRAME_HEADER_LENGTH 17u /* a three-component frame header's length field: 8 + 3 x 3 */

static uint32_t big_endian16(const uint8_t *bytes)
{
    return ((uint32_t)bytes[0] << 8) | bytes[1];
}

/* The frame header's fields (after its length): precision, height, width, component count, then per component its
 * id, sampling and quantisation table. */
static int read_frame(const uint8_t *frame, uint32_t min_width, uint32_t min_height, uint32_t max_side,
                      jpeg_shape_t *shape)
{
    const uint32_t height = big_endian16(frame + 1), width = big_endian16(frame + 3), sampling = frame[7];
    if (frame[0] != 8 || frame[5] != 3 || (sampling != 0x11u && sampling != 0x21u && sampling != 0x22u)
        || frame[10] != 0x11u || frame[13] != 0x11u || width < min_width || height < min_height || width > max_side
        || height > max_side || (width * height) % 8u != 0) {
        return 0;
    }
    shape->width = width;
    shape->height = height;
    shape->mcu_width = (sampling >> 4) * 8u;
    shape->mcu_height = (sampling & 15u) * 8u;
    return 1;
}

int jpeg_check(const uint8_t *data, uint32_t size, uint32_t min_width, uint32_t min_height, uint32_t max_side,
               jpeg_shape_t *shape)
{
    jpeg_shape_t found;
    uint32_t at = 2;
    int framed = 0;
    if (data == NULL || shape == NULL || size < 4 || data[0] != 0xFF || data[1] != 0xD8 || data[size - 2] != 0xFF
        || data[size - 1] != 0xD9) {
        return 0;
    }
    while (at + 4 <= size - 2) { /* a marker and its length, before the EOI */
        const uint32_t marker = data[at + 1];
        uint32_t length;
        if (data[at] != 0xFF) {
            return 0;
        }
        if (marker == 0xFF) { /* a fill byte */
            ++at;
            continue;
        }
        if (marker == 0x00 || marker == 0x01 || (marker >= 0xD0 && marker <= 0xD9)) {
            return 0; /* a stuffed byte, TEM, a restart, SOI or EOI has no place before the scan */
        }
        length = big_endian16(data + at + 2);
        if (length < 2 || length > size - 2 - at - 2) { /* the segment must end before the EOI */
            return 0;
        }
        if (marker == MARKER_SOS) {
            if (!framed) {
                return 0;
            }
            *shape = found;
            return 1;
        }
        if (marker >= 0xC0 && marker <= 0xCF && marker != MARKER_DHT && marker != MARKER_JPG && marker != MARKER_DAC) {
            if (marker != MARKER_SOF0 || framed || length != FRAME_HEADER_LENGTH
                || !read_frame(data + at + 4, min_width, min_height, max_side, &found)) {
                return 0; /* not baseline, a second frame header, or not a frame the decoder takes */
            }
            framed = 1;
        }
        at += 2 + length;
    }
    return 0; /* no scan */
}
