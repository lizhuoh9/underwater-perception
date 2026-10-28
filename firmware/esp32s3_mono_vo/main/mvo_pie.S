/*
 * The Lucas-Kanade kernels of core/mono_vo.h (mvo_kernels_t) with the ESP32-S3's vector instructions (PIE), eight
 * 16-bit lanes at a time. They give exactly the integers of the portable mvo_lk_template_c and mvo_lk_sums_c: each
 * pixel times its Q14 weight shifted right by 7 lane by lane (EE.VMUL.S16 with SAR = 7), lane sums that stay within
 * 16 bits (an interpolation is at most 255 << 7), and window sums accumulated exactly in the 40-bit ACCX register.
 * mvo_pie_block copies a window at the image border as mvo_block_c does, a 16-byte vector per row.
 *
 * Windowed ABI, leaf functions. q0 and q1 load and widen pixels, q2 holds a weight in every lane, q4 and q5 the lanes
 * 0-7 and 8-15 of an interpolated row; a8-a11 point at the four weights.
 */

    .set    WINDOW_ROW, 32          /* bytes per window row: MVO_LANES int16 */
    .set    WINDOW_ARRAY, 480       /* bytes per window array: MVO_SIDE rows */

/* q4/q5 (first) or q4/q5 += (the 16 pixels from p widened to 16 bits) * (the weight in every lane of wq), >> 7 per
 * lane. a14 is scratch. */
    .macro  TERM p, wq, first
    mov     a14, \p
    ee.ld.128.usar.ip q0, a14, 16   /* the aligned block holding p; SAR_BYTE = p mod 16 */
    ee.vld.128.ip   q1, a14, 16     /* and the next one */
    ee.src.q        q0, q0, q1      /* the 16 bytes from p */
    ee.zero.q       q1
    ee.vzip.8       q0, q1          /* q0: pixels 0-7, q1: pixels 8-15, as 16-bit lanes */
    .if \first
    ee.vmul.s16     q4, q0, \wq
    ee.vmul.s16     q5, q1, \wq
    .else
    ee.vmul.s16     q0, q0, \wq
    ee.vmul.s16     q1, q1, \wq
    ee.vadds.s16    q4, q4, q0
    ee.vadds.s16    q5, q5, q1
    .endif
    .endm

/* q4/q5 = the bilinear interpolation of the 16 lanes from pixel p: p, p + 1, p + stride, p + stride + 1 with the four
 * weights, each loaded into q2 for its term. a14 and a15 are scratch; p and stride are kept. */
    .macro  INTERP p, stride
    ee.vldbc.16     q2, a8
    TERM    \p, q2, 1
    addi    a15, \p, 1
    ee.vldbc.16     q2, a9
    TERM    a15, q2, 0
    add     a15, \p, \stride
    ee.vldbc.16     q2, a10
    TERM    a15, q2, 0
    addi    a15, a15, 1
    ee.vldbc.16     q2, a11
    TERM    a15, q2, 0
    .endm

/* INTERP with the weights already in every lane of q2, q3, q6 and q7. */
    .macro  INTERP_HELD p, stride
    TERM    \p, q2, 1
    addi    a15, \p, 1
    TERM    a15, q3, 0
    add     a15, \p, \stride
    TERM    a15, q6, 0
    addi    a15, a15, 1
    TERM    a15, q7, 0
    .endm

/* The four weight pointers from a5. */
    .macro  WEIGHTS
    mov     a8, a5
    addi    a9, a5, 2
    addi    a10, a5, 4
    addi    a11, a5, 6
    .endm

/* ACCX, a 40-bit sum, sign-extended into the int64 at out + offset. a14 and a15 are scratch. */
    .macro  STORE_ACCX out, offset
    rur.accx_0      a14
    rur.accx_1      a15
    slli    a15, a15, 24
    srai    a15, a15, 24
    s32i    a14, \out, \offset
    s32i    a15, \out, \offset + 4
    .endm

/* ACCX = the dot product of the 30 vectors (a window array) from pa and pb, which advance past them. q0-q3 and a5
 * are scratch. */
    .macro  DOT30 pa, pb
    ee.zero.accx
    ee.vld.128.ip   q0, \pa, 16
    ee.vld.128.ip   q1, \pb, 16
    movi    a5, 14
1:
    ee.vld.128.ip   q2, \pa, 16
    ee.vmulas.s16.accx.ld.ip q3, \pb, 16, q0, q1
    ee.vld.128.ip   q0, \pa, 16
    ee.vmulas.s16.accx.ld.ip q1, \pb, 16, q2, q3
    addi    a5, a5, -1
    bnez    a5, 1b
    ee.vld.128.ip   q2, \pa, 16
    ee.vmulas.s16.accx.ld.ip q3, \pb, 16, q0, q1
    ee.vmulas.s16.accx q2, q3
    .endm


/* void mvo_pie_lk_sums(const mvo_window_t *window, const uint8_t *src, int stride, const int16_t weights[4],
 *                      int64_t sums[2])
 * a2 window, a3 src (the window's top-left pixel), a4 stride, a5 weights, a6 sums. */
    .text
    .align  4
    .global mvo_pie_lk_sums
    .type   mvo_pie_lk_sums, @function
mvo_pie_lk_sums:
    entry   sp, 512                 /* [sp, sp + 480): patch - interpolation, a window array */
    ssai    7                       /* EE.VMUL.S16 shifts its products right by SAR */
    WEIGHTS
    ee.vldbc.16     q2, a8          /* the weights, held for the whole window */
    ee.vldbc.16     q3, a9
    ee.vldbc.16     q6, a10
    ee.vldbc.16     q7, a11
    mov     a12, a2                 /* patch */
    movi    a13, WINDOW_ARRAY       /* gx */
    add     a13, a13, a2
    mov     a7, sp
    ee.zero.accx
    movi    a5, 15                  /* MVO_SIDE rows */
1:
    INTERP_HELD a3, a4
    ee.vld.128.ip   q0, a12, 16
    ee.vsubs.s16    q0, q0, q4      /* patch - interpolation, lanes 0-7 */
    ee.vst.128.ip   q0, a7, 16
    ee.vld.128.ip   q1, a13, 16
    ee.vmulas.s16.accx q0, q1
    ee.vld.128.ip   q0, a12, 16
    ee.vsubs.s16    q0, q0, q5      /* lanes 8-15; lane 15 meets gx = 0 */
    ee.vst.128.ip   q0, a7, 16
    ee.vld.128.ip   q1, a13, 16
    ee.vmulas.s16.accx q0, q1
    add     a3, a3, a4
    addi    a5, a5, -1
    bnez    a5, 1b
    STORE_ACCX a6, 0
    mov     a7, sp                  /* the differences again, times gy */
    movi    a13, 2 * WINDOW_ARRAY
    add     a13, a13, a2
    DOT30   a7, a13
    STORE_ACCX a6, 8
    retw
    .size   mvo_pie_lk_sums, . - mvo_pie_lk_sums


/* void mvo_pie_lk_template(mvo_window_t *window, const uint8_t *src, int stride, const int16_t weights[4],
 *                          int64_t g[3])
 * a2 window, a3 src (one up and one left of the window's top-left pixel), a4 stride, a5 weights, a6 g. The window's
 * pixel (i, j) is src + (j + 1) * stride + i + 1. */
    .text
    .align  4
    .global mvo_pie_lk_template
    .type   mvo_pie_lk_template, @function
mvo_pie_lk_template:
    entry   sp, 608                 /* [sp, sp + 544): rows -1..15 of the interpolation; then src and a lane mask */
    ssai    7
    WEIGHTS
    s32i    a3, sp, 544             /* src */
    movi    a14, -1                 /* sp + 560: keeps lanes 0-6 of a vector, clears lane 7 (a row's lane 15) */
    s32i    a14, sp, 560
    s32i    a14, sp, 564
    s32i    a14, sp, 568
    extui   a14, a14, 0, 16
    s32i    a14, sp, 572
    movi    a14, 560
    add     a14, a14, sp
    ee.vld.128.ip   q3, a14, 0

    /* 1. the interpolation at the window's columns for rows -1..15, into the buffer */
    addi    a3, a3, 1               /* the pixel above the window's top-left one */
    mov     a7, sp
    movi    a5, 17
1:
    INTERP  a3, a4
    ee.vst.128.ip   q4, a7, 16
    ee.vst.128.ip   q5, a7, 16
    add     a3, a3, a4
    addi    a5, a5, -1
    bnez    a5, 1b

    /* 2. patch = rows 0..14, gy = row j + 1 - row j - 1 */
    addi    a7, sp, WINDOW_ROW      /* row 0 */
    mov     a12, a2                 /* patch */
    movi    a13, 2 * WINDOW_ARRAY   /* gy */
    add     a13, a13, a2
    movi    a5, 15
2:
    mov     a14, a7
    ee.vld.128.ip   q0, a14, 16     /* row j */
    ee.vld.128.ip   q1, a14, 16
    ee.andq q1, q1, q3
    ee.vst.128.ip   q0, a12, 16
    ee.vst.128.ip   q1, a12, 16
    ee.vld.128.ip   q0, a14, 16     /* row j + 1 */
    ee.vld.128.ip   q1, a14, 16
    addi    a15, a7, -WINDOW_ROW    /* row j - 1 */
    ee.vld.128.ip   q4, a15, 16
    ee.vld.128.ip   q5, a15, 16
    ee.vsubs.s16    q0, q0, q4
    ee.vsubs.s16    q1, q1, q5
    ee.andq q1, q1, q3
    ee.vst.128.ip   q0, a13, 16
    ee.vst.128.ip   q1, a13, 16
    addi    a7, a7, WINDOW_ROW
    addi    a5, a5, -1
    bnez    a5, 2b

    /* 3. gx = the interpolation a column right minus a column left, row by row; the right one waits in gx */
    l32i    a3, sp, 544
    add     a3, a3, a4              /* row 0, a column left of the window: src + stride */
    movi    a13, WINDOW_ARRAY       /* gx */
    add     a13, a13, a2
    movi    a7, 15
3:
    addi    a12, a3, 2
    INTERP  a12, a4                 /* a column right */
    mov     a12, a13
    ee.vst.128.ip   q4, a12, 16
    ee.vst.128.ip   q5, a12, 16
    INTERP  a3, a4                  /* a column left */
    mov     a12, a13
    ee.vld.128.ip   q0, a12, 16
    ee.vld.128.ip   q1, a12, 16
    ee.vsubs.s16    q0, q0, q4
    ee.vsubs.s16    q1, q1, q5
    ee.andq q1, q1, q3
    ee.vst.128.ip   q0, a13, 16
    ee.vst.128.ip   q1, a13, 16
    add     a3, a3, a4
    addi    a7, a7, -1
    bnez    a7, 3b

    /* 4. g = gx.gx, gx.gy, gy.gy */
    movi    a12, WINDOW_ARRAY
    add     a12, a12, a2
    mov     a13, a12
    DOT30   a12, a13
    STORE_ACCX a6, 0
    movi    a12, WINDOW_ARRAY
    add     a12, a12, a2
    movi    a13, 2 * WINDOW_ARRAY
    add     a13, a13, a2
    DOT30   a12, a13
    STORE_ACCX a6, 8
    movi    a12, 2 * WINDOW_ARRAY
    add     a12, a12, a2
    mov     a13, a12
    DOT30   a12, a13
    STORE_ACCX a6, 16
    retw
    .size   mvo_pie_lk_template, . - mvo_pie_lk_template


/* void mvo_pie_block(uint8_t *out, int size, const uint8_t *img, int w, int h, const int origin[2])
 * a2 out (16-byte aligned), a3 size (16 or 18), a4 img, a5 w, a6 h, a7 origin. Each row of the block: one or two
 * 16-byte vector copies from the clamped image row at origin[0] (reading before or after the row when the block crosses
 * the left or right border), then the bytes left or right of the image set to the row's first or last pixel. The caller
 * must provide readable margins for those loads; main.c dispatches unsafe image edges to the C kernel. Rows of
 * out are 16 bytes apart for size 16 and 32 for size 18. */
    .text
    .align  4
    .global mvo_pie_block
    .type   mvo_pie_block, @function
mvo_pie_block:
    entry   sp, 32
    l32i    a8, a7, 0               /* x0 */
    l32i    a9, a7, 4               /* y0 */
    addi    a7, a3, 15
    srli    a7, a7, 4
    slli    a7, a7, 4               /* out's row stride */
    addi    a6, a6, -1              /* the last image row */
    movi    a10, 0                  /* the block row */
1:
    add     a11, a9, a10            /* its image row, clamped */
    movi    a14, 0
    max     a11, a11, a14
    min     a11, a11, a6
    mull    a12, a11, a5
    add     a12, a12, a4            /* the image row's first pixel */
    add     a13, a12, a8            /* the block's first pixel in that row, maybe outside it */
    ee.ld.128.usar.ip q0, a13, 16
    ee.vld.128.ip   q1, a13, 16
    ee.src.q        q2, q0, q1
    mov     a14, a2
    ee.vst.128.ip   q2, a14, 16
    beqi    a3, 16, 2f              /* size 18: 16 more bytes */
    ee.vld.128.ip   q0, a13, 16
    ee.src.q        q2, q1, q0
    ee.vst.128.ip   q2, a14, 16
2:
    bgez    a8, 4f                  /* left of the image: the row's first pixel */
    neg     a14, a8
    min     a14, a14, a3
    l8ui    a15, a12, 0
    mov     a13, a2
3:
    s8i     a15, a13, 0
    addi    a13, a13, 1
    addi    a14, a14, -1
    bnez    a14, 3b
4:
    add     a14, a8, a3             /* right of the image: the row's last pixel */
    sub     a14, a14, a5
    blti    a14, 1, 6f
    min     a14, a14, a3
    add     a13, a12, a5
    addi    a13, a13, -1
    l8ui    a15, a13, 0
    add     a13, a2, a3
    sub     a13, a13, a14
5:
    s8i     a15, a13, 0
    addi    a13, a13, 1
    addi    a14, a14, -1
    bnez    a14, 5b
6:
    add     a2, a2, a7
    addi    a10, a10, 1
    blt     a10, a3, 1b
    retw
    .size   mvo_pie_block, . - mvo_pie_block
