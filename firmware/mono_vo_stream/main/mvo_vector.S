/*
 * The Lucas-Kanade window kernels of core/mono_vo.h (mvo_kernels_t) with the ESP32-P4's vector instructions (PIE),
 * eight 16-bit lanes at a time, giving exactly the integers of the portable mvo_lk_template_c and mvo_lk_sums_c. Each
 * pixel times its Q14 weight is shifted right by 7 lane by lane (ESP.VMUL.S16 with SAR = 7), lane sums stay within 16
 * bits (an interpolation is at most 255 << 7), and window sums accumulate exactly in the 40-bit XACC register, stored
 * sign-extended to an int64 (ESP.ST.S.XACC.IP): a window sum is at most SIDE * SIDE products of two Q7 differences,
 * 361 * (255 << 7)^2 < 2^39 for the 19x19 window. The block copy stays the portable mvo_block_c (about 8 us a call on
 * the P4); no vector version was written.
 *
 * The window is the core's (core/mono_vo.h): 19x19, stored in rows of 24 lanes (three vectors). main.c checks that the
 * core's MVO_SIDE and MVO_LANES are the ones assembled here.
 *
 * Standard RISC-V calling convention, leaf functions using only a0-a5, t0-t6 and the stack. Vector loads and stores
 * take their address in s0, s1, a0-a5, s8-s11 or t3-t6, never sp, so stack buffers are reached through t3. q2, q3, q6
 * and q7 hold the four weights, one in every lane, for the whole call; q0 and q1 load and widen pixels; q4 and q5 hold
 * the lanes 0-7 and 8-15 of an interpolated row, and q4 again its lanes 16-23. A row's lanes from
 * SIDE on lie outside the window and are cleared with halfword stores.
 */

    .set    SIDE, 19                /* MVO_SIDE */
    .set    VECS, 3                 /* MVO_LANES / 8 */
    .set    WINDOW_ROW, 16 * VECS           /* bytes per window row: MVO_LANES int16 */
    .set    WINDOW_ARRAY, SIDE * WINDOW_ROW /* bytes per window array: MVO_SIDE rows */
    .set    INTERP_BYTES, (SIDE + 2) * WINDOW_ROW /* the template's interpolation, rows -1..SIDE */
    .set    DOT_VECTORS, SIDE * VECS        /* vectors per window array */
    .set    PAD_LANES, 8 * VECS - SIDE      /* lanes of a row outside the window */
    .if DOT_VECTORS % 2 == 0
    .error "DOT takes the vectors of a window array two at a time after the first: it needs an odd count"
    .endif

/* q4/q5 (first) or q4/q5 += (the 16 pixels from p widened to 16 bits) * (the weight in every lane of wq), >> 7 per
 * lane. t5 is scratch. */
    .macro  TERM p, wq, first
    mv      t5, \p
    esp.ld.128.usar.ip q0, t5, 16   /* the aligned block holding p; SAR_BYTES = p mod 16 */
    esp.vld.128.ip  q1, t5, 16      /* and the next one */
    esp.src.q       q0, q0, q1      /* the 16 bytes from p */
    esp.zero.q      q1
    esp.vzip.8      q0, q1          /* q0: pixels 0-7, q1: pixels 8-15, as 16-bit lanes */
    .if \first
    esp.vmul.s16    q4, q0, \wq
    esp.vmul.s16    q5, q1, \wq
    .else
    esp.vmul.s16    q0, q0, \wq
    esp.vmul.s16    q1, q1, \wq
    esp.vadd.s16    q4, q4, q0
    esp.vadd.s16    q5, q5, q1
    .endif
    .endm

/* q4 (first) or q4 += (the 8 pixels from p widened to 16 bits) * (the weight in every lane of wq), >> 7 per lane. t5
 * is scratch. */
    .macro  TERM8 p, wq, first
    mv      t5, \p
    esp.ld.128.usar.ip q0, t5, 16
    esp.vld.128.ip  q1, t5, 16
    esp.src.q       q0, q0, q1
    esp.zero.q      q1
    esp.vzip.8      q0, q1          /* q0: pixels 0-7 as 16-bit lanes */
    .if \first
    esp.vmul.s16    q4, q0, \wq
    .else
    esp.vmul.s16    q0, q0, \wq
    esp.vadd.s16    q4, q4, q0
    .endif
    .endm

/* q4/q5 = the bilinear interpolation of the 16 lanes from pixel p: p, p + 1, p + stride, p + stride + 1 with the
 * weights held in q2, q3, q6 and q7. t5 and t6 are scratch; p (not t5 or t6) and stride are kept. */
    .macro  INTERP p, stride
    TERM    \p, q2, 1
    addi    t6, \p, 1
    TERM    t6, q3, 0
    add     t6, \p, \stride
    TERM    t6, q6, 0
    addi    t6, t6, 1
    TERM    t6, q7, 0
    .endm

/* q4 = the bilinear interpolation of the 8 lanes from pixel p, as INTERP. */
    .macro  INTERP8 p, stride
    TERM8   \p, q2, 1
    addi    t6, \p, 1
    TERM8   t6, q3, 0
    add     t6, \p, \stride
    TERM8   t6, q6, 0
    addi    t6, t6, 1
    TERM8   t6, q7, 0
    .endm

/* SAR = 7 for ESP.VMUL.S16, and the four weights from w into every lane of q2, q3, q6 and q7. t5 is scratch. */
    .macro  SETUP w
    li      t5, 7
    esp.movx.w.sar t5
    esp.vldbc.16.ip q2, \w, 0
    addi    t5, \w, 2
    esp.vldbc.16.ip q3, t5, 0
    addi    t5, \w, 4
    esp.vldbc.16.ip q6, t5, 0
    addi    t5, \w, 6
    esp.vldbc.16.ip q7, t5, 0
    .endm

/* Zero the lanes outside the window of the row that ends at r. */
    .macro  CLEAR r
    .set    pad_at, -2 * PAD_LANES
    .rept   PAD_LANES
    sh      zero, pad_at(\r)
    .set    pad_at, pad_at + 2
    .endr
    .endm

/* XACC = the dot product of the DOT_VECTORS vectors (a window array) from pa and pb, which advance past them. q0, q1,
 * q4 and q5 and t0 are scratch. */
    .macro  DOT pa, pb
    esp.zero.xacc
    esp.vld.128.ip  q0, \pa, 16
    esp.vld.128.ip  q1, \pb, 16
    li      t0, (DOT_VECTORS - 1) / 2   /* DOT_VECTORS is odd */
1:
    esp.vld.128.ip  q4, \pa, 16
    esp.vmulas.s16.xacc.ld.ip q5, \pb, 16, q0, q1
    esp.vld.128.ip  q0, \pa, 16
    esp.vmulas.s16.xacc.ld.ip q1, \pb, 16, q4, q5
    addi    t0, t0, -1
    bnez    t0, 1b
    esp.vmulas.s16.xacc q0, q1
    .endm


/* void mvo_vector_lk_sums(const mvo_window_t *window, const uint8_t *src, int stride, const int16_t weights[4],
 *                         int64_t sums[2])
 * a0 window, a1 src (the window's top-left pixel), a2 stride, a3 weights, a4 sums. */
    .text
    .align  2
    .global mvo_vector_lk_sums
    .type   mvo_vector_lk_sums, @function
mvo_vector_lk_sums:
    addi    sp, sp, -WINDOW_ARRAY   /* [sp, sp + WINDOW_ARRAY): patch - interpolation, a window array */
    SETUP   a3
    mv      t2, a0                  /* the window, for gy */
    addi    a5, a0, WINDOW_ARRAY    /* gx; a0 walks the patch */
    mv      t3, sp
    esp.zero.xacc
    li      t1, SIDE                /* MVO_SIDE rows */
1:
    INTERP  a1, a2
    esp.vld.128.ip  q0, a0, 16
    esp.vsub.s16    q0, q0, q4      /* patch - interpolation, lanes 0-7 */
    esp.vst.128.ip  q0, t3, 16
    esp.vld.128.ip  q1, a5, 16
    esp.vmulas.s16.xacc q0, q1
    esp.vld.128.ip  q0, a0, 16
    esp.vsub.s16    q0, q0, q5      /* lanes 8-15 */
    esp.vst.128.ip  q0, t3, 16
    esp.vld.128.ip  q1, a5, 16
    esp.vmulas.s16.xacc q0, q1
    addi    a3, a1, 16              /* the weights are in q2, q3, q6, q7 now */
    INTERP8 a3, a2
    esp.vld.128.ip  q0, a0, 16
    esp.vsub.s16    q0, q0, q4      /* lanes 16-23 */
    esp.vst.128.ip  q0, t3, 16
    esp.vld.128.ip  q1, a5, 16
    esp.vmulas.s16.xacc q0, q1  /* the lanes outside the window meet gx = 0 */
    add     a1, a1, a2
    addi    t1, t1, -1
    bnez    t1, 1b
    esp.st.s.xacc.ip a4, 8          /* sums[0] */
    mv      t3, sp                  /* the differences again, times gy */
    addi    a5, t2, 2 * WINDOW_ARRAY
    DOT     t3, a5
    esp.st.s.xacc.ip a4, 8          /* sums[1] */
    addi    sp, sp, WINDOW_ARRAY
    ret
    .size   mvo_vector_lk_sums, . - mvo_vector_lk_sums


/* void mvo_vector_lk_template(mvo_window_t *window, const uint8_t *src, int stride, const int16_t weights[4],
 *                             int64_t g[3])
 * a0 window, a1 src (one up and one left of the window's top-left pixel), a2 stride, a3 weights, a4 g. The window's
 * pixel (i, j) is src + (j + 1) * stride + i + 1. */
    .text
    .align  2
    .global mvo_vector_lk_template
    .type   mvo_vector_lk_template, @function
mvo_vector_lk_template:
    addi    sp, sp, -INTERP_BYTES   /* [sp, sp + INTERP_BYTES): rows -1..SIDE of the interpolation */
    SETUP   a3

    /* 1. the interpolation at the window's columns for rows -1..SIDE, into the buffer */
    addi    a5, a1, 1               /* the pixel above the window's top-left one */
    mv      t3, sp
    li      t1, SIDE + 2
1:
    INTERP  a5, a2
    esp.vst.128.ip  q4, t3, 16
    esp.vst.128.ip  q5, t3, 16
    addi    a3, a5, 16
    INTERP8 a3, a2
    esp.vst.128.ip  q4, t3, 16
    add     a5, a5, a2
    addi    t1, t1, -1
    bnez    t1, 1b

    /* 2. patch = rows 0..SIDE-1, gy = row j + 1 - row j - 1 */
    addi    t3, sp, WINDOW_ROW      /* row 0 */
    mv      t4, a0                  /* patch */
    addi    a5, a0, 2 * WINDOW_ARRAY /* gy */
    li      t1, SIDE
2:
    mv      t5, t3
    esp.vld.128.ip  q0, t5, 16      /* row j */
    esp.vld.128.ip  q1, t5, 16
    esp.vst.128.ip  q0, t4, 16
    esp.vst.128.ip  q1, t4, 16
    esp.vld.128.ip  q0, t5, 16
    esp.vst.128.ip  q0, t4, 16
    CLEAR   t4
    esp.vld.128.ip  q0, t5, 16      /* row j + 1 */
    esp.vld.128.ip  q1, t5, 16
    addi    t6, t3, -WINDOW_ROW     /* row j - 1 */
    esp.vld.128.ip  q4, t6, 16
    esp.vld.128.ip  q5, t6, 16
    esp.vsub.s16    q0, q0, q4
    esp.vsub.s16    q1, q1, q5
    esp.vst.128.ip  q0, a5, 16
    esp.vst.128.ip  q1, a5, 16
    esp.vld.128.ip  q0, t5, 16
    esp.vld.128.ip  q4, t6, 16
    esp.vsub.s16    q0, q0, q4
    esp.vst.128.ip  q0, a5, 16
    CLEAR   a5
    addi    t3, t3, WINDOW_ROW
    addi    t1, t1, -1
    bnez    t1, 2b

    /* 3. gx = the interpolation a column right minus a column left, row by row; the right one waits in gx */
    add     t3, a1, a2              /* row 0, a column left of the window: src + stride */
    addi    t4, a0, WINDOW_ARRAY    /* gx */
    li      t1, SIDE
3:
    addi    a5, t3, 2
    INTERP  a5, a2                  /* a column right */
    mv      a1, t4
    esp.vst.128.ip  q4, a1, 16
    esp.vst.128.ip  q5, a1, 16
    addi    a3, a5, 16
    INTERP8 a3, a2
    esp.vst.128.ip  q4, a1, 16
    INTERP  t3, a2                  /* a column left */
    mv      a1, t4
    esp.vld.128.ip  q0, a1, 16
    esp.vld.128.ip  q1, a1, 16
    esp.vsub.s16    q0, q0, q4
    esp.vsub.s16    q1, q1, q5
    esp.vst.128.ip  q0, t4, 16
    esp.vst.128.ip  q1, t4, 16
    addi    a3, t3, 16
    INTERP8 a3, a2
    esp.vld.128.ip  q0, a1, 16
    esp.vsub.s16    q0, q0, q4
    esp.vst.128.ip  q0, t4, 16
    CLEAR   t4
    add     t3, t3, a2
    addi    t1, t1, -1
    bnez    t1, 3b

    /* 4. g = gx.gx, gx.gy, gy.gy */
    addi    a1, a0, WINDOW_ARRAY
    mv      a5, a1
    DOT     a1, a5
    esp.st.s.xacc.ip a4, 8
    addi    a1, a0, WINDOW_ARRAY
    addi    a5, a0, 2 * WINDOW_ARRAY
    DOT     a1, a5
    esp.st.s.xacc.ip a4, 8
    addi    a1, a0, 2 * WINDOW_ARRAY
    mv      a5, a1
    DOT     a1, a5
    esp.st.s.xacc.ip a4, 8
    addi    sp, sp, INTERP_BYTES
    ret
    .size   mvo_vector_lk_template, . - mvo_vector_lk_template
