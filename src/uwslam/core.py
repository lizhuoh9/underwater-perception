"""How the C core (core/mono_vo*.c, core/mono_map.c, core/mono_cloud.c) is compiled on the PC.

The local gcc builds it into a shared library with the floating-point flags the ESP32-S3 build uses
(-ffp-contract=off, no fast-math), so the board reproduces the PC's results bit for bit.
"""

import sys

CFLAGS = ["-O2", "-std=c99", "-Wall", "-Wextra", "-Werror", "-pedantic", "-ffp-contract=off"]
LIBRARY_SUFFIX, LINK_FLAGS = {"win32": (".dll", ["-static-libgcc"]), "darwin": (".dylib", [])}.get(sys.platform, (".so", ["-fPIC"]))
