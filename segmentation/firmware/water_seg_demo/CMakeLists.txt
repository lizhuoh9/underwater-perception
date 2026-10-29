cmake_minimum_required(VERSION 3.16)

set(EXTRA_COMPONENT_DIRS "${CMAKE_CURRENT_LIST_DIR}/../components")
include($ENV{IDF_PATH}/tools/cmake/project.cmake)

# esp-dl 3.3.12 allocates with MALLOC_CAP_SIMD, which ESP-IDF only defines from v5.4. With
# IDF 5.3 this demo maps it to MALLOC_CAP_8BIT; built that way, the ESP32-P4 passed esp-dl's
# self-test and gave the PC's mask (results/esp32p4_export/board_timing.log). Firmware for
# real use should run IDF >= 5.4.
if (IDF_VERSION_MAJOR EQUAL 5 AND IDF_VERSION_MINOR LESS 4)
    idf_build_set_property(COMPILE_DEFINITIONS "MALLOC_CAP_SIMD=MALLOC_CAP_8BIT" APPEND)
endif()

project(water_seg_demo)
