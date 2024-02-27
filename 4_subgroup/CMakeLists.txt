cmake_minimum_required(VERSION 3.15.1)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} EXAMPLE_SCR)

add_executable(subgroup ${EXAMPLE_SCR})