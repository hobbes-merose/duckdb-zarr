cmake_minimum_required(VERSION 3.5)

# Set extension name here
set(TARGET_NAME zarr)

# DuckDB's extension distribution supports vcpkg. As such, dependencies can be
# added in ./vcpkg.json and then used in cmake with find_package. Feel free to
# remove or replace with other dependencies. Note that it should also be removed
# from vcpkg.json to prevent needlessly installing it..
find_package(OpenSSL REQUIRED)

# yyjson is included from DuckDB's third_party directory
set(YYJSON_INCLUDE_DIR "${DUCKDB_SRCDIR}/third_party/yyjson/include")

# Include DuckDB headers for yyjson
include_directories(${DUCKDB_SRCDIR}/third_party/yyjson/include)
# Include compression library headers
include_directories(${DUCKDB_SRCDIR}/third_party/lz4)
include_directories(${DUCKDB_SRCDIR}/third_party/zstd/include)
include_directories(${DUCKDB_SRCDIR}/third_party/miniz)

# Arrow is built into DuckDB core - include directories are handled by DuckDB
# The Arrow headers are already available through DuckDB's build system

set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

project(${TARGET_NAME})
include_directories(src/include)
include_directories(${YYJSON_INCLUDE_DIR})

set(EXTENSION_SOURCES
    src/zarr_extension.cpp
    src/zarr_metadata.cpp
    src/zarr/metadata.cpp
    src/zarr/chunk.cpp
)

set(EXTENSION_HEADERS
    src/include/zarr_extension.hpp
    src/include/zarr/metadata.hpp
    src/include/zarr/chunk.hpp
    src/zarr/arrow_pivot.hpp
)

build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES} ${EXTENSION_HEADERS})
build_loadable_extension(${TARGET_NAME} " " ${EXTENSION_SOURCES} ${EXTENSION_HEADERS})

# Link OpenSSL and yyjson in both the static library as the loadable extension
target_link_libraries(${EXTENSION_NAME} OpenSSL::SSL OpenSSL::Crypto
                      duckdb_yyjson)
target_link_libraries(${LOADABLE_EXTENSION_NAME} OpenSSL::SSL OpenSSL::Crypto
                      duckdb_yyjson)

install(
  TARGETS ${EXTENSION_NAME}
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")
