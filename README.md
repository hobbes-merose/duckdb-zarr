# duckdb-zarr

A DuckDB extension for querying Zarr-format scientific arrays with SQL.

## What It Does

**duckdb-zarr** extends DuckDB with the ability to query Zarr arrays directly using SQL. [Zarr](https://zarr.dev/) is a format for storing chunked, compressed N-dimensional arrays, widely used in scientific computing and big data applications—particularly in climate science, astronomy, genomics, and remote sensing.

With this extension, you can:

- Query Zarr array metadata (dimensions, chunk structure, data type, attributes)
- Read array data directly into DuckDB tables
- Combine Zarr data with other data sources using standard SQL joins

### The Pivot Concept

Zarr arrays are n-dimensional gridded data (like satellite imagery, climate model outputs, or sensor readings). This extension "pivots" these multi-dimensional arrays into relational tables—treating each cell in the array as a row, with coordinates as columns. This lets you use familiar SQL to query data that would otherwise require specialized array-processing tools.

### Why This Is Useful

Scientific datasets are often stored in Zarr format because it handles multi-terabyte to petabyte-scale arrays efficiently through chunking and compression. However, traditional SQL databases don't understand Zarr. This extension bridges that gap—allowing data scientists and engineers to:

- Query Zarr metadata without loading entire arrays into memory
- Perform SQL analytics on scientific data alongside structured data
- Use DuckDB's fast SQL engine to filter and aggregate chunked array data
- Integrate Zarr data pipelines with existing SQL-based workflows

This project is related to [xarray-sql](https://github.com/alxmrs/xarray-sql) and [zarr-datafusion](https://github.com/alxmrs/zarr-datafusion), which provide similar functionality for other query engines.

## Quick Start

### Installation

**Pre-built binaries** (coming soon):
```sql
INSTALL zarr FROM community;
LOAD zarr;
```

**Build from source** (see Development Setup below)

### Basic Usage

```sql
-- Load the extension
LOAD 'duckdb_zarr';

-- Read metadata from a Zarr array
SELECT * FROM read_zarr_metadata('/path/to/my/array.zarr');

-- Read array data as a table
SELECT * FROM read_zarr('/path/to/my/array.zarr');

-- Query specific dimensions or filter data
SELECT * FROM read_zarr('/path/to/my/array.zarr') 
WHERE dimension_0 > 100 AND dimension_1 < 50;
```

## Development Setup

### Prerequisites

- **C++ Compiler**: GCC 9+, Clang 10+, or MSVC 2019+
- **CMake**: Version 3.5 or higher
- **Git**
- **vcpkg**: For dependency management
- **DuckDB**: Source code (the extension builds against DuckDB)

### Dependencies

- **OpenSSL**: For secure connections
- **nlohmann_json**: JSON parsing library (via vcpkg)

### Build Instructions

```bash
# 1. Clone the repository
git clone https://github.com/hobbes-merose/duckdb-zarr.git
cd duckdb-zarr

# 2. Set up vcpkg (if not already configured)
export VCPKG_ROOT=/path/to/vcpkg

# 3. Configure with CMake
mkdir build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake \
         -DCMAKE_BUILD_TYPE=Release

# 4. Build the extension
cmake --build . --config Release

# 5. Run tests (optional)
ctest --output-on-failure
```

### Running the Extension

After building, load the extension in DuckDB:

```bash
# Start DuckDB CLI with the extension loaded
./build/release/duckdb -c "LOAD '/path/to/libduckdb_zarr.so'; SELECT * FROM read_zarr_metadata('/data/array.zarr');"
```

## Project Status

**Current Phase: Phase 1 - Zarr Metadata Parser**

This extension is under active development. Track our progress and planned features via the [issue tracker](https://github.com/hobbes-merose/duckdb-zarr/issues).

### Table Functions

- `read_zarr_metadata(path)` — Returns metadata about a Zarr array (dimensions, chunks, dtype, attributes)
- `read_zarr(path)` — Reads array data as a SQL table

## Contributing

Contributions are welcome! Whether you want to fix a bug, add a feature, or improve documentation, help is appreciated.

### How to Contribute

1. **Fork** the repository
2. **Create a feature branch**: `git checkout -b feature/your-feature-name`
3. **Make your changes** and add tests
4. **Push** to your fork and **open a Pull Request**

### Pull Request Process

1. Ensure all tests pass (`ctest --output-on-failure`)
2. Update documentation for any changed functionality
3. PRs should target the `main` branch
4. Follow the coding standards below

### Coding Standards

- **Language**: C++ (following DuckDB's C++ style)
- **Formatting**: Clang-format with DuckDB's configuration
- **Testing**: Add unit tests for new functionality
- **Commit Messages**: Clear, descriptive messages explaining *what* and *why*

### Related Projects

- [xarray-sql](https://github.com/alxmrs/xarray-sql) — xarray integration for DuckDB
- [zarr-datafusion](https://github.com/alxmrs/zarr-datafusion) — Zarr support for DataFusion
- [DuckDB](https://github.com/duckdb/duckdb) — In-process SQL OLAP database

## License

MIT License - see [LICENSE](LICENSE) for full details.
