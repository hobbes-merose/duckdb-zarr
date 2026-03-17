# Zarr Extension Design Document

**Project:** duckdb-zarr - DuckDB Extension for n-Dimensional Array Data
**Status:** Design Draft
**Version:** 0.1.0

---

## 1. Problem Statement

### Why Query Arrays with SQL?

Scientific data is increasingly stored in cloud-native formats like Zarr, which organize data as n-dimensional arrays with chunked, compressed storage. This is particularly common in:

- **Weather and climate data**: NOAA, ECMWF, and climate reanalysis datasets
- **Satellite imagery**: Landsat, Sentinel, MODIS
- **Oceanography**: Ocean model outputs, sea surface temperature
- **Medical imaging**: CT, MRI scans stored as 3D arrays
- **Machine learning**: Training data stored as tensors

### The Challenge

Current workflows require:

1. Loading entire arrays into memory (impossible for large datasets)
2. Using Python/R libraries with specialized APIs
3. Pre-processing data before analysis
4. Writing custom code for each query pattern

### The Opportunity

By exposing Zarr arrays as SQL tables in DuckDB, users can:

- Query massive datasets directly from cloud storage
- Use familiar SQL syntax for array operations
- Leverage DuckDB's pushdown predicates and column pruning
- Combine array data with traditional relational data
- Benefit from DuckDB's excellent performance characteristics

---

## 2. Architecture Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         DuckDB Core                             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Query Planning & Execution                  │   │
│  │  • SQL parsing, logical planning, optimization          │   │
│  │  • Predicate pushdown, column pruning                  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Zarr Extension (this project)               │   │
│  │  ┌─────────────────────────────────────────────────────┐ │   │
│  │  │              Table Function: read_zarr              │ │   │
│  │  │  • CreateTableFunction() - Register function       │ │   │
│  │  │  • bind() - Validate params, infer schema          │ │   │
│  │  │  • scan() - Return DataChunk iterator              │ │   │
│  │  │  • GlobalState - Parallel scan coordination        │ │   │
│  │  └─────────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
└──────────────────────────────┼───────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    duckdb-zarr Extension                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  DuckDB Integration Layer                                │   │
│  │  • Table function, vector conversion                    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  zarrs_ffi (single header library)                     │   │
│  │  • Metadata APIs (shape, dtype, chunks)                │   │
│  │  • Chunk retrieval & decompression (blosc2, zstd)      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Arrow-Based Pivot (~100 LOC)                           │   │
│  │  • Strided memory access for array→table transform     │   │
│  │  • Dictionary encoding for coordinates                  │   │
│  └─────────────────────────────────────────────────────────┘   │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Zarr Storage Layer                           │
│  • Remote: S3, GCS, Azure Blob                                  │
│  • Local: Filesystem                                           │
│  • Formats: Zarr v2/v3                                         │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Function Call**: User calls `SELECT * FROM read_zarr(path='...', array='temp')`
2. **Binding**: DuckDB calls table function `bind()` to validate parameters and define return schema
3. **Planning**: DuckDB optimizes query (predicate pushdown, column pruning)
4. **Scanning**: Zarr function returns a `DataChunk` iterator via parallel scan
5. **Conversion**: Zarr data (via zarrs_ffi + Arrow pivot) is converted to DuckDB vectors

---

## 2.2 How It Works

Zarr stores multidimensional data in chunked arrays. For example, a weather dataset might have:
- Coordinate arrays: time, lat, lon (1D)
- Data arrays: temperature, humidity (3D: time × lat × lon)

This library flattens the 3D structure into rows where each row represents one grid cell:

```
Zarr Store (3D)           →    SQL Table (2D)
─────────────────────────────────────────────────────
temperature[t, lat, lon]  →    | timestamp | lat | lon | temperature |
humidity[t, lat, lon]     →    | 0         | 0   | 0   | 43          |
                            →    | 0         | 0   | 1   | 51          |
                            →    | ...       | ... | ... | ...         |
```

### Assumptions

1. **Coordinates are 1D arrays** — Any array with a single dimension is treated as a coordinate
2. **Data variables are nD arrays** — Arrays with multiple dimensions. Their dimensionality must equal the number of coordinate arrays
3. **Cartesian product structure** — Data variables represent the Cartesian product of all coordinates
4. **Dimension ordering** — Coordinates are sorted alphabetically, and data variable dimensions follow this same order

### Features

- **Zarr v2 and v3 support**: Compatible with both Zarr format versions
- **Schema inference**: Automatically infers Arrow schema from Zarr metadata
- **Projection pushdown**: Only reads arrays that are needed for the query
- **Memory efficient coordinates**: Uses Arrow DictionaryArray for coordinate columns (~75% memory savings)
- **SQL interface**: Full DataFusion SQL support (filtering, aggregation, joins, etc.)

---

## 2.1 Implementation Approach

### zarrs_ffi Integration

We are using [zarrs_ffi](https://zarrs.dev/zarrs_ffi/) as the Zarr reader instead of building a native C++ implementation from scratch. This C library provides C/C++ bindings for the [zarrs](https://github.com/LDeakin/zarrs) Rust crate, which is a high-performance Zarr implementation.

**Why zarrs_ffi?**

- **Comprehensive codec support**: Built-in support for blosc2, zstd, gzip, lz4, and other codecs without needing to reimplement decompression
- **Zarr v2 and v3 support**: Native support for both Zarr format versions including sharded arrays
- **Active development**: Well-maintained with proper versioning and error handling
- **Remote storage**: Support for filesystem storage (can be extended to S3/GCS via storage transformers)
- **Performance**: Optimized Rust implementation for chunk retrieval and decompression

**zarrs_ffi Build Instructions:**

zarrs_ffi is distributed as a single-header C library. To build it for use in the duckdb-zarr extension:

```bash
# Clone the zarrs repository
git clone https://github.com/LDeakin/zarrs.git
cd zarrs

# Build the FFI bindings (produces libzarrs.a static library)
cargo build --release --features=ffi

# The static library will be at:
# target/release/libzarrs.a
```

**Linking with DuckDB Extension:**

Add the following to your extension's CMakeLists.txt:
```cmake
# Link against zarrs_ffi static library
target_link_libraries(duckdb_zarr_extension INTERFACE zarrs)
```

Alternatively, use the prebuilt release artifacts from the [zarrs releases page](https://github.com/LDeakin/zarrs/releases).

**What zarrs_ffi provides:**

| Feature | zarrs_ffi API |
|---------|---------------|
| Array opening | `zarrsOpenArrayRW()` |
| Metadata APIs | `zarrsArrayGetDimensionality()`, `zarrsArrayGetShape()`, `zarrsArrayGetDataType()` |
| Chunk information | `zarrsArrayGetChunkGridShape()`, `zarrsArrayGetChunkOrigin()`, `zarrsArrayGetChunkShape()` |
| Chunk retrieval | `zarrsArrayRetrieveChunk()`, `zarrsArrayRetrieveSubset()` |
| Sharding support | `zarrsCreateShardIndexCache()` for efficient shard access |

**What we still need to build ourselves:**

1. **DuckDB integration**: Table function registration, vector conversion, predicate pushdown
2. **Arrow-based pivot**: Convert n-dimensional array data to tabular format using Arrow buffers

This hybrid approach gives us the best of both worlds: production-ready chunk handling and metadata APIs from zarrs_ffi while maintaining full control over the SQL interface and data transformation logic.

### Implementation Plan

The implementation is divided into 3 phases, targeting approximately **400 lines of C++ code** total (reduced from ~950 LOC):

#### Phase 1: DuckDB Integration & zarrs_ffi Wrapper (~150 LOC)
- Table function registration (read_zarr)
- Schema inference from zarrs_ffi metadata APIs
- DataChunk conversion to DuckDB vectors
- Predicate pushdown support
- Projection pushdown (only read requested columns)

#### Phase 2: Chunk Reader via zarrs_ffi (~100 LOC)
- Use zarrs_ffi C library for chunk retrieval and decompression
- Open arrays with `zarrsOpenArrayRW()` for read access
- Retrieve chunk data with `zarrsArrayRetrieveChunk()` (individual chunks)
- Retrieve subsets with `zarrsArrayRetrieveSubset()` (optimized for query pushdown)
- Leverage built-in codec support: blosc2, zstd, gzip, lz4
- Handle sharded arrays with `zarrsCreateShardIndexCache()`

#### Phase 3: Arrow-Based Pivot (~100 LOC)
- Core array-to-table transform algorithm using Arrow
- Use Arrow buffers directly for efficiency
- Dictionary encoding for coordinate columns (~75% memory savings)
- Strided memory access to convert chunk data to row-oriented format

### Dependencies

| Dependency | Purpose | Status |
|------------|---------|--------|
| zarrs_ffi | Zarr reading (chunk retrieval, decompression, Zarr v2/v3, metadata APIs) | Build-time: cargo/rust, Runtime: prebuilt static library |
| Arrow C++ | Arrow integration | Already a DuckDB dependency |
| DuckDB yyjson | JSON metadata parsing | Already in DuckDB tree |

**Note:** We no longer need nlohmann/json, c-blosc2, or libzstd as separate dependencies - zarrs_ffi handles all compression/decompression internally.

**Build Integration Note:**

Using zarrs_ffi requires adding Rust/cargo to the DuckDB extension build:
- **Build-time**: Compile zarrs_ffi as a static library and link into the extension
- **Runtime**: zarrs_ffi is embedded in the extension binary (no separate runtime dependencies)
- **Alternative**: Use the prebuilt zarrs_ffi static library from the zarrs release artifacts

The zarrs_ffi library handles:
- Chunk retrieval from storage
- Decompression (blosc2, zstd, gzip, lz4 - all codecs built into zarrs)
- Zarr v2 and v3 metadata and data handling
- Sharded array support
- Metadata APIs (shape, dtype, dimensionality, chunk shape) - eliminates need for custom JSON parsing

#### zarrs_ffi Build Instructions

zarrs_ffi is a single-header C library that wraps the zarrs Rust crate. To build it:

1. **Install Rust** (if not already installed):
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. **Build zarrs_ffi**:
   ```bash
   # Clone and build the zarrs_ffi crate
   git clone https://github.com/LDeakin/zarrs_ffi.git
   cd zarrs_ffi
   cargo build --release
   
   # This produces a static library (target/release/libzarrs_ffi.a)
   ```

3. **Integration**: The resulting static library can be linked directly into the DuckDB extension. No runtime Rust dependencies needed.

**Key APIs**:
```cpp
// Open array
int zarrsOpenArrayRW(const char *storage, const char *path, ZarrsArray *array);

// Metadata APIs (eliminate need for custom JSON parsing)
int zarrsArrayGetDimensionality(ZarrsArray array, size_t *dimensionality);
int zarrsArrayGetShape(ZarrsArray array, uint64_t *shape);
int zarrsArrayGetDataType(ZarrsArray array, char **dtype);
int zarrsArrayGetChunkGridShape(ZarrsArray array, uint64_t *chunk_grid_shape);

// Chunk retrieval
int zarrsArrayRetrieveChunk(ZarrsArray array, size_t dimensionality,
                            const uint64_t *indices, size_t chunk_size, uint8_t *buffer);
```

### The Pivot Algorithm

The pivot algorithm is the heart of the transformation - converting n-dimensional array data into a tabular format for SQL queries. The core implementation uses strided memory access (about 30 lines):

For each cell in the output table:
1. Calculate the n-dimensional index from the row position
2. Determine which chunk contains that index
3. Calculate the offset within the chunk
4. Apply strided access to read the value

This is simply strided memory access - the same technique used by NumPy arrays. No complex data structures or algorithms required.

#### zarrs_ffi Integration Pattern

The zarrs_ffi integration follows a clean separation of concerns:

```
┌─────────────────────────────────────────────────────────────────┐
│                    duckdb-zarr Extension                        │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  DuckDB Integration Layer                                  │ │
│  │  • Table function registration (read_zarr)                │ │
│  │  • Schema inference from metadata JSON                    │ │
│  │  • Vector construction from chunk data                     │ │
│  └───────────────────────────────────────────────────────────┘ │
│                              │                                   │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  Pivot Algorithm (custom C++)                             │ │
│  │  • Strided memory access for array→table transform       │ │
│  │  • Coordinate expansion (DictionaryArray optimization)   │ │
│  └───────────────────────────────────────────────────────────┘ │
│                              │                                   │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  zarrs_ffi (C library via FFI)                           │ │
│  │  • zarrsOpenArrayRW() - Open array handle                │ │
│  │  • zarrsArrayGetMetadataString() - Get metadata JSON     │ │
│  │  • zarrsArrayRetrieveChunk() - Get decompressed chunk    │ │
│  │  • zarrsArrayRetrieveSubset() - Get decompressed subset   │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

**Typical query flow:**

1. Call `zarrsOpenArrayRW(storage, path, &array)` to open the Zarr array
2. Call `zarrsArrayGetMetadataString(array, true, &metadata_json)` to get metadata
3. Parse metadata JSON to infer DuckDB schema (dtype → SQL type, shape → row estimation)
4. For each query batch:
   - Determine which chunks overlap with the query region
   - Call `zarrsArrayRetrieveChunk()` or `zarrsArrayRetrieveSubset()` to get decompressed data
   - Apply pivot algorithm to convert chunk data to row format
   - Convert to DuckDB vectors

**Key insight:** zarrs_ffi returns *already decompressed* data. The pivot algorithm operates on plain in-memory buffers with no additional decompression overhead.

#### Memory-Efficient Coordinate Expansion

The zarr-datafusion crate demonstrates a key optimization: using DictionaryArray for coordinate columns (~75% memory savings). When flattening n-dimensional data, coordinate values repeat many times. Using dictionary encoding instead of plain arrays significantly reduces memory usage:

| Approach | Memory for 10M rows |
|----------|-------------------|
| Plain arrays | ~300 MB |
| DictionaryArray | ~75 MB |

Our C++ implementation will use similar techniques for memory efficiency.

#### Comparison with xarray-sql's iter_record_batches()

Our C++ implementation is inspired by xarray-sql's production-ready `iter_record_batches()` method, which uses strided index arithmetic to convert array data to tabular format. The key difference is that we implement this approach in native C++ to eliminate Python interpreter overhead:

| Aspect | xarray-sql `iter_record_batches()` | duckdb-zarr (C++) |
|--------|-----------------------------------|-----------|
| **Implementation** | Python with numpy index arithmetic | Native C++ with strided memory access |
| **Memory** | Creates intermediate numpy arrays (~2x overhead) | Zero-copy where possible, ~2x overhead |
| **Dependencies** | Requires xarray, numpy, PyArrow | Pure C++ with DuckDB integration |

The key insight from xarray-sql's `iter_record_batches()` is their use of strided index arithmetic: `coord_idx = (row_idx // stride[k]) % shape[k]`. Our C++ implementation applies the same principle but eliminates the Python interpreter overhead.

## 3. Key Interfaces

### DuckDB Extension Entry Point

The extension follows the standard DuckDB extension pattern:

```cpp
#include <duckdb.hpp>

extern "C" {
DUCKDB_CPP_EXTENSION_ENTRY(zarr, loader) {
    duckdb::LoadInternal(loader);
}
}
```

This registers the extension and its functions with DuckDB at load time.

### Table Function Implementation

DuckDB extensions use **Table Functions** to expose custom data sources. The core pattern:

```cpp
#include <duckdb/function/table_function.hpp>

using namespace duckdb;

// 1. Define the table function
static void ZarrScanFunction(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
    auto &bind_data = (ZarrScanBindData &)*data.bind_data;
    auto &global_state = (ZarrScanGlobalState &)*data.global_state;

    // Read from Zarr via native C++ implementation
    // Convert to DataChunk (DuckDB vectors)
    // ...
}

// 2. Bind function - validates params and defines schema
static unique_ptr<FunctionData> ZarrScanBind(ClientContext &context, TableFunctionBindInput &input,
                                              vector<LogicalType> &return_types, vector<string> &names) {
    // Extract path, array name from input.named_params
    // Read Zarr metadata to determine schema
    // Populate return_types and names (dimension columns + data columns)
    return make_uniq<ZarrScanBindData>(...);
}

// 3. Register the table function
void LoadInternal(ExtensionLoader &loader) {
    TableFunction read_zarr("read_zarr", {}, ZarrScanFunction, ZarrScanBind);
    read_zarr.named_params["path"] = LogicalType::VARCHAR;
    read_zarr.named_params["array"] = LogicalType::VARCHAR;
    loader.RegisterFunction(read_zarr);
}
```

### Parallel Table Scan (Advanced)

For performance with large datasets, implement `GlobalState` for parallel coordination:

```cpp
// GlobalState for parallel scanning
struct ZarrScanGlobalState : public GlobalState {
    idx_t current_chunk_idx = 0;
    mutex lock;
    // Track which Zarr chunks have been read
};

// Table function with parallel support
static unique_ptr<GlobalState> ZarrScanInitGlobal(ClientContext &context, TableFunctionInitInput &input) {
    return make_uniq<ZarrScanGlobalState>();
}

static void ZarrScanFunction(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
    auto &global_state = (ZarrScanGlobalState &)*data.global_state;

    // Thread-safe chunk reading
    lock_guard<mutex> lg(global_state.lock);
    // Read next chunk from Zarr...
}
```

### Zarr to Table Mapping

A Zarr array maps to a SQL table as follows:

| Zarr Concept | SQL Concept |
|---------------|--------------|
| Array | Table |
| Dimensions (axes) | Columns (coordinate dimensions) |
| Attributes | Table metadata / comments |
| Chunks | Internal storage (not directly visible) |
| Data variables | Non-coordinate columns |

#### Example: Temperature Array

```
Zarr Array (temperature):
  Shape: (time=1000, lat=180, lon=360)
  Chunks: (100, 30, 60)
  dtype: float32

SQL Table (temperature):
  time    | lat    | lon    | temp
  --------|--------|--------|-------
  0       | -90.0  | 0.0    | 273.15
  0       | -89.0  | 0.0    | 272.90
  ...     | ...    | ...    | ...
```

### Scan Execution Flow

```cpp
// In table function implementation
static void ZarrScanFunction(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
    auto &bind_data = (ZarrScanBindData &)*data.bind_data;

    // 1. Extract parameters from bind data
    auto &zarr_path = bind_data.zarr_path;
    auto &array_name = bind_data.array_name;
    auto column_ids = output.ColumnIds();

    // 2. Use native C++ implementation to read Zarr data
    //    (ZarrMetadataParser -> ChunkReader -> PivotAlgorithm)
    auto zarr_metadata = ZarrMetadataParser::Parse(zarr_path, array_name);
    auto chunk_reader = ChunkReader(zarr_path, zarr_metadata);
    auto pivot = PivotAlgorithm(zarr_metadata);

    // 3. Read chunks and pivot to row-oriented format
    auto chunks = chunk_reader.ReadRelevantChunks(bind_data.predicate);
    pivot.PivotToDuckDBVectors(chunks, output, column_ids);
}
```

### Predicate Pushdown

DuckDB automatically pushes down predicates to the table function. Access them via the `data` parameter:

```cpp
static unique_ptr<FunctionData> ZarrScanBind(ClientContext &context, TableFunctionBindInput &input,
                                              vector<LogicalType> &return_types, vector<string> &names) {
    // DuckDB passes filter expressions to the scan function
    // We extract them from input.filters

    // Example: WHERE time BETWEEN 100 AND 200 AND temp > 300
    // input.filters contains these expressions
    // Convert to Zarr predicate for chunk selection

    auto predicate = ConvertDuckDBFilters(input.filters, schema);
    return make_uniq<ZarrScanBindData>(..., std::move(predicate));
}
```

---

## 4. User-Facing API

### Querying Zarr Arrays

The extension exposes a table function `read_zarr` that users call directly:

```sql
-- Basic scan (infers schema from Zarr metadata)
SELECT * FROM read_zarr(
    path => 's3://noaa-gfs-pds/gfs.20210101/00/temp',
    array => 'temperature'
) LIMIT 100;

-- With explicit column selection
SELECT time, lat, lon, temperature
FROM read_zarr(
    path => 's3://my-bucket/weather.zarr',
    array => 'surface'
)
WHERE time = 0 AND lat BETWEEN 40 AND 50;

-- Creating a view for repeated queries
CREATE VIEW temperature AS
SELECT * FROM read_zarr(
    path => 's3://noaa-gfs-pds/gfs.20210101/00/temp',
    array => 'temperature'
);
```

### Querying Array Data

```sql
-- Basic query
SELECT * FROM temperature WHERE time = 0 LIMIT 100;

-- Aggregate over dimensions
SELECT AVG(temperature) FROM temperature
WHERE time BETWEEN 0 AND 100
GROUP BY lat, lon;

-- Spatial query (DuckDB's GIS functions work!)
SELECT ST_Point(lon, lat) as geom, temperature
FROM temperature
WHERE time = 0 AND lat BETWEEN 40 AND 50 AND lon BETWEEN -120 AND -100;

-- Time series extraction
SELECT time, temperature FROM temperature
WHERE lat = 45.0 AND lon = -122.0
ORDER BY time;
```

### Extension Management

```sql
-- Load extension
LOAD zarr;

-- Show extension version
SELECT zarr_version();

-- List arrays in a Zarr store
SELECT * FROM read_zarr_metadata('/path/to/store.zarr');
```

---

## 5. Dependencies

### Core Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| DuckDB | (submodule) | Core database, extension framework |
| zarrs_ffi | (build-time) | Zarr reading (chunk retrieval, decompression, Zarr v2/v3, metadata APIs) |
| Arrow C++ | (in DuckDB tree) | Columnar data format integration |

### Dependency Graph

```
duckdb
  │
  └── duckdb_zarr_extension
        │
        ├── zarrs_ffi (Zarr chunk retrieval, decompression, metadata APIs)
        │     │
        │     └── Built via cargo, linked as static library
        │
        └── Arrow C++ (data conversion)
              │
              └── Already a DuckDB dependency

```

### Simplified Architecture Components

With zarrs_ffi handling chunk retrieval and metadata, our implementation focuses on the integration layer:

```cpp
// DuckDB Integration Layer - Table function and vector conversion
class ZarrScanFunction {
public:
    static unique_ptr<FunctionData> Bind(ClientContext &context, ...);
    static void Function(ClientContext &context, TableFunctionInput &data, DataChunk &output);
};

// Arrow-Based Pivot - Array to table transform
class ArrowPivot {
public:
    ArrowPivot(const ZarrArrayMetadata &metadata);
    void PivotToDuckDBVectors(const uint8_t* chunk_data, DataChunk &output);
    // Uses Arrow buffers directly with dictionary encoding for coordinates
};
```

**Key simplifications:**

1. **No custom JSON parsing**: Use zarrs_ffi metadata APIs (`zarrsArrayGetShape()`, `zarrsArrayGetDataType()`) instead
2. **No custom decompression**: zarrs_ffi returns decompressed data
3. **Arrow-native pivot**: Use Arrow buffers directly for efficiency

---

## 6. Roadmap / Milestones

### Milestone 1: Extension Foundation (Week 1-2)

**Goal**: Basic extension structure with Zarr discovery

- [ ] Set up C++ dependencies (nlohmann/json, c-blosc2)
- [ ] Implement basic table function registration
- [ ] Implement `read_zarr_metadata` function to discover arrays in a store
- [ ] Read Zarr metadata and expose as DuckDB result
- [ ] Basic test with local filesystem Zarr

**Deliverable**: `SELECT * FROM read_zarr_metadata('/path/to/zarr.zarr')`

### Milestone 2: Table Function Implementation (Week 3-4)

**Goal**: Basic table scans work

- [ ] Implement table function interface (bind + scan)
- [ ] Integrate native C++ ZarrMetadataParser, ChunkReader, PivotAlgorithm
- [ ] Convert native C++ data to DuckDB vectors
- [ ] Support simple queries (SELECT without filters)
- [ ] Test with small Zarr arrays

**Deliverable**: `SELECT * FROM read_zarr(path='/path', array='temp')` returns data

### Milestone 3: Predicate Pushdown (Week 5-6)

**Goal**: Efficient queries with filters

- [ ] Access DuckDB filter expressions via TableFunctionBindInput
- [ ] Translate to native C++ predicates for chunk selection
- [ ] Implement time/dimension range filtering
- [ ] Column pruning (only read requested columns)
- [ ] Chunk cache for frequently accessed data
- [ ] Benchmark with large datasets

**Deliverable**: `SELECT * FROM read_zarr(...) WHERE time BETWEEN x AND y` only reads relevant chunks

### Milestone 4: Cloud Storage Support (Week 7-8)

**Goal**: Works with S3, GCS, Azure

- [ ] Configure object_store for cloud providers
- [ ] Handle authentication (IAM, credentials)
- [ ] Async I/O for remote reads
- [ ] Connection pooling
- [ ] Test with public cloud datasets

**Deliverable**: Query data directly from `s3://bucket/data.zarr`

### Milestone 5: Advanced Features (Week 9-10)

**Goal**: Production-ready features

- [ ] Support for Zarr v3 format
- [ ] Multi-variable arrays
- [ ] Coordinate variable handling
- [ ] Null/missing value handling
- [ ] Error handling and diagnostics

**Deliverable**: Full-featured production extension

### Milestone 6: Integration & Distribution (Week 11-12)

**Goal**: Release-ready

- [ ] CI/CD pipeline for multi-platform builds
- [ ] Submit to DuckDB community extensions
- [ ] Documentation and examples
- [ ] Performance benchmarking
- [ ] Compatibility testing with DuckDB versions

**Deliverable**: `INSTALL zarr FROM community; LOAD zarr;` followed by `SELECT * FROM read_zarr(...)`

---

## 7. Technical Considerations

### Chunking Strategy

Zarr arrays use chunked storage. The extension must:
1. **Read minimal chunks**: Only fetch chunks that contain data matching query predicates
2. **Cache chunks**: Keep frequently accessed chunks in memory
3. **Parallel reads**: Read multiple chunks concurrently from remote stores

### Type Mapping

| Zarr dtype | DuckDB Type |
|------------|-------------|
| int8 | TINYINT |
| int16 | SMALLINT |
| int32 | INTEGER |
| int64 | BIGINT |
| uint8 | UTINYINT |
| float32 | FLOAT |
| float64 | DOUBLE |
| string | VARCHAR |
| bytes | BLOB |

### Memory Management

- **DataChunk-based**: Convert Arrow RecordBatches to DuckDB DataChunks
- **Streaming**: Process large arrays in chunks to avoid memory pressure
- **Parallel execution**: Use DuckDB's GlobalState for parallel scan coordination

### Extension Loading

The extension uses DuckDB's standard extension mechanism:

```cpp
extern "C" {
DUCKDB_CPP_EXTENSION_ENTRY(zarr, loader) {
    duckdb::LoadInternal(loader);
}
}
```

This allows:
- Static linking into DuckDB binary
- Dynamic loading as `.duckdb_extension` file
- Installation via `INSTALL zarr FROM community`

---

## 8. Related Projects

- **xarray-sql** (https://github.com/alxmrs/xarray-sql): Parent project, Python library
- **zarr-datafusion** (https://lib.rs/crates/zarr-datafusion): Rust crate for Zarr + DataFusion
- **DuckDB** (https://github.com/duckdb/duckdb): Analytical database
- **DuckDB Extension Template** (https://github.com/duckdb/extension-template): This project's foundation
- **Zarr** (https://zarr.dev): N-dimensional array storage format
- **Apache Arrow** (https://arrow.apache.org): Columnar data format
- **DataFusion** (https://datafusion.apache.org): Query engine
- **[IceChunk](https://icechunk.io/)** (https://icechunk.io/): Cloud-native columnar storage (potential future backend)

---

## 9. Open Questions

1. **Multidimensional primary keys**: How to handle tables where the "primary key" is a tuple of dimension values?

2. **Schema evolution**: How to handle Zarr arrays that change over time (new variables, changed chunks)?

3. **Nested arrays**: Zarr supports arrays of arrays. Should these be exposed as JSON or nested DuckDB types?

4. **Write support**: Read-only initially, but should we plan for write support?

5. **Aggregation pushdown**: Should aggregations (SUM, AVG) be pushed to the Zarr layer where possible?

6. **Time handling**: Zarr doesn't have a native datetime type. How to handle time coordinates?

---

*Document Version: 0.1.0*
*Last Updated: 2026-03-13*
