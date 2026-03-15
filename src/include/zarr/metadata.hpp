#pragma once

#include "duckdb.hpp"
#include <string>
#include <vector>
#include <optional>
#include <variant>

namespace duckdb {
namespace zarr {

/// Represents the data type of a Zarr array element
enum class ZarrDtype {
	INT8,
	INT16,
	INT32,
	INT64,
	UINT8,
	UINT16,
	UINT32,
	UINT64,
	FLOAT32,
	FLOAT64,
	STRING,
	BYTES,
	BOOL,
	COMPLEX64,
	COMPLEX128,
	UNKNOWN
};

/// Represents a codec configuration (for Zarr v3)
struct CodecConfig {
	std::string name;
	std::string configuration;
};

/// Represents the metadata for a Zarr array
struct ZarrArrayMetadata {
	// Common fields
	std::string name;
	std::vector<idx_t> shape;
	ZarrDtype dtype;
	std::string zarr_format; // "v2" or "v3"

	// Chunking
	std::vector<idx_t> chunks;

	// Fill value
	std::optional<std::string> fill_value;
	std::optional<idx_t> fill_value_native;

	// Compression (v2)
	std::optional<std::string> compressor; // JSON string of compressor config

	// Codecs (v3)
	std::vector<CodecConfig> codecs;

	// Order (v2): C (row-major) or F (column-major)
	std::optional<std::string> order;

	// Filters (v2)
	std::optional<std::string> filters;

	// Storage transformer (v3)
	std::optional<std::string> storage_transformers;

	// Attributes (user-defined metadata)
	std::optional<std::string> attributes;

	// Computed fields
	idx_t element_size; // bytes per element
	idx_t chunk_size;   // total elements per chunk
	idx_t num_chunks;   // total number of chunks
};

/// Metadata parser for Zarr arrays
class ZarrMetadataParser {
public:
	/// Parse metadata from a zarray file path (Zarr v2)
	static ZarrArrayMetadata ParseV2(const std::string& zarray_path);

	/// Parse metadata from a z.json file path (Zarr v3)
	static ZarrArrayMetadata ParseV3(const std::string& z_json_path);

	/// Auto-detect format and parse metadata
	static ZarrArrayMetadata Parse(const std::string& path);

	/// Parse metadata from JSON string (for testing)
	static std::string ParseFromString(const std::string& json_str);

	/// Convert Zarr dtype string to enum
	static ZarrDtype ParseDtype(const std::string& dtype_str);

	/// Convert ZarrDtype enum to DuckDB logical type
	static LogicalType ToDuckDBType(ZarrDtype dtype);

private:
	/// Validate metadata consistency
	static void Validate(const ZarrArrayMetadata& metadata);
};

} // namespace zarr
} // namespace duckdb
