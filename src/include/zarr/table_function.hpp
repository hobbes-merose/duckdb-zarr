#pragma once

#include "duckdb.hpp"
#include "zarr/metadata.hpp"
#include "zarr/chunk.hpp"
#include "zarr/pivot.hpp"
#include <string>
#include <memory>

namespace duckdb {
namespace zarr {

/// Table function binding information
struct ZarrBindData {
	std::string path;                      // Path to Zarr store
	std::string array_name;                 // Array name
	ZarrArrayMetadata metadata;            // Parsed metadata
	std::vector<CoordinateArray> coordinates; // Coordinate arrays
	vector<string> column_names;           // Output column names
	vector<LogicalType> column_types;      // Output column types
};

/// Global state for parallel scanning
class ZarrGlobalState {
public:
	std::vector<std::vector<idx_t>> chunk_indices;  // All chunk indices to scan
	size_t current_chunk;                            // Current chunk position
	Mutex lock;                                     // For thread-safe access
	
	ZarrGlobalState(const ZarrArrayMetadata& metadata);
	
	/// Get the next chunk indices to scan
	bool GetNextChunk(std::vector<idx_t>& indices);
};

/// Table function for reading Zarr arrays
class ZarrTableFunction {
public:
	/// Register the read_zarr table function with DuckDB
	static void Register(DatabaseInstance& db);
	
private:
	/// Table function bind callback
	static unique_ptr<FunctionData> Bind(ClientContext& context, TableFunctionBindInput& input,
	                                      vector<LogicalType>& return_types, vector<string>& names);
	
	/// Table function init global state callback
	static unique_ptr<GlobalState> InitGlobalState(ClientContext& context, 
	                                                TableFunctionInitInput& input);
	
	/// Table function scan callback
	static void Scan(ClientContext& context, TableFunctionScanInput& input, DataChunk& output);
	
	/// Scan a single chunk and populate output
	static void ScanChunk(ClientContext& context, const ZarrBindData& bind_data,
	                     const std::vector<idx_t>& chunk_indices, DataChunk& output);
	
	/// Infer schema from Zarr store
	static void InferSchema(const std::string& path, const std::string& array_name,
	                       ZarrArrayMetadata& metadata, vector<string>& names,
	                       vector<LogicalType>& types);
	
	/// Load coordinate arrays from Zarr store
	static std::vector<CoordinateArray> LoadCoordinates(const std::string& path,
	                                                     const ZarrArrayMetadata& metadata);
};

} // namespace zarr
} // namespace duckdb
