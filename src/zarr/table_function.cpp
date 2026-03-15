#include "zarr/table_function.hpp"
#include "duckdb/common/file_system.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/database.hpp"
#include <filesystem>

namespace duckdb {
namespace zarr {

// ============================================================================
// ZarrGlobalState Implementation
// ============================================================================

ZarrGlobalState::ZarrGlobalState(const ZarrArrayMetadata& metadata)
    : current_chunk(0) {
	
	// Calculate number of chunks in each dimension
	vector<idx_t> num_chunks;
	for (size_t i = 0; i < metadata.shape.size(); i++) {
		idx_t n = (metadata.shape[i] + metadata.chunks[i] - 1) / metadata.chunks[i];
		num_chunks.push_back(n);
	}
	
	// Generate all chunk indices using recursion
	function<void(size_t, vector<idx_t>)> generate;
	generate = [&](size_t dim, vector<idx_t> indices) {
		if (dim == num_chunks.size()) {
			chunk_indices.push_back(indices);
		} else {
			for (idx_t i = 0; i < num_chunks[dim]; i++) {
				indices.push_back(i);
				generate(dim + 1, indices);
				indices.pop_back();
			}
		}
	};
	
	generate(0, {});
}

bool ZarrGlobalState::GetNextChunk(std::vector<idx_t>& indices) {
	lock.lock();
	if (current_chunk >= chunk_indices.size()) {
		lock.unlock();
		return false;
	}
	indices = chunk_indices[current_chunk];
	current_chunk++;
	lock.unlock();
	return true;
}

// ============================================================================
// ZarrTableFunction Implementation
// ============================================================================

void ZarrTableFunction::Register(DatabaseInstance& db) {
	// Create the table function
	TableFunction read_zarr("read_zarr", 
	                        {LogicalType::VARCHAR, LogicalType::VARCHAR},  // path, array_name
	                        Bind,
	                        InitGlobalState,
	                        Scan);
	
	// Add named parameters
	read_zarr.named_parameters["path"] = LogicalType::VARCHAR;
	read_zarr.named_parameters["array"] = LogicalType::VARCHAR;
	
	// Register with DuckDB
	auto& catalog = Catalog::GetSystemCatalog(db);
	CreateTableFunctionInfo info(read_zarr);
	catalog.CreateTableFunction(db, info);
}

unique_ptr<FunctionData> ZarrTableFunction::Bind(
    ClientContext& context,
    TableFunctionBindInput& input,
    vector<LogicalType>& return_types,
    vector<string>& names) {
	
	// Get parameters
	string path = input.named_params.at("path").ToString();
	string array_name = input.named_params.at("array").ToString();
	
	// Load and parse metadata
	ZarrArrayMetadata metadata;
	try {
		// Try to find zarray or z.json file
		auto& fs = FileSystem::GetFileSystem("");
		string zarray_path = path + "/" + array_name + "/.zarray";
		string zjson_path = path + "/" + array_name + "/z.json";
		
		if (fs.FileExists(zarray_path)) {
			metadata = ZarrMetadataParser::ParseV2(zarray_path);
		} else if (fs.FileExists(zjson_path)) {
			metadata = ZarrMetadataParser::ParseV3(zjson_path);
		} else {
			throw std::runtime_error("Could not find Zarr array at: " + path + "/" + array_name);
		}
	} catch (const std::exception& e) {
		throw std::runtime_error("Failed to parse Zarr metadata: " + string(e.what()));
	}
	
	// Build column names and types
	// For now, we'll create columns for coordinates + one value column
	vector<string> column_names;
	vector<LogicalType> column_types;
	
	// TODO: Load coordinate arrays and add them as columns
	// For now, just add position columns based on dimensions
	for (size_t i = 0; i < metadata.shape.size(); i++) {
		column_names.push_back("dim_" + std::to_string(i));
		column_types.push_back(LogicalType::UBIGINT);
	}
	
	// Add the data variable column
	column_names.push_back(array_name);
	column_types.push_back(ZarrMetadataParser::ToDuckDBType(metadata.dtype));
	
	// Return the bind data
	auto bind_data = make_uniq<ZarrBindData>();
	bind_data->path = path;
	bind_data->array_name = array_name;
	bind_data->metadata = metadata;
	bind_data->column_names = column_names;
	bind_data->column_types = column_types;
	
	return_types = column_types;
	names = column_names;
	
	return bind_data;
}

unique_ptr<GlobalState> ZarrTableFunction::InitGlobalState(
    ClientContext& context,
    TableFunctionInitInput& input) {
	
	auto& bind_data = input.bind_data->Cast<ZarrBindData>();
	auto global_state = make_uniq<ZarrGlobalState>(bind_data.metadata);
	
	return global_state;
}

void ZarrTableFunction::Scan(
    ClientContext& context,
    TableFunctionScanInput& input,
    DataChunk& output) {
	
	auto& bind_data = input.bind_data->Cast<ZarrBindData>();
	auto& global_state = input.global_state->Cast<ZarrGlobalState>();
	
	// Get next chunk to scan
	vector<idx_t> chunk_indices;
	if (!global_state.GetNextChunk(chunk_indices)) {
		// No more chunks
		return;
	}
	
	// Scan the chunk
	ScanChunk(context, bind_data, chunk_indices, output);
}

void ZarrTableFunction::ScanChunk(
    ClientContext& context,
    const ZarrBindData& bind_data,
    const std::vector<idx_t>& chunk_indices,
    DataChunk& output) {
	
	// TODO: Implement chunk reading and pivoting
	// For now, just set all values to null/zero
	
	// This would involve:
	// 1. Reading the chunk data using ChunkReader
	// 2. Pivoting the chunk data using PivotAlgorithm
	// 3. Converting to DuckDB vectors
	
	// For now, throw an exception to indicate this isn't implemented
	throw std::runtime_error("Chunk scanning not yet implemented - "
	                          "Phase 4 DuckDB integration is in progress");
}

void ZarrTableFunction::InferSchema(
    const std::string& path,
    const std::string& array_name,
    ZarrArrayMetadata& metadata,
    vector<string>& names,
    vector<LogicalType>& types) {
	
	// Load metadata
	auto& fs = FileSystem::GetFileSystem("");
	string zarray_path = path + "/" + array_name + "/.zarray";
	string zjson_path = path + "/" + array_name + "/z.json";
	
	if (fs.FileExists(zarray_path)) {
		metadata = ZarrMetadataParser::ParseV2(zarray_path);
	} else if (fs.FileExists(zjson_path)) {
		metadata = ZarrMetadataParser::ParseV3(zjson_path);
	} else {
		throw std::runtime_error("Could not find Zarr array");
	}
	
	// Build schema
	for (size_t i = 0; i < metadata.shape.size(); i++) {
		names.push_back("dim_" + std::to_string(i));
		types.push_back(LogicalType::UBIGINT);
	}
	
	names.push_back(array_name);
	types.push_back(ZarrMetadataParser::ToDuckDBType(metadata.dtype));
}

std::vector<CoordinateArray> ZarrTableFunction::LoadCoordinates(
    const std::string& path,
    const ZarrArrayMetadata& metadata) {
	
	// TODO: Load coordinate arrays from Zarr store
	// Coordinate arrays are 1D arrays in the Zarr store
	// They should be named according to the dimension they represent
	
	return {};  // Return empty for now
}

} // namespace zarr
} // namespace duckdb
