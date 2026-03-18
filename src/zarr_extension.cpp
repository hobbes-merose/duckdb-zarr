#define DUCKDB_EXTENSION_MAIN

#include "zarr_extension.hpp"
#include "zarr_metadata.hpp"
#include "zarr/arrow_pivot.hpp"
#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/parser/parsed_data/create_scalar_function_info.hpp"
#include "duckdb/common/vector.hpp"

// OpenSSL linked through vcpkg
#include <openssl/opensslv.h>

namespace duckdb {

//! Bind data for the read_zarr_metadata table function
struct ReadZarrMetadataBindData : public TableFunctionData {
	std::vector<ZarrArrayMetadata> arrays;
};

//! Bind data for the read_zarr table function
struct ReadZarrBindData : public TableFunctionData {
	std::string path;
	ZarrArrayMetadata array_metadata;
};

//! Global state for the read_zarr table function
struct ReadZarrGlobalState : public GlobalTableFunctionState {
	size_t current_chunk_index = 0;
	std::vector<std::vector<idx_t>> chunk_indices;
};

//! Global state for the read_zarr_metadata table function
struct ReadZarrMetadataGlobalState : public GlobalTableFunctionState {
	size_t current_index = 0;
};

inline void ZarrScalarFun(DataChunk &args, ExpressionState &state, Vector &result) {
	(void)args;
	(void)state;
	auto &name_vector = args.data[0];
	UnaryExecutor::Execute<string_t, string_t>(name_vector, result, args.size(), [&](string_t name) {
		return StringVector::AddString(result, "zarr " + name.GetString() + " 🦆");
	});
}

inline void ZarrOpenSSLVersionScalarFun(DataChunk &args, ExpressionState &state, Vector &result) {
	(void)args;
	(void)state;
	auto &name_vector = args.data[0];
	UnaryExecutor::Execute<string_t, string_t>(name_vector, result, args.size(), [&](string_t name) {
		return StringVector::AddString(result, "zarr " + name.GetString() + ", my linked OpenSSL version is " +
		                                           OPENSSL_VERSION_TEXT);
	});
}

// ============================================================================
// read_zarr_metadata table function
// ============================================================================

static unique_ptr<FunctionData> ReadZarrMetadataBind(ClientContext &context, TableFunctionBindInput &input,
                                                     vector<LogicalType> &return_types, vector<string> &names) {
	(void)context;
	auto result = make_uniq<ReadZarrMetadataBindData>();

	// Get the path from the function arguments
	if (input.inputs.empty()) {
		throw InvalidInputException("read_zarr_metadata requires a path argument");
	}
	auto path = input.inputs[0].GetValue<string>();

	// Parse Zarr metadata
	try {
		result->arrays = ParseZarrMetadata(path);
	} catch (const std::exception &e) {
		throw InvalidInputException("Failed to parse Zarr metadata: " + std::string(e.what()));
	}

	// Define return types
	return_types.push_back(LogicalType::VARCHAR); // name
	return_types.push_back(LogicalType::VARCHAR); // shape
	return_types.push_back(LogicalType::VARCHAR); // dtype
	return_types.push_back(LogicalType::VARCHAR); // chunks
	return_types.push_back(LogicalType::VARCHAR); // compressor
	return_types.push_back(LogicalType::INTEGER); // zarr_version
	return_types.push_back(LogicalType::VARCHAR); // fill_value

	names.push_back("name");
	names.push_back("shape");
	names.push_back("dtype");
	names.push_back("chunks");
	names.push_back("compressor");
	names.push_back("zarr_version");
	names.push_back("fill_value");

	return std::move(result);
}

static unique_ptr<GlobalTableFunctionState> ReadZarrMetadataInit(ClientContext &context,
                                                                 TableFunctionInitInput &input) {
	(void)context;
	(void)input;
	return make_uniq<ReadZarrMetadataGlobalState>();
}

static void ReadZarrMetadataFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	(void)context;
	auto &bind_data = data_p.bind_data->Cast<ReadZarrMetadataBindData>();
	auto &state = data_p.global_state->Cast<ReadZarrMetadataGlobalState>();

	if (state.current_index >= bind_data.arrays.size()) {
		output.SetCardinality(0);
		return;
	}

	// Determine how many rows to output
	idx_t count = std::min((idx_t)STANDARD_VECTOR_SIZE, (idx_t)(bind_data.arrays.size() - state.current_index));

	// Allocate columns
	output.SetCardinality(count);
	auto &name_col = output.data[0];
	auto &shape_col = output.data[1];
	auto &dtype_col = output.data[2];
	auto &chunks_col = output.data[3];
	auto &compressor_col = output.data[4];
	auto &zarr_version_col = output.data[5];
	auto &fill_value_col = output.data[6];

	// Fill the columns
	for (idx_t i = 0; i < count; i++) {
		auto &array_meta = bind_data.arrays[state.current_index + i];

		// Name
		FlatVector::GetData<string_t>(name_col)[i] = StringVector::AddStringOrBlob(name_col, array_meta.name);

		// Shape - join as comma-separated string
		std::string shape_str;
		for (size_t j = 0; j < array_meta.shape.size(); j++) {
			if (j > 0)
				shape_str += ",";
			shape_str += std::to_string(array_meta.shape[j]);
		}
		FlatVector::GetData<string_t>(shape_col)[i] = StringVector::AddStringOrBlob(shape_col, shape_str);

		// Dtype
		FlatVector::GetData<string_t>(dtype_col)[i] = StringVector::AddStringOrBlob(dtype_col, array_meta.dtype);

		// Chunks - join as comma-separated string
		std::string chunks_str;
		for (size_t j = 0; j < array_meta.chunks.size(); j++) {
			if (j > 0)
				chunks_str += ",";
			chunks_str += std::to_string(array_meta.chunks[j]);
		}
		FlatVector::GetData<string_t>(chunks_col)[i] = StringVector::AddStringOrBlob(chunks_col, chunks_str);

		// Compressor
		FlatVector::GetData<string_t>(compressor_col)[i] =
		    StringVector::AddStringOrBlob(compressor_col, array_meta.compressor);

		// Zarr version
		FlatVector::GetData<string_t>(zarr_version_col)[i] =
		    StringVector::AddStringOrBlob(zarr_version_col, array_meta.zarr_version == 2 ? "2" : "3");

		// Fill value
		FlatVector::GetData<string_t>(fill_value_col)[i] =
		    StringVector::AddStringOrBlob(fill_value_col, array_meta.fill_value);
	}

	state.current_index += count;
}

static TableFunction GetReadZarrMetadataFunction() {
	return TableFunction("read_zarr_metadata", {LogicalType::VARCHAR}, ReadZarrMetadataFunction, ReadZarrMetadataBind,
	                     ReadZarrMetadataInit);
}

// ============================================================================
// read_zarr table function (Arrow-Based Pivot)
// ============================================================================

static unique_ptr<FunctionData> ReadZarrBind(ClientContext &context, TableFunctionBindInput &input,
                                              vector<LogicalType> &return_types, vector<string> &names) {
	(void)context;
	auto result = make_uniq<ReadZarrBindData>();

	// Get the path from the function arguments
	if (input.inputs.empty()) {
		throw InvalidInputException("read_zarr requires a path argument");
	}
	auto path = input.inputs[0].GetValue<string>();
	result->path = path;

	// Get optional array name (defaults to first array)
	std::string array_name;
	if (input.inputs.size() > 1) {
		array_name = input.inputs[1].GetValue<string>();
	}

	// Parse Zarr metadata to find the array
	try {
		auto arrays = ParseZarrMetadata(path);
		if (arrays.empty()) {
			throw InvalidInputException("No Zarr arrays found in path");
		}

		// Find the requested array or use the first one
		if (!array_name.empty()) {
			bool found = false;
			for (const auto& arr : arrays) {
				if (arr.name == array_name) {
					result->array_metadata = arr;
					found = true;
					break;
				}
			}
			if (!found) {
				throw InvalidInputException("Array not found: " + array_name);
			}
		} else {
			result->array_metadata = arrays[0];
		}
	} catch (const std::exception &e) {
		throw InvalidInputException("Failed to parse Zarr metadata: " + std::string(e.what()));
	}

	// Define return type based on array dtype
	auto duckdb_type = duckdb::zarr::ZarrMetadataParser::ToDuckDBType(result->array_metadata.dtype);
	return_types.push_back(duckdb_type);

	// Use array name or default
	names.push_back(result->array_metadata.name.empty() ? "value" : result->array_metadata.name);

	return std::move(result);
}

static unique_ptr<GlobalTableFunctionState> ReadZarrInit(ClientContext &context, TableFunctionInitInput &input) {
	(void)context;
	auto result = make_uniq<ReadZarrGlobalState>();
	auto &bind_data = input.bind_data->Cast<ReadZarrBindData>();

	// Generate all chunk indices for the array
	const auto& shape = bind_data.array_metadata.shape;
	const auto& chunks = bind_data.array_metadata.chunks;

	// Calculate number of chunks in each dimension
	std::vector<idx_t> num_chunks(shape.size());
	for (size_t i = 0; i < shape.size(); i++) {
		num_chunks[i] = (shape[i] + chunks[i] - 1) / chunks[i];
	}

	// Generate all chunk index combinations using recursion
	std::function<void(size_t, std::vector<idx_t>&)> generate_indices = 
		[&](size_t dim, std::vector<idx_t>& current) {
			if (dim == shape.size()) {
				result->chunk_indices.push_back(current);
			} else {
				for (idx_t i = 0; i < num_chunks[dim]; i++) {
					current.push_back(i);
					generate_indices(dim + 1, current);
					current.pop_back();
				}
			}
		};

	std::vector<idx_t> current;
	generate_indices(0, current);

	return std::move(result);
}

static void ReadZarrFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	(void)context;
	auto &bind_data = data_p.bind_data->Cast<ReadZarrBindData>();
	auto &state = data_p.global_state->Cast<ReadZarrGlobalState>();

	if (state.current_chunk_index >= state.chunk_indices.size()) {
		output.SetCardinality(0);
		return;
	}

	// Determine how many chunks to output
	idx_t count = std::min((idx_t)STANDARD_VECTOR_SIZE, 
	                       (idx_t)(state.chunk_indices.size() - state.current_chunk_index));

	output.SetCardinality(count);

	// Create chunk reader
	duckdb::zarr::ChunkReader chunk_reader(bind_data.path, bind_data.array_metadata);

	// Read chunks and convert to vectors using Arrow pivot
	auto &value_col = output.data[0];

	for (idx_t i = 0; i < count; i++) {
		auto& chunk_indices = state.chunk_indices[state.current_chunk_index + i];

		// Read the chunk using zarrs_ffi style chunk reader
		auto chunk_data = chunk_reader.ReadChunk(chunk_indices);

		// Use ArrowConverter to convert chunk data to vector
		auto vector = duckdb::zarr::ArrowConverter::ToVector(chunk_data, bind_data.array_metadata);

		// Copy the vector data to output (simplified - production would handle batched reading)
		// For now, just copy the first element if available
		if (chunk_data.num_elements > 0) {
			// Copy first element to output column
			switch (bind_data.array_metadata.dtype) {
			case duckdb::zarr::ZarrDtype::INT32: {
				auto* src = reinterpret_cast<const int32_t*>(chunk_data.data.data());
				FlatVector::GetData<int32_t>(value_col)[i] = src[0];
				break;
			}
			case duckdb::zarr::ZarrDtype::INT64: {
				auto* src = reinterpret_cast<const int64_t*>(chunk_data.data.data());
				FlatVector::GetData<int64_t>(value_col)[i] = src[0];
				break;
			}
			case duckdb::zarr::ZarrDtype::FLOAT32: {
				auto* src = reinterpret_cast<const float*>(chunk_data.data.data());
				FlatVector::GetData<float>(value_col)[i] = src[0];
				break;
			}
			case duckdb::zarr::ZarrDtype::FLOAT64: {
				auto* src = reinterpret_cast<const double*>(chunk_data.data.data());
				FlatVector::GetData<double>(value_col)[i] = src[0];
				break;
			}
			default:
				break;
			}
		}
	}

	state.current_chunk_index += count;
}

static TableFunction GetReadZarrFunction() {
	return TableFunction("read_zarr", {LogicalType::VARCHAR}, ReadZarrFunction, ReadZarrBind, ReadZarrInit);
}

static void LoadInternal(ExtensionLoader &loader) {
	// Register scalar functions
	auto zarr_scalar_function = ScalarFunction("zarr", {LogicalType::VARCHAR}, LogicalType::VARCHAR, ZarrScalarFun);
	loader.RegisterFunction(zarr_scalar_function);

	auto zarr_openssl_version_scalar_function = ScalarFunction("zarr_openssl_version", {LogicalType::VARCHAR},
	                                                           LogicalType::VARCHAR, ZarrOpenSSLVersionScalarFun);
	loader.RegisterFunction(zarr_openssl_version_scalar_function);

	// Register table functions
	loader.RegisterFunction(GetReadZarrMetadataFunction());
	loader.RegisterFunction(GetReadZarrFunction());
}

void ZarrExtension::Load(ExtensionLoader &loader) {
	LoadInternal(loader);
}
std::string ZarrExtension::Name() {
	return "zarr";
}

std::string ZarrExtension::Version() const {
#ifdef EXT_VERSION_ZARR
	return EXT_VERSION_ZARR;
#else
	return "";
#endif
}

} // namespace duckdb

extern "C" {

DUCKDB_CPP_EXTENSION_ENTRY(zarr, loader) {
	duckdb::LoadInternal(loader);
}
}
