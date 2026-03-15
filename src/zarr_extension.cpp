#define DUCKDB_EXTENSION_MAIN

#include "zarr_extension.hpp"
#include "zarr_metadata.hpp"
#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/parser/parsed_data/create_scalar_function_info.hpp"

// OpenSSL linked through vcpkg
#include <openssl/opensslv.h>

namespace duckdb {

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

struct ReadZarrMetadataState {
	std::vector<ZarrArrayMetadata> arrays;
	size_t current_index = 0;
};

static void ReadZarrMetadataFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &state = data_p.state_obj->Cast<ReadZarrMetadataState>();

	if (state.current_index >= state.arrays.size()) {
		output.SetCardinality(0);
		return;
	}

	// Determine how many rows to output
	idx_t count = std::min((idx_t)STANDARD_VECTOR_SIZE, (idx_t)(state.arrays.size() - state.current_index));

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
		auto &array_meta = state.arrays[state.current_index + i];

		// Name
		StringVector::SetString(name_col, i, array_meta.name);

		// Shape - join as comma-separated string
		std::string shape_str;
		for (size_t j = 0; j < array_meta.shape.size(); j++) {
			if (j > 0)
				shape_str += ",";
			shape_str += std::to_string(array_meta.shape[j]);
		}
		StringVector::SetString(shape_col, i, shape_str);

		// Dtype
		StringVector::SetString(dtype_col, i, array_meta.dtype);

		// Chunks - join as comma-separated string
		std::string chunks_str;
		for (size_t j = 0; j < array_meta.chunks.size(); j++) {
			if (j > 0)
				chunks_str += ",";
			chunks_str += std::to_string(array_meta.chunks[j]);
		}
		StringVector::SetString(chunks_col, i, chunks_str);

		// Compressor
		StringVector::SetString(compressor_col, i, array_meta.compressor);

		// Zarr version
		StringVector::SetString(zarr_version_col, i, array_meta.zarr_version == 2 ? "2" : "3");

		// Fill value
		StringVector::SetString(fill_value_col, i, array_meta.fill_value);
	}

	state.current_index += count;
}

static unique_ptr<FunctionData>
ReadZarrMetadataBind(ClientContext &context, TableFunctionBindInfo &bind_info, (void)context; (void)bind_info;
                     vector<LogicalType> & return_types, vector<string> &names) {
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

	return nullptr;
}

static unique_ptr<FunctionData> ReadZarrMetadataInit(ClientContext &context, TableFunctionInitInput &input) {
	auto state = make_uniq<ReadZarrMetadataState>();

	// Get the path from the function arguments
	auto &path_arg = input.input.inputs[0];
	auto path = path_arg.GetValue<string>();

	// Parse Zarr metadata
	try {
		state->arrays = ParseZarrMetadata(path);
	} catch (const std::exception &e) {
		throw InvalidException("Failed to parse Zarr metadata: " + std::string(e.what()));
	}

	state->current_index = 0;
	return std::move(state);
}

static TableFunction GetReadZarrMetadataFunction() {
	return TableFunction("read_zarr_metadata", {LogicalType::VARCHAR}, ReadZarrMetadataFunction, ReadZarrMetadataBind,
	                     ReadZarrMetadataInit);
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
