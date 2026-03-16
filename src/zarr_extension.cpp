#define DUCKDB_EXTENSION_MAIN

#include "zarr_extension.hpp"
#include "zarr_metadata.hpp"
#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/function/table_function.hpp"
#include <duckdb/parser/parsed_data/create_scalar_function_info.hpp>
#include <duckdb/parser/parsed_data/create_table_function_info.hpp>

// OpenSSL linked through vcpkg
#include <openssl/opensslv.h>

namespace duckdb {

//! Zarr metadata table function data
struct ZarrMetadataFunctionData : public FunctionData {
	std::vector<ZarrArrayMetadata> arrays;
	idx_t current_array_index = 0;
};

//! Zarr metadata table function bind
static unique_ptr<FunctionData> ZarrMetadataBind(ClientContext &context, TableFunctionBindInput &input,
                                                 vector<LogicalType> &return_types, vector<string> &names) {
	// Define the return schema
	return_types.push_back(LogicalType::VARCHAR);                   // name
	return_types.push_back(LogicalType::INTEGER);                   // zarr_version
	return_types.push_back(LogicalType::LIST(LogicalType::BIGINT)); // shape
	return_types.push_back(LogicalType::LIST(LogicalType::BIGINT)); // chunks
	return_types.push_back(LogicalType::VARCHAR);                   // dtype
	return_types.push_back(LogicalType::VARCHAR);                   // fill_value
	return_types.push_back(LogicalType::VARCHAR);                   // compressor
	return_types.push_back(LogicalType::VARCHAR);                   // order

	names.push_back("name");
	names.push_back("zarr_version");
	names.push_back("shape");
	names.push_back("chunks");
	names.push_back("dtype");
	names.push_back("fill_value");
	names.push_back("compressor");
	names.push_back("order");

	// Parse the path input
	auto &path_input = input.inputs[0];
	auto path = path_input.GetValue<std::string>();

	// Parse Zarr metadata
	auto arrays = ParseZarrMetadata(path);

	// Create function data
	auto function_data = make_uniq<ZarrMetadataFunctionData>();
	function_data->arrays = std::move(arrays);

	return function_data;
}

//! Zarr metadata table function
static void ZarrMetadataFunction(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &function_data = data.bind_data->Cast<ZarrMetadataFunctionData>();

	// Check if we have more arrays to return
	if (function_data.current_array_index >= function_data.arrays.size()) {
		return;
	}

	// Get current array
	auto &array_meta = function_data.arrays[function_data.current_array_index];
	function_data.current_array_index++;

	// Set output values
	output.SetValue(0, 0, Value(array_meta.name));
	output.SetValue(1, 0, Value(array_meta.zarr_version));

	// Shape as list
	vector<Value> shape_values;
	for (auto dim : array_meta.shape) {
		shape_values.push_back(Value(dim));
	}
	output.SetValue(2, 0, Value::LIST(LogicalType::BIGINT, shape_values));

	// Chunks as list
	vector<Value> chunk_values;
	for (auto chunk : array_meta.chunks) {
		chunk_values.push_back(Value(chunk));
	}
	output.SetValue(3, 0, Value::LIST(LogicalType::BIGINT, chunk_values));

	output.SetValue(4, 0, Value(array_meta.dtype));
	output.SetValue(5, 0, Value(array_meta.fill_value));
	output.SetValue(6, 0, Value(array_meta.compressor));
	output.SetValue(7, 0, Value(array_meta.order));

	output.SetCardinality(1);
}

inline void ZarrScalarFun(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &name_vector = args.data[0];
	UnaryExecutor::Execute<string_t, string_t>(name_vector, result, args.size(), [&](string_t name) {
		return StringVector::AddString(result, "zarr " + name.GetString() + " 🦆");
	});
}

inline void ZarrOpenSSLVersionScalarFun(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &name_vector = args.data[0];
	UnaryExecutor::Execute<string_t, string_t>(name_vector, result, args.size(), [&](string_t name) {
		return StringVector::AddString(result, "zarr " + name.GetString() + ", my linked OpenSSL version is " +
		                                           OPENSSL_VERSION_TEXT);
	});
}

static void LoadInternal(ExtensionLoader &loader) {
	// Register a scalar function
	auto zarr_scalar_function = ScalarFunction("zarr", {LogicalType::VARCHAR}, LogicalType::VARCHAR, ZarrScalarFun);
	loader.RegisterFunction(zarr_scalar_function);

	// Register another scalar function
	auto zarr_openssl_version_scalar_function = ScalarFunction("zarr_openssl_version", {LogicalType::VARCHAR},
	                                                           LogicalType::VARCHAR, ZarrOpenSSLVersionScalarFun);
	loader.RegisterFunction(zarr_openssl_version_scalar_function);

	// Register the zarr_metadata table function
	TableFunction zarr_metadata_function("zarr_metadata", {LogicalType::VARCHAR}, ZarrMetadataFunction,
	                                     ZarrMetadataBind);
	zarr_metadata_function.named_parameters["binary"] = LogicalType::BOOLEAN;
	loader.RegisterFunction(zarr_metadata_function);
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
