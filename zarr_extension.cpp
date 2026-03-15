#define DUCKDB_EXTENSION_MAIN

#include "zarr_extension.hpp"
#include "zarr/metadata.hpp"
#include "zarr/table_function.hpp"
#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/function/scalar_function.hpp"
#include <duckdb/parser/parsed_data/create_scalar_function_info.hpp>

// OpenSSL linked through vcpkg
#include <openssl/opensslv.h>

namespace duckdb {

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

// Test function: parse metadata from JSON string
inline void ZarrMetadataParseFun(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &input_vector = args.data[0];
	UnaryExecutor::Execute<string_t, string_t>(input_vector, result, args.size(), [&](string_t json_str) {
		try {
			auto metadata = zarr::ZarrMetadataParser::ParseFromString(json_str.GetString());
			return StringVector::AddString(result, metadata);
		} catch (const std::exception& e) {
			return StringVector::AddString(result, string("Error: ") + e.what());
		}
	});
}

// Test function: detect dtype from string
inline void ZarrDetectDtypeFun(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &input_vector = args.data[0];
	UnaryExecutor::Execute<string_t, string_t>(input_vector, result, args.size(), [&](string_t dtype_str) {
		auto dtype = zarr::ZarrMetadataParser::ParseDtype(dtype_str.GetString());
		string result_str;
		switch (dtype) {
			case zarr::ZarrDtype::INT8: result_str = "INT8"; break;
			case zarr::ZarrDtype::INT16: result_str = "INT16"; break;
			case zarr::ZarrDtype::INT32: result_str = "INT32"; break;
			case zarr::ZarrDtype::INT64: result_str = "INT64"; break;
			case zarr::ZarrDtype::UINT8: result_str = "UINT8"; break;
			case zarr::ZarrDtype::UINT16: result_str = "UINT16"; break;
			case zarr::ZarrDtype::UINT32: result_str = "UINT32"; break;
			case zarr::ZarrDtype::UINT64: result_str = "UINT64"; break;
			case zarr::ZarrDtype::FLOAT32: result_str = "FLOAT32"; break;
			case zarr::ZarrDtype::FLOAT64: result_str = "FLOAT64"; break;
			case zarr::ZarrDtype::STRING: result_str = "STRING"; break;
			case zarr::ZarrDtype::BYTES: result_str = "BYTES"; break;
			case zarr::ZarrDtype::BOOL: result_str = "BOOL"; break;
			case zarr::ZarrDtype::COMPLEX64: result_str = "COMPLEX64"; break;
			case zarr::ZarrDtype::COMPLEX128: result_str = "COMPLEX128"; break;
			default: result_str = "UNKNOWN"; break;
		}
		return StringVector::AddString(result, result_str);
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

	// Register metadata parsing functions (for testing Phase 1)
	auto zarr_metadata_parse_function = ScalarFunction("zarr_metadata_parse", {LogicalType::VARCHAR},
	                                                     LogicalType::VARCHAR, ZarrMetadataParseFun);
	loader.RegisterFunction(zarr_metadata_parse_function);

	// Register dtype detection function
	auto zarr_detect_dtype_function = ScalarFunction("zarr_detect_dtype", {LogicalType::VARCHAR},
	                                                   LogicalType::VARCHAR, ZarrDetectDtypeFun);
	loader.RegisterFunction(zarr_detect_dtype_function);
	
	// Register table function (Phase 4: DuckDB Integration)
	// Note: Table function registration requires access to DatabaseInstance
	// This will be done in the extension Load function
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
