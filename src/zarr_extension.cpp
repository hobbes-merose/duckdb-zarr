#define DUCKDB_EXTENSION_MAIN

#include "zarr_extension.hpp"
#include "zarr_scan.hpp"
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

static void LoadInternal(ExtensionLoader &loader) {
	// Register a scalar function
	auto zarr_scalar_function = ScalarFunction("zarr", {LogicalType::VARCHAR}, LogicalType::VARCHAR, ZarrScalarFun);
	loader.RegisterFunction(zarr_scalar_function);

	// Register another scalar function
	auto zarr_openssl_version_scalar_function = ScalarFunction("zarr_openssl_version", {LogicalType::VARCHAR},
	                                                           LogicalType::VARCHAR, ZarrOpenSSLVersionScalarFun);
	loader.RegisterFunction(zarr_openssl_version_scalar_function);

	// Register the read_zarr table function
	ZarrScanFunction::Register(loader);
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
