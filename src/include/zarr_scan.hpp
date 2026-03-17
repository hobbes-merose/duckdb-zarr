#ifndef ZARR_SCAN_HPP
#define ZARR_SCAN_HPP

#include <duckdb.hpp>

namespace duckdb {

class ZarrScanFunction {
public:
    static void Register(ExtensionLoader &loader);
    
private:
    // Bind function - validates params and defines schema
    static duckdb::unique_ptr<FunctionData> Bind(ClientContext &context, TableFunctionBindInput &input);
    
    // Global init - setup global state
    static unique_ptr<GlobalFunctionState> InitGlobal(ClientContext &context, TableFunctionInitInput &input);
    
    // Local init - setup local state  
    static unique_ptr<LocalFunctionState> InitLocal(ExecutionContext &context, TableFunctionInitInput &input);
    
    // Scan function - returns data
    static void Scan(ClientContext &context, TableFunctionInput &input, DataChunk &output);
    
    // Helper: Infer schema from zarrs_ffi metadata
    static void InferSchema(const string &path, const string &array_name, vector<ColumnDefinition> &columns);
    
    // Helper: Convert zarr dtype to DuckDB type
    static LogicalType ConvertDataType(const string &zarr_dtype);
};

} // namespace duckdb

#endif // ZARR_SCAN_HPP
