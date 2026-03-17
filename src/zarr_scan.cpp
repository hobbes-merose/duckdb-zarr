#include "zarr_scan.hpp"
#include "zarr_extension.hpp"
#include <duckdb/function/table_function.hpp>
#include <duckdb/main/client_context.hpp>
#include <duckdb/common/types/data_chunk.hpp>

namespace duckdb {

// TODO: Include zarrs_ffi header when available
// #include <zarrs.h>

void ZarrScanFunction::Register(ExtensionLoader &loader) {
    // Define return types: dim_0 (INTEGER), dim_1 (INTEGER), value (DOUBLE)
    // This is a stub schema - will be replaced with actual zarrs_ffi metadata
    vector<LogicalType> return_types = {LogicalType::INTEGER, LogicalType::INTEGER, LogicalType::DOUBLE};
    vector<string> return_names = {"dim_0", "dim_1", "value"};
    
    TableFunction read_zarr("read_zarr", {LogicalType::VARCHAR, LogicalType::VARCHAR}, Scan, Bind, InitGlobal, InitLocal);
    read_zarr.return_types = return_types;
    read_zarr.return_names = return_names;
    read_zarr.named_parameters["path"] = LogicalType::VARCHAR;
    read_zarr.named_parameters["array"] = LogicalType::VARCHAR;
    
    loader.RegisterFunction(read_zarr);
}

unique_ptr<FunctionData> ZarrScanFunction::Bind(ClientContext &context, TableFunctionBindInput &input) {
    auto path = input.inputs[0].GetValue<string>();
    auto array_name = input.inputs[1].GetValue<string>();
    
    // Infer schema from the Zarr array
    vector<ColumnDefinition> columns;
    InferSchema(path, array_name, columns);
    
    auto result = make_uniq<FunctionData>();
    return result;
}

unique_ptr<GlobalFunctionState> ZarrScanFunction::InitGlobal(ClientContext &context, TableFunctionInitInput &input) {
    return make_uniq<GlobalFunctionState>();
}

unique_ptr<LocalFunctionState> ZarrScanFunction::InitLocal(ExecutionContext &context, TableFunctionInitInput &input) {
    return make_uniq<LocalFunctionState>();
}

void ZarrScanFunction::Scan(ClientContext &context, TableFunctionInput &input, DataChunk &output) {
    // TODO: Use zarrs_ffi to read actual data
    // For now, produce stub data to verify the table function works
    
    // Get the number of rows to produce
    idx_t row_count = 10; // Stub: produce 10 rows
    
    // Produce stub output with 3 columns: dim_0, dim_1, value
    // (schema will be fixed when we integrate zarrs_ffi)
    for (idx_t col_idx = 0; col_idx < output.ColumnCount(); col_idx++) {
        auto &vector = output.data[col_idx];
        
        if (col_idx < 2) {
            // Dimension column - produce row indices
            for (idx_t i = 0; i < row_count; i++) {
                vector.SetValue(i, Value::INTEGER(static_cast<int32_t>(i)));
            }
        } else {
            // Data column - produce stub values
            for (idx_t i = 0; i < row_count; i++) {
                vector.SetValue(i, Value::DOUBLE(static_cast<double>(i * 1.5)));
            }
        }
    }
    
    output.SetCardinality(row_count);
}

void ZarrScanFunction::InferSchema(const string &path, const string &array_name, vector<ColumnDefinition> &columns) {
    // TODO: Use zarrs_ffi to get actual metadata:
    // - zarrsOpenArrayRW(path.c_str(), array_name.c_str(), &array)
    // - zarrsArrayGetDimensionality(array, &dimensionality)
    // - zarrsArrayGetShape(array, shape)
    // - zarrsArrayGetDataType(array, &dtype)
    
    // For now, produce stub schema: 2D array with dim_0, dim_1, and value
    columns.push_back(ColumnDefinition("dim_0", LogicalType::INTEGER));
    columns.push_back(ColumnDefinition("dim_1", LogicalType::INTEGER));
    columns.push_back(ColumnDefinition("value", LogicalType::DOUBLE));
}

LogicalType ZarrScanFunction::ConvertDataType(const string &zarr_dtype) {
    // Map zarr dtypes to DuckDB types
    if (zarr_dtype == "bool" || zarr_dtype == "b") {
        return LogicalType::BOOLEAN;
    } else if (zarr_dtype == "int8" || zarr_dtype == "i1") {
        return LogicalType::TINYINT;
    } else if (zarr_dtype == "int16" || zarr_dtype == "i2") {
        return LogicalType::SMALLINT;
    } else if (zarr_dtype == "int32" || zarr_dtype == "i4") {
        return LogicalType::INTEGER;
    } else if (zarr_dtype == "int64" || zarr_dtype == "i8") {
        return LogicalType::BIGINT;
    } else if (zarr_dtype == "float32" || zarr_dtype == "f4") {
        return LogicalType::FLOAT;
    } else if (zarr_dtype == "float64" || zarr_dtype == "f8") {
        return LogicalType::DOUBLE;
    }
    // Default to VARCHAR for unknown types
    return LogicalType::VARCHAR;
}

} // namespace duckdb
