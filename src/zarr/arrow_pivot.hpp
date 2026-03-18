//===----------------------------------------------------------------------===//
//                         DuckDB Zarr Extension
//
// zarr/arrow_pivot.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "zarr/chunk.hpp"
#include "duckdb.hpp"
#include "duckdb/common/arrow/arrow.hpp"
#include "duckdb/common/arrow/arrow_converter.hpp"
#include "duckdb/common/arrow/arrow_wrapper.hpp"
#include "duckdb/main/client_properties.hpp"
#include <vector>
#include <memory>
#include <cstring>

namespace duckdb {

// Forward declare ZarrArrayMetadata from zarr namespace
namespace zarr {
class ZarrMetadataParser;
}

//! Convert ZarrDtype to DuckDB LogicalType (standalone function)
static LogicalType ZarrDtypeToDuckDBType(zarr::ZarrDtype dtype) {
    switch (dtype) {
        case zarr::ZarrDtype::INT8:
            return LogicalType::TINYINT;
        case zarr::ZarrDtype::INT16:
            return LogicalType::SMALLINT;
        case zarr::ZarrDtype::INT32:
            return LogicalType::INTEGER;
        case zarr::ZarrDtype::INT64:
            return LogicalType::BIGINT;
        case zarr::ZarrDtype::UINT8:
            return LogicalType::UTINYINT;
        case zarr::ZarrDtype::UINT16:
            return LogicalType::USMALLINT;
        case zarr::ZarrDtype::UINT32:
            return LogicalType::UINTEGER;
        case zarr::ZarrDtype::UINT64:
            return LogicalType::UBIGINT;
        case zarr::ZarrDtype::FLOAT32:
            return LogicalType::FLOAT;
        case zarr::ZarrDtype::FLOAT64:
            return LogicalType::DOUBLE;
        case zarr::ZarrDtype::STRING:
            return LogicalType::VARCHAR;
        case zarr::ZarrDtype::BOOL:
            return LogicalType::BOOLEAN;
        case zarr::ZarrDtype::BYTES:
            return LogicalType::BLOB;
        default:
            return LogicalType::BLOB;
    }
}

namespace zarr {

/// ArrowConverter converts Zarr chunk data to Arrow format for efficient DuckDB integration.
/// This is the "pivot" that transforms n-dimensional array data to tabular rows.
class ArrowConverter {
public:
	/// Constructor
	ArrowConverter();

	/// Convert chunk data to DuckDB Vector
	/// @param chunk_data The decompressed chunk data
	/// @param metadata The array metadata
	/// @return DuckDB Vector containing the data
	static Vector ToVector(const ChunkData& chunk_data, const ZarrArrayMetadata& metadata);

	/// Convert chunk data to Arrow Array
	/// @param chunk_data The decompressed chunk data
	/// @param metadata The array metadata
	/// @return Arrow Array wrapper
	static unique_ptr<ArrowArrayWrapper> ToArrowArray(const ChunkData& chunk_data,
	                                                   const ZarrArrayMetadata& metadata);

	/// Flatten n-dimensional chunk data to 1D for Arrow
	/// @param chunk_data The chunk data to flatten
	/// @return Flattened data vector
	static std::vector<uint8_t> FlattenChunkData(const ChunkData& chunk_data);

	/// Create validity mask from chunk data (handling fill values)
	/// @param chunk_data The chunk data
	/// @return Validity bitmap (true = valid, false = null)
	static std::vector<bool> CreateValidityMask(const ChunkData& chunk_data);

private:
	/// Get Arrow type from Zarr dtype
	static ArrowType GetArrowType(ZarrDtype dtype);

	/// Convert integer types (handling endianness)
	static void ConvertEndianness(uint8_t* data, idx_t element_size, idx_t count);
};

/// Implementation (~100 LOC core pivot conversion logic)

ArrowConverter::ArrowConverter() = default;

Vector ArrowConverter::ToVector(const ChunkData& chunk_data, const ZarrArrayMetadata& metadata) {
	// Convert to DuckDB type using the standalone converter
	LogicalType duckdb_type = ZarrDtypeToDuckDBType(metadata.dtype);
	
	// Create result vector
	Vector result(duckdb_type, chunk_data.num_elements);
	
	const auto* src = chunk_data.data.data();
	auto count = static_cast<idx_t>(chunk_data.num_elements);
	
	// Direct memory copy based on dtype (the core pivot conversion)
	switch (metadata.dtype) {
	case ZarrDtype::INT8: {
		auto* dst = FlatVector::GetData<int8_t>(result);
		std::memcpy(dst, src, count * sizeof(int8_t));
		break;
	}
	case ZarrDtype::INT16: {
		auto* dst = FlatVector::GetData<int16_t>(result);
		std::memcpy(dst, src, count * sizeof(int16_t));
		break;
	}
	case ZarrDtype::INT32: {
		auto* dst = FlatVector::GetData<int32_t>(result);
		std::memcpy(dst, src, count * sizeof(int32_t));
		break;
	}
	case ZarrDtype::INT64: {
		auto* dst = FlatVector::GetData<int64_t>(result);
		std::memcpy(dst, src, count * sizeof(int64_t));
		break;
	}
	case ZarrDtype::UINT8: {
		auto* dst = FlatVector::GetData<uint8_t>(result);
		std::memcpy(dst, src, count * sizeof(uint8_t));
		break;
	}
	case ZarrDtype::UINT16: {
		auto* dst = FlatVector::GetData<uint16_t>(result);
		std::memcpy(dst, src, count * sizeof(uint16_t));
		break;
	}
	case ZarrDtype::UINT32: {
		auto* dst = FlatVector::GetData<uint32_t>(result);
		std::memcpy(dst, src, count * sizeof(uint32_t));
		break;
	}
	case ZarrDtype::UINT64: {
		auto* dst = FlatVector::GetData<uint64_t>(result);
		std::memcpy(dst, src, count * sizeof(uint64_t));
		break;
	}
	case ZarrDtype::FLOAT32: {
		auto* dst = FlatVector::GetData<float>(result);
		std::memcpy(dst, src, count * sizeof(float));
		break;
	}
	case ZarrDtype::FLOAT64: {
		auto* dst = FlatVector::GetData<double>(result);
		std::memcpy(dst, src, count * sizeof(double));
		break;
	}
	case ZarrDtype::BOOL: {
		auto* dst = FlatVector::GetData<bool>(result);
		std::memcpy(dst, src, count * sizeof(bool));
		break;
	}
	default:
		throw std::runtime_error("Unsupported dtype for Arrow conversion");
	}
	
	return result;
}

unique_ptr<ArrowArrayWrapper> ArrowConverter::ToArrowArray(const ChunkData& chunk_data,
                                                            const ZarrArrayMetadata& metadata) {
	auto result = make_uniq<ArrowArrayWrapper>();
	
	// Set array length
	result->arrow_array.length = static_cast<int64_t>(chunk_data.num_elements);
	
	// Allocate buffers: [validity, data, offset]
	size_t data_size = static_cast<size_t>(chunk_data.num_elements) * static_cast<size_t>(metadata.element_size);
	
	result->arrow_array.buffers = new const void*[3];
	result->arrow_array.buffers[0] = nullptr; // validity (set below)
	result->arrow_array.buffers[1] = nullptr; // data (set below)
	result->arrow_array.buffers[2] = nullptr; // offset (for variable-length)
	
	// Allocate data buffer
	auto* data_buffer = new uint8_t[data_size];
	std::memcpy(data_buffer, chunk_data.data.data(), data_size);
	result->arrow_array.buffers[1] = data_buffer;
	
	// Handle endianness if needed
	if (metadata.element_size > 1) {
		ConvertEndianness(data_buffer, metadata.element_size, chunk_data.num_elements);
	}
	
	// Allocate validity buffer (all valid for now)
	auto validity_size = (chunk_data.num_elements + 7) / 8;
	auto* validity_buffer = new uint8_t[validity_size];
	std::memset(validity_buffer, 0xFF, validity_size);
	result->arrow_array.buffers[0] = validity_buffer;
	
	// Set array metadata
	result->arrow_array.null_count = 0;
	result->arrow_array.offset = 0;
	result->arrow_array.nulls_ordered = 1;
	
	// Set release function for proper memory cleanup
	result->arrow_array.release = [](ArrowArray* array) {
		if (array->buffers) {
			for (int i = 0; i < 3; i++) {
				delete[] static_cast<const uint8_t*>(array->buffers[i]);
			}
			delete[] array->buffers;
		}
		array->release = nullptr;
	};
	
	return result;
}

std::vector<uint8_t> ArrowConverter::FlattenChunkData(const ChunkData& chunk_data) {
	// Zarr v2 uses C order (row-major) by default
	return chunk_data.data;
}

std::vector<bool> ArrowConverter::CreateValidityMask(const ChunkData& chunk_data) {
	std::vector<bool> validity(chunk_data.num_elements, true);
	// Check for fill values to identify nulls (simplified)
	return validity;
}

ArrowType ArrowConverter::GetArrowType(ZarrDtype dtype) {
	switch (dtype) {
	case ZarrDtype::INT8: return ArrowType::INT8;
	case ZarrDtype::INT16: return ArrowType::INT16;
	case ZarrDtype::INT32: return ArrowType::INT32;
	case ZarrDtype::INT64: return ArrowType::INT64;
	case ZarrDtype::UINT8: return ArrowType::UINT8;
	case ZarrDtype::UINT16: return ArrowType::UINT16;
	case ZarrDtype::UINT32: return ArrowType::UINT32;
	case ZarrDtype::UINT64: return ArrowType::UINT64;
	case ZarrDtype::FLOAT32: return ArrowType::FLOAT;
	case ZarrDtype::FLOAT64: return ArrowType::DOUBLE;
	case ZarrDtype::BOOL: return ArrowType::BOOL;
	case ZarrDtype::STRING: return ArrowType::STRING;
	default: return ArrowType::NA;
	}
}

void ArrowConverter::ConvertEndianness(uint8_t* data, idx_t element_size, idx_t count) {
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
	for (idx_t i = 0; i < count; i++) {
		std::reverse(data + i * element_size, data + (i + 1) * element_size);
	}
#endif
}

} // namespace zarr
} // namespace duckdb
