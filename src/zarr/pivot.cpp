#include "zarr/pivot.hpp"
#include <algorithm>
#include <cstring>

namespace duckdb {
namespace zarr {

// ============================================================================
// PivotAlgorithm Implementation
// ============================================================================

PivotAlgorithm::PivotAlgorithm(const ZarrArrayMetadata& metadata, 
                               const std::vector<CoordinateArray>& coordinates)
    : metadata_(metadata), coordinates_(coordinates), shape_(metadata.shape), chunks_(metadata.chunks) {
}

std::vector<idx_t> PivotAlgorithm::CalculateStrides() const {
	// Stride for dimension i = product of chunk sizes for dimensions 0 to i-1
	std::vector<idx_t> strides(shape_.size(), 1);
	for (size_t i = 1; i < shape_.size(); i++) {
		strides[i] = strides[i - 1] * chunks_[i - 1];
	}
	return strides;
}

idx_t PivotAlgorithm::CalculateLinearIndex(const std::vector<idx_t>& indices) const {
	idx_t linear_idx = 0;
	idx_t stride = 1;
	for (size_t i = 0; i < indices.size(); i++) {
		linear_idx += indices[i] * stride;
		stride *= chunks_[i];
	}
	return linear_idx;
}

std::vector<idx_t> PivotAlgorithm::CalculateMultidimensionalIndex(idx_t linear_idx) const {
	std::vector<idx_t> indices(shape_.size());
	idx_t remaining = linear_idx;
	for (size_t i = 0; i < shape_.size(); i++) {
		// stride[k] = product of chunks[0]..chunks[k-1]
		idx_t stride = 1;
		for (size_t j = 0; j < i; j++) {
			stride *= chunks_[j];
		}
		indices[i] = remaining / stride;
		remaining = remaining % stride;
	}
	return indices;
}

template<typename T>
T PivotAlgorithm::ReadValue(const ChunkData& chunk_data, const std::vector<idx_t>& indices) const {
	// Calculate offset within chunk using row-major order
	idx_t offset = 0;
	idx_t stride = 1;
	for (size_t i = 0; i < indices.size(); i++) {
		offset += indices[i] * stride;
		stride *= chunk_data.shape[i];
	}
	
	// Read the value
	T* data_ptr = reinterpret_cast<T*>(chunk_data.data.data());
	return data_ptr[offset];
}

idx_t PivotAlgorithm::GetTotalRows() const {
	// Total rows = product of chunk dimensions
	idx_t total = 1;
	for (auto chunk : chunks_) {
		total *= chunk;
	}
	return total;
}

idx_t PivotAlgorithm::GetNumColumns() const {
	// Columns = coordinates + data variables (1 for now - the main array)
	return coordinates_.size() + 1;
}

std::vector<std::string> PivotAlgorithm::GetColumnNames() const {
	std::vector<std::string> names;
	for (const auto& coord : coordinates_) {
		names.push_back(coord.name);
	}
	// Add data variable name
	names.push_back(metadata_.name.empty() ? "value" : metadata_.name);
	return names;
}

std::vector<LogicalType> PivotAlgorithm::GetColumnTypes() const {
	std::vector<LogicalType> types;
	for (const auto& coord : coordinates_) {
		types.push_back(ZarrMetadataParser::ToDuckDBType(coord.dtype));
	}
	// Add data variable type
	types.push_back(ZarrMetadataParser::ToDuckDBType(metadata_.dtype));
	return types;
}

std::vector<std::vector<uint8_t>> PivotAlgorithm::PivotChunk(
    const ChunkData& chunk_data,
    const std::vector<idx_t>& chunk_indices) {
	
	// Number of output columns
	size_t num_cols = coordinates_.size() + 1;
	
	// Number of rows in this chunk
	idx_t num_rows = 1;
	for (auto chunk : chunks_) {
		num_rows *= chunk;
	}
	
	// Allocate output columns
	std::vector<std::vector<uint8_t>> result(num_cols);
	size_t element_size = metadata_.element_size;
	
	for (size_t col = 0; col < num_cols; col++) {
		result[col].resize(static_cast<size_t>(num_rows) * element_size);
	}
	
	// For each row in the chunk output
	for (idx_t row = 0; row < num_rows; row++) {
		// Calculate the multidimensional index within the chunk
		std::vector<idx_t> chunk_idx = CalculateMultidimensionalIndex(row);
		
		// Calculate the global array indices by adding chunk offset
		std::vector<idx_t> global_indices(shape_.size());
		for (size_t i = 0; i < shape_.size(); i++) {
			global_indices[i] = chunk_indices[i] * chunks_[i] + chunk_idx[i];
		}
		
		// For coordinate columns, look up the coordinate value
		for (size_t coord_idx = 0; coord_idx < coordinates_.size(); coord_idx++) {
			const auto& coord = coordinates_[coord_idx];
			// The coordinate value is at position global_indices[coord_idx]
			idx_t coord_pos = global_indices[coord_idx];
			
			// Copy coordinate value to output
			void* src = const_cast<uint8_t*>(coord.data.data()) + coord_pos * coord.element_size;
			void* dst = result[coord_idx].data() + static_cast<size_t>(row) * coord.element_size;
			std::memcpy(dst, src, coord.element_size);
		}
		
		// For data column, read value from chunk data
		// This is the key pivot operation: convert chunk-relative indices to chunk data offset
		idx_t data_col_idx = coordinates_.size();
		void* dst = result[data_col_idx].data() + static_cast<size_t>(row) * element_size;
		
		// Read the value from chunk data at chunk-relative position
		switch (metadata_.dtype) {
			case ZarrDtype::INT8: {
				int8_t val = ReadValue<int8_t>(chunk_data, chunk_idx);
				std::memcpy(dst, &val, element_size);
				break;
			}
			case ZarrDtype::INT16: {
				int16_t val = ReadValue<int16_t>(chunk_data, chunk_idx);
				std::memcpy(dst, &val, element_size);
				break;
			}
			case ZarrDtype::INT32: {
				int32_t val = ReadValue<int32_t>(chunk_data, chunk_idx);
				std::memcpy(dst, &val, element_size);
				break;
			}
			case ZarrDtype::INT64: {
				int64_t val = ReadValue<int64_t>(chunk_data, chunk_idx);
				std::memcpy(dst, &val, element_size);
				break;
			}
			case ZarrDtype::UINT8: {
				uint8_t val = ReadValue<uint8_t>(chunk_data, chunk_idx);
				std::memcpy(dst, &val, element_size);
				break;
			}
			case ZarrDtype::UINT16: {
				uint16_t val = ReadValue<uint16_t>(chunk_data, chunk_idx);
				std::memcpy(dst, &val, element_size);
				break;
			}
			case ZarrDtype::UINT32: {
				uint32_t val = ReadValue<uint32_t>(chunk_data, chunk_idx);
				std::memcpy(dst, &val, element_size);
				break;
			}
			case ZarrDtype::UINT64: {
				uint64_t val = ReadValue<uint64_t>(chunk_data, chunk_idx);
				std::memcpy(dst, &val, element_size);
				break;
			}
			case ZarrDtype::FLOAT32: {
				float val = ReadValue<float>(chunk_data, chunk_idx);
				std::memcpy(dst, &val, element_size);
				break;
			}
			case ZarrDtype::FLOAT64: {
				double val = ReadValue<double>(chunk_data, chunk_idx);
				std::memcpy(dst, &val, element_size);
				break;
			}
			default:
				// For unknown types, zero-initialize
				std::memset(dst, 0, element_size);
				break;
		}
	}
	
	return result;
}

// ============================================================================
// PivotToVectors Implementation
// ============================================================================

void PivotToVectors(
    const std::vector<std::vector<uint8_t>>& pivoted_data,
    const std::vector<LogicalType>& column_types,
    std::vector<Vector>& result) {
	
	size_t num_cols = pivoted_data.size();
	result.resize(num_cols);
	
	for (size_t col = 0; col < num_cols; col++) {
		auto& vec = result[col];
		vec.Reference(pivoted_data[col].data(), 0, pivoted_data[col].size());
		// Set the type - this would require more complex integration with DuckDB
		// For now, this is a placeholder
	}
}

} // namespace zarr
} // namespace duckdb
