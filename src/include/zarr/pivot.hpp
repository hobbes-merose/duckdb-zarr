#pragma once

#include "zarr/metadata.hpp"
#include "zarr/chunk.hpp"
#include "duckdb.hpp"
#include <vector>
#include <cstdint>

namespace duckdb {
namespace zarr {

/// Represents a coordinate array (1D array from Zarr store)
struct CoordinateArray {
	std::string name;           // Coordinate array name (e.g., "lat", "lon")
	std::vector<uint8_t> data; // Raw coordinate data
	idx_t size;                // Number of elements
	ZarrDtype dtype;           // Data type
	idx_t element_size;        // Bytes per element
};

/// Represents the pivot configuration
struct PivotConfig {
	std::vector<CoordinateArray> coordinates;  // Coordinate arrays (1D)
	std::vector<ZarrArrayMetadata> data_arrays; // Data arrays (nD)
	bool use_dictionary_encoding;                // Use dictionary encoding for coordinates
};

/// Pivot algorithm for converting n-dimensional arrays to tabular format
class PivotAlgorithm {
public:
	/// Constructor
	PivotAlgorithm(const ZarrArrayMetadata& metadata, const std::vector<CoordinateArray>& coordinates);

	/// Pivot a single chunk to row-oriented format
	/// @param chunk_data The chunk data to pivot
	/// @param chunk_indices The indices of the chunk in the array
	/// @return Vector of column data (each column is a vector of values)
	std::vector<std::vector<uint8_t>> PivotChunk(
		const ChunkData& chunk_data,
		const std::vector<idx_t>& chunk_indices);

	/// Calculate the total number of rows in the output
	idx_t GetTotalRows() const;

	/// Get the number of output columns (coordinates + data variables)
	idx_t GetNumColumns() const;

	/// Get column names
	std::vector<std::string> GetColumnNames() const;

	/// Get column types
	std::vector<LogicalType> GetColumnTypes() const;

private:
	ZarrArrayMetadata metadata_;
	std::vector<CoordinateArray> coordinates_;
	std::vector<idx_t> shape_;
	std::vector<idx_t> chunks_;

	/// Calculate stride for each dimension
	std::vector<idx_t> CalculateStrides() const;

	/// Calculate the linear index from multidimensional indices
	idx_t CalculateLinearIndex(const std::vector<idx_t>& indices) const;

	/// Calculate multidimensional indices from linear index
	std::vector<idx_t> CalculateMultidimensionalIndex(idx_t linear_idx) const;

	/// Read a value from chunk data at the given indices
	template<typename T>
	T ReadValue(const ChunkData& chunk_data, const std::vector<idx_t>& indices) const;
};

/// Convert pivot output to DuckDB vectors
void PivotToVectors(
	const std::vector<std::vector<uint8_t>>& pivoted_data,
	const std::vector<LogicalType>& column_types,
	std::vector<Vector>& result);

} // namespace zarr
} // namespace duckdb
