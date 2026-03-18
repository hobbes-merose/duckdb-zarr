#pragma once

#include "zarr/metadata.hpp"
#include "duckdb.hpp"
#include <string>
#include <vector>
#include <memory>
#include <optional>

namespace duckdb {
namespace zarr {

/// Represents decompressed chunk data
struct ChunkData {
	std::vector<uint8_t> data;           // Decompressed data
	std::vector<idx_t> shape;            // Shape of this chunk
	idx_t num_elements;                   // Number of elements in chunk
	ZarrDtype dtype;                      // Data type
	idx_t element_size;                   // Bytes per element
};

/// Supported compression codecs
enum class CompressionCodec {
	NONE,    // No compression
	BLOSC,   // Blosc compressor
	ZSTD,    // Zstd compression
	LZ4,     // LZ4 compression
	GZIP,    // Gzip compression
	UNKNOWN  // Unknown compression
};

/// Configuration for a compression codec
struct CodecConfiguration {
	CompressionCodec codec;
	// Codec-specific settings (e.g., compression level)
	int compression_level = 3;
};

/// Chunk reader for reading and decompressing Zarr chunk data
class ChunkReader {
public:
	/// Constructor
	ChunkReader(const std::string& path, const ZarrArrayMetadata& metadata);

	/// Read a specific chunk by indices
	ChunkData ReadChunk(const std::vector<idx_t>& chunk_indices);

	/// Read multiple chunks
	std::vector<ChunkData> ReadChunks(const std::vector<std::vector<idx_t>>& chunk_indices);

	/// Get the path to a chunk file
	std::string GetChunkPath(const std::vector<idx_t>& chunk_indices);

	/// Decompress raw data based on codec
	static std::vector<uint8_t> Decompress(
		const uint8_t* compressed_data,
		size_t compressed_size,
		size_t uncompressed_size,
		const CodecConfiguration& config);

	/// Detect compression codec from metadata
	static CompressionCodec DetectCodec(const ZarrArrayMetadata& metadata);

	/// Parse codec configuration from metadata
	static CodecConfiguration ParseCodecConfig(const ZarrArrayMetadata& metadata);

private:
	std::string base_path_;
	ZarrArrayMetadata metadata_;
	CodecConfiguration codec_config_;

	/// Read raw bytes from a file
	std::vector<uint8_t> ReadFile(const std::string& path);

	/// Decompress using blosc
	static std::vector<uint8_t> DecompressBlosc(
		const uint8_t* compressed_data,
		size_t compressed_size,
		size_t uncompressed_size);

	/// Decompress using zstd
	static std::vector<uint8_t> DecompressZstd(
		const uint8_t* compressed_data,
		size_t compressed_size,
		size_t uncompressed_size);

	/// Decompress using lz4
	static std::vector<uint8_t> DecompressLz4(
		const uint8_t* compressed_data,
		size_t compressed_size,
		size_t uncompressed_size);

	/// Decompress using gzip
	static std::vector<uint8_t> DecompressGzip(
		const uint8_t* compressed_data,
		size_t compressed_size,
		size_t uncompressed_size);
};

} // namespace zarr
} // namespace duckdb
