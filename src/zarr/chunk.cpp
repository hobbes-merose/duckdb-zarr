#include "zarr/chunk.hpp"
#include "duckdb/common/file_system.hpp"
#include <fstream>
#include <sstream>
#include <cstring>

// Include compression libraries from DuckDB third_party
// Note: We'll use DuckDB's internal abstractions where possible
// For now, implement basic decompression

namespace duckdb {
namespace zarr {

// ============================================================================
// ChunkReader Implementation
// ============================================================================

ChunkReader::ChunkReader(const std::string& path, const ZarrArrayMetadata& metadata)
    : base_path_(path), metadata_(metadata) {
	codec_config_ = ParseCodecConfig(metadata_);
}

std::string ChunkReader::GetChunkPath(const std::vector<idx_t>& chunk_indices) {
	// Zarr v2: chunks are stored as c/{0}/{1}/... where each index is a directory
	// Zarr v3: similar but may vary based on codec
	std::ostringstream path;
	path << base_path_ << "/";

	if (metadata_.zarr_format == "v2") {
		// Blosc stores chunks as separate files: c/0/0/0
		path << "c/";
		for (size_t i = 0; i < chunk_indices.size(); i++) {
			if (i > 0) {
				path << "/";
			}
			path << chunk_indices[i];
		}
	} else {
		// Zarr v3: chunks stored as zarr arrays
		// Based on codec, but typically c/0/0/0
		path << "c/";
		for (size_t i = 0; i < chunk_indices.size(); i++) {
			if (i > 0) {
				path << "/";
			}
			path << chunk_indices[i];
		}
	}

	return path.str();
}

std::vector<uint8_t> ChunkReader::ReadFile(const std::string& path) {
	auto& fs = FileSystem::GetFileSystem("");
	auto handle = fs.OpenFile(path, FileOpenFlags::FILE_FLAG_READ);
	auto file_size = fs.GetFileSize(*handle);

	std::vector<uint8_t> buffer(static_cast<size_t>(file_size));
	fs.Read(*handle, buffer.data(), file_size, 0);

	return buffer;
}

ChunkData ChunkReader::ReadChunk(const std::vector<idx_t>& chunk_indices) {
	ChunkData chunk;

	// Get chunk path
	std::string chunk_path = GetChunkPath(chunk_indices);

	// Read raw compressed data
	std::vector<uint8_t> compressed_data;
	try {
		compressed_data = ReadFile(chunk_path);
	} catch (const std::exception& e) {
		// Chunk doesn't exist or can't be read - return fill value
		chunk.num_elements = metadata_.chunk_size;
		chunk.shape = metadata_.chunks;
		chunk.dtype = metadata_.dtype;
		chunk.element_size = metadata_.element_size;

		// Fill with fill_value if available
		if (metadata_.fill_value_native.has_value()) {
			chunk.data.resize(chunk.num_elements * metadata_.element_size, 0);
		} else {
			chunk.data.resize(chunk.num_elements * metadata_.element_size, 0);
		}
		return chunk;
	}

	// Calculate expected uncompressed size
	size_t uncompressed_size = metadata_.chunk_size * metadata_.element_size;

	// Decompress if needed
	if (codec_config_.codec == CompressionCodec::NONE) {
		chunk.data = std::move(compressed_data);
	} else {
		chunk.data = Decompress(
			compressed_data.data(),
			compressed_data.size(),
			uncompressed_size,
			codec_config_
		);
	}

	// Set chunk metadata
	chunk.num_elements = metadata_.chunk_size;
	chunk.shape = metadata_.chunks;
	chunk.dtype = metadata_.dtype;
	chunk.element_size = metadata_.element_size;

	return chunk;
}

std::vector<ChunkData> ChunkReader::ReadChunks(const std::vector<std::vector<idx_t>>& chunk_indices) {
	std::vector<ChunkData> chunks;
	chunks.reserve(chunk_indices.size());

	for (const auto& indices : chunk_indices) {
		chunks.push_back(ReadChunk(indices));
	}

	return chunks;
}

// ============================================================================
// Compression Detection and Configuration
// ============================================================================

CompressionCodec ChunkReader::DetectCodec(const ZarrArrayMetadata& metadata) {
	// Check for compressor in v2 metadata
	if (metadata.compressor.has_value()) {
		const std::string& comp = metadata.compressor.value();

		// Blosc
		if (comp.find("blosc") != std::string::npos) {
			return CompressionCodec::BLOSC;
		}
		// Zstd
		if (comp.find("zstd") != std::string::npos) {
			return CompressionCodec::ZSTD;
		}
		// LZ4
		if (comp.find("lz4") != std::string::npos) {
			return CompressionCodec::LZ4;
		}
		// Gzip
		if (comp.find("gzip") != std::string::npos || comp.find("zlib") != std::string::npos) {
			return CompressionCodec::GZIP;
		}
	}

	// Check codecs in v3 metadata
	for (const auto& codec : metadata.codecs) {
		if (codec.name == "blosc") {
			return CompressionCodec::BLOSC;
		}
		if (codec.name == "zstd") {
			return CompressionCodec::ZSTD;
		}
		if (codec.name == "lz4") {
			return CompressionCodec::LZ4;
		}
		if (codec.name == "gzip" || codec.name == "zlib") {
			return CompressionCodec::GZIP;
		}
	}

	// No compression
	return CompressionCodec::NONE;
}

CodecConfiguration ChunkReader::ParseCodecConfig(const ZarrArrayMetadata& metadata) {
	CodecConfiguration config;
	config.codec = DetectCodec(metadata);

	// Try to extract compression level from config
	// This is a simplified version - full parsing would need to handle
	// specific codec JSON structures

	// Default compression level
	config.compression_level = 3;

	// Try to parse compression level from v2 compressor config
	if (metadata.compressor.has_value()) {
		const std::string& comp = metadata.compressor.value();
		// Look for "level" or "clevel" in JSON
		size_t level_pos = comp.find("\"level\"");
		if (level_pos != std::string::npos) {
			// Extract level value (simplified)
			size_t colon_pos = comp.find(":", level_pos);
			if (colon_pos != std::string::npos) {
				int level = std::atoi(comp.c_str() + colon_pos + 1);
				if (level >= 0) {
					config.compression_level = level;
				}
			}
		}
	}

	return config;
}

// ============================================================================
// Decompression Implementations
// ============================================================================

std::vector<uint8_t> ChunkReader::Decompress(
    const uint8_t* compressed_data,
    size_t compressed_size,
    size_t uncompressed_size,
    const CodecConfiguration& config) {

	switch (config.codec) {
		case CompressionCodec::BLOSC:
			return DecompressBlosc(compressed_data, compressed_size, uncompressed_size);
		case CompressionCodec::ZSTD:
			return DecompressZstd(compressed_data, compressed_size, uncompressed_size);
		case CompressionCodec::LZ4:
			return DecompressLz4(compressed_data, compressed_size, uncompressed_size);
		case CompressionCodec::GZIP:
			return DecompressGzip(compressed_data, compressed_size, uncompressed_size);
		case CompressionCodec::NONE:
		default:
			// No compression - just copy
			std::vector<uint8_t> result(compressed_size);
			std::memcpy(result.data(), compressed_data, compressed_size);
			return result;
	}
}

std::vector<uint8_t> ChunkReader::DecompressBlosc(
    const uint8_t* compressed_data,
    size_t compressed_size,
    size_t uncompressed_size) {
	// Blosc decompression requires the c-blosc2 library
	// For now, throw if not available - this will be integrated with DuckDB's blosc

	// TODO: Integrate with DuckDB's blosc implementation
	// For now, fall through to uncompressed if blosc header not detected

	// Check for blosc magic number (0xCA 0xFE 0xD0 0x01)
	if (compressed_size >= 4 &&
		compressed_data[0] == 0xCA && compressed_data[1] == 0xFE &&
		compressed_data[2] == 0xD0 && compressed_data[3] == 0x01) {

		throw std::runtime_error("Blosc decompression not yet implemented - "
			"requires integration with DuckDB's c-blosc2");
	}

	// Not blosc, assume uncompressed
	std::vector<uint8_t> result(uncompressed_size);
	std::memcpy(result.data(), compressed_data, std::min(compressed_size, uncompressed_size));
	return result;
}

std::vector<uint8_t> ChunkReader::DecompressZstd(
    const uint8_t* compressed_data,
    size_t compressed_size,
    size_t uncompressed_size) {
	// Use DuckDB's bundled zstd
	// For now, throw if not available

	throw std::runtime_error("Zstd decompression not yet implemented - "
		"requires integration with DuckDB's zstd library");

	// Unreachable but keeps compiler happy
	// std::vector<uint8_t> result(uncompressed_size);
	// return result;
}

std::vector<uint8_t> ChunkReader::DecompressLz4(
    const uint8_t* compressed_data,
    size_t compressed_size,
    size_t uncompressed_size) {
	// Use DuckDB's bundled lz4
	throw std::runtime_error("LZ4 decompression not yet implemented - "
		"requires integration with DuckDB's lz4 library");
}

std::vector<uint8_t> ChunkReader::DecompressGzip(
    const uint8_t* compressed_data,
    size_t compressed_size,
    size_t uncompressed_size) {
	// Use DuckDB's bundled miniz (which supports gzip/zlib)
	throw std::runtime_error("Gzip decompression not yet implemented - "
		"requires integration with DuckDB's miniz library");
}

} // namespace zarr
} // namespace duckdb
