#include "zarr/metadata.hpp"
#include "duckdb/common/file_system.hpp"
#include "yyjson.hpp"
#include <sstream>

namespace duckdb {
namespace zarr {

// Helper: trim whitespace from string
static string Trim(const string& s) {
    size_t start = s.find_first_not_of(" \t\n\r");
    if (start == string::npos) return "";
    size_t end = s.find_last_not_of(" \t\n\r");
    return s.substr(start, end - start + 1);
}

// Helper: convert yyjson val to string
static string JsonGetString(yyjson_val* obj, const char* key) {
    yyjson_val* val = yyjson_obj_get(obj, key);
    if (!val || !yyjson_is_str(val)) return "";
    return string(yyjson_get_str(val), yyjson_get_len(val));
}

// Helper: get integer from JSON
static idx_t JsonGetInt(yyjson_val* obj, const char* key, idx_t default_val = 0) {
    yyjson_val* val = yyjson_obj_get(obj, key);
    if (!val) return default_val;
    if (yyjson_is_int(val)) {
        return static_cast<idx_t>(yyjson_get_int(val));
    }
    if (yyjson_is_uint(val)) {
        return static_cast<idx_t>(yyjson_get_uint(val));
    }
    return default_val;
}

// Helper: parse dtype string
ZarrDtype ZarrMetadataParser::ParseDtype(const string& dtype_str) {
    // Handle endianness prefix (e.g., '>i4' or '<f8')
    string dtype = dtype_str;
    size_t pos = 0;
    if (dtype.length() > 1 && (dtype[0] == '>' || dtype[0] == '<' || dtype[0] == '=')) {
        pos = 1;
    }
    // Also handle byte order markers
    if (dtype.length() > 2 && dtype[pos] == '|') {
        pos++;
    }

    string base = dtype.substr(pos);

    if (base == "i1" || base == "int8") return ZarrDtype::INT8;
    if (base == "i2" || base == "int16") return ZarrDtype::INT16;
    if (base == "i4" || base == "int32") return ZarrDtype::INT32;
    if (base == "i8" || base == "int64") return ZarrDtype::INT64;
    if (base == "u1" || base == "uint8") return ZarrDtype::UINT8;
    if (base == "u2" || base == "uint16") return ZarrDtype::UINT16;
    if (base == "u4" || base == "uint32") return ZarrDtype::UINT32;
    if (base == "u8" || base == "uint64") return ZarrDtype::UINT64;
    if (base == "f4" || base == "float32") return ZarrDtype::FLOAT32;
    if (base == "f8" || base == "float64") return ZarrDtype::FLOAT64;
    if (base == "U" || base == "str" || base == "unicode") return ZarrDtype::STRING;
    if (base == "S" || base == "bytes") return ZarrDtype::BYTES;
    if (base == "b" || base == "bool") return ZarrDtype::BOOL;
    if (base == "c8") return ZarrDtype::COMPLEX64;
    if (base == "c16") return ZarrDtype::COMPLEX128;

    return ZarrDtype::UNKNOWN;
}

// Helper: get element size in bytes
static idx_t GetElementSize(ZarrDtype dtype) {
    switch (dtype) {
        case ZarrDtype::INT8:
        case ZarrDtype::UINT8:
        case ZarrDtype::BOOL:
            return 1;
        case ZarrDtype::INT16:
        case ZarrDtype::UINT16:
            return 2;
        case ZarrDtype::INT32:
        case ZarrDtype::UINT32:
        case ZarrDtype::FLOAT32:
            return 4;
        case ZarrDtype::INT64:
        case ZarrDtype::UINT64:
        case ZarrDtype::FLOAT64:
            return 8;
        case ZarrDtype::COMPLEX64:
            return 8;
        case ZarrDtype::COMPLEX128:
            return 16;
        case ZarrDtype::STRING:
        case ZarrDtype::BYTES:
            return 0; // Variable size
        default:
            return 0;
    }
}

// Convert ZarrDtype to DuckDB LogicalType
LogicalType ZarrMetadataParser::ToDuckDBType(ZarrDtype dtype) {
    switch (dtype) {
        case ZarrDtype::INT8:
            return LogicalType::TINYINT;
        case ZarrDtype::INT16:
            return LogicalType::SMALLINT;
        case ZarrDtype::INT32:
            return LogicalType::INTEGER;
        case ZarrDtype::INT64:
            return LogicalType::BIGINT;
        case ZarrDtype::UINT8:
            return LogicalType::UTINYINT;
        case ZarrDtype::UINT16:
            return LogicalType::USMALLINT;
        case ZarrDtype::UINT32:
            return LogicalType::UINTEGER;
        case ZarrDtype::UINT64:
            return LogicalType::UBIGINT;
        case ZarrDtype::FLOAT32:
            return LogicalType::FLOAT;
        case ZarrDtype::FLOAT64:
            return LogicalType::DOUBLE;
        case ZarrDtype::STRING:
            return LogicalType::VARCHAR;
        case ZarrDtype::BOOL:
            return LogicalType::BOOLEAN;
        case ZarrDtype::BYTES:
            return LogicalType::BLOB;
        default:
            return LogicalType::BLOB;
    }
}

// Validate metadata consistency
void ZarrMetadataParser::Validate(const ZarrArrayMetadata& metadata) {
    if (metadata.shape.empty()) {
        throw std::runtime_error("Zarr array shape cannot be empty");
    }
    if (metadata.chunks.empty()) {
        throw std::runtime_error("Zarr array chunks cannot be empty");
    }
    if (metadata.shape.size() != metadata.chunks.size()) {
        throw std::runtime_error("Zarr array shape and chunks must have same dimensionality");
    }
    if (metadata.element_size == 0 && metadata.dtype != ZarrDtype::STRING && metadata.dtype != ZarrDtype::BYTES) {
        throw std::runtime_error("Unknown or unsupported Zarr dtype");
    }
}

// Parse Zarr v2 metadata from .zarray file
ZarrArrayMetadata ZarrMetadataParser::ParseV2(const string& zarray_path) {
    ZarrArrayMetadata metadata;
    metadata.zarr_format = "v2";

    // Read file content
    auto& fs = FileSystem::GetFileSystem("");
    auto handle = fs.OpenFile(zarray_path, FileOpenFlags::FILE_FLAG_READ);
    auto file_size = fs.GetFileSize(*handle);

    vector<char> buffer(static_cast<size_t>(file_size) + 1);
    fs.Read(*handle, buffer.data(), file_size, 0);
    buffer[file_size] = '\0';

    // Parse JSON with yyjson
    yyjson_doc* doc = yyjson_read(buffer.data(), file_size, 0);
    if (!doc) {
        throw std::runtime_error("Failed to parse Zarr metadata JSON: " + zarray_path);
    }

    yyjson_val* root = yyjson_doc_get_root(doc);

    // Extract fields
    metadata.name = JsonGetString(root, "name");

    // Shape (array of dimensions)
    yyjson_val* shape_val = yyjson_obj_get(root, "shape");
    if (shape_val && yyjson_is_arr(shape_val)) {
        size_t idx, len;
        yyjson_val* dim_val;
        yyjson_arr_foreach(shape_val, idx, len, dim_val) {
            metadata.shape.push_back(static_cast<idx_t>(yyjson_get_int(dim_val)));
        }
    }

    // Chunks
    yyjson_val* chunks_val = yyjson_obj_get(root, "chunks");
    if (chunks_val && yyjson_is_arr(chunks_val)) {
        size_t idx, len;
        yyjson_val* dim_val;
        yyjson_arr_foreach(chunks_val, idx, len, dim_val) {
            metadata.chunks.push_back(static_cast<idx_t>(yyjson_get_int(dim_val)));
        }
    }

    // Dtype
    string dtype_str = JsonGetString(root, "dtype");
    metadata.dtype = ParseDtype(dtype_str);
    metadata.element_size = GetElementSize(metadata.dtype);

    // Fill value
    yyjson_val* fill_val = yyjson_obj_get(root, "fill_value");
    if (fill_val) {
        if (yyjson_is_str(fill_val)) {
            metadata.fill_value = string(yyjson_get_str(fill_val), yyjson_get_len(fill_val));
        } else if (yyjson_is_int(fill_val)) {
            metadata.fill_value = std::to_string(yyjson_get_int(fill_val));
        } else if (yyjson_is_real(fill_val)) {
            metadata.fill_value = std::to_string(yyjson_get_real(fill_val));
        }
    }

    // Compressor
    yyjson_val* compressor_val = yyjson_obj_get(root, "compressor");
    if (compressor_val) {
        // Serialize compressor config to string
        char* json_str = yyjson_val_write(compressor_val, 0, nullptr);
        if (json_str) {
            metadata.compressor = json_str;
            free(json_str);
        }
    }

    // Order
    metadata.order = JsonGetString(root, "order");

    // Filters
    yyjson_val* filters_val = yyjson_obj_get(root, "filters");
    if (filters_val) {
        char* json_str = yyjson_val_write(filters_val, 0, nullptr);
        if (json_str) {
            metadata.filters = json_str;
            free(json_str);
        }
    }

    // Attributes
    yyjson_val* attrs_val = yyjson_obj_get(root, "attributes");
    if (attrs_val) {
        char* json_str = yyjson_val_write(attrs_val, 0, nullptr);
        if (json_str) {
            metadata.attributes = json_str;
            free(json_str);
        }
    }

    yyjson_doc_free(doc);

    // Calculate chunk size
    metadata.chunk_size = 1;
    for (auto chunk_dim : metadata.chunks) {
        metadata.chunk_size *= chunk_dim;
    }

    // Calculate total chunks
    metadata.num_chunks = 1;
    for (size_t i = 0; i < metadata.shape.size(); i++) {
        idx_t dim_chunks = (metadata.shape[i] + metadata.chunks[i] - 1) / metadata.chunks[i];
        metadata.num_chunks *= dim_chunks;
    }

    Validate(metadata);
    return metadata;
}

// Parse Zarr v3 metadata from z.json
ZarrArrayMetadata ZarrMetadataParser::ParseV3(const string& z_json_path) {
    ZarrArrayMetadata metadata;
    metadata.zarr_format = "v3";

    // Read file content
    auto& fs = FileSystem::GetFileSystem("");
    auto handle = fs.OpenFile(z_json_path, FileOpenFlags::FILE_FLAG_READ);
    auto file_size = fs.GetFileSize(*handle);

    vector<char> buffer(static_cast<size_t>(file_size) + 1);
    fs.Read(*handle, buffer.data(), file_size, 0);
    buffer[file_size] = '\0';

    // Parse JSON with yyjson
    yyjson_doc* doc = yyjson_read(buffer.data(), file_size, 0);
    if (!doc) {
        throw std::runtime_error("Failed to parse Zarr v3 metadata JSON: " + z_json_path);
    }

    yyjson_val* root = yyjson_doc_get_root(doc);

    // Extract fields
    metadata.name = JsonGetString(root, "name");

    // Shape
    yyjson_val* shape_val = yyjson_obj_get(root, "shape");
    if (shape_val && yyjson_is_arr(shape_val)) {
        size_t idx, len;
        yyjson_val* dim_val;
        yyjson_arr_foreach(shape_val, idx, len, dim_val) {
            metadata.shape.push_back(static_cast<idx_t>(yyjson_get_int(dim_val)));
        }
    }

    // Chunk shape (called "chunk_grid" in v3)
    yyjson_val* chunk_grid_val = yyjson_obj_get(root, "chunk_grid");
    if (chunk_grid_val) {
        yyjson_val* chunk_shape_val = yyjson_obj_get(chunk_grid_val, "chunk_shape");
        if (chunk_shape_val && yyjson_is_arr(chunk_shape_val)) {
            size_t idx, len;
            yyjson_val* dim_val;
            yyjson_arr_foreach(chunk_shape_val, idx, len, dim_val) {
                metadata.chunks.push_back(static_cast<idx_t>(yyjson_get_int(dim_val)));
            }
        }
    }

    // Fallback: look for "chunk_shape" directly
    if (metadata.chunks.empty()) {
        yyjson_val* chunk_shape_val = yyjson_obj_get(root, "chunk_shape");
        if (chunk_shape_val && yyjson_is_arr(chunk_shape_val)) {
            size_t idx, len;
            yyjson_val* dim_val;
            yyjson_arr_foreach(chunk_shape_val, idx, len, dim_val) {
                metadata.chunks.push_back(static_cast<idx_t>(yyjson_get_int(dim_val)));
            }
        }
    }

    // Dtype
    string dtype_str = JsonGetString(root, "dtype");
    metadata.dtype = ParseDtype(dtype_str);
    metadata.element_size = GetElementSize(metadata.dtype);

    // Fill value
    yyjson_val* fill_val = yyjson_obj_get(root, "fill_value");
    if (fill_val) {
        if (yyjson_is_str(fill_val)) {
            metadata.fill_value = string(yyjson_get_str(fill_val), yyjson_get_len(fill_val));
        } else if (yyjson_is_null(fill_val)) {
            metadata.fill_value = "null";
        } else if (yyjson_is_int(fill_val)) {
            metadata.fill_value = std::to_string(yyjson_get_int(fill_val));
        } else if (yyjson_is_real(fill_val)) {
            metadata.fill_value = std::to_string(yyjson_get_real(fill_val));
        }
    }

    // Codecs
    yyjson_val* codecs_val = yyjson_obj_get(root, "codecs");
    if (codecs_val && yyjson_is_arr(codecs_val)) {
        size_t idx, len;
        yyjson_val* codec_val;
        yyjson_arr_foreach(codecs_val, idx, len, codec_val) {
            CodecConfig codec;
            codec.name = JsonGetString(codec_val, "codec");
            char* json_str = yyjson_val_write(codec_val, 0, nullptr);
            if (json_str) {
                codec.configuration = json_str;
                free(json_str);
            }
            metadata.codecs.push_back(codec);
        }
    }

    // Attributes
    yyjson_val* attrs_val = yyjson_obj_get(root, "attributes");
    if (attrs_val) {
        char* json_str = yyjson_val_write(attrs_val, 0, nullptr);
        if (json_str) {
            metadata.attributes = json_str;
            free(json_str);
        }
    }

    yyjson_doc_free(doc);

    // Calculate chunk size
    metadata.chunk_size = 1;
    for (auto chunk_dim : metadata.chunks) {
        metadata.chunk_size *= chunk_dim;
    }

    // Calculate total chunks
    metadata.num_chunks = 1;
    for (size_t i = 0; i < metadata.shape.size(); i++) {
        idx_t dim_chunks = (metadata.shape[i] + metadata.chunks[i] - 1) / metadata.chunks[i];
        metadata.num_chunks *= dim_chunks;
    }

    Validate(metadata);
    return metadata;
}

// Auto-detect format and parse metadata
ZarrArrayMetadata ZarrMetadataParser::Parse(const string& path) {
    auto& fs = FileSystem::GetFileSystem("");

    // Try .zarray first (Zarr v2)
    string zarray_path = path;
    if (fs.FileExists(zarray_path)) {
        // Check if it's a directory (Zarr v2 stores metadata in .zarray file)
        if (!fs.DirectoryExists(zarray_path)) {
            return ParseV2(zarray_path);
        }
    }

    // Try z.json (Zarr v3)
    string z_json_path = path + "/z.json";
    if (fs.FileExists(z_json_path)) {
        return ParseV3(z_json_path);
    }

    // Try .zarray in directory (Zarr v2)
    string zarray_in_dir = path + "/.zarray";
    if (fs.FileExists(zarray_in_dir)) {
        return ParseV2(zarray_in_dir);
    }

    throw std::runtime_error("Cannot determine Zarr format: " + path);
}

// Parse metadata from JSON string (for testing)
string ZarrMetadataParser::ParseFromString(const string& json_str) {
    // For testing, just echo back the parsed JSON in a normalized form
    yyjson_doc* doc = yyjson_read(json_str.c_str(), json_str.size(), 0);
    if (!doc) {
        throw std::runtime_error("Failed to parse JSON string");
    }

    yyjson_val* root = yyjson_doc_get_root(doc);

    // Extract and return key fields
    string shape_str = JsonGetString(root, "shape");
    string chunks_str = JsonGetString(root, "chunks");
    string dtype_str = JsonGetString(root, "dtype");

    yyjson_doc_free(doc);

    // Return a formatted string
    std::ostringstream result;
    result << "{\"shape\":" << (shape_str.empty() ? "[]" : shape_str)
           << ",\"chunks\":" << (chunks_str.empty() ? "[]" : chunks_str)
           << ",\"dtype\":\"" << dtype_str << "\"}";
    return result.str();
}

} // namespace zarr
} // namespace duckdb
