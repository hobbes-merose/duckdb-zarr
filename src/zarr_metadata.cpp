#include "zarr_metadata.hpp"
#include "duckdb/common/common.hpp"
#include "duckdb/common/file_system.hpp"
#include <nlohmann/json.hpp>

namespace duckdb {

ZarrMetadata::ZarrMetadata() : is_valid_(false) {
}

void ZarrMetadata::Parse(const std::string &path) {
    path_ = path;
    arrays_.clear();
    is_valid_ = false;
    error_message_.clear();
    
    try {
        // Get the file system
        auto &fs = FileSystem::GetFileSystem(nullptr);
        
        // Try to read the zarr.json file (Zarr v3 group metadata)
        // or check if it's a directory with .zarray files (Zarr v2)
        
        std::string metadata_path = path;
        if (!path.ends_with("/") && !path.ends_with(".zarr")) {
            // Handle cases where path might not have trailing slash
        }
        
        // First, try to read zarr.json (Zarr v3)
        std::string zarr_json_path = path;
        if (!zarr_json_path.ends_with("/")) {
            zarr_json_path += "/";
        }
        zarr_json_path += "zarr.json";
        
        try {
            auto file_handle = fs.OpenFile(zarr_json_path, FileOpenFlags::FILE_FLAGS_READ);
            idx_t file_size = fs.GetFileSize(*file_handle);
            std::string content(file_size, '\0');
            fs.Read(*file_handle, char_ptr_cast(content.data()), file_size);
            
            // Parse as JSON
            auto json = nlohmann::json::parse(content);
            
            // Check if it's a group or array
            if (json.contains("zarr_format") && json["zarr_format"] == 3) {
                // Zarr v3 format
                if (json.contains("metadata")) {
                    auto &metadata = json["metadata"];
                    
                    // Check if it's an array (has shape)
                    if (metadata.contains("shape")) {
                        auto array_meta = ParseZarrJson(json, "");
                        if (array_meta) {
                            arrays_.push_back(*array_meta);
                        }
                    }
                    // Check for arrays in subgroups
                    if (json.contains("attributes") && json["attributes"].contains("zarr")) {
                        auto &zarr_attrs = json["attributes"]["zarr"];
                        if (zarr_attrs.contains("children")) {
                            // Has child arrays/groups
                        }
                    }
                }
            }
            
            is_valid_ = true;
            return;
        } catch (std::exception &) {
            // zarr.json doesn't exist, try Zarr v2 approach
        }
        
        // For Zarr v2, we need to list files in the directory
        // and find .zarray files (array metadata)
        std::vector<std::string> zarray_files;
        
        try {
            fs.ListFiles(path, [&](const std::string &fname, bool is_dir) {
                if (fname == ".zarray") {
                    zarray_files.push_back("");
                } else if (fname.find("/.zarray") != std::string::npos) {
                    // Nested array
                    zarray_files.push_back(fname.substr(0, fname.size() - 8)); // Remove /.zarray
                }
            });
        } catch (std::exception &e) {
            error_message_ = "Failed to list directory: " + std::string(e.what());
            return;
        }
        
        // Parse .zarray for each array found
        for (const auto &array_path : zarray_files) {
            std::string zarray_path = path;
            if (!zarray_path.ends_with("/")) {
                zarray_path += "/";
            }
            zarray_path += array_path;
            if (!array_path.empty() && !zarray_path.ends_with("/")) {
                zarray_path += "/";
            }
            zarray_path += ".zarray";
            
            try {
                auto file_handle = fs.OpenFile(zarray_path, FileOpenFlags::FILE_FLAGS_READ);
                idx_t file_size = fs.GetFileSize(*file_handle);
                std::string content(file_size, '\0');
                fs.Read(*file_handle, char_ptr_cast(content.data()), file_size);
                
                std::string name = array_path;
                if (name.starts_with("/")) {
                    name = name.substr(1);
                }
                if (name.ends_with("/.zarray")) {
                    name = name.substr(0, name.size() - 8);
                }
                
                arrays_.push_back(ParseZarray(content, name));
            } catch (std::exception &e) {
                // Skip this file if we can't read it
                continue;
            }
        }
        
        // If we found arrays, mark as valid
        if (!arrays_.empty()) {
            is_valid_ = true;
        } else {
            error_message_ = "No zarr arrays found in path: " + path;
        }
        
    } catch (std::exception &e) {
        error_message_ = std::string(e.what());
        is_valid_ = false;
    }
}

ZarrArrayMetadata ZarrMetadata::ParseZarray(const std::string &zarray_content, const std::string &name) {
    ZarrArrayMetadata meta;
    meta.name = name;
    meta.zarr_version = 2;
    meta.compressor = "none";
    
    try {
        auto json = nlohmann::json::parse(zarray_content);
        
        // Shape
        if (json.contains("shape")) {
            for (const auto &dim : json["shape"]) {
                meta.shape.push_back(dim.get<int64_t>());
            }
        }
        
        // Chunks
        if (json.contains("chunks")) {
            for (const auto &chunk : json["chunks"]) {
                meta.chunks.push_back(chunk.get<int64_t>());
            }
        }
        
        // Dtype
        if (json.contains("dtype")) {
            meta.dtype = NormalizeDtype(json["dtype"].get<std::string>());
        }
        
        // Fill value
        if (json.contains("fill_value")) {
            meta.fill_value = FillValueToString(json["fill_value"]);
        }
        
        // Compressor
        if (json.contains("compressor")) {
            auto &compressor = json["compressor"];
            if (compressor.is_object() && compressor.contains("cname")) {
                meta.compressor = compressor["cname"].get<std::string>();
            } else if (compressor.is_string()) {
                meta.compressor = compressor.get<std::string>();
            }
        }
        
        // Dimension separator
        if (json.contains("dimension_separator")) {
            meta.dimension_separator = json["dimension_separator"].get<std::string>();
        }
        
    } catch (std::exception &) {
        // Return what we have
    }
    
    return meta;
}

std::optional<ZarrArrayMetadata> ZarrMetadata::ParseZarrJson(const nlohmann::json &json, const std::string &name) {
    if (!json.contains("metadata")) {
        return std::nullopt;
    }
    
    auto &metadata = json["metadata"];
    ZarrArrayMetadata meta;
    meta.name = name;
    meta.zarr_version = 3;
    
    // Zarr v3 stores shape and dtype in different locations
    if (metadata.contains("shape")) {
        for (const auto &dim : metadata["shape"]) {
            if (dim.is_number()) {
                meta.shape.push_back(dim.get<int64_t>());
            } else if (dim.is_null()) {
                // Unlimited dimension
                meta.shape.push_back(-1);
            }
        }
    }
    
    if (metadata.contains("chunk_grid")) {
        auto &chunk_grid = metadata["chunk_grid"];
        if (chunk_grid.contains("chunk_shape")) {
            for (const auto &chunk : chunk_grid["chunk_shape"]) {
                meta.chunks.push_back(chunk.get<int64_t>());
            }
        }
    }
    
    if (metadata.contains("data_type")) {
        meta.dtype = NormalizeDtype(metadata["data_type"].get<std::string>());
    }
    
    if (metadata.contains("fill_value")) {
        meta.fill_value = FillValueToString(metadata["fill_value"]);
    }
    
    // Codecs
    if (metadata.contains("codecs")) {
        std::vector<std::string> codecs;
        for (const auto &codec : metadata["codecs"]) {
            if (codec.is_string()) {
                codecs.push_back(codec.get<std::string>());
            } else if (codec.is_object() && codec.contains("name")) {
                codecs.push_back(codec["name"].get<std::string>());
            }
        }
        // Join codecs
        for (size_t i = 0; i < codecs.size(); i++) {
            if (i > 0) meta.compressor += ",";
            meta.compressor += codecs[i];
        }
    }
    
    return meta;
}

std::string ZarrMetadata::NormalizeDtype(const std::string &dtype) {
    // Zarr uses numpy-style dtype strings, simplify them
    std::string result = dtype;
    
    // Handle common types
    if (dtype == "<i1" || dtype == "i1" || dtype == "int8") return "int8";
    if (dtype == "<i2" || dtype == "i2" || dtype == "int16") return "int16";
    if (dtype == "<i4" || dtype == "i4" || dtype == "int32") return "int32";
    if (dtype == "<i8" || dtype == "i8" || dtype == "int64") return "int64";
    if (dtype == "<u1" || dtype == "u1" || dtype == "uint8") return "uint8";
    if (dtype == "<u2" || dtype == "u2" || dtype == "uint16") return "uint16";
    if (dtype == "<u4" || dtype == "u4" || dtype == "uint32") return "uint32";
    if (dtype == "<u8" || dtype == "u8" || dtype == "uint64") return "uint64";
    if (dtype == "<f2" || dtype == "f2" || dtype == "float16") return "float16";
    if (dtype == "<f4" || dtype == "f4" || dtype == "float32") return "float32";
    if (dtype == "<f8" || dtype == "f8" || dtype == "float64") return "float64";
    if (dtype == "|b1" || dtype == "bool") return "bool";
    if (dtype == "|S1" || dtype == "S1" || dtype == "char") return "char";
    if (dtype == "|O" || dtype == "object") return "object";
    
    return result;
}

std::string ZarrMetadata::FillValueToString(const nlohmann::json &fill_value) {
    if (fill_value.is_number()) {
        return fill_value.dump();
    } else if (fill_value.is_string()) {
        return fill_value.get<std::string>();
    } else if (fill_value.is_null()) {
        return "NaN";
    } else if (fill_value.is_boolean()) {
        return fill_value.get<bool>() ? "true" : "false";
    }
    return fill_value.dump();
}

const std::vector<ZarrArrayMetadata> &ZarrMetadata::GetArrays() const {
    return arrays_;
}

const std::string &ZarrMetadata::GetPath() const {
    return path_;
}

bool ZarrMetadata::IsValid() const {
    return is_valid_;
}

const std::string &ZarrMetadata::GetError() const {
    return error_message_;
}

} // namespace duckdb
