#pragma once

#include "duckdb.hpp"
#include <string>
#include <vector>
#include <memory>
#include <optional>

namespace duckdb {

//! Zarr array metadata structure
struct ZarrArrayMetadata {
    std::string name;          //! Array name (path within store)
    std::vector<int64_t> shape; //! Array shape
    std::string dtype;          //! Data type (e.g., "<i4", "<f8")
    std::vector<int64_t> chunks; //! Chunk shape
    std::string compressor;    //! Compressor name (e.g., "blosc", "zstd")
    int zarr_version;          //! Zarr format version (2 or 3)
    std::string fill_value;    //! Fill value as string
    std::optional<std::string> dimension_separator;
};

//! Zarr metadata parser class
class ZarrMetadata {
public:
    ZarrMetadata();
    
    //! Parse metadata from a zarr store path
    void Parse(const std::string &path);
    
    //! Get list of arrays in the store
    const std::vector<ZarrArrayMetadata> &GetArrays() const;
    
    //! Get the store path
    const std::string &GetPath() const;
    
    //! Check if the store is valid
    bool IsValid() const;
    
    //! Get error message if parsing failed
    const std::string &GetError() const;

private:
    std::string path_;
    std::vector<ZarrArrayMetadata> arrays_;
    bool is_valid_;
    std::string error_message_;
    
    //! Parse Zarr v2 metadata (.zarray file)
    ZarrArrayMetadata ParseZarray(const std::string &zarray_content, const std::string &name);
    
    //! Parse Zarr v3 metadata (zarr.json)
    std::optional<ZarrArrayMetadata> ParseZarrJson(const nlohmann::json &json, const std::string &name);
    
    //! Normalize dtype string
    std::string NormalizeDtype(const std::string &dtype);
    
    //! Convert fill_value to string
    std::string FillValueToString(const nlohmann::json &fill_value);
};

} // namespace duckdb
