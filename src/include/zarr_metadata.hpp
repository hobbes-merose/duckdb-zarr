//===----------------------------------------------------------------------===//
//                         DuckDB
//
// zarr_metadata.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb.hpp"
#include <vector>
#include <string>

namespace duckdb {

//! Structure to hold parsed Zarr array metadata
struct ZarrArrayMetadata {
	//! Array name (empty for root array)
	std::string name;
	//! Zarr format version (2 or 3)
	int zarr_version;
	//! Array shape (dimensions)
	std::vector<int64_t> shape;
	//! Chunk dimensions
	std::vector<int64_t> chunks;
	//! Data type (normalized)
	std::string dtype;
	//! Fill value (as string)
	std::string fill_value;
	//! Compressor/codecs (comma-separated)
	std::string compressor;
	//! Order (C or F)
	std::string order;
};

//! Zarr metadata parser class
class ZarrMetadata {
public:
	ZarrMetadata();

	//! Parse Zarr metadata from a path
	void Parse(const std::string &path);

	//! Get the parsed arrays
	const std::vector<ZarrArrayMetadata> &GetArrays() const;

	//! Get the original path
	const std::string &GetPath() const;

	//! Check if parsing was valid
	bool IsValid() const;

	//! Get error message if parsing failed
	const std::string &GetError() const;

private:
	//! Parse a .zarray file (Zarr v2)
	ZarrArrayMetadata ParseZarray(const std::string &zarray_content, const std::string &name);

	//! Parse a zarr.json file (Zarr v3)
	void ParseZarrJson(const std::string &zarr_json_content, const std::string &path);

	//! Normalize dtype to a standard format
	std::string NormalizeDtype(const std::string &dtype);

	//! Convert fill_value to string
	std::string FillValueToString(yyjson_val *fill_value);

private:
	//! Original path
	std::string path_;
	//! Parsed arrays
	std::vector<ZarrArrayMetadata> arrays_;
	//! Whether parsing was successful
	bool is_valid_;
	//! Error message if parsing failed
	std::string error_message_;
};

//! Free function to parse Zarr metadata from a path
std::vector<ZarrArrayMetadata> ParseZarrMetadata(const std::string &path);

} // namespace duckdb
