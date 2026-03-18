#include "zarr_metadata.hpp"
#include "duckdb/common/common.hpp"
#include "duckdb/common/file_system.hpp"
#include "duckdb/common/local_file_system.hpp"
#include "duckdb/common/string_util.hpp"
#include "yyjson.hpp"

using duckdb_yyjson::yyjson_val;

namespace duckdb {

//! Free function to parse Zarr metadata from a path
std::vector<ZarrArrayMetadata> ParseZarrMetadata(const std::string &path) {
	ZarrMetadata parser;
	parser.Parse(path);
	return parser.GetArrays();
}

ZarrMetadata::ZarrMetadata() : is_valid_(false) {
}

void ZarrMetadata::Parse(const std::string &path) {
	path_ = path;
	arrays_.clear();
	is_valid_ = false;
	error_message_.clear();

	try {
		// Get the file system
		LocalFileSystem fs;

		// Try to read the zarr.json file (Zarr v3 group metadata)
		// or check if it's a directory with .zarray files (Zarr v2)

		std::string metadata_path = path;
		if (!StringUtil::EndsWith(path, "/") && !StringUtil::EndsWith(path, ".zarr")) {
			// Handle cases where path might not have trailing slash
		}

		// First, try to read zarr.json (Zarr v3)
		std::string zarr_json_path = path;
		if (!StringUtil::EndsWith(zarr_json_path, "/")) {
			zarr_json_path += "/";
		}
		zarr_json_path += "zarr.json";

		try {
			auto file_handle = fs.OpenFile(zarr_json_path, FileOpenFlags::FILE_FLAGS_READ);
			idx_t file_size = fs.GetFileSize(*file_handle);
			std::string content(file_size, '\0');
			fs.Read(*file_handle, char_ptr_cast(content.data()), file_size);

			// Parse as JSON using yyjson
			duckdb_yyjson::yyjson_read_err err;
			auto *doc = duckdb_yyjson::yyjson_read(content.c_str(), content.size(), 0, &err);
			if (!doc) {
				throw std::runtime_error(err.msg);
			}

			auto *root = duckdb_yyjson::yyjson_doc_get_root(doc);

			// Check if it's a group or array
			auto *zarr_format_val = duckdb_yyjson::yyjson_obj_get(root, "zarr_format");
			if (zarr_format_val && duckdb_yyjson::yyjson_get_int(zarr_format_val) == 3) {
				// Zarr v3 format
				auto *metadata_val = duckdb_yyjson::yyjson_obj_get(root, "metadata");
				if (metadata_val && duckdb_yyjson::yyjson_is_obj(metadata_val)) {
					// Check if it's an array (has shape)
					auto *shape_val = duckdb_yyjson::yyjson_obj_get(metadata_val, "shape");
					if (shape_val) {
						ParseZarrJson(root, "");
					}
					// Check for arrays in subgroups
					auto *attributes_val = duckdb_yyjson::yyjson_obj_get(root, "attributes");
					if (attributes_val) {
						auto *zarr_attrs_val = duckdb_yyjson::yyjson_obj_get(attributes_val, "zarr");
						if (zarr_attrs_val) {
							auto *children_val = duckdb_yyjson::yyjson_obj_get(zarr_attrs_val, "children");
							// Has child arrays/groups
							(void)children_val;
						}
					}
				}
			}

			duckdb_yyjson::yyjson_doc_free(doc);
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
			if (!StringUtil::EndsWith(zarray_path, "/")) {
				zarray_path += "/";
			}
			zarray_path += array_path;
			if (!array_path.empty() && !StringUtil::EndsWith(zarray_path, "/")) {
				zarray_path += "/";
			}
			zarray_path += ".zarray";

			try {
				auto file_handle = fs.OpenFile(zarray_path, FileOpenFlags::FILE_FLAGS_READ);
				idx_t file_size = fs.GetFileSize(*file_handle);
				std::string content(file_size, '\0');
				fs.Read(*file_handle, char_ptr_cast(content.data()), file_size);

				std::string name = array_path;
				if (StringUtil::StartsWith(name, "/")) {
					name = name.substr(1);
				}
				if (StringUtil::EndsWith(name, "/.zarray")) {
					name = name.substr(0, name.size() - 8);
				}

				arrays_.push_back(ParseZarray(content, name));
			} catch (std::exception &e) {
				// Skip this file if we can't read it
				(void)e;
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
		duckdb_yyjson::yyjson_read_err err;
		auto *doc = duckdb_yyjson::yyjson_read(zarray_content.c_str(), zarray_content.size(), 0, &err);
		if (!doc) {
			return meta;
		}

		auto *json = duckdb_yyjson::yyjson_doc_get_root(doc);

		// Shape
		auto *shape_val = duckdb_yyjson::yyjson_obj_get(json, "shape");
		if (shape_val && duckdb_yyjson::yyjson_is_arr(shape_val)) {
			size_t idx, max;
			yyjson_val *dim;
			duckdb_yyjson::yyjson_arr_foreach(shape_val, idx, max, dim) {
				if (duckdb_yyjson::yyjson_is_int(dim)) {
					meta.shape.push_back(duckdb_yyjson::yyjson_get_int(dim));
				}
			}
		}

		// Chunks
		auto *chunks_val = duckdb_yyjson::yyjson_obj_get(json, "chunks");
		if (chunks_val && duckdb_yyjson::yyjson_is_arr(chunks_val)) {
			size_t idx, max;
			yyjson_val *chunk;
			duckdb_yyjson::yyjson_arr_foreach(chunks_val, idx, max, chunk) {
				if (duckdb_yyjson::yyjson_is_int(chunk)) {
					meta.chunks.push_back(duckdb_yyjson::yyjson_get_int(chunk));
				}
			}
		}

		// Dtype
		auto *dtype_val = duckdb_yyjson::yyjson_obj_get(json, "dtype");
		if (dtype_val && duckdb_yyjson::yyjson_is_str(dtype_val)) {
			meta.dtype = NormalizeDtype(duckdb_yyjson::yyjson_get_str(dtype_val));
		}

		// Fill value
		auto *fill_value_val = duckdb_yyjson::yyjson_obj_get(json, "fill_value");
		if (fill_value_val) {
			meta.fill_value = FillValueToString(fill_value_val);
		}

		// Compressor
		auto *compressor_val = duckdb_yyjson::yyjson_obj_get(json, "compressor");
		if (compressor_val) {
			if (duckdb_yyjson::yyjson_is_obj(compressor_val)) {
				auto *cname_val = duckdb_yyjson::yyjson_obj_get(compressor_val, "cname");
				if (cname_val && duckdb_yyjson::yyjson_is_str(cname_val)) {
					meta.compressor = duckdb_yyjson::yyjson_get_str(cname_val);
				}
			} else if (duckdb_yyjson::yyjson_is_str(compressor_val)) {
				meta.compressor = duckdb_yyjson::yyjson_get_str(compressor_val);
			}
		}

		duckdb_yyjson::yyjson_doc_free(doc);

	} catch (std::exception &) {
		// Return what we have
	}

	return meta;
}

void ZarrMetadata::ParseZarrJson(duckdb_yyjson::yyjson_val *json, const std::string &name) {
	auto *metadata_val = duckdb_yyjson::yyjson_obj_get(json, "metadata");
	if (!metadata_val || !duckdb_yyjson::yyjson_is_obj(metadata_val)) {
		return;
	}

	ZarrArrayMetadata meta;
	meta.name = name;
	meta.zarr_version = 3;

	// Zarr v3 stores shape and dtype in different locations
	auto *shape_val = duckdb_yyjson::yyjson_obj_get(metadata_val, "shape");
	if (shape_val && duckdb_yyjson::yyjson_is_arr(shape_val)) {
		size_t idx, max;
		yyjson_val *dim;
		duckdb_yyjson::yyjson_arr_foreach(shape_val, idx, max, dim) {
			if (duckdb_yyjson::yyjson_is_int(dim)) {
				meta.shape.push_back(duckdb_yyjson::yyjson_get_int(dim));
			} else if (duckdb_yyjson::yyjson_is_null(dim)) {
				// Unlimited dimension
				meta.shape.push_back(-1);
			}
		}
	}

	auto *chunk_grid_val = duckdb_yyjson::yyjson_obj_get(metadata_val, "chunk_grid");
	if (chunk_grid_val && duckdb_yyjson::yyjson_is_obj(chunk_grid_val)) {
		auto *chunk_shape_val = duckdb_yyjson::yyjson_obj_get(chunk_grid_val, "chunk_shape");
		if (chunk_shape_val && duckdb_yyjson::yyjson_is_arr(chunk_shape_val)) {
			size_t idx, max;
			yyjson_val *chunk;
			duckdb_yyjson::yyjson_arr_foreach(chunk_shape_val, idx, max, chunk) {
				if (duckdb_yyjson::yyjson_is_int(chunk)) {
					meta.chunks.push_back(duckdb_yyjson::yyjson_get_int(chunk));
				}
			}
		}
	}

	auto *data_type_val = duckdb_yyjson::yyjson_obj_get(metadata_val, "data_type");
	if (data_type_val && duckdb_yyjson::yyjson_is_str(data_type_val)) {
		meta.dtype = NormalizeDtype(duckdb_yyjson::yyjson_get_str(data_type_val));
	}

	auto *fill_value_val = duckdb_yyjson::yyjson_obj_get(metadata_val, "fill_value");
	if (fill_value_val) {
		meta.fill_value = FillValueToString(fill_value_val);
	}

	// Codecs
	auto *codecs_val = duckdb_yyjson::yyjson_obj_get(metadata_val, "codecs");
	if (codecs_val && duckdb_yyjson::yyjson_is_arr(codecs_val)) {
		std::vector<std::string> codecs;
		size_t idx, max;
		yyjson_val *codec;
		duckdb_yyjson::yyjson_arr_foreach(codecs_val, idx, max, codec) {
			if (duckdb_yyjson::yyjson_is_str(codec)) {
				codecs.push_back(duckdb_yyjson::yyjson_get_str(codec));
			} else if (duckdb_yyjson::yyjson_is_obj(codec)) {
				auto *name_val = duckdb_yyjson::yyjson_obj_get(codec, "name");
				if (name_val && duckdb_yyjson::yyjson_is_str(name_val)) {
					codecs.push_back(duckdb_yyjson::yyjson_get_str(name_val));
				}
			}
		}
		// Join codecs
		for (size_t i = 0; i < codecs.size(); i++) {
			if (i > 0)
				meta.compressor += ",";
			meta.compressor += codecs[i];
		}
	}

	arrays_.push_back(meta);
}

std::string ZarrMetadata::NormalizeDtype(const std::string &dtype) {
	// Handle common types
	if (dtype == "<i1" || dtype == "i1" || dtype == "int8")
		return "int8";
	if (dtype == "<i2" || dtype == "i2" || dtype == "int16")
		return "int16";
	if (dtype == "<i4" || dtype == "i4" || dtype == "int32")
		return "int32";
	if (dtype == "<i8" || dtype == "i8" || dtype == "int64")
		return "int64";
	if (dtype == "<u1" || dtype == "u1" || dtype == "uint8")
		return "uint8";
	if (dtype == "<u2" || dtype == "u2" || dtype == "uint16")
		return "uint16";
	if (dtype == "<u4" || dtype == "u4" || dtype == "uint32")
		return "uint32";
	if (dtype == "<u8" || dtype == "u8" || dtype == "uint64")
		return "uint64";
	if (dtype == "<f2" || dtype == "f2" || dtype == "float16")
		return "float16";
	if (dtype == "<f4" || dtype == "f4" || dtype == "float32")
		return "float32";
	if (dtype == "<f8" || dtype == "f8" || dtype == "float64")
		return "float64";
	if (dtype == "|b1" || dtype == "bool")
		return "bool";
	if (dtype == "|S1" || dtype == "S1" || dtype == "char")
		return "char";
	if (dtype == "|O" || dtype == "object")
		return "object";

	return dtype;
}

std::string ZarrMetadata::FillValueToString(duckdb_yyjson::yyjson_val *fill_value) {
	if (duckdb_yyjson::yyjson_is_num(fill_value)) {
		if (duckdb_yyjson::yyjson_is_int(fill_value)) {
			return std::to_string(duckdb_yyjson::yyjson_get_int(fill_value));
		} else if (duckdb_yyjson::yyjson_is_real(fill_value)) {
			return std::to_string(duckdb_yyjson::yyjson_get_real(fill_value));
		}
	} else if (duckdb_yyjson::yyjson_is_str(fill_value)) {
		return duckdb_yyjson::yyjson_get_str(fill_value);
	} else if (duckdb_yyjson::yyjson_is_null(fill_value)) {
		return "NaN";
	} else if (duckdb_yyjson::yyjson_is_bool(fill_value)) {
		return duckdb_yyjson::yyjson_get_bool(fill_value) ? "true" : "false";
	}
	// Fallback: convert to string
	char *json_str = duckdb_yyjson::yyjson_val_write(fill_value, 0, nullptr);
	if (json_str) {
		std::string result(json_str);
		free(json_str);
		return result;
	}
	return "";
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
