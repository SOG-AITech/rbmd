#pragma once
#include <algorithm>
#include <cctype>
#include <cmath>
#include <experimental/filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "../common/object.h"
#include "../common/types.h"
#include "common/json.hpp"

class ConfigData : public Object {
 public:
  /**
   * @brief constructor
   * @param file json config file
   */
  ConfigData(const std::string& file) {
    if (!IsJsonFile(file)) {
      //_console->error("{} is not json file!", file);
      return;
    }

    _config_file = ResolveAbsolutePath(file);
    if (_config_file.empty()) {
      _config_file = file;
    }

    std::experimental::filesystem::path config_path(_config_file);
    if (config_path.has_parent_path()) {
      _config_dir = config_path.parent_path().string();
    } else {
      _config_dir = std::experimental::filesystem::current_path().string();
    }

    ParseJsonFile(_config_file);
  }

  ~ConfigData() = default;

 public:
  /**
   * @brief get value
   * @tparam T
   * @tparam ...Args
   * @param key
   * @param ...args
   * @return value
   */
  template <typename T, typename... Args>
  T Get(std::string key, Args&&... args) {
    nlohmann::ordered_json json_node = _json_node;
    const std::vector<std::string> path{std::forward<Args>(args)...};

    auto getNode = [this, &json_node](const auto& arg) {
      if (json_node.contains(arg) && json_node[arg].is_object()) {
        json_node = json_node[arg];
      } else {
        //_console->error("{} is not a object!", arg);
        return;
      }
    };

    (getNode(std::forward<Args>(args)), ...);

    try {
      if (json_node.contains(key)) {
        auto value = json_node[key].get<T>();
        if constexpr (std::is_same_v<T, std::string>) {
          return NormalizeConfigOption(std::move(value), key, path);
        }
        return value;
      } else {
        throw std::runtime_error("no key named: " + key);
      }
    } catch (const std::exception&) {
      // log
      //_console->error("no key named: {}", key);
      throw;
    }
  }

  template <typename T, typename... Args>
  std::vector<T> GetArray(std::string key, Args&&... args) {
    nlohmann::ordered_json json_node = _json_node;

    auto getNode = [this, &json_node](const auto& arg) {
      if (json_node.contains(arg) && json_node[arg].is_object()) {
        json_node = json_node[arg];
      } else {
        //_console->error("{} is not a object!", arg);
        return;
      }
    };

    (getNode(std::forward<Args>(args)), ...);


    if (json_node.contains(key)) {
      nlohmann::ordered_json value = json_node[key];
      if (value.is_array()) {
        std::vector<T> result;
        for (const auto& item : value) {
          result.push_back(item.get<T>());  // 将数组元素转换为 T 类型
        }
        return result;
      } else {
        throw std::runtime_error(key + " is not an array");
      }
    } else {
      throw std::runtime_error("no key named: " + key);
    }
  }

  template <typename T, typename... Args>
  T GetJudge(std::string key, Args&&... args) {
    nlohmann::ordered_json json_node = _json_node;
    bool valid_path = true;

    auto checkNode = [&](const auto& arg) {
      if (!valid_path) return;
      if (json_node.contains(arg) && json_node[arg].is_object()) {
        json_node = json_node[arg];
      } else {
        valid_path = false;
      }
    };

    (checkNode(std::forward<Args>(args)), ...);

    if (!valid_path || !json_node.contains(key)) {
      return T{};
    }

    try {
      return json_node[key].get<T>();
    } catch (...) {
      return T{};
    }
  }

  /**
   * @brief get json node
   * @param key
   * @return json node
   */
  auto& GetJsonNode(const std::string& key) {
    if (!_json_node.contains(key) || !_json_node[key].is_object()) {
      //_console->warn("Can not find key: {}", key);
    }

    return _json_node[key];
  }

  /**
   * @brief Check whether there is key Node
   * @param key node name
   * @return true or false
   */
  bool HasNode(const std::string& key) { return _json_node.contains(key); }

  bool PathExists(std::initializer_list<std::string> path) {
    nlohmann::ordered_json current = _json_node;
    for (const auto& key : path) {
      if (!current.contains(key)) {
        return false;
      }
      current = current[key];
    }
    return true;
  }

  const std::string& GetConfigFile() const { return _config_file; }

  const std::string& GetConfigDir() const { return _config_dir; }

  std::string ResolvePath(const std::string& path) const {
    namespace fs = std::experimental::filesystem;
    if (path.empty()) {
      return path;
    }

    fs::path candidate(path);
    if (candidate.is_absolute()) {
      return NormalizePath(candidate);
    }

    if (!_config_dir.empty()) {
      fs::path attempt = fs::path(_config_dir) / candidate;
      if (fs::exists(attempt)) {
        return NormalizePath(attempt);
      }
      return attempt.string();
    }

    fs::path current_attempt = fs::current_path() / candidate;
    if (fs::exists(current_attempt)) {
      return NormalizePath(current_attempt);
    }
    return current_attempt.string();
  }

 private:
  /**
   * @brief Check whether the file is json file
   * @param file file path
   * @return true or false
   */
  bool IsJsonFile(const std::string& file) {
    auto length = file.length();
    return (length >= 5 && file.substr(length - 5) == ".json");
  }

  static std::string NormalizePath(
      const std::experimental::filesystem::path& path) {
    namespace fs = std::experimental::filesystem;
    try {
      if (fs::exists(path)) {
        return fs::canonical(path).string();
      }
    } catch (const std::exception&) {
    }
    return path.string();
  }

  static std::string ResolveAbsolutePath(const std::string& path) {
    namespace fs = std::experimental::filesystem;
    if (path.empty()) {
      return path;
    }
    fs::path input(path);
    if (input.is_absolute()) {
      return NormalizePath(input);
    }
    return NormalizePath(fs::current_path() / input);
  }

  static std::string NormalizeConfigOption(
      std::string value, const std::string& key,
      const std::vector<std::string>& path) {
    const auto path_is = [&path](std::initializer_list<const char*> expected) {
      if (path.size() != expected.size()) {
        return false;
      }
      return std::equal(path.begin(), path.end(), expected.begin(),
                        [](const std::string& lhs, const char* rhs) {
                          return lhs == rhs;
                        });
    };
    const auto convert_case = [&value](bool uppercase) {
      std::transform(value.begin(), value.end(), value.begin(),
                     [uppercase](unsigned char ch) {
                       return static_cast<char>(uppercase ? std::toupper(ch)
                                                          : std::tolower(ch));
                     });
    };

    if (path_is({"hyper_parameters", "force_field"})) {
      if (key == "type") {
        convert_case(true);
        // Keep the spelling expected by the existing force dispatch tables.
        if (value == "TERSOFF") {
          value = "Tersoff";
        }
      } else if (key == "dihedral_type" || key == "improper_type") {
        convert_case(false);
      }
    } else if (path_is({"hyper_parameters", "neighbor"})) {
      if (key == "type") {
        convert_case(true);
      } else if (key == "energy_rbl_flag") {
        convert_case(false);
      }
    } else if (path_is({"hyper_parameters", "coulomb"})) {
      if (key == "type") {
        convert_case(true);
      } else if (key == "energy_rbe_flag") {
        convert_case(false);
      }
    } else if (path_is({"init_configuration", "read_data"}) &&
               key == "atom_style") {
      convert_case(false);
    } else if (path_is({"execution"})) {
      if (key == "ensemble" || key == "temp_ctrl_type" ||
          key == "press_ctrl_type") {
        convert_case(true);
      } else if (key == "integration_type" || key == "com_bias") {
        convert_case(false);
      }
    }

    return value;
  }

  /**
   * @brief parse json file
   * @param file
   */
  void ParseJsonFile(const std::string& file) {
    std::ifstream filestream(file);
    if (!filestream.is_open()) {
      throw std::runtime_error("failed to open config file: " + file);
    }

    try {
      _json_node = nlohmann::ordered_json::parse(
          filestream,
          nullptr,  // callback
          true,     // allow_exceptions
          true      // ignore_comments
      );
    } catch (const nlohmann::json::parse_error& e) {
      throw std::runtime_error(
          std::string("配置文件 JSON 语法错误: ") + e.what());
    }
  }

 private:
  //Json::Value _json_node;
  nlohmann::ordered_json _json_node;
  std::string _config_file;
  std::string _config_dir;
};
