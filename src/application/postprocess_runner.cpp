#include "postprocess_runner.h"

#include <cstdlib>
#include <exception>
#include <experimental/filesystem>
#include <sstream>
#include <string>
#include <vector>

#include "common/mpi_root_guard.hpp"
#include "data_manager/include/config_data.h"
#include "output/include/Logger.hpp"

namespace {
namespace fs = std::experimental::filesystem;

std::string ShellQuote(const std::string& value) {
  std::string quoted = "'";
  for (const char ch : value) {
    if (ch == '\'') {
      quoted += "'\\''";
    } else {
      quoted += ch;
    }
  }
  quoted += "'";
  return quoted;
}

bool IsExistingRegularFile(const std::string& path) {
  try {
    return fs::exists(path) && fs::is_regular_file(path);
  } catch (const std::exception&) {
    return false;
  }
}

std::string FindRegularFileInParents(const std::string& base_dir,
                                     const std::string& relative_path) {
  fs::path current(base_dir.empty() ? fs::current_path() : fs::path(base_dir));
  for (int depth = 0; depth < 8; ++depth) {
    const fs::path candidate = current / relative_path;
    if (IsExistingRegularFile(candidate.string())) {
      return candidate.string();
    }
    if (!current.has_parent_path() || current == current.parent_path()) {
      break;
    }
    current = current.parent_path();
  }
  return "";
}

std::string ExecutableDir() {
  try {
    const fs::path exe = fs::read_symlink("/proc/self/exe");
    if (exe.has_parent_path()) {
      return exe.parent_path().string();
    }
  } catch (const std::exception&) {
  }
  return "";
}

std::vector<std::string> ReleaseRoots() {
  std::vector<std::string> roots;
  const std::string exe_dir = ExecutableDir();
  if (exe_dir.empty()) {
    return roots;
  }

  roots.push_back(exe_dir);
  try {
    const fs::path parent = fs::path(exe_dir).parent_path();
    if (!parent.empty() && parent != exe_dir) {
      roots.push_back(parent.string());
    }
  } catch (const std::exception&) {
  }
  return roots;
}

std::string FindBundledPython() {
  for (const std::string& root : ReleaseRoots()) {
    const fs::path candidate = fs::path(root) / "python" / "bin" / "python";
    if (IsExistingRegularFile(candidate.string())) {
      return candidate.string();
    }
  }
  return "";
}

std::string ResolvePython(const nlohmann::ordered_json& postprocess) {
  if (postprocess.is_object()) {
    const std::string configured = postprocess.value("python", "");
    if (!configured.empty()) {
      return configured;
    }
  }

  const std::string bundled = FindBundledPython();
  if (!bundled.empty()) {
    return bundled;
  }

  return "python3";
}

std::string ResolveScript(const ConfigData& config, const std::string& script,
                          bool allow_bundled_default) {
  if (IsExistingRegularFile(script)) {
    return script;
  }

  const std::string config_relative_script = config.ResolvePath(script);
  if (IsExistingRegularFile(config_relative_script)) {
    return config_relative_script;
  }

  const std::string config_parent_script =
      FindRegularFileInParents(config.GetConfigDir(), script);
  if (!config_parent_script.empty()) {
    return config_parent_script;
  }

  const std::string cwd_parent_script = FindRegularFileInParents(".", script);
  if (!cwd_parent_script.empty()) {
    return cwd_parent_script;
  }

  if (allow_bundled_default) {
    for (const std::string& root : ReleaseRoots()) {
      const fs::path bundled_script =
          fs::path(root) / "analysis" / "postprocess_freud.py";
      if (IsExistingRegularFile(bundled_script.string())) {
        return bundled_script.string();
      }
    }
  }

  return "";
}

bool HasAnalysisOutput(const nlohmann::ordered_json& outputs) {
  return (outputs.contains("rdf_out") && outputs["rdf_out"].is_object()) ||
         (outputs.contains("msd_out") && outputs["msd_out"].is_object()) ||
         (outputs.contains("vacf_out") && outputs["vacf_out"].is_object());
}
}  // namespace

void PostprocessRunner::Run(const std::shared_ptr<ConfigData>& config) const {
  if (!rbmd::mpi::ShouldWriteRootOnlyOutput()) {
    return;
  }

  if (!config || !config->PathExists({"outputs"})) {
    return;
  }

  auto& outputs = config->GetJsonNode("outputs");
  if (!HasAnalysisOutput(outputs)) {
    return;
  }

  nlohmann::ordered_json postprocess = nlohmann::ordered_json::object();
  if (outputs.contains("analysis_postprocess") &&
      outputs["analysis_postprocess"].is_object()) {
    postprocess = outputs["analysis_postprocess"];
  }

  const bool enabled = postprocess.value("enabled", true);
  if (!enabled) {
    Logger::Instance().info("Analysis postprocess disabled by configuration.");
    return;
  }

  const std::string backend = postprocess.value("backend", "freud");
  if (backend != "freud") {
    Logger::Instance().error(
        "Unsupported analysis postprocess backend '{}'. Only 'freud' is supported.",
        backend);
    return;
  }

  const std::string python = ResolvePython(postprocess);
  const bool script_configured = postprocess.contains("script");
  const std::string requested_script =
      postprocess.value("script", "scripts/postprocess_freud.py");
  const bool allow_bundled_script =
      !script_configured || requested_script == "scripts/postprocess_freud.py";
  const std::string script =
      ResolveScript(*config, requested_script, allow_bundled_script);
  const std::string trajectory = postprocess.value("trajectory", "rbmd.trj");
  const std::string outdir = postprocess.value("outdir", ".");

  if (script.empty()) {
    Logger::Instance().error(
        "Analysis postprocess script '{}' was not found. Set outputs.analysis_postprocess.script to an absolute path.",
        requested_script);
    return;
  }

  std::ostringstream command;
  command << ShellQuote(python) << ' ' << ShellQuote(script)
          << " --config " << ShellQuote(config->GetConfigFile())
          << " --traj " << ShellQuote(trajectory)
          << " --outdir " << ShellQuote(outdir);

  Logger::Instance().info("Running analysis postprocess: {}", command.str());
  const int exit_code = std::system(command.str().c_str());
  if (exit_code != 0) {
    Logger::Instance().error(
        "Analysis postprocess failed with exit code {}.", exit_code);
  }
}
