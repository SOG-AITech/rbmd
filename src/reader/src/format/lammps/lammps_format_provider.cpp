#include <algorithm>
#include <cctype>
#include <memory>
#include <string>
#include <string_view>

#include "core/structure_reader.h"
#include "formats/lammps/lammps_full_reader.h"
#include "formats/lammps/lammps_format_provider.h"
#include "io/icursor.h"
#include "io/isource.h"
#include "io/resource_locator.h"

namespace reader {

namespace {

std::string Trim(std::string_view value) {
  while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front()))) {
    value.remove_prefix(1);
  }
  while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) {
    value.remove_suffix(1);
  }
  return std::string(value);
}

bool IsComment(std::string_view value) {
  value = Trim(value);
  return value.empty() || value.front() == '#';
}

}  // namespace

std::string LammpsFormatProvider::name() const {
  return "LAMMPS";
}

int LammpsFormatProvider::probe(const ISource& src, const IResourceLocator&) const {
  auto cursor = src.cursor(0);
  if (!cursor) {
    return 0;
  }

  std::string_view line;
  int score = 0;
  int inspected = 0;
  while (inspected++ < 64 && cursor->nextLine(line)) {
    auto trimmed = Trim(line);
    if (trimmed.empty()) {
      continue;
    }
    if (trimmed.find("LAMMPS") != std::string_view::npos) {
      score += 5;
    }
    if (trimmed.find(" atoms") != std::string_view::npos) {
      score += 10;
    }
    if (trimmed.find(" atom types") != std::string_view::npos) {
      score += 10;
    }
    if (trimmed == "Masses" || trimmed.rfind("Masses", 0) == 0) {
      score += 20;
      break;
    }
  }

  return score;
}

std::shared_ptr<StructureReader> LammpsFormatProvider::createStructureReader(
    std::shared_ptr<ISource> src,
    std::shared_ptr<IResourceLocator> locator,
    std::string file_path,
    std::shared_ptr<MDDataOwner> owner) const {
  return std::make_shared<LammpsFullReader>(std::move(src), std::move(locator),
                                            std::move(file_path), std::move(owner));
}

}  // namespace reader
