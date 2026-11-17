#pragma once
#include <array>
#include <cmath>
#include <cstdint>
#include <istream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace rbmd { namespace revision {
struct ReferenceForce {
  std::array<double, 3> position, force;
  std::size_t index;
};
using ReferenceForces = std::map<std::int64_t, ReferenceForce>;

struct ReferenceFrame {
  std::int64_t step = -1, count = 0;
  std::array<double, 3> lower{}, length{};
};

// A visitor keeps production storage local to each rank and never buffers a frame.
template<class Visitor>
ReferenceFrame VisitReferenceDump(std::istream& input, std::int64_t requested_step,
                                 std::int64_t offset, Visitor visit, bool sorted = true) {
  auto line = [&]() {
    std::string value;
    if (!std::getline(input, value)) throw std::runtime_error("Truncated reference force dump");
    if (!value.empty() && value.back() == '\r') value.pop_back();
    return value;
  };
  auto integer = [](const std::string& value) {
    std::size_t end = 0;
    const auto number = std::stoll(value, &end);
    if (end != value.size()) throw std::runtime_error("Invalid integer in reference dump");
    return number;
  };
  auto number = [](const std::string& value) {
    std::size_t end = 0;
    const double result = std::stod(value, &end);
    if (end != value.size() || !std::isfinite(result))
      throw std::runtime_error("Invalid/non-finite value in reference dump");
    return result;
  };
  auto words = [](const std::string& value) {
    std::istringstream in(value);
    std::vector<std::string> result;
    for (std::string word; in >> word;) result.push_back(word);
    return result;
  };
  while (input.peek() != std::char_traits<char>::eof()) {
    if (line() != "ITEM: TIMESTEP") throw std::runtime_error("Expected LAMMPS text custom dump");
    const auto step = integer(line());
    if (line() != "ITEM: NUMBER OF ATOMS") throw std::runtime_error("Missing reference atom count");
    const auto count = integer(line());
    if (count <= 0) throw std::runtime_error("Selected reference frame is empty");
    ReferenceFrame frame; frame.step = step; frame.count = count;
    if (line() != "ITEM: BOX BOUNDS pp pp pp") throw std::runtime_error("Missing reference box");
    for (int d = 0; d < 3; ++d) {
      const auto bounds = words(line());
      if (bounds.size() != 2) throw std::runtime_error("Reference requires orthogonal periodic box");
      frame.lower[d] = number(bounds[0]);
      frame.length[d] = number(bounds[1]) - frame.lower[d];
      if (!(frame.length[d] > 0) || !std::isfinite(frame.length[d]))
        throw std::runtime_error("Invalid reference box length");
    }
    const auto header = line();
    if (header.find("ITEM: ATOMS ") != 0) throw std::runtime_error("Missing reference atom columns");
    const auto fields = words(header.substr(12));
    std::map<std::string, std::size_t> columns;
    for (std::size_t i = 0; i < fields.size(); ++i)
      if (!columns.emplace(fields[i], i).second) throw std::runtime_error("Duplicate reference column");
    for (const auto* key : {"id", "x", "y", "z", "fx", "fy", "fz"})
      if (!columns.count(key)) throw std::runtime_error("Reference requires id x y z fx fy fz");
    std::int64_t previous_id = 0;
    const bool select = requested_step == -1 || step == requested_step;
    for (long long i = 0; i < count; ++i) {
      const auto values = words(line());
      if (values.size() != fields.size()) throw std::runtime_error("Malformed reference atom row");
      if (!select) continue;
      const auto raw_id = integer(values[columns.at("id")]);
      if ((offset > 0 && raw_id > std::numeric_limits<std::int64_t>::max() - offset) ||
          (offset < 0 && raw_id < std::numeric_limits<std::int64_t>::min() - offset))
        throw std::runtime_error("Reference ID offset overflow");
      ReferenceForce atom;
      const char* xyz[] = {"x", "y", "z"};
      const char* fxyz[] = {"fx", "fy", "fz"};
      for (int d = 0; d < 3; ++d) {
        atom.position[d] = number(values[columns.at(xyz[d])]);
        atom.force[d] = number(values[columns.at(fxyz[d])]);
      }
      const auto id = raw_id + offset;
      if (sorted && i && id <= previous_id)
        throw std::runtime_error("Reference IDs must be unique and sorted (dump_modify sort id)");
      previous_id = id;
      atom.index = static_cast<std::size_t>(i);
      visit(id, atom);
    }
    if (select) {
      return frame;
    }
  }
  throw std::runtime_error("Reference timestep not found");
}

inline ReferenceForces ReadReferenceDump(std::istream& input, std::int64_t requested_step,
                                        std::int64_t offset, std::int64_t& selected_step) {
  ReferenceForces result;
  const auto frame = VisitReferenceDump(input, requested_step, offset,
      [&](std::int64_t id, const ReferenceForce& atom) {
        if (!result.emplace(id, atom).second) throw std::runtime_error("Duplicate reference atom ID");
      }, false);
  selected_step = frame.step;
  return result;
}

struct OnlineForceMoments {
  std::array<double, 3> mean_error{};
  double m2 = 0;
  std::uint64_t samples = 0;
  void Add(const std::array<double, 3>& error) {
    ++samples;
    for (int d = 0; d < 3; ++d) {
      if (!std::isfinite(error[d])) throw std::runtime_error("Non-finite force error");
      const double delta = error[d] - mean_error[d];
      mean_error[d] += delta / samples;
      m2 += delta * (error[d] - mean_error[d]);
    }
    if (!std::isfinite(m2)) throw std::runtime_error("Force variance overflow");
  }
  double BiasSquared() const {
    double sum = 0;
    for (double value : mean_error) sum += value * value;
    return sum;
  }
};

inline bool ReferencePositionMatches(const ReferenceFrame& box, int d,
                                     double position, double reference, double tolerance) {
  if (!std::isfinite(position) || !std::isfinite(reference)) return false;
  double delta = position - reference;
  delta -= box.length[d] * std::round(delta / box.length[d]);
  return std::abs(delta) <= tolerance;
}

inline double ReferenceErrorSquared(const ReferenceForce& ref,
                                    const std::array<double, 3>& position,
                                    const std::array<double, 3>& force, double tolerance) {
  double error = 0;
  for (int d = 0; d < 3; ++d) {
    if (!std::isfinite(position[d]) || !std::isfinite(force[d]))
      throw std::runtime_error("Non-finite sampled force or position");
    // shortcut: compare Cartesian coordinates directly; add periodic mapping only if required.
    if (std::abs(position[d] - ref.position[d]) > tolerance)
      throw std::runtime_error("Reference position mismatch; fixed configuration required");
    const double delta = force[d] - ref.force[d];
    error += delta * delta;
  }
  if (!std::isfinite(error)) throw std::runtime_error("Force error overflow");
  return error;
}
}} // namespace rbmd::revision
