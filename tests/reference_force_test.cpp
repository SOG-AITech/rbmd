// Standalone host regression test; not compiled as part of the implementation-only request.
// c++ -std=c++17 -Isrc tests/reference_force_test.cpp -o /tmp/reference_force_test
#include "common/reference_force.h"
#include <cassert>
#include <functional>

using namespace rbmd::revision;
std::string Frame(long long step, const std::vector<std::string>& atoms,
                  const std::string& fields = "id x y z fx fy fz") {
  std::string text = "ITEM: TIMESTEP\n" + std::to_string(step) +
      "\nITEM: NUMBER OF ATOMS\n" + std::to_string(atoms.size()) +
      "\nITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\nITEM: ATOMS " + fields + "\n";
  for (const auto& atom : atoms) text += atom + "\n";
  return text;
}
void Rejects(const std::function<void()>& call) {
  bool rejected = false;
  try { call(); } catch (const std::exception&) { rejected = true; }
  assert(rejected);
}
int main() {
  std::int64_t step = -1;
  std::istringstream dump(Frame(0, {"1 0 0 0 9 0 0"}) +
      Frame(10, {"2 2 0 0 1 0 0 1", "0 1 0 0 0 0 0 1"}, "fy id z fx x fz y type"));
  auto ref = ReadReferenceDump(dump, 10, -1, step);
  assert(step == 10 && ref.size() == 2 && ref.at(0).index != ref.at(1).index);
  const double e1 = ReferenceErrorSquared(ref.at(0), {{0,0,0}}, {{1,0,0}}, 1.e-6);
  const double e2 = ReferenceErrorSquared(ref.at(0), {{0,0,0}}, {{3,0,0}}, 1.e-6);
  assert(e1 == 1 && e2 == 9 && std::sqrt((e1+e2)/2) == std::sqrt(5.0));
  assert(ReferenceErrorSquared(ref.at(1), {{1,0,0}}, {{0,2,0}}, 1.e-6) == 0);
  Rejects([&] { ReferenceErrorSquared(ref.at(0), {{1,0,0}}, {{1,0,0}}, 1.e-6); });
  Rejects([&] { ReferenceErrorSquared(ref.at(0), {{0,0,0}},
                                    {{std::numeric_limits<double>::infinity(),0,0}}, 1.e-6); });
  std::istringstream first(Frame(5, {"1 0 0 0 0 0 0"}));
  assert(ReadReferenceDump(first, -1, 0, step).size() == 1 && step == 5);
  for (const auto& text : {
      Frame(0, {"1 0 0 0 0 0 0", "1 0 0 0 0 0 0"}),
      Frame(0, {"1 0 0 0 nan 0 0"}),
      Frame(0, {"1 0 0 0 0 0"}),
      Frame(0, {"1 0 0 0 0 0 0"}, "id xs ys zs fx fy fz"),
      Frame(0, {}), std::string("ITEM: TIMESTEP\n0\n")}) {
    Rejects([&] { std::istringstream in(text); ReadReferenceDump(in, -1, 0, step); });
  }
  Rejects([&] { std::istringstream in(Frame(0, {"1 0 0 0 0 0 0"}));
               ReadReferenceDump(in, 99, 0, step); });
  Rejects([&] { std::istringstream in(Frame(0, {"9223372036854775807 0 0 0 0 0 0"}));
               ReadReferenceDump(in, 0, 1, step); });
}
