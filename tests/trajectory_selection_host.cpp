#include <cassert>
#include <iostream>
#include "output/include/trajectory_selection.h"
using namespace rbmd;
using namespace rbmd::trajectory;
using Json = nlohmann::ordered_json;

int main() {
  auto o = ParseOptions(Json::parse(R"({"fields":["id","type","x","xu"],"selection":{"types":[3,3],"regions":[{"x":[0,2],"y":[-1,1]},{"x":[10,12]}],"mode":"initial"}})"));
  assert(o.types == std::vector<Id>{2});
  assert(o.initial && o.regions.size() == 2);
  Id ids[] = {3, 7, 9, 12, 13, 20};
  Id types[] = {2, 2, 1, 2, 2, 2};
  Real x[] = {0, 2, 1, 11, 1, 50};
  Real y[] = {0, 0, 0, 20, 1, 0};
  Real z[] = {0, 0, 0, 0, 0, 0};
  Predicate p;
  p.atom_ids=ids; p.atom_types=types; p.x=x; p.y=y; p.z=z;
  p.types=o.types.data(); p.num_types=o.types.size();
  p.regions=o.regions.data(); p.num_regions=o.regions.size();
  assert(p(0) && !p(1) && !p(2) && p(3) && !p(4) && !p(5));
  // IDs survive motion, reorder and transfer to a different rank's array.
  Id frozen[] = {3,12};
  p.ids=frozen; p.num_ids=2; p.filter_ids=true;
  p.num_types=0; p.num_regions=0;
  x[0]=100;
  assert(p(0) && p(3) && !p(5));
  Id migrated[] = {20,12,7};
  p.atom_ids=migrated;
  assert(!p(0) && p(1) && !p(2));
  p.num_ids=0;
  assert(!p(1));
#if USE_64BIT_IDS
  auto large=ReadIds(Json::parse("[10000000000,2147483649]"));
  assert(large[1] == 9999999999LL);
  assert(ContainsId(large.data(),large.size(),9999999999LL));
#endif
  for (const char* invalid : {
      R"({"selection":{}})", R"({"selection":{"types":[]}})",
      R"({"selection":{"ids":[0]}})", R"({"selection":{"ids":[1.5]}})",
      R"({"selection":{"ids":[18446744073709551615]}})",
      R"({"selection":{"region":{"x":[2,1]}}})",
      R"({"selection":{"region":{"x":[1]}}})",
      R"({"selection":{"region":{"xx":[0,1]}}})",
      R"({"selection":{"region":{},"regions":[{"x":[0,1]}]}})",
      R"({"selection":{"mode":"typo","types":[1]}})",
      R"({"selection":{"type":[1]}})", R"({"fields":["id","id"]})",
      R"({"fields":["foo"]})", R"({"fields":[]})", R"({"max_atoms":0})",
      R"({"max_atoms":2147483648})"}) {
    bool threw=false;
    try { ParseOptions(Json::parse(invalid)); } catch(const std::exception&) { threw=true; }
    assert(threw);
  }
  std::cout << "trajectory selection host tests passed (Real=" << sizeof(Real)
            << ", Id=" << sizeof(Id) << ")\n";
}
