// clang++ -std=c++17 -O2 -Isrc tests/rbe_low_modes_test.cpp -o /tmp/rbe_low_modes_test
#include "common/rbe_low_modes.h"
#include <cassert>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>

using Sampler = rbmd::RbeLowModes;
using Mode = Sampler::Mode;
using Obs = std::array<double, 25>; // 6 atom forces, reciprocal energy, 6 virials
const std::array<double, 3> box{{7, 9, 11}};
const double alpha = 0.3, pi = std::acos(-1.0);
const std::array<std::array<double, 3>, 6> positions{{
    {{.2,.4,.8}}, {{2.1,3.3,1.7}}, {{5.1,1.1,7.2}},
    {{3.2,7.4,9.1}}, {{6.2,5.8,4.7}}, {{1.4,2.8,6.3}}}};
const std::array<double, 6> charges{{1,-1,1,-1,1,-1}};
double k2(const Mode& m) {
  return std::pow(2*pi*m.x/box[0],2)+std::pow(2*pi*m.y/box[1],2)+std::pow(2*pi*m.z/box[2],2);
}
Obs observable(const Mode& m) {
  Obs out{};
  const std::array<double,3> k{{2*pi*m.x/box[0],2*pi*m.y/box[1],2*pi*m.z/box[2]}};
  const double kk=k2(m), volume=box[0]*box[1]*box[2];
  double re=0, im=0;
  std::array<double,6> phase;
  for(int i=0;i<6;++i) {
    phase[i]=0;
    for(int d=0;d<3;++d) phase[i]+=k[d]*positions[i][d];
    re+=charges[i]*std::cos(phase[i]); im+=charges[i]*std::sin(phase[i]);
  }
  for(int i=0;i<6;++i) for(int d=0;d<3;++d)
    out[3*i+d]=-4*pi*charges[i]/volume/kk*(std::cos(phase[i])*im-std::sin(phase[i])*re)*k[d];
  out[18]=2*pi/volume*(re*re+im*im)/kk;
  const int a[6]={0,1,2,0,0,1}, b[6]={0,1,2,1,2,2};
  for(int j=0;j<6;++j) out[19+j]=out[18]*((a[j]==b[j]?1.:0.)-2*k[a[j]]*k[b[j]]*(1/(4*alpha)+1/kk));
  return out;
}
int main() {
  Sampler invalid;
  for(double bad : {-1.0, std::numeric_limits<double>::infinity()}) {
    bool threw=false; try { invalid.Setup(box,alpha,bad); } catch(const std::invalid_argument&) { threw=true; }
    assert(threw);
  }
  bool threw=false; try { invalid.Setup(box,alpha,100); } catch(const std::invalid_argument&) { threw=true; }
  assert(threw);
  std::vector<Mode> modes;
  Obs exact{};
  std::array<int,3> bounds;
  for(int d=0;d<3;++d) bounds[d]=std::ceil(std::sqrt(192*alpha)*box[d]/(2*pi));
  for(int x=-bounds[0];x<=bounds[0];++x)
    for(int y=-bounds[1];y<=bounds[1];++y)
      for(int z=-bounds[2];z<=bounds[2];++z) if(x||y||z) {
        Mode m{x,y,z,0}; m.weight=std::exp(-k2(m)/(4*alpha)); modes.push_back(m);
        auto o=observable(m); for(int j=0;j<25;++j) exact[j]+=m.weight*o[j];
      }
  const int batch=32, repeats=12000;
  std::cout<<"K_max,deterministic,tail_samples,tail_mass,force_variance_predicted,force_variance_measured,force_mean_error_rms,max_observable_mean_z\n";
  for(double cutoff : {0., 2*pi/box[2], 1.0, 1.6, 2.2}) {
    Sampler sampler;
    assert(sampler.Setup(box,alpha,cutoff));
    assert(!sampler.Setup(box,alpha,cutoff));
    std::set<std::array<int,3>> low;
    Obs deterministic{}, tail_mean{}, second{};
    for(auto m:sampler.Low()) {
      assert(k2(m)<=cutoff*cutoff && (m.x||m.y||m.z));
      assert(low.insert(std::array<int,3>{{m.x,m.y,m.z}}).second);
      auto o=observable(m); for(int j=0;j<25;++j) deterministic[j]+=m.weight*o[j];
    }
    double mass=0;
    for(auto m:modes) {
      if(k2(m)<=cutoff*cutoff) { assert(low.count({{m.x,m.y,m.z}})); continue; }
      mass+=m.weight; auto o=observable(m);
      for(int j=0;j<25;++j) { tail_mean[j]+=m.weight*o[j]; second[j]+=m.weight*o[j]*o[j]; }
    }
    assert(std::abs(mass-sampler.TailMass())<1e-12*mass);
    Obs variance{}, mean{}, m2{};
    for(int j=0;j<25;++j) variance[j]=(mass*second[j]-tail_mean[j]*tail_mean[j])/batch;
    std::mt19937 rng(20261012);
    for(int r=1;r<=repeats;++r) {
      Obs estimate=deterministic;
      for(int p=0;p<batch;++p) {
        auto m=sampler.DrawTail(rng,batch);
        assert(k2(m)>cutoff*cutoff && (m.x||m.y||m.z));
        auto o=observable(m); for(int j=0;j<25;++j) estimate[j]+=m.weight*o[j];
      }
      for(int j=0;j<25;++j) { double delta=estimate[j]-mean[j]; mean[j]+=delta/r; m2[j]+=delta*(estimate[j]-mean[j]); }
    }
    double predicted=0,measured=0,error=0,max_z=0;
    for(int j=0;j<25;++j) {
      double z=std::abs(mean[j]-exact[j])/std::sqrt(variance[j]/repeats);
      max_z=std::max(max_z,z); assert(z<6);
      if(j<18) { predicted+=variance[j]/6; measured+=m2[j]/(repeats-1)/6; error+=std::pow(mean[j]-exact[j],2)/6; }
    }
    assert(std::abs(measured/predicted-1)<.06);
    std::cout<<std::setprecision(10)<<cutoff<<','<<low.size()<<','<<batch<<','<<mass<<','<<predicted<<','<<measured<<','<<std::sqrt(error)<<','<<max_z<<'\n';
    auto changed=box; changed[0]*=1.7;
    assert(sampler.Setup(changed,alpha,cutoff));
    std::mt19937 first(19),second_rng(19);
    for(int i=0;i<100;++i) {
      auto a=sampler.DrawTail(first,batch),b=sampler.DrawTail(second_rng,batch);
      assert(a.x==b.x && a.y==b.y && a.z==b.z && a.weight==b.weight);
      double kk=std::pow(2*pi*a.x/changed[0],2)+std::pow(2*pi*a.y/changed[1],2)+std::pow(2*pi*a.z/changed[2],2);
      assert(kk>cutoff*cutoff);
    }
  }
}
