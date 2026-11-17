#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>
#include <memory>
#include <functional>
#include <vector>
#include "common/types.h"
#define __host__
#define __global__
#define __device__
#define __shared__
#define BLOCK_SIZE 1
#define SQRT std::sqrt
#define MIN(a,b) ((a)<(b)?(a):(b))
#define MAX(a,b) ((a)>(b)?(a):(b))
#define CHECK_KERNEL(...) __VA_ARGS__
namespace device { struct DEVICE_GPU {}; }
struct Dim { unsigned int x=0; } blockIdx, threadIdx, blockDim{1};
struct Box { double length=32; };
struct LinkedCellDeviceDataPtr {
  rbmd::Id _d_per_dimension_cells[3]{1,1,1};
  rbmd::Id _d_nativate_cells_start[3]{0,0,0};
  rbmd::Id _d_nativate_cells_dims[3]{1,1,1};
  bool _d_covers_whole_domain[3]{true,true,true};
};
struct Real7 {
  rbmd::Real v[7]{};
  rbmd::Real& operator[](int i) { return v[i]; }
  rbmd::Real operator[](int i) const { return v[i]; }
};
template<class T,int> struct BLOCKREDUCE {
  struct TempStorage{};
  explicit BLOCKREDUCE(TempStorage&) {}
  T Sum(T x){return x;}
};
void MinImageDistance(Box box, rbmd::Real& x, rbmd::Real& y, rbmd::Real& z) {
  for(auto* v:{&x,&y,&z}) *v-=box.length*std::nearbyint(*v/box.length);
}
#include "production_eam.h"
#include "production_halo.h"
namespace thrust {
template<class T> using plus=std::plus<T>;
template<class I,class T,class Op> T reduce(I b,I e,T init,Op op) {return std::accumulate(b,e,init,op);}
template<class T> T* raw_pointer_cast(T* p){return p;}
}
namespace op { template<class> struct FixRBLForceOp {
  void operator()(rbmd::Id n,rbmd::Real x,rbmd::Real y,rbmd::Real z,
                  rbmd::Real* fx,rbmd::Real* fy,rbmd::Real* fz) {
    for(rbmd::Id i=0;i<n;++i){fx[i]-=x;fy[i]-=y;fz[i]-=z;}
  }
}; }
struct Data {std::vector<rbmd::Real> _d_fx,_d_fy,_d_fz;};
struct EAM {
  std::shared_ptr<Data> _device_data=std::make_shared<Data>();
  rbmd::Real _corr_value_x{},_corr_value_y{},_corr_value_z{};
  void ApplyRblForceCorrection(rbmd::Id num_atoms);
};
#include "production_rbl_correction.h"

void close(double a,double b) {
  if(std::abs(a-b)>=1.e-5) {
    std::cerr<<"mismatch: actual="<<a<<" expected="<<b<<'\n';
    assert(false);
  }
}
int main() {
  // Analytic synthetic potential: rho(r)=1, F(rho)=rho, phi(r)=2/r.
  // E = directed-neighbor count + sum_{i<j} 2/r_ij.
  Real7 rho[16], embedding[16], pair[16];
  for(int m=0;m<16;++m) {
    rho[m][6]=1;
    embedding[m][2]=1; embedding[m][5]=1; embedding[m][6]=m-1;
    pair[m][6]=2;
  }
  EAMParameters params{1,1,15,0,16,16};
  rbmd::Real x[]{0,2,5}, y[]{0,0,0}, z[]{0,0,0};
  rbmd::Id ids[]{10000000000LL,42,10000000007LL}, types[]{0,0,0};
  rbmd::Id start[]{0,2,4}, end[]{2,4,6}, neighbors[]{1,2,0,2,0,1};
  rbmd::Real fp[]{-99,-99,-99}, fx[3]{},fy[3]{},fz[3]{};
  rbmd::Energy eembed[3]{}, epair[3]{};
  for(unsigned i=0;i<3;++i) {
    blockIdx.x=i;
    op::FpVerlet({},params,10,3,types,ids,start,end,neighbors,rho,embedding,x,y,z,fp,eembed);
    close(fp[i],1);
  }
  for(unsigned i=0;i<3;++i) {
    blockIdx.x=i;
    op::EAMForceVerlet({},params,10,3,types,ids,start,end,neighbors,rho,pair,x,y,z,fp,fx,fy,fz,epair);
  }
  close(std::accumulate(eembed,eembed+3,0.0),6);
  close(std::accumulate(epair,epair+3,0.0),1+0.4+2.0/3);
  close(fx[0],-0.5-0.08); close(fx[1],0.5-2.0/9); close(fx[2],0.08+2.0/9);
  // Independently permuted rank: one owned atom, two ghosts. Only owned output
  // may be written. F'(rho) for ghost atoms is supplied by its owner.
  rbmd::Real rx[]{5,0,2},rfp[]{-99,fp[0],fp[1]},rfx[]{123,123,123},rfy[3]{},rfz[3]{};
  rbmd::Energy re[1]{},rp[1]{};
  blockIdx.x=0;
  op::FpVerlet({},params,10,1,types,ids,start,end,neighbors,rho,embedding,rx,y,z,rfp,re);
  op::EAMForceVerlet({},params,10,1,types,ids,start,end,neighbors,rho,pair,rx,y,z,rfp,rfx,rfy,rfz,rp);
  close(re[0],eembed[2]); close(rp[0],epair[2]); close(rfx[0],fx[2]); close(rfx[1],123);
  // Direct traversal must reproduce density and energy without neighbor arrays.
  rbmd::Id cells[]{0,0,0}, cellstart[]{0},cellend[]{3},sorted[]{0,1,2};
  LinkedCellDeviceDataPtr linked;
  op::EAMDirectTraversalData traversal{cells,cellstart,cellend,sorted,&linked,1,0};
  rbmd::Energy de[3]{}, dp[3]{};
  for(unsigned i=0;i<3;++i) {
    blockIdx.x=i;
    op::FpDirect({},params,10,3,traversal,rho,embedding,x,y,z,fp,de);
    op::EAMEnergyDirect({},params,10,3,traversal,rho,pair,x,y,z,fp,dp);
    close(de[i],eembed[i]);close(dp[i],epair[i]);
  }
  // Cell mapping must match an independent pure-modulo reference, including
  // mixed native-domain mapping and offsets spanning many periodic images.
  auto check_cell_iterator=[](rbmd::Id dim_x,rbmd::Id dim_y,rbmd::Id dim_z,
                              rbmd::Id start_x,rbmd::Id start_y,rbmd::Id start_z,
                              rbmd::Id native_x,rbmd::Id native_y,rbmd::Id native_z,
                              bool cover_x,bool cover_y,bool cover_z,
                              rbmd::Id center_x,rbmd::Id center_y,rbmd::Id center_z,
                              rbmd::Id radius) {
    LinkedCellDeviceDataPtr data;
    data._d_per_dimension_cells[0]=dim_x;
    data._d_per_dimension_cells[1]=dim_y;
    data._d_per_dimension_cells[2]=dim_z;
    data._d_nativate_cells_start[0]=start_x;
    data._d_nativate_cells_start[1]=start_y;
    data._d_nativate_cells_start[2]=start_z;
    data._d_nativate_cells_dims[0]=native_x;
    data._d_nativate_cells_dims[1]=native_y;
    data._d_nativate_cells_dims[2]=native_z;
    data._d_covers_whole_domain[0]=cover_x;
    data._d_covers_whole_domain[1]=cover_y;
    data._d_covers_whole_domain[2]=cover_z;
    const rbmd::Id width=2*radius+1;
    const rbmd::Id count=width*width*width;
    const rbmd::Id center=(center_z*dim_y+center_y)*dim_x+center_x;
    auto pure_wrap=[](long long value,rbmd::Id size) {
      long long result=value%static_cast<long long>(size);
      if(result<0) result+=size;
      return static_cast<rbmd::Id>(result);
    };
    auto map=[&](rbmd::Id cell,long long offset,rbmd::Id dimension,
                 rbmd::Id native_start,rbmd::Id native_width,bool covers) {
      return covers
          ? pure_wrap(static_cast<long long>(cell)-native_start+offset,
                      native_width)+native_start
          : pure_wrap(static_cast<long long>(cell)+offset,dimension);
    };
    for(rbmd::Id stencil=0;stencil<count;++stencil) {
      const long long ox=stencil%width-radius;
      const long long oy=(stencil/width)%width-radius;
      const long long oz=stencil/(width*width)-radius;
      const rbmd::Id expected_x=map(center_x,ox,dim_x,start_x,native_x,cover_x);
      const rbmd::Id expected_y=map(center_y,oy,dim_y,start_y,native_y,cover_y);
      const rbmd::Id expected_z=map(center_z,oz,dim_z,start_z,native_z,cover_z);
      const rbmd::Id expected=(expected_z*dim_y+expected_y)*dim_x+expected_x;
      assert(op::EAMDirectNeighborCell(&data,center,stencil,radius)==expected);
    }
  };
  check_cell_iterator(7,5,4,2,0,1,3,5,2,true,false,true,3,4,2,6);
  check_cell_iterator(4,3,2,0,0,0,4,3,2,false,false,false,0,2,1,3);
  check_cell_iterator(1,1,1,0,0,0,1,1,1,true,true,true,0,0,0,0);
  // Direct RBL force sampling is defined by the flattened stencil/cell order,
  // including shell entries skipped after the sample cap is reached.  Use
  // unequal cell populations and a shell whose first and third forces cancel:
  // changing traversal order changes the capped result for atom 0.
  constexpr rbmd::Id direct_atoms=5, direct_storage=7;
  rbmd::Real mx[]{0,4,-4,1,8,-1,-5};
  rbmd::Real my[direct_storage]{},mz[direct_storage]{};
  rbmd::Id mcells[]{0,1,2,0,1};
  rbmd::Id mstart[]{0,2,4},mend[]{2,4,7};
  rbmd::Id msorted[]{0,3,1,4,2,5,6};
  LinkedCellDeviceDataPtr mlinked;
  mlinked._d_per_dimension_cells[0]=3;
  mlinked._d_nativate_cells_dims[0]=3;
  op::EAMDirectTraversalData mtraversal{
      mcells,mstart,mend,msorted,&mlinked,3,1};
  rbmd::Real mfp[direct_storage];
  std::fill(mfp,mfp+direct_storage,rbmd::Real(1));
  rbmd::Energy me[direct_atoms]{},mp[direct_atoms]{},mfused[direct_atoms]{};
  auto visit_direct=[&](rbmd::Id atom,auto&& visitor) {
    const rbmd::Id cell=mcells[atom];
    const rbmd::Id ordered_cells[]{(cell+2)%3,cell,(cell+1)%3};
    for(rbmd::Id neighbor_cell:ordered_cells)
      for(rbmd::Id pos=mstart[neighbor_cell];pos<mend[neighbor_cell];++pos) {
        const rbmd::Id neighbor=msorted[pos];
        if(neighbor!=atom) visitor(neighbor);
      }
  };
  for(unsigned i=0;i<direct_atoms;++i) {
    blockIdx.x=i;
    op::FpDirect({},params,6,direct_atoms,mtraversal,rho,embedding,
                 mx,my,mz,mfp,me);
    op::EAMEnergyDirect({},params,6,direct_atoms,mtraversal,rho,pair,
                        mx,my,mz,mfp,mp);
    op::FpDirect<true>({},params,6,direct_atoms,mtraversal,rho,embedding,
                       mx,my,mz,mfp,me,pair,mfused);
    rbmd::Real expected_rho=0;
    rbmd::Energy expected_pair=0;
    visit_direct(i,[&](rbmd::Id neighbor) {
      const rbmd::Real distance=std::abs(mx[i]-mx[neighbor]);
      if(distance<6) {
        expected_rho+=1;
        expected_pair+=1/distance;
      }
    });
    close(mfp[i],1);
    close(me[i],expected_rho);
    close(mp[i],expected_pair);
    close(mfused[i],mp[i]);
  }
  auto check_direct_force=[&](rbmd::Real core_cutoff,rbmd::Id frequency,
                              rbmd::Id sample_cap) {
    rbmd::Real mfx[direct_atoms]{},mfy[direct_atoms]{},mfz[direct_atoms]{};
    for(unsigned i=0;i<direct_atoms;++i) {
      blockIdx.x=i;
      op::EAMForceRBLDirect({},params,core_cutoff,6,direct_atoms,sample_cap,frequency,
                            mtraversal,rho,pair,mx,my,mz,mfp,mfx,mfy,mfz);
      rbmd::Real expected=0;
      rbmd::Id shell_index=0,sampled=0;
      visit_direct(i,[&](rbmd::Id neighbor) {
        const rbmd::Real dx=mx[i]-mx[neighbor];
        const rbmd::Real distance=std::abs(dx);
        if(distance<core_cutoff) {
          expected+=2*dx/(distance*distance*distance);
        } else if(distance<6) {
          if(shell_index%frequency==0 && sampled<sample_cap) {
            expected+=frequency*2*dx/(distance*distance*distance);
            ++sampled;
          }
          ++shell_index;
        }
      });
      close(mfx[i],expected);close(mfy[i],0);close(mfz[i],0);
    }
    return mfx[0];
  };
  close(check_direct_force(2,1,99),0.08);
  close(check_direct_force(2,1,2),0.08+0.125);
  close(check_direct_force(2,2,1),0.25);
  close(check_direct_force(2,2,2),0);
  close(check_direct_force(2,3,1),0.375);
  // With core == cutoff the shell is empty and sample_cap == 0 is inert.
  close(check_direct_force(6,1,0),0.08);
  // Scalar payload is never translated by periodic coordinate shifts; each
  // stage can forward a ghost updated in the preceding stage.
  std::array<std::array<rbmd::Real,4>,4> values{};
  values[0][0]=7.5;
  rbmd::Real shifts[]{10000},payload[1]{};
  for(int stage=0;stage<3;++stage) {
    rbmd::Id src[]{stage},dst[]{stage+1};
    halo_test::PackForwardFields pack{src,shifts,shifts,shifts,{}, {},payload,1,1};
    pack.source[0]=values[stage].data();pack.shift_dimension[0]=-1;pack(0);
    halo_test::ApplyForwardFields apply{payload,dst,{},1,1};
    apply.destination[0]=values[stage+1].data();apply(0);
    close(values[stage+1][stage+1],7.5);
    dst[0]=-1;payload[0]=123;apply(0);close(values[stage+1][stage+1],7.5);
  }
  EAM correction;
  correction._device_data->_d_fx={1,3,1000};
  correction._device_data->_d_fy={2,6,1000};
  correction._device_data->_d_fz={3,9,1000};
  correction.ApplyRblForceCorrection(2);
  close(correction._corr_value_x,2);close(correction._corr_value_y,4);close(correction._corr_value_z,6);
  close(correction._device_data->_d_fx[0],-1);close(correction._device_data->_d_fx[2],1000);
  // FP64 block outputs preserve a small difference above a large baseline.
  rbmd::Energy blocks[]{4.e10,1.25,-4.e10};
  close(std::accumulate(blocks,blocks+3,0.0),1.25);
  std::cout<<"PASS "<<(sizeof(rbmd::Real)==8?"double":"float")
    <<": production EAM kernels/wrappers compile, analytic energy/force, owned/ghost permutation, direct traversal, FP64 block sums\n";
}
