#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
using I=rbmd::Id; using R=rbmd::Real;
static void require(bool x,const std::string& m){if(!x)throw std::runtime_error(m);}
static void hipok(hipError_t e,const char* m){if(e!=hipSuccess)throw std::runtime_error(std::string(m)+": "+hipGetErrorString(e));}
template<class T> struct Dev {
 T* p=nullptr;std::size_t n;
 explicit Dev(std::size_t c):n(std::max<std::size_t>(c,1)){hipok(hipMalloc(reinterpret_cast<void**>(&p),n*sizeof(T)),"malloc");}
 explicit Dev(const std::vector<T>& v):Dev(v.size()){put(v);}
 ~Dev(){if(p)hipFree(p);} Dev(const Dev&)=delete;Dev&operator=(const Dev&)=delete;
 void put(const std::vector<T>& v){require(v.size()<=n,"upload bound");if(!v.empty())hipok(hipMemcpy(p,v.data(),v.size()*sizeof(T),hipMemcpyHostToDevice),"upload");}
 void fill(T x){put(std::vector<T>(n,x));}
 std::vector<T> get(std::size_t c=0)const{if(!c)c=n;require(c<=n,"download bound");std::vector<T> v(c);hipok(hipMemcpy(v.data(),p,c*sizeof(T),hipMemcpyDeviceToHost),"download");return v;}
};
struct Data {
 std::vector<I> ids{100,200},types{1,1},candidates{1},core;
 std::vector<R> q{R(1),R(-1)},x{R(0),R(2)},y{R(0),R(0)},z{R(0),R(0)};
 Box box;
 Data(){box._type=Box::BoxType::ORTHOGONAL;box._pbc_x=box._pbc_y=box._pbc_z=true;for(int a=0;a<3;++a){box._length[a]=R(100);box._length_inv[a]=R(.01);box._coord_min[a]=R(-50);box._coord_max[a]=R(50);}}
};
struct Session {
 Data& f;bool implicit;I prefix;R threshold,physical;
 Dev<I> ids,types,rows,cs,ce,cn,maxcore,cb,cecore,core,corecount,required,realloc,error;
 Dev<R> q,x,y,z;
 Session(Data& d,bool im,R t,R p):f(d),implicit(im),prefix(im?0:2),threshold(t),physical(p),
  ids(d.ids),types(d.types),rows(d.candidates.size()+prefix),cs(std::vector<I>{prefix}),ce(std::vector<I>{prefix+I(d.candidates.size())}),cn(std::vector<I>{I(d.candidates.size())}),maxcore(std::vector<I>{32}),cb(std::vector<I>{0}),cecore(std::vector<I>{I(d.core.size())}),core(32),corecount(1),required(1),realloc(1),error(1),q(d.q),x(d.x),y(d.y),z(d.z){
  require(d.core.size()<=32,"fixture core cap");std::vector<I> r(prefix,I(-777));r.insert(r.end(),d.candidates.begin(),d.candidates.end());rows.put(r);core.fill(I(-999));core.put(d.core);}
 void coords(){x.put(f.x);y.put(f.y);z.put(f.z);}
 I produce(I tpp){required.fill(0);realloc.fill(0);core.fill(I(-999));corecount.fill(0);cecore.fill(0);
  op::FilterRblNeighborCandidatesOp<device::DEVICE_GPU>()(threshold,R(256),1,ids.p,x.p,y.p,z.p,cs.p,ce.p,rows.p,cn.p,I(f.candidates.size()),implicit,maxcore.p,cb.p,cecore.p,core.p,corecount.p,0,nullptr,nullptr,nullptr,required.p,f.box,realloc.p,1,1,tpp);
  hipok(hipDeviceSynchronize(),"actual filter sync");require(!realloc.get()[0],"unexpected producer capacity failure");const I n=corecount.get()[0];require(n>=0&&n<=I(f.candidates.size())&&cecore.get()[0]==n&&required.get()[0]==n,"producer metadata");return n;}
 void replace_core(const std::vector<I>& row){require(row.size()<=32,"replacement core cap");core.fill(I(-999));core.put(row);cecore.put(std::vector<I>{I(row.size())});}
 void witness(bool cvff,I expected_stored){Dev<unsigned int> claimed(1);Dev<I> wi(22);Dev<R> va(39);claimed.fill(0);wi.fill(I(-987));va.fill(R(-654));
  op::DiagnoseRblFixedpMissingCoreOp<device::DEVICE_GPU>()(1,I(f.ids.size()),ids.p,x.p,y.p,z.p,cs.p,ce.p,rows.p,cn.p,I(f.candidates.size()),implicit,rows.n,cb.p,cecore.p,core.p,core.n,threshold,physical,f.box,cvff,claimed.p,wi.p+1,va.p+1);
  hipok(hipDeviceSynchronize(),"actual witness sync");auto id=wi.get();auto value=va.get();require(claimed.get()[0]==1,"failure witness missing");require(id.front()==I(-987)&&id.back()==I(-987)&&value.front()==R(-654)&&value.back()==R(-654),"witness buffer canary");require(id[1+18]==expected_stored,"witness actual stored localidx");
  if(expected_stored>=0){require(value[1+28]==f.x[expected_stored]&&value[1+29]==f.y[expected_stored]&&value[1+30]==f.z[expected_stored],"witness actual stored coordinates");require(id[1+19]==0,"failed different input must have equivalence kind0");}}
 I prepare(bool cvff){error.fill(0);auto before=core.get();op::PrepareRblFixedpRowsOp<device::DEVICE_GPU>()(1,I(f.ids.size()),ids.p,types.p,q.p,x.p,y.p,z.p,cs.p,ce.p,rows.p,cn.p,I(f.candidates.size()),implicit,rows.n,cb.p,cecore.p,core.p,core.n,threshold,physical,f.box,cvff,error.p);
  hipok(hipDeviceSynchronize(),"actual prepare sync");require(before==core.get(),"Prepare changed core list");auto r=rows.get();for(I j=0;j<prefix;++j)require(r[j]==I(-777),"CSR prefix changed");return error.get()[0];}
};
static std::size_t producer_cases=0,producer_accepts=0,producer_rejects=0;
static void producer_contract(){
 const R coords[2][3]={{R(2.4755117893218994),R(2.683236837387085),R(5.1344113349914551)},{R(-.84582889080047607),R(3.0803050994873047),R(4.4773197174072266)}};
 const R threshold[2]={R(6.3)*R(6.3)+EPSILON,R(5.5)*R(5.5)+EPSILON};
 const R physical[2]={R(36),R(5.2)*R(5.2)};
 for(int point=0;point<2;++point)for(bool cvff:{false,true})for(bool implicit:{false,true})for(I tpp:{4,8,16,32,64}){
  Data d;d.x[1]=coords[point][0];d.y[1]=coords[point][1];d.z[1]=coords[point][2];Session s(d,implicit,threshold[point],physical[point]);
  // The REAL GPU producer decides padding membership; no CPU/FMA oracle overrides it.
  for(int move=-8;move<=8;++move){R value=coords[point][2];for(int j=0;j<std::abs(move);++j)value=std::nextafter(value,move<0?-std::numeric_limits<R>::infinity():std::numeric_limits<R>::infinity());d.z[1]=value;s.coords();const I n=s.produce(tpp);require(n==0||n==1,"single physical producer count");++producer_cases;n?++producer_accepts:++producer_rejects;require(!s.prepare(cvff),"actual producer vs v4 Prepare contract mismatch");}
 }
 require(producer_accepts&&producer_rejects,"edge sweep did not cover both cache membership sides");
}
static void hard_guards(){for(bool cvff:{false,true})for(bool implicit:{false,true}){
 {Data d;Session s(d,implicit,R(6.3)*R(6.3)+EPSILON,R(36));require(s.produce(8)==1&&!s.prepare(cvff),"valid real core");s.replace_core({});require(s.prepare(cvff)==I(2048),"missing physical core must be 2048");}
 {Data d;d.x[1]=R(6.1);Session s(d,implicit,R(6.3)*R(6.3)+EPSILON,R(36));require(s.produce(8)==1&&!s.prepare(cvff),"valid padding");s.replace_core({});require(s.prepare(cvff)==I(128),"missing padding-only core must remain 128");}
 {Data d;d.x[1]=R(6);Session s(d,implicit,R(6.3)*R(6.3)+EPSILON,R(36));require(s.produce(8)==1&&!s.prepare(cvff),"exact core boundary valid");s.replace_core({});require(s.prepare(cvff)==I(2048),"exact core boundary coverage");}
 {Data d;d.core={1,1};Session s(d,implicit,R(6.3)*R(6.3)+EPSILON,R(36));require(s.prepare(cvff)==I(32),"duplicate stored core must remain 32");}
 }
 // A same-GID but force-ineligible alias cannot stand in for the physical core.
 {Data d;d.ids.push_back(200);d.types.push_back(1);d.q.push_back(-1);d.x.push_back(R(202));d.y.push_back(0);d.z.push_back(0);d.candidates={1,2};d.core={2};Session s(d,false,R(6.3)*R(6.3)+EPSILON,R(36));require(s.prepare(false)==I(2048),"wrong single-wrap alias must be 2048");require(!s.prepare(true),"CVFF RINT equivalent alias should remain supported");}
}
static void structural_inputs(){
 for(bool cvff:{false,true})for(bool implicit:{false,true}){
  Data d;d.ids.push_back(200);d.types.push_back(1);d.q.push_back(-1);d.x.push_back(R(2));d.y.push_back(0);d.z.push_back(0);d.candidates={1};d.core={2};
  Session same(d,implicit,R(6.3)*R(6.3)+EPSILON,R(36));require(!same.prepare(cvff),"exact raw inputs must establish coverage");
  d.x[2]=R(102);Session image(d,implicit,R(6.3)*R(6.3)+EPSILON,R(36));require(!image.prepare(cvff),"exact MIC inputs must establish coverage");
  d.x[2]=0;d.y[2]=R(2);Session wrong(d,implicit,R(6.3)*R(6.3)+EPSILON,R(36));require(wrong.prepare(cvff)==I(2048),"equal norm different direction must not establish coverage");wrong.witness(cvff,2);
  d.x[2]=R(2);d.y[2]=0;d.q[2]=R(-2);Session charge(d,implicit,R(6.3)*R(6.3)+EPSILON,R(36));require(charge.prepare(cvff)==I(4),"alias charge mismatch must fail");
  d.q[2]=R(-1);d.types[2]=2;Session type(d,implicit,R(6.3)*R(6.3)+EPSILON,R(36));require(type.prepare(cvff)==I(4),"alias type mismatch must fail");
 }
 const R pts[3][7]={{R(123.445839),R(80.7254562),R(241.19899),R(120.381302),R(85.1668701),R(243.822433),R(271)},
  {R(38.1199493),R(73.9849472),R(80.3792343),R(38.4486313),R(79.2780838),R(81.8368607),R(90)},
  {R(449.347473),R(389.213562),R(370.747406),R(444.995026),R(386.325043),R(373.699127),R(583.851807)}};
 for(int c=0;c<3;++c){Data d;d.x={pts[c][0],pts[c][3]};d.y={pts[c][1],pts[c][4]};d.z={pts[c][2],pts[c][5]};d.core={1};
  for(int axis=0;axis<3;++axis){d.box._length[axis]=pts[c][6];d.box._length_inv[axis]=R(1)/pts[c][6];d.box._coord_min[axis]=0;d.box._coord_max[axis]=pts[c][6];}
  Session s(d,false,c==1?R(42.2500992):R(39.6901016),c==1?R(30.25):R(36));require(!s.prepare(c==1),"witness-position identical input coverage");
 }
}
int main(){try{int count=0;hipok(hipGetDeviceCount(&count),"device count");require(count>0,"NO GPU: test cannot pass");hipok(hipSetDevice(0),"set device");hipDeviceProp_t p;hipok(hipGetDeviceProperties(&p,0),"properties");std::cout<<"GPU_DEVICE="<<p.name<<" WARP="<<p.warpSize<<" REAL_BYTES="<<sizeof(R)<<" ID_BYTES="<<sizeof(I)<<'\n';require(sizeof(R)==4,"production fixture requires FP32 Real");producer_contract();hard_guards();structural_inputs();hipok(hipDeviceSynchronize(),"final sync");std::cout<<"PRODUCER_CASES="<<producer_cases<<" ACCEPTS="<<producer_accepts<<" REJECTS="<<producer_rejects<<'\n';std::cout<<"PASS_REAL_HIP_FIXEDP_V5_CORE_ALIAS_INPUTS\n";return 0;}catch(const std::exception& e){std::cerr<<"FAIL_REAL_HIP_FIXEDP_V5_CORE_ALIAS_INPUTS: "<<e.what()<<'\n';return 1;}}
