#pragma once
// In-process protocol test, NOT a substitute for real MPI/GPU integration.
#include <algorithm>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <map>
#include <mutex>
#include <vector>
using MPI_Comm=int; using MPI_Datatype=int; using MPI_Request=int;
struct MPI_Status {int MPI_SOURCE=0, MPI_TAG=0;};
constexpr int MPI_SUCCESS=0, MPI_COMM_NULL=0, MPI_COMM_WORLD=1;
constexpr int MPI_BYTE=1,MPI_INT=2,MPI_FLOAT=3,MPI_DOUBLE=4,MPI_LONG_LONG=5;
constexpr int MPI_SUM=1,MPI_MIN=2,MPI_MAX=3,MPI_ANY_TAG=-1,MPI_REQUEST_NULL=0;
inline MPI_Status* MPI_STATUS_IGNORE=nullptr;
inline MPI_Status* MPI_STATUSES_IGNORE=nullptr;
namespace mock_mpi {
inline int size=4;
inline thread_local int rank=0;
inline thread_local std::size_t sequence=0;
inline thread_local int dims[3]{4,1,1};
inline std::mutex mutex;
inline std::condition_variable changed;
struct Slot {
  int kind=-1,count=0,type=0,op=0,root=0,arrived=0;
  std::vector<const void*> inputs;
  std::vector<void*> outputs;
};
inline std::map<std::size_t,Slot> slots;
inline std::size_t width(int t) {
  switch(t) {case MPI_BYTE:return 1;case MPI_INT:return sizeof(int);
    case MPI_FLOAT:return sizeof(float);case MPI_DOUBLE:return sizeof(double);
    case MPI_LONG_LONG:return sizeof(long long);default:assert(false);return 0;}
}
template<class T> void reduce(Slot& s) {
  for(int i=0;i<s.count;++i) {
    T value=static_cast<const T*>(s.inputs[0])[i];
    for(int r=1;r<size;++r) {
      T v=static_cast<const T*>(s.inputs[r])[i];
      if(s.op==MPI_SUM)value+=v;
      else if(s.op==MPI_MIN)value=std::min(value,v);
      else value=std::max(value,v);
    }
    for(auto out:s.outputs)static_cast<T*>(out)[i]=value;
  }
}
inline int collective(int kind,const void* in,void* out,int count,int type,int root=0,int op=0) {
  std::unique_lock<std::mutex> lock(mutex);
  auto& s=slots[sequence++];
  if(s.kind==-1) {
    s.kind=kind;s.count=count;s.type=type;s.root=root;s.op=op;
    s.inputs.resize(size);s.outputs.resize(size);
  }
  assert(s.kind==kind && s.count==count && s.type==type && s.root==root && s.op==op);
  s.inputs[rank]=in;s.outputs[rank]=out;
  if(++s.arrived==size) {
    auto bytes=width(type)*count;
    if(bytes) {
      if(kind==0)for(int r=0;r<size;++r)if(r!=root)std::memcpy(s.outputs[r],s.inputs[root],bytes);
      if(kind==1)for(auto target:s.outputs)for(int r=0;r<size;++r)
        std::memcpy(static_cast<char*>(target)+r*bytes,s.inputs[r],bytes);
      if(kind==2) {
        if(type==MPI_LONG_LONG)reduce<long long>(s);
        else if(type==MPI_INT)reduce<int>(s);
        else if(type==MPI_DOUBLE)reduce<double>(s);
        else if(type==MPI_FLOAT)reduce<float>(s);
        else assert(false);
      }
    }
    changed.notify_all();
  } else if(!changed.wait_for(lock,std::chrono::seconds(10),[&]{return s.arrived==size;})) {
    std::abort();
  }
  return MPI_SUCCESS;
}
struct Message {const void* data;int count,type,peer,tag;bool send;};
inline std::vector<Message> messages;
}
inline int MPI_Initialized(int* x){*x=1;return 0;}
inline int MPI_Finalized(int* x){*x=0;return 0;}
inline int MPI_Comm_rank(MPI_Comm,int* x){*x=mock_mpi::rank;return 0;}
inline int MPI_Comm_size(MPI_Comm,int* x){*x=mock_mpi::size;return 0;}
inline int MPI_Comm_dup(MPI_Comm x,MPI_Comm* y){*y=x;return 0;}
inline int MPI_Comm_free(MPI_Comm* x){*x=0;return 0;}
inline int MPI_Dims_create(int n,int,int* d){d[0]=n;d[1]=1;d[2]=1;return 0;}
inline int MPI_Cart_create(MPI_Comm,int,const int* d,const int*,int,MPI_Comm* c){std::copy(d,d+3,mock_mpi::dims);*c=2;return 0;}
inline int MPI_Cart_rank(MPI_Comm,const int* d,int* r){*r=(d[0]*mock_mpi::dims[1]+d[1])*mock_mpi::dims[2]+d[2];return 0;}
inline int MPI_Bcast(void* p,int n,MPI_Datatype t,int root,MPI_Comm){return mock_mpi::collective(0,p,p,n,t,root);}
inline int MPI_Allgather(const void* p,int n,MPI_Datatype t,void* q,int m,MPI_Datatype u,MPI_Comm){assert(n==m && t==u);return mock_mpi::collective(1,p,q,n,t);}
inline int MPI_Allreduce(const void* p,void* q,int n,MPI_Datatype t,int op,MPI_Comm){return mock_mpi::collective(2,p,q,n,t,0,op);}
inline int MPI_Irecv(void* p,int n,MPI_Datatype t,int peer,int tag,MPI_Comm,MPI_Request* r){mock_mpi::messages.push_back({p,n,t,peer,tag,false});*r=1;return 0;}
inline int MPI_Isend(const void* p,int n,MPI_Datatype t,int peer,int tag,MPI_Comm,MPI_Request* r){mock_mpi::messages.push_back({p,n,t,peer,tag,true});*r=1;return 0;}
inline int MPI_Waitall(int n,MPI_Request* r,MPI_Status*){for(int i=0;i<n;++i)r[i]=0;return 0;}
inline int MPI_Type_free(MPI_Datatype*){return 0;}
// Legacy scatter/streaming are compiled here but are not simulated by this test.
inline int MPI_Scatter(...){std::abort();}
inline int MPI_Scatterv(...){std::abort();}
inline int MPI_Ssend(...){std::abort();}
inline int MPI_Probe(...){std::abort();}
inline int MPI_Get_count(...){std::abort();}
inline int MPI_Recv(...){std::abort();}
