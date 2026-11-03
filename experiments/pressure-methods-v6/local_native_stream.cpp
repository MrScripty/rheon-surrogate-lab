// Exact same three-stage stencil and per-layer replicated boundaries as v1,
// scheduled through three-plane rings to avoid full-grid hidden activations.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <new>
#include <vector>
struct Workspace {std::vector<float> q,a,b,c;int qz[3]={-1,-1,-1};};
extern "C" void* rheon_local_create(){return new(std::nothrow) Workspace;}
extern "C" void rheon_local_destroy(void*p){delete static_cast<Workspace*>(p);}
static void plane(const float*const input[3],float* __restrict out,const float* __restrict w,int ci,int co,int ny,int nx){
    const size_t n=size_t(ny)*nx;std::fill(out,out+n*co,0.f);
    for(int o=0;o<co;++o)for(int i=0;i<ci;++i)for(int dz=0;dz<3;++dz)
      for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx){
        const float weight=w[((o*ci+i)*3+dz)*9+(dy+1)*3+dx+1];
        for(int y=0;y<ny;++y){
          const float* __restrict src=input[dz]+i*n+size_t(std::clamp(y+dy,0,ny-1))*nx;
          float* __restrict dst=out+o*n+size_t(y)*nx;
          dst[0]+=weight*src[std::clamp(dx,0,nx-1)];
          for(int x=1;x<nx-1;++x)dst[x]+=weight*src[x+dx];
          if(nx>1)dst[nx-1]+=weight*src[std::clamp(nx-1+dx,0,nx-1)];
        }
      }
}
extern "C" int rheon_local_apply(void*p,const double*r,double factor,double scale,const float*w1,const float*w2,const float*w3,int nz,int ny,int nx,double*out){
    if(!p||!r||!out||!w1||!w2||!w3||nz<1||ny<1||nx<1||nz>256||ny>256||nx>256||!std::isfinite(factor)||factor<=0||!std::isfinite(scale)||scale<=0)return 1;
    try{
      auto&s=*static_cast<Workspace*>(p);const size_t n=size_t(ny)*nx;
      s.q.resize(3*n);s.a.resize(3*4*n);s.b.resize(3*4*n);s.c.resize(n);
      for(int k=0;k<3;++k)s.qz[k]=-1;
      bool finite=true;double gauge=0;
      auto qplane=[&](int z)->const float*{
        z=std::clamp(z,0,nz-1);int slot=z%3;float*dest=s.q.data()+slot*n;
        if(s.qz[slot]!=z){for(size_t j=0;j<n;++j){dest[j]=float((r[size_t(z)*n+j]*factor)/scale);if(!std::isfinite(dest[j]))finite=false;}s.qz[slot]=z;}
        return dest;
      };
      auto aplane=[&](int z){const float*in[3]={qplane(z-1),qplane(z),qplane(z+1)};plane(in,s.a.data()+(z%3)*4*n,w1,1,4,ny,nx);};
      auto bplane=[&](int z){const float*in[3];for(int dz=-1;dz<=1;++dz)in[dz+1]=s.a.data()+(std::clamp(z+dz,0,nz-1)%3)*4*n;plane(in,s.b.data()+(z%3)*4*n,w2,4,4,ny,nx);};
      auto cplane=[&](int z){
        const float*in[3];for(int dz=-1;dz<=1;++dz)in[dz+1]=s.b.data()+(std::clamp(z+dz,0,nz-1)%3)*4*n;
        plane(in,s.c.data(),w3,4,1,ny,nx);
        for(size_t j=0;j<n;++j){const size_t k=size_t(z)*n+j;float q=float((r[k]*factor)/scale);double value=double(q*.125f+s.c[j])*scale;if(k==0)gauge=value;out[k]=value-gauge;if(!std::isfinite(out[k]))finite=false;}
      };
      for(int z=0;z<nz;++z){aplane(z);if(z>=1)bplane(z-1);if(z>=2)cplane(z-2);}
      bplane(nz-1);if(nz>=2)cplane(nz-2);cplane(nz-1);
      return finite?0:2;
    }catch(...){return 4;}
}
