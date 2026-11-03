// One bounded native call: physical features, three replicated convolutions,
// fixed Jacobi skip, float64 reconstruction, and pressure gauge removal.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <new>
#include <vector>
struct Workspace { std::vector<float> q,a,b,c; };
extern "C" void* rheon_local_create() { return new(std::nothrow) Workspace; }
extern "C" void rheon_local_destroy(void* p) { delete static_cast<Workspace*>(p); }
static void conv(const float* __restrict input,float* __restrict output,
                 const float* __restrict w,int ci,int co,int nz,int ny,int nx) {
    const size_t n=size_t(nz)*ny*nx;
    std::fill(output,output+n*co,0.f);
    for(int o=0;o<co;++o) for(int i=0;i<ci;++i)
      for(int dz=-1;dz<=1;++dz) for(int dy=-1;dy<=1;++dy) for(int dx=-1;dx<=1;++dx) {
        const float weight=w[((o*ci+i)*3+dz+1)*9+(dy+1)*3+dx+1];
        for(int z=0;z<nz;++z) for(int y=0;y<ny;++y) {
          const int zz=std::clamp(z+dz,0,nz-1),yy=std::clamp(y+dy,0,ny-1);
          const float* __restrict src=input+i*n+(size_t(zz)*ny+yy)*nx;
          float* __restrict dst=output+o*n+(size_t(z)*ny+y)*nx;
          dst[0]+=weight*src[std::clamp(dx,0,nx-1)];
          for(int x=1;x<nx-1;++x) dst[x]+=weight*src[x+dx];
          if(nx>1) dst[nx-1]+=weight*src[std::clamp(nx-1+dx,0,nx-1)];
        }
      }
}
extern "C" int rheon_local_apply(void* p,const double* r,double factor,double scale,
  const float* w1,const float* w2,const float* w3,int nz,int ny,int nx,double* out) {
    if(!p||!r||!out||!w1||!w2||!w3||nz<1||ny<1||nx<1||nz>256||ny>256||nx>256||
       !std::isfinite(factor)||factor<=0||!std::isfinite(scale)||scale<=0) return 1;
    try {
      auto& s=*static_cast<Workspace*>(p);const size_t n=size_t(nz)*ny*nx;
      s.q.resize(n);s.a.resize(4*n);s.b.resize(4*n);s.c.resize(n);
      for(size_t j=0;j<n;++j) {s.q[j]=float((r[j]*factor)/scale);if(!std::isfinite(s.q[j]))return 2;}
      conv(s.q.data(),s.a.data(),w1,1,4,nz,ny,nx);
      conv(s.a.data(),s.b.data(),w2,4,4,nz,ny,nx);
      conv(s.b.data(),s.c.data(),w3,4,1,nz,ny,nx);
      const double gauge=double(s.q[0]*.125f+s.c[0])*scale;
      for(size_t j=0;j<n;++j) {out[j]=double(s.q[j]*.125f+s.c[j])*scale-gauge;if(!std::isfinite(out[j]))return 3;}
      return 0;
    } catch(...) {return 4;}
}
