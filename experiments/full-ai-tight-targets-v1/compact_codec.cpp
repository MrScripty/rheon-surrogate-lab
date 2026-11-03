// Execute the compact sealed-domain field representation with explicit types.
// No dynamics, interpolation, pressure inverse or reference-dependent repair.
#include <cmath>
#include <cstdint>

extern "C" int rheon_compact_pack(const double* p,const float* vx,const float* vy,
    const float* vz,const float* tracer,double* pressure,float* carriers,
    int64_t nx,int64_t ny,int64_t nz,double pscale,float vscale,float tscale) {
    if(!p||!vx||!vy||!vz||!tracer||!pressure||!carriers||nx<1||ny<1||nz<1||
       !std::isfinite(pscale)||!std::isfinite(vscale)||!std::isfinite(tscale)||pscale<=0||vscale<=0||tscale<=0) return 1;
    const int64_t volume=nx*ny*nz;
    for(int64_t z=0;z<nz;++z) for(int64_t y=0;y<ny;++y)
      if(vx[(z*ny+y)*(nx+1)+nx]!=0) return 3;
    for(int64_t z=0;z<nz;++z) for(int64_t x=0;x<nx;++x)
      if(vy[(z*(ny+1)+ny)*nx+x]!=0) return 3;
    for(int64_t y=0;y<ny;++y) for(int64_t x=0;x<nx;++x)
      if(vz[(nz*ny+y)*nx+x]!=0) return 3;
    for(int64_t z=0;z<nz;++z) for(int64_t y=0;y<ny;++y) for(int64_t x=0;x<nx;++x) {
        const int64_t i=(z*ny+y)*nx+x;
        const float a=vx[(z*ny+y)*(nx+1)+x];
        const float b=vy[(z*(ny+1)+y)*nx+x];
        const float c=vz[i];
        if(!std::isfinite(p[i])||!std::isfinite(a)||!std::isfinite(b)||!std::isfinite(c)||!std::isfinite(tracer[i])) return 2;
        if((x==0&&a!=0)||(y==0&&b!=0)||(z==0&&c!=0)) return 3;
        pressure[i]=p[i]/pscale;
        carriers[i]=a/vscale;carriers[volume+i]=b/vscale;
        carriers[2*volume+i]=c/vscale;carriers[3*volume+i]=tracer[i]/tscale;
    }
    return 0;
}

extern "C" int rheon_compact_unpack(const double* pressure,const float* carriers,
    double* p,float* vx,float* vy,float* vz,float* tracer,
    int64_t nx,int64_t ny,int64_t nz,double pscale,float vscale,float tscale) {
    if(!pressure||!carriers||!p||!vx||!vy||!vz||!tracer||nx<1||ny<1||nz<1||
       !std::isfinite(pscale)||!std::isfinite(vscale)||!std::isfinite(tscale)||pscale<=0||vscale<=0||tscale<=0) return 1;
    const int64_t volume=nx*ny*nz;const double gauge=pressure[0]*pscale;
    for(int64_t z=0;z<nz;++z) for(int64_t y=0;y<ny;++y)
      vx[(z*ny+y)*(nx+1)+nx]=0;
    for(int64_t z=0;z<nz;++z) for(int64_t x=0;x<nx;++x)
      vy[(z*(ny+1)+ny)*nx+x]=0;
    for(int64_t y=0;y<ny;++y) for(int64_t x=0;x<nx;++x)
      vz[(nz*ny+y)*nx+x]=0;
    for(int64_t z=0;z<nz;++z) for(int64_t y=0;y<ny;++y) for(int64_t x=0;x<nx;++x) {
        const int64_t i=(z*ny+y)*nx+x;
        const float a=carriers[i],b=carriers[volume+i],c=carriers[2*volume+i],t=carriers[3*volume+i];
        if(!std::isfinite(pressure[i])||!std::isfinite(a)||!std::isfinite(b)||!std::isfinite(c)||!std::isfinite(t)) return 2;
        p[i]=pressure[i]*pscale-gauge;
        vx[(z*ny+y)*(nx+1)+x]=x==0?0:a*vscale;
        vy[(z*(ny+1)+y)*nx+x]=y==0?0:b*vscale;
        vz[i]=z==0?0:c*vscale;tracer[i]=t*tscale;
    }
    return 0;
}
