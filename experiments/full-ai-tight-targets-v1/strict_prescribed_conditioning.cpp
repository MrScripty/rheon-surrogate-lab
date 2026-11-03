// Fixed prescribed external forcing only. No learned fluid or pressure solve.
#include <algorithm>
#include <cmath>
#include <cstdint>
struct Shape {int x,y,z;int64_t size()const{return int64_t(x)*y*z;}};
static Shape shape(int k,int nx,int ny,int nz){return {nx+(k==0),ny+(k==1),nz+(k==2)};}
extern "C" int rheon_conditioning_force(int nx,int ny,int nz,int steps,int stride,
    const double* offsets,const double* waves,const double* amplitude,const double* direction,
    double swirl,double fluctuation,double dt,double* output){
    if(nx<1||ny<1||nz<1||steps<1||steps>4||stride<=std::max(nx,std::max(ny,nz))||!offsets||!waves||!amplitude||!direction||!output)return 1;
    const int64_t lengths[3]={shape(0,nx,ny,nz).size(),shape(1,nx,ny,nz).size(),shape(2,nx,ny,nz).size()};
    const int64_t block=lengths[0]+lengths[1]+lengths[2];
    for(int step=0;step<steps;++step){int64_t base=step*block;
        for(int d=0;d<3;++d){const auto s=shape(d,nx,ny,nz);
            for(int z=0;z<s.z;++z)for(int y=0;y<s.y;++y)for(int x=0;x<s.x;++x){
                const int xyz[3]={x,y,z};const int normal=xyz[d];const int64_t at=base+x+int64_t(s.x)*(y+int64_t(s.y)*z);
                if(normal==0||normal==(d==0?nx:d==1?ny:nz)){output[at]=0;continue;}
                double off[3],r2=0,wave=0;
                for(int a=0;a<3;++a){const int64_t index=((d*3+a)*steps+step)*stride+xyz[a];off[a]=offsets[index];r2=r2+off[a]*off[a];wave=wave+waves[index];}
                const double taper=1.-r2/16.,envelope=r2<16?std::exp(-.5*r2)*(taper*taper):0.;const int a=(d+1)%3,b=(d+2)%3;
                const double rotation=direction[a]*off[b]-direction[b]*off[a];
                const double value=((amplitude[step]*envelope)*((direction[d]+swirl*rotation)+fluctuation*(wave/3.)))*dt;
                if(!std::isfinite(value))return 2;output[at]=value;
            }
            base+=lengths[d];
        }
    }
    return 0;
}
