// Known external forcing only: no physical state evolution or pressure solve.
#include <algorithm>
#include <cmath>
#include <cstdint>

static double next_value(uint64_t& state) {
    state += UINT64_C(0x9e3779b97f4a7c15);
    uint64_t value = state;
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return double((value ^ (value >> 31)) >> 11) / double(UINT64_C(1) << 53);
}

extern "C" int rheon_scheduled_force_substeps(int nx,int ny,int nz,int density,int step,int substeps,uint64_t seed,
    const double* center_normalized,double pulse,double magnitude,double radius_fraction,
    double swirl,double fluctuation,double frequency,
    const float* bx,const float* by,const float* bz,
    double* fx,double* fy,double* fz,float* ox,float* oy,float* oz) {
    if(nx<5||ny<5||nz<5||nx>256||ny>256||nz>256||density<1||density>1000||step<0||step>=3600||
       (substeps!=1&&substeps!=2&&substeps!=4&&substeps!=8)||
       !center_normalized||!bx||!by||!bz||!fx||!fy||!fz||!ox||!oy||!oz)return 1;
    const int counts[3]={nx,ny,nz};
    const double tau=6.283185307179586476925286766559;
    const double h=1./density,dt=1./(60.*substeps),time=double(step)*dt;
    double direction[3],norm=0.,extents[3],center[3];
    for(int a=0;a<3;++a){direction[a]=2.*next_value(seed)-1.;norm=norm+direction[a]*direction[a];}
    norm=std::max(std::sqrt(norm),1e-12);
    for(int a=0;a<3;++a){direction[a]=direction[a]/norm;extents[a]=double(counts[a])*h;center[a]=extents[a]*center_normalized[a];}
    const double phase=next_value(seed)*tau,omega=tau*frequency;
    const double radius=radius_fraction*std::min(extents[0],std::min(extents[1],extents[2]));
    const float* bases[3]={bx,by,bz};double* forces[3]={fx,fy,fz};float* outputs[3]={ox,oy,oz};
    for(int d=0;d<3;++d){
        const int sx=nx+(d==0),sy=ny+(d==1),sz=nz+(d==2);
        for(int z=0;z<sz;++z)for(int y=0;y<sy;++y)for(int x=0;x<sx;++x){
            const int p[3]={x,y,z};const int64_t at=x+int64_t(sx)*(y+int64_t(sy)*z);
            if(p[d]==0||p[d]==counts[d]){forces[d][at]=0.;outputs[d][at]=bases[d][at];continue;}
            double pos[3],offset[3],r2=0.;
            for(int a=0;a<3;++a){pos[a]=0.+(double(p[a])+(a==d?0.:.5))*h;offset[a]=(pos[a]-center[a])/radius;r2=r2+offset[a]*offset[a];}
            const double cutoff=1.-r2/16.;
            const double envelope=r2<16.?std::exp(-.5*r2)*(cutoff*cutoff):0.;
            const int a=(d+1)%3,b=(d+2)%3;
            const double rotation=direction[a]*offset[b]-direction[b]*offset[a];
            double wave=0.;
            for(int k=0;k<3;++k)wave=wave+std::sin(tau*pos[k]/extents[k]+omega*time*double(k+1)+phase+double(d));
            wave=wave/3.;
            const double acceleration=magnitude*pulse*envelope*(direction[d]+swirl*rotation+fluctuation*wave);
            forces[d][at]=dt*acceleration;
            outputs[d][at]=float(double(bases[d][at])+forces[d][at]);
        }
    }
    return 0;
}
