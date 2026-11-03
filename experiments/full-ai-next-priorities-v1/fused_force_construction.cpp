// Execution of the existing prescribed force formula; no learned dynamics here.
#include <cmath>
#include <cstdint>

extern "C" int rheon_prescribed_force(const double* offsets, const double* waves,
    const double* amplitude, const double* direction, float* result,
    int64_t nx, int64_t ny, int64_t nz, int64_t k, int64_t stride,
    double swirl, double fluctuation) {
    if (!offsets || !waves || !amplitude || !direction || !result ||
        nx<2 || ny<2 || nz<2 || k<1 || k>4 || stride<nx || stride<ny || stride<nz) return 1;
    const int64_t volume=nx*ny*nz;
    for(int64_t s=0;s<k;++s) for(int64_t d=0;d<3;++d)
      for(int64_t z=0;z<nz;++z) for(int64_t y=0;y<ny;++y) for(int64_t x=0;x<nx;++x) {
        const int64_t coordinates[3]={x,y,z};
        const int64_t counts[3]={nx,ny,nz};
        const int64_t index=(s*3+d)*volume+(z*ny+y)*nx+x;
        if(coordinates[d]==0 || coordinates[d]==counts[d]-1){result[index]=0;continue;}
        double offset[3];double r2=0;double wave=0;
        for(int64_t a=0;a<3;++a){
            const int64_t at=((d*3+a)*k+s)*stride+coordinates[a];
            offset[a]=offsets[at];r2=r2+offset[a]*offset[a];wave=wave+waves[at];
        }
        wave=wave*(1.0/3.0);
        const double taper=1.0-r2*(1.0/16.0);
        const double envelope=r2<16.0 ? std::exp(-0.5*r2)*(taper*taper):0.0;
        const int64_t a=(d+1)%3,b=(d+2)%3;
        const double rotation=direction[a]*offset[b]-direction[b]*offset[a];
        const double angular=(direction[d]+swirl*rotation)+fluctuation*wave;
        const double value=((amplitude[s]*envelope)*angular)*(1.0/60.0);
        if(!std::isfinite(value)) return 2;
        result[index]=static_cast<float>(value);
      }
    return 0;
}
