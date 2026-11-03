// Fused execution of learned coordinate proposals and fixed sampling layers.
// All point, gate, mixing, viscosity and flux coefficients are caller-supplied
// learned tensors. No native trajectory routine or pressure inverse appears.
#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

struct Shape { int x,y,z; int64_t size()const{return int64_t(x)*y*z;} };
static Shape shape(int k,int nx,int ny,int nz){return {nx+(k==0),ny+(k==1),nz+(k==2)};}
static inline __m256d selectv(__m256d a,__m256d b,__m256d condition){return _mm256_blendv_pd(a,b,condition);}
static inline __m256d bounded(__m256d value,double high){
    auto low=_mm256_setzero_pd(),upper=_mm256_set1_pd(high);
    value=selectv(value,low,_mm256_cmp_pd(value,low,_CMP_LT_OQ));
    return selectv(value,upper,_mm256_cmp_pd(value,upper,_CMP_GT_OQ));
}
static bool uniform(__m128i x){return _mm_movemask_epi8(_mm_cmpeq_epi32(x,_mm_shuffle_epi32(x,0)))==65535;}
static bool consecutive(__m128i x){return _mm_movemask_epi8(_mm_cmpeq_epi32(x,_mm_add_epi32(_mm_shuffle_epi32(x,0),_mm_setr_epi32(0,1,2,3))))==65535;}
template<class T> static __m256d gather4(const T* field,__m128i index,bool contiguous){
    const int start=_mm_cvtsi128_si32(index);
    if constexpr(std::is_same_v<T,float>)return _mm256_cvtps_pd(contiguous?_mm_loadu_ps(field+start):_mm_i32gather_ps(field,index,4));
    else return contiguous?_mm256_loadu_pd(field+start):_mm256_i32gather_pd(field,index,8);
}
template<class T> static __m256d at4(const T* field,Shape s,__m256d x,__m256d y,__m256d z){
    const auto ix=_mm256_cvttpd_epi32(bounded(x,s.x-1)),iy=_mm256_cvttpd_epi32(bounded(y,s.y-1)),iz=_mm256_cvttpd_epi32(bounded(z,s.z-1));
    const auto index=_mm_add_epi32(ix,_mm_mullo_epi32(_mm_set1_epi32(s.x),_mm_add_epi32(iy,_mm_mullo_epi32(_mm_set1_epi32(s.y),iz))));
    return gather4(field,index,consecutive(ix)&&uniform(iy)&&uniform(iz));
}
template<bool clip> static __attribute__((noinline)) __m256d sample4(const float* field,Shape s,__m256d x,__m256d y,__m256d z){
    x=bounded(x,s.x-1);y=bounded(y,s.y-1);z=bounded(z,s.z-1);
    const auto lx=_mm256_cvttpd_epi32(x),ly=_mm256_cvttpd_epi32(y),lz=_mm256_cvttpd_epi32(z);
    const auto hx=_mm_min_epi32(_mm_add_epi32(lx,_mm_set1_epi32(1)),_mm_set1_epi32(s.x-1));
    const auto hy=_mm_min_epi32(_mm_add_epi32(ly,_mm_set1_epi32(1)),_mm_set1_epi32(s.y-1));
    const auto hz=_mm_min_epi32(_mm_add_epi32(lz,_mm_set1_epi32(1)),_mm_set1_epi32(s.z-1));
    const auto fx=_mm256_sub_pd(x,_mm256_cvtepi32_pd(lx)),fy=_mm256_sub_pd(y,_mm256_cvtepi32_pd(ly)),fz=_mm256_sub_pd(z,_mm256_cvtepi32_pd(lz));
    const bool regular=consecutive(lx)&&consecutive(hx)&&uniform(ly)&&uniform(hy)&&uniform(lz)&&uniform(hz);
    const auto one=_mm256_set1_pd(1.);auto value=_mm256_setzero_pd();
    auto lower=_mm256_set1_pd(std::numeric_limits<double>::infinity()),upper=_mm256_set1_pd(-std::numeric_limits<double>::infinity());
    // Separate proven-consecutive addresses from the irregular gather path.
    // Donor order, weight products, sum order and clipping are unchanged.
    auto accumulate=[&](__m256d v,int bx,int by,int bz){
        auto w=one;
        w=_mm256_mul_pd(w,bx?fx:_mm256_sub_pd(one,fx));
        w=_mm256_mul_pd(w,by?fy:_mm256_sub_pd(one,fy));
        w=_mm256_mul_pd(w,bz?fz:_mm256_sub_pd(one,fz));
        value=_mm256_add_pd(value,_mm256_mul_pd(w,v));
        if constexpr(clip){lower=selectv(lower,v,_mm256_cmp_pd(v,lower,_CMP_LT_OQ));upper=selectv(upper,v,_mm256_cmp_pd(v,upper,_CMP_GT_OQ));}
    };
    if(regular){
        const int x0=_mm_cvtsi128_si32(lx),y0=_mm_cvtsi128_si32(ly),z0=_mm_cvtsi128_si32(lz);
        const int64_t y_stride=int64_t(s.x)*(_mm_cvtsi128_si32(hy)-y0);
        const int64_t z_stride=int64_t(s.x)*s.y*(_mm_cvtsi128_si32(hz)-z0);
        const float* origin=field+x0+int64_t(s.x)*(y0+int64_t(s.y)*z0);
        for(int bz=0;bz<2;++bz)for(int by=0;by<2;++by){
            const float* row=origin+bz*z_stride+by*y_stride;
            accumulate(_mm256_cvtps_pd(_mm_loadu_ps(row)),0,by,bz);
            accumulate(_mm256_cvtps_pd(_mm_loadu_ps(row+1)),1,by,bz);
        }
    }else{
        for(int bz=0;bz<2;++bz)for(int by=0;by<2;++by)for(int bx=0;bx<2;++bx){
            const auto ix=bx?hx:lx,iy=by?hy:ly,iz=bz?hz:lz;
            const auto index=_mm_add_epi32(ix,_mm_mullo_epi32(_mm_set1_epi32(s.x),_mm_add_epi32(iy,_mm_mullo_epi32(_mm_set1_epi32(s.y),iz))));
            accumulate(gather4(field,index,false),bx,by,bz);
        }
    }
    if constexpr(clip){value=selectv(value,lower,_mm256_cmp_pd(value,lower,_CMP_LT_OQ));value=selectv(value,upper,_mm256_cmp_pd(value,upper,_CMP_GT_OQ));}
    return value;
}
template<class T> static double at(const T* a,Shape s,int x,int y,int z){
    x=std::clamp(x,0,s.x-1);y=std::clamp(y,0,s.y-1);z=std::clamp(z,0,s.z-1);return double(a[x+s.x*(int64_t(y)+int64_t(s.y)*z)]);
}
static double sample(const float* a,Shape s,double x,double y,double z){
    x=std::clamp(x,0.,double(s.x-1));y=std::clamp(y,0.,double(s.y-1));z=std::clamp(z,0.,double(s.z-1));
    const int lx=int(x),ly=int(y),lz=int(z),hx=std::min(lx+1,s.x-1),hy=std::min(ly+1,s.y-1),hz=std::min(lz+1,s.z-1);
    const double fx=x-lx,fy=y-ly,fz=z-lz;double value=0,lo=INFINITY,hi=-INFINITY;
    for(int bz=0;bz<2;++bz)for(int by=0;by<2;++by)for(int bx=0;bx<2;++bx){
        const double v=at(a,s,bx?hx:lx,by?hy:ly,bz?hz:lz);double w=1;
        w=w*(bx?fx:1-fx);w=w*(by?fy:1-fy);w=w*(bz?fz:1-fz);
        value=value+w*v;lo=std::min(lo,v);hi=std::max(hi,v);
    }
    return std::min(std::max(value,lo),hi);
}
// Static integer offsets do not require floating-point coordinate clamps.
// Preserve coefficient order and all floating-point multiply/add operations.
template<class T> static __m256d linear4(const T*const* fields,const Shape* shapes,int head,
    int x,int y,int z,const int* rows,const int* channels,const int* offsets,const double* weights){
    auto total=_mm256_setzero_pd();
    for(int i=rows[head];i<rows[head+1];++i){
        const int c=channels[i];const Shape s=shapes[c];
        const int yy=std::clamp(y+offsets[3*i+1],0,s.y-1),zz=std::clamp(z+offsets[3*i+2],0,s.z-1);
        const int xx=x+offsets[3*i];const T* row=fields[c]+int64_t(s.x)*(yy+int64_t(s.y)*zz);
        __m256d value;
        if(xx>=0&&xx+3<s.x){
            if constexpr(std::is_same_v<T,float>)value=_mm256_cvtps_pd(_mm_loadu_ps(row+xx));
            else value=_mm256_loadu_pd(row+xx);
        }else{
            const auto ix=_mm_min_epi32(_mm_max_epi32(_mm_setr_epi32(xx,xx+1,xx+2,xx+3),_mm_setzero_si128()),_mm_set1_epi32(s.x-1));
            value=gather4(row,ix,false);
        }
        total=_mm256_add_pd(total,_mm256_mul_pd(_mm256_set1_pd(weights[i]),value));
    }
    return total;
}

template<class T> static double linear(const T*const* fields,const Shape* shapes,int head,
    int x,int y,int z,const int* rows,const int* channels,const int* offsets,const double* weights){
    double total=0;for(int i=rows[head];i<rows[head+1];++i)total=total+weights[i]*at(fields[channels[i]],shapes[channels[i]],x+offsets[3*i],y+offsets[3*i+1],z+offsets[3*i+2]);return total;
}

extern "C" int rheon_learned_sampling(int nx,int ny,int nz,const float* vx,const float* vy,const float* vz,const float* tracer,
    float* ox,float* oy,float* oz,float* ot,const int* rows,const int* channels,const int* offsets,const double* weights,
    const double* gate,const double* mix,double dt_over_h){
    if(nx<1||ny<1||nz<1||nx>1024||ny>1024||nz>1024||!vx||!vy||!vz||!tracer||!ox||!oy||!oz||!ot||!rows||!channels||!offsets||!weights||!gate||!mix||!std::isfinite(dt_over_h)||dt_over_h<=0)return 1;
    const float* fields[4]={vx,vy,vz,tracer};float* outputs[4]={ox,oy,oz,ot};Shape dims[4];
    for(int k=0;k<4;++k){dims[k]=shape(k,nx,ny,nz);for(int64_t i=0;i<dims[k].size();++i)if(!std::isfinite(fields[k][i])||std::abs(fields[k][i])>1e6)return 2;}
    for(int i=0;i<12;++i)if(!std::isfinite(gate[i])||std::abs(gate[i])>4)return 3;
    for(int i=0;i<36;++i)if(!std::isfinite(mix[i])||std::abs(mix[i])>4)return 3;
    const auto alpha=_mm256_set1_pd(dt_over_h);
    for(int k=0;k<4;++k){const auto s=dims[k];
        for(int z=0;z<s.z;++z)for(int y=0;y<s.y;++y){int x=0;
            for(;x+4<=s.x;x+=4){
                __m256d base[3]={_mm256_setr_pd(x,x+1,x+2,x+3),_mm256_set1_pd(y),_mm256_set1_pd(z)},query[3],mid[3];
                for(int a=0;a<3;++a){
                    const auto p=linear4(fields,dims,k*3+a,x,y,z,rows,channels,offsets,weights);
                    query[a]=_mm256_add_pd(_mm256_add_pd(base[a],_mm256_set1_pd(k==a?0.:.5)),_mm256_mul_pd(_mm256_mul_pd(alpha,p),_mm256_set1_pd(gate[k*3+a])));
                }
                for(int c=0;c<3;++c)mid[c]=sample4<true>(fields[c],dims[c],_mm256_sub_pd(query[0],_mm256_set1_pd(c==0?0.:.5)),_mm256_sub_pd(query[1],_mm256_set1_pd(c==1?0.:.5)),_mm256_sub_pd(query[2],_mm256_set1_pd(c==2?0.:.5)));
                __m256d final[3];
                for(int a=0;a<3;++a){auto v=_mm256_setzero_pd();for(int c=0;c<3;++c)v=_mm256_add_pd(v,_mm256_mul_pd(_mm256_set1_pd(mix[(k*3+a)*3+c]),mid[c]));
                    auto delta=_mm256_mul_pd(alpha,v);delta=selectv(delta,_mm256_set1_pd(-1),_mm256_cmp_pd(delta,_mm256_set1_pd(-1),_CMP_LT_OQ));delta=selectv(delta,_mm256_set1_pd(1),_mm256_cmp_pd(delta,_mm256_set1_pd(1),_CMP_GT_OQ));final[a]=_mm256_add_pd(base[a],delta);}
                auto value=_mm256_cvtpd_ps(sample4<false>(fields[k],s,final[0],final[1],final[2]));
                if(k<3){const auto normal=base[k],maximum=_mm256_set1_pd(k==0?nx:k==1?ny:nz);const auto wall=_mm256_or_pd(_mm256_cmp_pd(normal,_mm256_setzero_pd(),_CMP_EQ_OQ),_mm256_cmp_pd(normal,maximum,_CMP_EQ_OQ));
                    value=_mm_blendv_ps(value,_mm_setzero_ps(),_mm_castsi128_ps(_mm256_cvttpd_epi32(selectv(_mm256_setzero_pd(),_mm256_set1_pd(-1),wall))));}
                _mm_storeu_ps(outputs[k]+x+int64_t(s.x)*(y+int64_t(s.y)*z),value);
            }
            for(;x<s.x;++x){double base[3]={double(x),double(y),double(z)},query[3],mid[3],final[3];
                for(int a=0;a<3;++a)query[a]=(base[a]+(k==a?0.:.5))+(dt_over_h*linear(fields,dims,k*3+a,x,y,z,rows,channels,offsets,weights))*gate[k*3+a];
                for(int c=0;c<3;++c)mid[c]=sample(fields[c],dims[c],query[0]-(c==0?0.:.5),query[1]-(c==1?0.:.5),query[2]-(c==2?0.:.5));
                for(int a=0;a<3;++a){double v=0;for(int c=0;c<3;++c)v=v+mix[(k*3+a)*3+c]*mid[c];final[a]=base[a]+std::clamp(dt_over_h*v,-1.,1.);}
                const bool wall=k<3&&(base[k]==0||base[k]==(k==0?nx:k==1?ny:nz));
                outputs[k][x+int64_t(s.x)*(y+int64_t(s.y)*z)]=wall?0.f:float(sample(fields[k],s,final[0],final[1],final[2]));
            }
        }
    }
    return 0;
}

// Generic sparse learned stages: 0 viscosity, 1 flux, 2 momentum. Coefficients
// retain source-channel/offset provenance; no analytical stencil is assigned.
template<class T> static int spatial_stage(int kind,int nx,int ny,int nz,const T*const* input,float*const* output,
    double* q,const float*const* base,const int* rows,const int* channels,const int* offsets,const double* weights,double scale){
    Shape sources[3];for(int c=0;c<3;++c)sources[c]=kind==2?Shape{nx,ny,nz}:shape(c,nx,ny,nz);
    const int heads=kind==1?1:3;
    for(int d=0;d<heads;++d){const Shape s=kind==1?Shape{nx,ny,nz}:shape(d,nx,ny,nz);
        for(int z=0;z<s.z;++z)for(int y=0;y<s.y;++y){int x=0;
            for(;x+4<=s.x;x+=4){const int64_t at=x+int64_t(s.x)*(y+int64_t(s.y)*z);
                auto value=_mm256_mul_pd(_mm256_set1_pd(scale),linear4(input,sources,d,x,y,z,rows,channels,offsets,weights));
                if(kind==1)_mm256_storeu_pd(q+at,value);
                else{value=_mm256_add_pd(_mm256_cvtps_pd(_mm_loadu_ps(base[d]+at)),value);auto stored=_mm256_cvtpd_ps(value);
                    const auto normal=d==0?_mm256_setr_pd(x,x+1,x+2,x+3):_mm256_set1_pd(d==1?y:z);
                    const auto wall=_mm256_or_pd(_mm256_cmp_pd(normal,_mm256_setzero_pd(),_CMP_EQ_OQ),_mm256_cmp_pd(normal,_mm256_set1_pd(d==0?nx:d==1?ny:nz),_CMP_EQ_OQ));
                    stored=_mm_blendv_ps(stored,_mm_setzero_ps(),_mm_castsi128_ps(_mm256_cvttpd_epi32(selectv(_mm256_setzero_pd(),_mm256_set1_pd(-1),wall))));
                    _mm_storeu_ps(output[d]+at,stored);}
            }
            for(;x<s.x;++x){const int64_t at=x+int64_t(s.x)*(y+int64_t(s.y)*z);const double value=scale*linear(input,sources,d,x,y,z,rows,channels,offsets,weights);
                if(kind==1)q[at]=value;else{const int normal=d==0?x:d==1?y:z;const bool wall=normal==0||normal==(d==0?nx:d==1?ny:nz);output[d][at]=wall?0.f:float(double(base[d][at])+value);}}
        }
    }
    return 0;
}
extern "C" int rheon_learned_spatial_stage(int kind,int nx,int ny,int nz,const void* a,const void* b,const void* c,
    float* ox,float* oy,float* oz,double* q,const float* bx,const float* by,const float* bz,
    const int* rows,const int* channels,const int* offsets,const double* weights,double scale){
    if(kind<0||kind>2||nx<1||ny<1||nz<1||nx>1024||ny>1024||nz>1024||!a||!rows||!channels||!offsets||!weights||!std::isfinite(scale))return 1;
    float* output[3]={ox,oy,oz};const float* base[3]={bx,by,bz};
    if(kind!=1&&(!ox||!oy||!oz||!bx||!by||!bz))return 1;
    if(kind==1&&!q)return 1;
    if(kind==2){const double* input[3]={static_cast<const double*>(a),nullptr,nullptr};return spatial_stage(kind,nx,ny,nz,input,output,q,base,rows,channels,offsets,weights,scale);}
    if(!b||!c)return 1;const float* input[3]={static_cast<const float*>(a),static_cast<const float*>(b),static_cast<const float*>(c)};
    return spatial_stage(kind,nx,ny,nz,input,output,q,base,rows,channels,offsets,weights,scale);
}

// External prescribed conditioning, evaluated once for the entire block.
extern "C" int rheon_conditioning_force(int nx,int ny,int nz,int steps,int stride,
    const double* offsets,const double* waves,const double* amplitude,const double* direction,
    double swirl,double fluctuation,double dt,double* output){
    if(nx<1||ny<1||nz<1||steps<1||steps>4||stride<=std::max(nx,std::max(ny,nz))||!offsets||!waves||!amplitude||!direction||!output)return 1;
    const int64_t lengths[3]={shape(0,nx,ny,nz).size(),shape(1,nx,ny,nz).size(),shape(2,nx,ny,nz).size()};
    const int64_t block=lengths[0]+lengths[1]+lengths[2];std::vector<double> gaussian(9*steps*stride);
    for(int64_t i=0;i<9*steps*stride;++i)gaussian[i]=std::exp(-.5*offsets[i]*offsets[i]);
    for(int step=0;step<steps;++step){int64_t base=step*block;
        for(int d=0;d<3;++d){const auto s=shape(d,nx,ny,nz);
            for(int z=0;z<s.z;++z)for(int y=0;y<s.y;++y)for(int x=0;x<s.x;++x){
                const int xyz[3]={x,y,z};const int normal=xyz[d];const int64_t at=base+x+int64_t(s.x)*(y+int64_t(s.y)*z);
                if(normal==0||normal==(d==0?nx:d==1?ny:nz)){output[at]=0;continue;}
                double off[3],r2=0,wave=0,g=1;
                for(int a=0;a<3;++a){const int64_t index=((d*3+a)*steps+step)*stride+xyz[a];off[a]=offsets[index];r2=r2+off[a]*off[a];wave=wave+waves[index];g=g*gaussian[index];}
                const double taper=1.-r2/16.,envelope=r2<16?g*(taper*taper):0.;const int a=(d+1)%3,b=(d+2)%3;
                const double rotation=direction[a]*off[b]-direction[b]*off[a];
                const double value=((amplitude[step]*envelope)*((direction[d]+swirl*rotation)+fluctuation*(wave/3.)))*dt;
                if(!std::isfinite(value))return 2;output[at]=value;
            }
            base+=lengths[d];
        }
    }
    return 0;
}
