// Four independent neural field queries per AVX2 vector. Arithmetic order,
// finite/support guards, donor bounds and float32 materialization are retained.
#include <immintrin.h>
#include <climits>
#include <type_traits>
#define rheon_neural_field_readout rheon_scalar_field_readout
#include "../full-ai-next-priorities-v1/fused_field_readout.cpp"
#undef rheon_neural_field_readout

#ifndef __AVX2__
#error "This separately qualified execution variant requires AVX2."
#endif

static inline __m256d choose(__m256d a,__m256d b,__m256d condition) {
    return _mm256_blendv_pd(a,b,condition);
}
static inline __m256d clamp_position(__m256d value,double maximum) {
    const auto zero=_mm256_setzero_pd(),upper=_mm256_set1_pd(maximum);
    value=choose(value,zero,_mm256_cmp_pd(value,zero,_CMP_LT_OQ));
    return choose(value,upper,_mm256_cmp_pd(value,upper,_CMP_GT_OQ));
}
template<class T> static inline __m256d load_shifts(const T* source) {
    if constexpr(std::is_same_v<T,float>)return _mm256_cvtps_pd(_mm_loadu_ps(source));
    else return _mm256_loadu_pd(source);
}
template<class T> static int valid_shifts(const T* shifts,int64_t count) {
    int64_t i=0;
    if constexpr(std::is_same_v<T,float>) {
        for(;i+8<=count;i+=8) {
            const auto a=_mm256_andnot_ps(_mm256_set1_ps(-0.0f),_mm256_loadu_ps(shifts+i));
            if(_mm256_movemask_ps(_mm256_cmp_ps(a,_mm256_set1_ps(1.0f),_CMP_LE_OQ))!=255)
                for(int j=0;j<8;++j){if(!std::isfinite(shifts[i+j]))return 3;if(std::abs(shifts[i+j])>1)return 4;}
        }
    } else {
        for(;i+4<=count;i+=4) {
            const auto a=_mm256_andnot_pd(_mm256_set1_pd(-0.0),_mm256_loadu_pd(shifts+i));
            if(_mm256_movemask_pd(_mm256_cmp_pd(a,_mm256_set1_pd(1.0),_CMP_LE_OQ))!=15)
                for(int j=0;j<4;++j){if(!std::isfinite(shifts[i+j]))return 3;if(std::abs(shifts[i+j])>1)return 4;}
        }
    }
    for(;i<count;++i){if(!std::isfinite(shifts[i]))return 3;if(std::abs(shifts[i])>1)return 4;}
    return 0;
}

template<class T> static float scalar_query(const float* fields,const T* shifts,
    int64_t field,int64_t z,int64_t y,int64_t x,int64_t nz,int64_t ny,int64_t nx) {
    const int64_t size=nz*ny*nx,index=x+nx*(y+ny*z),start=field*size;
    const double px=std::clamp(double(x)+double(shifts[(field*3)*size+index]),0.,double(nx-1));
    const double py=std::clamp(double(y)+double(shifts[(field*3+1)*size+index]),0.,double(ny-1));
    const double pz=std::clamp(double(z)+double(shifts[(field*3+2)*size+index]),0.,double(nz-1));
    const int64_t lx=static_cast<int64_t>(px),ly=static_cast<int64_t>(py),lz=static_cast<int64_t>(pz);
    const int64_t hx=std::min(lx+1,nx-1),hy=std::min(ly+1,ny-1),hz=std::min(lz+1,nz-1);
    const double fx=px-double(lx),fy=py-double(ly),fz=pz-double(lz);
    double value=0,lower=std::numeric_limits<double>::infinity(),upper=-lower;
    for(int bz=0;bz<2;++bz)for(int by=0;by<2;++by)for(int bx=0;bx<2;++bx) {
        const auto ix=bx?hx:lx,iy=by?hy:ly,iz=bz?hz:lz;
        const double sample=double(fields[start+ix+nx*(iy+ny*iz)]);
        double weight=1.;weight=weight*(bx?fx:1.-fx);weight=weight*(by?fy:1.-fy);weight=weight*(bz?fz:1.-fz);
        value=value+weight*sample;lower=std::min(lower,sample);upper=std::max(upper,sample);
    }
    return float(std::min(std::max(value,lower),upper));
}

template<class T> static int vector_queries(const float* fields,const T* shifts,float* output,
    int64_t nz,int64_t ny,int64_t nx) {
    const int64_t size=nz*ny*nx;
    int64_t at=0;
    for(;at+8<=4*size;at+=8) {
        const auto a=_mm256_andnot_ps(_mm256_set1_ps(-0.0f),_mm256_loadu_ps(fields+at));
        if(_mm256_movemask_ps(_mm256_cmp_ps(a,_mm256_set1_ps(std::numeric_limits<float>::max()),_CMP_LE_OQ))!=255)return 2;
    }
    for(;at<4*size;++at)if(!std::isfinite(fields[at]))return 2;
    const int guard=valid_shifts(shifts,12*size);if(guard)return guard;
    const auto one=_mm256_set1_pd(1.);
    const auto stride_x=_mm_set1_epi32(static_cast<int>(nx)),stride_y=_mm_set1_epi32(static_cast<int>(ny));
    const auto last_x=_mm_set1_epi32(static_cast<int>(nx-1)),last_y=_mm_set1_epi32(static_cast<int>(ny-1)),last_z=_mm_set1_epi32(static_cast<int>(nz-1));
    for(int64_t field=0;field<4;++field)for(int64_t z=0;z<nz;++z)for(int64_t y=0;y<ny;++y) {
        int64_t x=0;
        for(;x+4<=nx;x+=4) {
            const int64_t index=x+nx*(y+ny*z);
            const auto px=clamp_position(_mm256_add_pd(_mm256_setr_pd(double(x),double(x+1),double(x+2),double(x+3)),load_shifts(shifts+(field*3)*size+index)),double(nx-1));
            const auto py=clamp_position(_mm256_add_pd(_mm256_set1_pd(double(y)),load_shifts(shifts+(field*3+1)*size+index)),double(ny-1));
            const auto pz=clamp_position(_mm256_add_pd(_mm256_set1_pd(double(z)),load_shifts(shifts+(field*3+2)*size+index)),double(nz-1));
            const auto lx=_mm256_cvttpd_epi32(px),ly=_mm256_cvttpd_epi32(py),lz=_mm256_cvttpd_epi32(pz);
            const auto hx=_mm_min_epi32(_mm_add_epi32(lx,_mm_set1_epi32(1)),last_x);
            const auto hy=_mm_min_epi32(_mm_add_epi32(ly,_mm_set1_epi32(1)),last_y);
            const auto hz=_mm_min_epi32(_mm_add_epi32(lz,_mm_set1_epi32(1)),last_z);
            const auto fx=_mm256_sub_pd(px,_mm256_cvtepi32_pd(lx));
            const auto fy=_mm256_sub_pd(py,_mm256_cvtepi32_pd(ly));
            const auto fz=_mm256_sub_pd(pz,_mm256_cvtepi32_pd(lz));
            auto value=_mm256_setzero_pd(),lower=_mm256_set1_pd(std::numeric_limits<double>::infinity()),upper=_mm256_set1_pd(-std::numeric_limits<double>::infinity());
            for(int bz=0;bz<2;++bz)for(int by=0;by<2;++by)for(int bx=0;bx<2;++bx) {
                const auto ix=bx?hx:lx,iy=by?hy:ly,iz=bz?hz:lz;
                const auto donor=_mm_add_epi32(ix,_mm_mullo_epi32(stride_x,_mm_add_epi32(iy,_mm_mullo_epi32(stride_y,iz))));
                const auto sample=_mm256_cvtps_pd(_mm_i32gather_ps(fields+field*size,donor,4));
                auto weight=one;
                weight=_mm256_mul_pd(weight,bx?fx:_mm256_sub_pd(one,fx));
                weight=_mm256_mul_pd(weight,by?fy:_mm256_sub_pd(one,fy));
                weight=_mm256_mul_pd(weight,bz?fz:_mm256_sub_pd(one,fz));
                value=_mm256_add_pd(value,_mm256_mul_pd(weight,sample));
                lower=choose(lower,sample,_mm256_cmp_pd(sample,lower,_CMP_LT_OQ));
                upper=choose(upper,sample,_mm256_cmp_pd(sample,upper,_CMP_GT_OQ));
            }
            value=choose(value,lower,_mm256_cmp_pd(value,lower,_CMP_LT_OQ));
            value=choose(value,upper,_mm256_cmp_pd(value,upper,_CMP_GT_OQ));
            _mm_storeu_ps(output+field*size+index,_mm256_cvtpd_ps(value));
        }
        for(;x<nx;++x)output[field*size+x+nx*(y+ny*z)]=scalar_query(fields,shifts,field,z,y,x,nz,ny,nx);
    }
    return 0;
}

extern "C" int rheon_neural_field_readout(const float* fields,const void* shifts,float* output,
    int64_t nz,int64_t ny,int64_t nx,int shift_kind) {
    if(!fields||!shifts||!output||nz<1||ny<1||nx<1||nz>2048||ny>2048||nx>2048||(shift_kind!=0&&shift_kind!=1))return 1;
    if(nz*ny*nx>INT_MAX)return rheon_scalar_field_readout(fields,shifts,output,nz,ny,nx,shift_kind);
    return shift_kind==0?vector_queries(fields,static_cast<const float*>(shifts),output,nz,ny,nx):
                        vector_queries(fields,static_cast<const double*>(shifts),output,nz,ny,nx);
}
