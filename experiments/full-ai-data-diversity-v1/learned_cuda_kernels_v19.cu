// Generated frozen learned-checkpoint specialization. Inherited dynamic-parameter comments apply to reference kernels.
// Version5: generic interpolation in physical units. All proposal/gather
// coefficients are restored learned inputs; no physical solver is present.
// Generic execution of caller-supplied learned coefficients. No physical
// stencil, pressure inverse, numerical solver, or fitted constants appear.
struct Shape { int x,y,z; };
__device__ Shape dims(int k,int nx,int ny,int nz) { return {nx+(k==0),ny+(k==1),nz+(k==2)}; }
__device__ long long volume(Shape s) { return (long long)s.x*s.y*s.z; }
__device__ int bounded(int x,int n) { return x<0?0:(x>=n?n-1:x); }
__device__ double bounded_position(double x,int n) { return x<0.?0.:(x>double(n-1)?double(n-1):x); }
__device__ double read_float(const float* a,Shape s,int x,int y,int z) {
    x=bounded(x,s.x);y=bounded(y,s.y);z=bounded(z,s.z);
    return double(a[x+(long long)s.x*(y+(long long)s.y*z)]);
}
__device__ double sample_reference(const float* a,Shape s,double x,double y,double z,bool clip) {
    // Propagate NaN coordinates without issuing an out-of-range memory access.
    if(x!=x || y!=y || z!=z) { union {unsigned long long u;double d;} v;v.u=0x7ff8000000000000ULL;return v.d; }
    x=bounded_position(x,s.x);y=bounded_position(y,s.y);z=bounded_position(z,s.z);
    int lx=int(x),ly=int(y),lz=int(z),hx=bounded(lx+1,s.x),hy=bounded(ly+1,s.y),hz=bounded(lz+1,s.z);
    double fx=x-lx,fy=y-ly,fz=z-lz,value=0.,lower=1.7976931348623157e308,upper=-1.7976931348623157e308;
    for(int bz=0;bz<2;++bz)for(int by=0;by<2;++by)for(int bx=0;bx<2;++bx) {
        double donor=read_float(a,s,bx?hx:lx,by?hy:ly,bz?hz:lz),weight=1.;
        weight=weight*(bx?fx:1.-fx);weight=weight*(by?fy:1.-fy);weight=weight*(bz?fz:1.-fz);
        value=value+weight*donor;
        if(clip) {lower=donor<lower?donor:lower;upper=donor>upper?donor:upper;}
    }
    if(clip) {value=value<lower?lower:value;value=value>upper?upper:value;}
    return value;
}

// A single changed interpolation candidate: seven endpoint-aware F64 lerps.
// All donor values/coordinates are caller inputs; this has no physical stencil.
__device__ __forceinline__ double nested_lerp(double a,double b,double f) {
 if(f==0.)return a;
 if(f==1.)return b;
 return __fma_rn(f,__dsub_rn(b,a),a);
}
__device__ double donor_reference(const float* d,double fx,double fy,double fz,bool clip) {
 double value=0.,lower=1.7976931348623157e308,upper=-1.7976931348623157e308;
 for(int bz=0;bz<2;++bz)for(int by=0;by<2;++by)for(int bx=0;bx<2;++bx) {
  double donor=double(d[4*bz+2*by+bx]),weight=1.;
  weight=weight*(bx?fx:1.-fx);weight=weight*(by?fy:1.-fy);weight=weight*(bz?fz:1.-fz);
  value=value+weight*donor;
  if(clip){lower=donor<lower?donor:lower;upper=donor>upper?donor:upper;}
 }
 if(clip){value=value<lower?lower:value;value=value>upper?upper:value;}return value;
}
__device__ double nested_donors(const float* d,double fx,double fy,double fz,bool clip,int& fallback) {
 fallback=0;
 bool valid=fx>=0.&&fx<=1.&&fy>=0.&&fy<=1.&&fz>=0.&&fz<=1.;
 #pragma unroll
 for(int i=0;i<8;++i)valid=valid&&((__float_as_uint(d[i])&0x7f800000U)!=0x7f800000U);
 if(!valid){fallback=1;return donor_reference(d,fx,fy,fz,clip);}
 double x0=nested_lerp(double(d[0]),double(d[1]),fx),x1=nested_lerp(double(d[2]),double(d[3]),fx);
 double x2=nested_lerp(double(d[4]),double(d[5]),fx),x3=nested_lerp(double(d[6]),double(d[7]),fx);
 double value=nested_lerp(nested_lerp(x0,x1,fy),nested_lerp(x2,x3,fy),fz);
 if(clip){double lower=double(d[0]),upper=lower;
  #pragma unroll
  for(int i=1;i<8;++i){double v=double(d[i]);lower=v<lower?v:lower;upper=v>upper?v:upper;}
  value=value<lower?lower:value;value=value>upper?upper:value;
 }
 return value;
}
__device__ double sample(const float* a,Shape s,double x,double y,double z,bool clip) {
 if(x!=x||y!=y||z!=z){union{unsigned long long u;double d;}v;v.u=0x7ff8000000000000ULL;return v.d;}
 x=bounded_position(x,s.x);y=bounded_position(y,s.y);z=bounded_position(z,s.z);
 int lx=int(x),ly=int(y),lz=int(z),hx=bounded(lx+1,s.x),hy=bounded(ly+1,s.y),hz=bounded(lz+1,s.z);
 float d[8];
 #pragma unroll
 for(int bz=0;bz<2;++bz) {
  #pragma unroll
  for(int by=0;by<2;++by) {
   #pragma unroll
   for(int bx=0;bx<2;++bx)d[4*bz+2*by+bx]=a[(bx?hx:lx)+(long long)s.x*((by?hy:ly)+(long long)s.y*(bz?hz:lz))];
  }
 }
 int fallback;return nested_donors(d,x-lx,y-ly,z-lz,clip,fallback);
}
extern "C" __global__ void nested_scalar_primitive(long long count,const float* donors,const double* fractions,const int* clips,
 double* output,double* reference,int* fallback) {
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 const float* d=donors+8*i;const double* f=fractions+3*i;int fb;
 output[i]=nested_donors(d,f[0],f[1],f[2],clips[i]!=0,fb);
 reference[i]=donor_reference(d,f[0],f[1],f[2],clips[i]!=0);fallback[i]=fb;
}
extern "C" __global__ void nested_grid_primitive(long long count,const float* field,int nx,int ny,int nz,
 const double* coordinates,const int* clips,double* output,double* reference) {
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 Shape s={nx,ny,nz};const double* q=coordinates+3*i;bool clip=clips[i]!=0;
 output[i]=sample(field,s,q[0],q[1],q[2],clip);reference[i]=sample_reference(field,s,q[0],q[1],q[2],clip);
}

// Serialized managed learned parameters, manifest SHA256 6642fc1817f1b1cbfd8feca8c9c50245634a91fb3cdf33b0558172a8be8ffb5e
__device__ __forceinline__ double bound_point_weight(int i) {
 switch(i) {
 case 0:return -0x1.0000000000000p+0;
 case 1:return -0x1.0000000000000p-2;
 case 2:return -0x1.0000000000000p-2;
 case 3:return -0x1.0000000000000p-2;
 case 4:return -0x1.0000000000000p-2;
 case 5:return -0x1.0000000000000p-2;
 case 6:return -0x1.0000000000000p-2;
 case 7:return -0x1.0000000000000p-2;
 case 8:return -0x1.0000000000000p-2;
 case 9:return -0x1.0000000000000p-2;
 case 10:return -0x1.0000000000000p-2;
 case 11:return -0x1.0000000000000p-2;
 case 12:return -0x1.0000000000000p-2;
 case 13:return -0x1.0000000000000p+0;
 case 14:return -0x1.0000000000000p-2;
 case 15:return -0x1.0000000000000p-2;
 case 16:return -0x1.0000000000000p-2;
 case 17:return -0x1.0000000000000p-2;
 case 18:return -0x1.0000000000000p-2;
 case 19:return -0x1.0000000000000p-2;
 case 20:return -0x1.0000000000000p-2;
 case 21:return -0x1.0000000000000p-2;
 case 22:return -0x1.0000000000000p-2;
 case 23:return -0x1.0000000000000p-2;
 case 24:return -0x1.0000000000000p-2;
 case 25:return -0x1.0000000000000p-2;
 case 26:return -0x1.0000000000000p+0;
 case 27:return -0x1.0000000000000p-1;
 case 28:return -0x1.0000000000000p-1;
 case 29:return -0x1.0000000000000p-1;
 case 30:return -0x1.0000000000000p-1;
 case 31:return -0x1.0000000000000p-1;
 case 32:return -0x1.0000000000000p-1;
 default:return __longlong_as_double(0x7ff8000000000000ULL);
 }
}
__device__ __forceinline__ double bound_gate(int i) {
 switch(i) {
 case 0:return 0x1.0000000002000p-1;
 case 1:return 0x1.0000000004000p-1;
 case 2:return 0x1.0000000004000p-1;
 case 3:return 0x1.0000000002000p-1;
 case 4:return 0x1.0000000004000p-1;
 case 5:return 0x1.0000000004000p-1;
 case 6:return 0x1.0000000004000p-1;
 case 7:return 0x1.0000000006000p-1;
 case 8:return 0x1.0000000004000p-1;
 case 9:return 0x1.0000000002000p-1;
 case 10:return 0x1.0000000004000p-1;
 case 11:return 0x1.0000000002000p-1;
 default:return __longlong_as_double(0x7ff8000000000000ULL);
 }
}
__device__ __forceinline__ double bound_mix(int i) {
 switch(i) {
 case 0:return -0x1.0000000000000p+0;
 case 1:return -0x0.0p+0;
 case 2:return 0x0.0p+0;
 case 3:return -0x0.0p+0;
 case 4:return -0x1.0000000000000p+0;
 case 5:return -0x0.0p+0;
 case 6:return -0x0.0p+0;
 case 7:return -0x0.0p+0;
 case 8:return -0x1.0000000000000p+0;
 case 9:return -0x1.0000000000000p+0;
 case 10:return -0x0.0p+0;
 case 11:return 0x0.0p+0;
 case 12:return 0x0.0p+0;
 case 13:return -0x1.0000000000000p+0;
 case 14:return 0x0.0p+0;
 case 15:return 0x0.0p+0;
 case 16:return -0x0.0p+0;
 case 17:return -0x1.0000000000000p+0;
 case 18:return -0x1.0000000000000p+0;
 case 19:return -0x0.0p+0;
 case 20:return -0x0.0p+0;
 case 21:return 0x0.0p+0;
 case 22:return -0x1.0000000000000p+0;
 case 23:return 0x0.0p+0;
 case 24:return 0x0.0p+0;
 case 25:return -0x0.0p+0;
 case 26:return -0x1.0000000000000p+0;
 case 27:return -0x1.0000000000000p+0;
 case 28:return 0x0.0p+0;
 case 29:return -0x0.0p+0;
 case 30:return 0x0.0p+0;
 case 31:return -0x1.0000000000000p+0;
 case 32:return 0x0.0p+0;
 case 33:return 0x0.0p+0;
 case 34:return -0x0.0p+0;
 case 35:return -0x1.0000000000000p+0;
 default:return __longlong_as_double(0x7ff8000000000000ULL);
 }
}
template<int HEAD>
__device__ __forceinline__ double bound_point_head(const float*const* a,const Shape* shapes,int x,int y,int z) {
 double total=0.;
 if constexpr(HEAD==0) {
  total=total+bound_point_weight(0)*read_float(a[0],shapes[0],x+(0),y+(0),z+(0));
 }
 else if constexpr(HEAD==1) {
  total=total+bound_point_weight(1)*read_float(a[1],shapes[1],x+(-1),y+(0),z+(0));
  total=total+bound_point_weight(2)*read_float(a[1],shapes[1],x+(0),y+(0),z+(0));
  total=total+bound_point_weight(3)*read_float(a[1],shapes[1],x+(-1),y+(1),z+(0));
  total=total+bound_point_weight(4)*read_float(a[1],shapes[1],x+(0),y+(1),z+(0));
 }
 else if constexpr(HEAD==2) {
  total=total+bound_point_weight(5)*read_float(a[2],shapes[2],x+(-1),y+(0),z+(0));
  total=total+bound_point_weight(6)*read_float(a[2],shapes[2],x+(0),y+(0),z+(0));
  total=total+bound_point_weight(7)*read_float(a[2],shapes[2],x+(-1),y+(0),z+(1));
  total=total+bound_point_weight(8)*read_float(a[2],shapes[2],x+(0),y+(0),z+(1));
 }
 else if constexpr(HEAD==3) {
  total=total+bound_point_weight(9)*read_float(a[0],shapes[0],x+(0),y+(-1),z+(0));
  total=total+bound_point_weight(10)*read_float(a[0],shapes[0],x+(1),y+(-1),z+(0));
  total=total+bound_point_weight(11)*read_float(a[0],shapes[0],x+(0),y+(0),z+(0));
  total=total+bound_point_weight(12)*read_float(a[0],shapes[0],x+(1),y+(0),z+(0));
 }
 else if constexpr(HEAD==4) {
  total=total+bound_point_weight(13)*read_float(a[1],shapes[1],x+(0),y+(0),z+(0));
 }
 else if constexpr(HEAD==5) {
  total=total+bound_point_weight(14)*read_float(a[2],shapes[2],x+(0),y+(-1),z+(0));
  total=total+bound_point_weight(15)*read_float(a[2],shapes[2],x+(0),y+(0),z+(0));
  total=total+bound_point_weight(16)*read_float(a[2],shapes[2],x+(0),y+(-1),z+(1));
  total=total+bound_point_weight(17)*read_float(a[2],shapes[2],x+(0),y+(0),z+(1));
 }
 else if constexpr(HEAD==6) {
  total=total+bound_point_weight(18)*read_float(a[0],shapes[0],x+(0),y+(0),z+(-1));
  total=total+bound_point_weight(19)*read_float(a[0],shapes[0],x+(1),y+(0),z+(-1));
  total=total+bound_point_weight(20)*read_float(a[0],shapes[0],x+(0),y+(0),z+(0));
  total=total+bound_point_weight(21)*read_float(a[0],shapes[0],x+(1),y+(0),z+(0));
 }
 else if constexpr(HEAD==7) {
  total=total+bound_point_weight(22)*read_float(a[1],shapes[1],x+(0),y+(0),z+(-1));
  total=total+bound_point_weight(23)*read_float(a[1],shapes[1],x+(0),y+(1),z+(-1));
  total=total+bound_point_weight(24)*read_float(a[1],shapes[1],x+(0),y+(0),z+(0));
  total=total+bound_point_weight(25)*read_float(a[1],shapes[1],x+(0),y+(1),z+(0));
 }
 else if constexpr(HEAD==8) {
  total=total+bound_point_weight(26)*read_float(a[2],shapes[2],x+(0),y+(0),z+(0));
 }
 else if constexpr(HEAD==9) {
  total=total+bound_point_weight(27)*read_float(a[0],shapes[0],x+(0),y+(0),z+(0));
  total=total+bound_point_weight(28)*read_float(a[0],shapes[0],x+(1),y+(0),z+(0));
 }
 else if constexpr(HEAD==10) {
  total=total+bound_point_weight(29)*read_float(a[1],shapes[1],x+(0),y+(0),z+(0));
  total=total+bound_point_weight(30)*read_float(a[1],shapes[1],x+(0),y+(1),z+(0));
 }
 else if constexpr(HEAD==11) {
  total=total+bound_point_weight(31)*read_float(a[2],shapes[2],x+(0),y+(0),z+(0));
  total=total+bound_point_weight(32)*read_float(a[2],shapes[2],x+(0),y+(0),z+(1));
 }
 return total;
}
__device__ double dynamic_point_reference(const float*const* a,const Shape* shapes,int head,int x,int y,int z,
    const int* rows,const int* channels,const int* offsets,const double* weights) {
    double total=0.;
    for(int i=rows[head];i<rows[head+1];++i) {
        int c=channels[i];
        double v=read_float(a[c],shapes[c],x+offsets[3*i],y+offsets[3*i+1],z+offsets[3*i+2]);
        total=total+weights[i]*v;
    }
    return total;
}
__device__ double point_linear(const float*const* a,const Shape* shapes,int head,int x,int y,int z,
 const int* rows,const int* channels,const int* offsets,const double* weights) {
 switch(head) {
 case 0:return bound_point_head<0>(a,shapes,x,y,z);
 case 1:return bound_point_head<1>(a,shapes,x,y,z);
 case 2:return bound_point_head<2>(a,shapes,x,y,z);
 case 3:return bound_point_head<3>(a,shapes,x,y,z);
 case 4:return bound_point_head<4>(a,shapes,x,y,z);
 case 5:return bound_point_head<5>(a,shapes,x,y,z);
 case 6:return bound_point_head<6>(a,shapes,x,y,z);
 case 7:return bound_point_head<7>(a,shapes,x,y,z);
 case 8:return bound_point_head<8>(a,shapes,x,y,z);
 case 9:return bound_point_head<9>(a,shapes,x,y,z);
 case 10:return bound_point_head<10>(a,shapes,x,y,z);
 case 11:return bound_point_head<11>(a,shapes,x,y,z);
 default:return __longlong_as_double(0x7ff8000000000000ULL);
 }
}
extern "C" __global__ void learned_transport(int nx,int ny,int nz,
    const float* vx,const float* vy,const float* vz,const float* tracer,
    float* ox,float* oy,float* oz,float* ot,
    const int* rows,const int* channels,const int* offsets,const double* weights,
    const double* gate,const double* mix,double dt,double h) {
    long long index=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    Shape shapes[4];long long length[4],total=0;
    for(int k=0;k<4;++k) {shapes[k]=dims(k,nx,ny,nz);length[k]=volume(shapes[k]);total+=length[k];}
    if(index>=total)return;
    int kind=0;while(index>=length[kind]) {index-=length[kind];++kind;}
    Shape s=shapes[kind];int x=index%s.x,y=(index/s.x)%s.y,z=index/((long long)s.x*s.y);
    float* out[4]={ox,oy,oz,ot};const float* input[4]={vx,vy,vz,tracer};
    int xyz[3]={x,y,z};
    if(kind<3 && (xyz[kind]==0 || xyz[kind]==(kind==0?nx:kind==1?ny:nz))) {out[kind][index]=0.f;return;}
    double base[3]={double(x),double(y),double(z)},position[3],query[3],middle[3],final[3];
    double upper[3]={double(nx)*h,double(ny)*h,double(nz)*h};
    for(int a=0;a<3;++a)position[a]=(base[a]+(kind==a?0.:.5))*h;
    for(int a=0;a<3;++a) {
        double p=point_linear(input,shapes,kind*3+a,x,y,z,rows,channels,offsets,weights);
        double world=position[a]+(dt*bound_gate(kind*3+a))*p;
        world=world<0.?0.:(world>upper[a]?upper[a]:world);
        query[a]=world/h;
    }
    for(int c=0;c<3;++c)middle[c]=sample(input[c],shapes[c],query[0]-(c==0?0.:.5),query[1]-(c==1?0.:.5),query[2]-(c==2?0.:.5),true);
    for(int a=0;a<3;++a) {
        double value=0.;for(int c=0;c<3;++c)value=value+bound_mix((kind*3+a)*3+c)*middle[c];
        double delta=dt*value;delta=delta< -h?-h:delta;delta=delta>h?h:delta;
        double world=position[a]+delta;world=world<0.?0.:(world>upper[a]?upper[a]:world);
        final[a]=world/h-(kind==a?0.:.5);
    }
    // Identical clipping policy to the qualified scalar-tail deployment.
    bool clip=(s.x%4)!=0 && x>=s.x-(s.x%4);
    out[kind][index]=float(sample(input[kind],s,final[0],final[1],final[2],clip));
}
extern "C" __global__ void learned_linear(int nx,int ny,int nz,
    const void* f0,const void* f1,const void* f2,
    int ax,int ay,int az,int bx,int by,int bz,int cx,int cy,int cz,
    int dtype,int head,int selected,double sign,
    const int* rows,const int* channels,const int* offsets,const double* weights,double* output) {
    long long index=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(index>=(long long)nx*ny*nz)return;
    int x=index%nx,y=(index/nx)%ny,z=index/((long long)nx*ny);
    const void* fields[3]={f0,f1,f2};Shape shapes[3]={{ax,ay,az},{bx,by,bz},{cx,cy,cz}};double total=0.;
    for(int i=rows[head];i<rows[head+1];++i) {
        int c=channels[i];if(selected>=0 && c!=selected)continue;
        Shape s=shapes[c];int xx=bounded(x+offsets[3*i],s.x),yy=bounded(y+offsets[3*i+1],s.y),zz=bounded(z+offsets[3*i+2],s.z);
        long long at=xx+(long long)s.x*(yy+(long long)s.y*zz);
        double value=dtype?((const double*)fields[c])[at]:double(((const float*)fields[c])[at]);
        total=total+(sign*weights[i])*value;
    }
    output[index]=total;
}

// Generic pointwise affine/readout execution. No spatial stencil or pressure
// inverse is constructed. All deltas come from the restored learned stages.
extern "C" __global__ void affine_three(int nx,int ny,int nz,
    const float* b0,const float* b1,const float* b2,
    const double* d0,const double* d1,const double* d2,
    void* o0,void* o1,void* o2,double scale,int multiply_only,int as_double,int wall) {
    long long index=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    Shape shapes[3];long long len[3],total=0;
    for(int k=0;k<3;++k){shapes[k]=dims(k,nx,ny,nz);len[k]=volume(shapes[k]);total+=len[k];}
    if(index>=total)return;
    int kind=0;while(index>=len[kind]){index-=len[kind];++kind;}
    const float* base[3]={b0,b1,b2};const double* delta[3]={d0,d1,d2};void* out[3]={o0,o1,o2};
    double value=delta[kind][index]*scale;
    if(!multiply_only)value=double(base[kind][index])+value;
    if(wall){Shape sh=shapes[kind];int pos=kind==0?int(index%sh.x):kind==1?int((index/sh.x)%sh.y):int(index/((long long)sh.x*sh.y));
        int end=kind==0?nx:kind==1?ny:nz;if(pos==0||pos==end)value=0.;}
    if(as_double)((double*)out[kind])[index]=value;else((float*)out[kind])[index]=float(value);
}
extern "C" __global__ void combine_ordered_three(long long count,
    const double* a,const double* b,const double* c,double scale,double denominator,double* out) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    double value=0.;value=value+scale*a[i];value=value+scale*b[i];value=value+scale*c[i];
    out[i]=value/denominator;
}
extern "C" __global__ void difference_scale(long long count,const double* a,const double* b,double scale,double* out) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i<count)out[i]=(a[i]-b[i])*scale;
}
extern "C" __global__ void sum_gauge(long long count,const double* a,const double* b,double* out) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i<count){double value=a[i]+b[i],origin=a[0]+b[0];out[i]=value-origin;}
}

// Generic multi-head learned CSR followed by affine readout. All offsets,
// channels and weights are caller-supplied restored network data.
extern "C" __global__ void learned_gather_three_affine(int nx,int ny,int nz,
 const void* f0,const void* f1,const void* f2,
 int ax,int ay,int az,int bx,int by,int bz,int cx,int cy,int cz,int dtype,
 const float* b0,const float* b1,const float* b2,
 void* o0,void* o1,void* o2,
 const int* rows,const int* channels,const int* offsets,const double* weights,
 double sign,double scale,int multiply_only,int as_double) {
 long long index=(long long)blockIdx.x*blockDim.x+threadIdx.x;
 Shape targets[3];long long lens[3],length=0;
 for(int k=0;k<3;++k){targets[k]=dims(k,nx,ny,nz);lens[k]=volume(targets[k]);length+=lens[k];}
 if(index>=length)return;int head=0;while(index>=lens[head]){index-=lens[head];++head;}
 Shape target=targets[head];int x=index%target.x,y=(index/target.x)%target.y,z=index/((long long)target.x*target.y);
 const void* fields[3]={f0,f1,f2};Shape shapes[3]={{ax,ay,az},{bx,by,bz},{cx,cy,cz}};
 double total=0.;
 for(int i=rows[head];i<rows[head+1];++i){
  int c=channels[i];Shape sh=shapes[c];int xx=bounded(x+offsets[3*i],sh.x),yy=bounded(y+offsets[3*i+1],sh.y),zz=bounded(z+offsets[3*i+2],sh.z);
  long long at=xx+(long long)sh.x*(yy+(long long)sh.y*zz);
  double v=dtype?((const double*)fields[c])[at]:double(((const float*)fields[c])[at]);
  total=total+(sign*weights[i])*v;
 }
 double value=total*scale;
 const float* bases[3]={b0,b1,b2};if(!multiply_only)value=double(bases[head][index])+value;
 int pos=head==0?x:head==1?y:z;int end=head==0?nx:head==1?ny:nz;
 if(pos==0||pos==end)value=0.;
 void* outputs[3]={o0,o1,o2};
 if(as_double)((double*)outputs[head])[index]=value;else((float*)outputs[head])[index]=float(value);
}
extern "C" __global__ void learned_gather_ordered_rhs(int nx,int ny,int nz,
 const float* f0,const float* f1,const float* f2,
 int ax,int ay,int az,int bx,int by,int bz,int cx,int cy,int cz,int head,
 const int* rows,const int* channels,const int* offsets,const double* weights,
 double sign,double scale,double denominator,double* output) {
 long long index=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(index>=(long long)nx*ny*nz)return;
 int x=index%nx,y=(index/nx)%ny,z=index/((long long)nx*ny);
 const float* fields[3]={f0,f1,f2};Shape shapes[3]={{ax,ay,az},{bx,by,bz},{cx,cy,cz}};
 double t0=0.,t1=0.,t2=0.;
 for(int i=rows[head];i<rows[head+1];++i){
  int c=channels[i];Shape sh=shapes[c];int xx=bounded(x+offsets[3*i],sh.x),yy=bounded(y+offsets[3*i+1],sh.y),zz=bounded(z+offsets[3*i+2],sh.z);
  long long at=xx+(long long)sh.x*(yy+(long long)sh.y*zz);double v=double(fields[c][at]);
  double term=(sign*weights[i])*v;
  if(c==0)t0=t0+term;else if(c==1)t1=t1+term;else t2=t2+term;
 }
 double value=0.;value=value+scale*t0;value=value+scale*t1;value=value+scale*t2;
 output[index]=__ddiv_rn(value,denominator);
}
extern "C" __global__ void learned_gather_difference_scale(int nx,int ny,int nz,
 const void* f0,const void* f1,const void* f2,
 int ax,int ay,int az,int bx,int by,int bz,int cx,int cy,int cz,int dtype,int head,
 const int* rows,const int* channels,const int* offsets,const double* weights,
 double sign,const double* base,double scale,double* output) {
 long long index=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(index>=(long long)nx*ny*nz)return;
 int x=index%nx,y=(index/nx)%ny,z=index/((long long)nx*ny);
 const void* fields[3]={f0,f1,f2};Shape shapes[3]={{ax,ay,az},{bx,by,bz},{cx,cy,cz}};double total=0.;
 for(int i=rows[head];i<rows[head+1];++i){
  int c=channels[i];Shape sh=shapes[c];int xx=bounded(x+offsets[3*i],sh.x),yy=bounded(y+offsets[3*i+1],sh.y),zz=bounded(z+offsets[3*i+2],sh.z);
  long long at=xx+(long long)sh.x*(yy+(long long)sh.y*zz);
  double v=dtype?((const double*)fields[c])[at]:double(((const float*)fields[c])[at]);
  total=total+(sign*weights[i])*v;
 }
 output[index]=(base[index]-total)*scale;
}

// Integer geometry specialization only. Learned CSR data remain arguments.
struct GeometrySourceTable {
    Shape shapes[4]; const float* input[4]; float* output[4];
    long long length[4]; long long total;
};
static_assert(sizeof(GeometrySourceTable)==152, "GeometrySourceTable ABI");
template<int NX,int NY,int NZ,int HEAD>
__device__ __forceinline__ double bound_geometry_point_head(const GeometrySourceTable& sources,int x,int y,int z) {
 double total=0.;
 if constexpr(HEAD==0) {
  total=total+bound_point_weight(0)*read_float(sources.input[0],Shape{NX+1,NY+0,NZ+0},x+(0),y+(0),z+(0));
 }
 else if constexpr(HEAD==1) {
  total=total+bound_point_weight(1)*read_float(sources.input[1],Shape{NX+0,NY+1,NZ+0},x+(-1),y+(0),z+(0));
  total=total+bound_point_weight(2)*read_float(sources.input[1],Shape{NX+0,NY+1,NZ+0},x+(0),y+(0),z+(0));
  total=total+bound_point_weight(3)*read_float(sources.input[1],Shape{NX+0,NY+1,NZ+0},x+(-1),y+(1),z+(0));
  total=total+bound_point_weight(4)*read_float(sources.input[1],Shape{NX+0,NY+1,NZ+0},x+(0),y+(1),z+(0));
 }
 else if constexpr(HEAD==2) {
  total=total+bound_point_weight(5)*read_float(sources.input[2],Shape{NX+0,NY+0,NZ+1},x+(-1),y+(0),z+(0));
  total=total+bound_point_weight(6)*read_float(sources.input[2],Shape{NX+0,NY+0,NZ+1},x+(0),y+(0),z+(0));
  total=total+bound_point_weight(7)*read_float(sources.input[2],Shape{NX+0,NY+0,NZ+1},x+(-1),y+(0),z+(1));
  total=total+bound_point_weight(8)*read_float(sources.input[2],Shape{NX+0,NY+0,NZ+1},x+(0),y+(0),z+(1));
 }
 else if constexpr(HEAD==3) {
  total=total+bound_point_weight(9)*read_float(sources.input[0],Shape{NX+1,NY+0,NZ+0},x+(0),y+(-1),z+(0));
  total=total+bound_point_weight(10)*read_float(sources.input[0],Shape{NX+1,NY+0,NZ+0},x+(1),y+(-1),z+(0));
  total=total+bound_point_weight(11)*read_float(sources.input[0],Shape{NX+1,NY+0,NZ+0},x+(0),y+(0),z+(0));
  total=total+bound_point_weight(12)*read_float(sources.input[0],Shape{NX+1,NY+0,NZ+0},x+(1),y+(0),z+(0));
 }
 else if constexpr(HEAD==4) {
  total=total+bound_point_weight(13)*read_float(sources.input[1],Shape{NX+0,NY+1,NZ+0},x+(0),y+(0),z+(0));
 }
 else if constexpr(HEAD==5) {
  total=total+bound_point_weight(14)*read_float(sources.input[2],Shape{NX+0,NY+0,NZ+1},x+(0),y+(-1),z+(0));
  total=total+bound_point_weight(15)*read_float(sources.input[2],Shape{NX+0,NY+0,NZ+1},x+(0),y+(0),z+(0));
  total=total+bound_point_weight(16)*read_float(sources.input[2],Shape{NX+0,NY+0,NZ+1},x+(0),y+(-1),z+(1));
  total=total+bound_point_weight(17)*read_float(sources.input[2],Shape{NX+0,NY+0,NZ+1},x+(0),y+(0),z+(1));
 }
 else if constexpr(HEAD==6) {
  total=total+bound_point_weight(18)*read_float(sources.input[0],Shape{NX+1,NY+0,NZ+0},x+(0),y+(0),z+(-1));
  total=total+bound_point_weight(19)*read_float(sources.input[0],Shape{NX+1,NY+0,NZ+0},x+(1),y+(0),z+(-1));
  total=total+bound_point_weight(20)*read_float(sources.input[0],Shape{NX+1,NY+0,NZ+0},x+(0),y+(0),z+(0));
  total=total+bound_point_weight(21)*read_float(sources.input[0],Shape{NX+1,NY+0,NZ+0},x+(1),y+(0),z+(0));
 }
 else if constexpr(HEAD==7) {
  total=total+bound_point_weight(22)*read_float(sources.input[1],Shape{NX+0,NY+1,NZ+0},x+(0),y+(0),z+(-1));
  total=total+bound_point_weight(23)*read_float(sources.input[1],Shape{NX+0,NY+1,NZ+0},x+(0),y+(1),z+(-1));
  total=total+bound_point_weight(24)*read_float(sources.input[1],Shape{NX+0,NY+1,NZ+0},x+(0),y+(0),z+(0));
  total=total+bound_point_weight(25)*read_float(sources.input[1],Shape{NX+0,NY+1,NZ+0},x+(0),y+(1),z+(0));
 }
 else if constexpr(HEAD==8) {
  total=total+bound_point_weight(26)*read_float(sources.input[2],Shape{NX+0,NY+0,NZ+1},x+(0),y+(0),z+(0));
 }
 else if constexpr(HEAD==9) {
  total=total+bound_point_weight(27)*read_float(sources.input[0],Shape{NX+1,NY+0,NZ+0},x+(0),y+(0),z+(0));
  total=total+bound_point_weight(28)*read_float(sources.input[0],Shape{NX+1,NY+0,NZ+0},x+(1),y+(0),z+(0));
 }
 else if constexpr(HEAD==10) {
  total=total+bound_point_weight(29)*read_float(sources.input[1],Shape{NX+0,NY+1,NZ+0},x+(0),y+(0),z+(0));
  total=total+bound_point_weight(30)*read_float(sources.input[1],Shape{NX+0,NY+1,NZ+0},x+(0),y+(1),z+(0));
 }
 else if constexpr(HEAD==11) {
  total=total+bound_point_weight(31)*read_float(sources.input[2],Shape{NX+0,NY+0,NZ+1},x+(0),y+(0),z+(0));
  total=total+bound_point_weight(32)*read_float(sources.input[2],Shape{NX+0,NY+0,NZ+1},x+(0),y+(0),z+(1));
 }
 return total;
}
template<int NX,int NY,int NZ>
__device__ __forceinline__ double geometry_point(const GeometrySourceTable& sources,int head,int x,int y,int z,
 const int* rows,const int* channels,const int* offsets,const double* weights) {
 switch(head) {
 case 0:return bound_geometry_point_head<NX,NY,NZ,0>(sources,x,y,z);
 case 1:return bound_geometry_point_head<NX,NY,NZ,1>(sources,x,y,z);
 case 2:return bound_geometry_point_head<NX,NY,NZ,2>(sources,x,y,z);
 case 3:return bound_geometry_point_head<NX,NY,NZ,3>(sources,x,y,z);
 case 4:return bound_geometry_point_head<NX,NY,NZ,4>(sources,x,y,z);
 case 5:return bound_geometry_point_head<NX,NY,NZ,5>(sources,x,y,z);
 case 6:return bound_geometry_point_head<NX,NY,NZ,6>(sources,x,y,z);
 case 7:return bound_geometry_point_head<NX,NY,NZ,7>(sources,x,y,z);
 case 8:return bound_geometry_point_head<NX,NY,NZ,8>(sources,x,y,z);
 case 9:return bound_geometry_point_head<NX,NY,NZ,9>(sources,x,y,z);
 case 10:return bound_geometry_point_head<NX,NY,NZ,10>(sources,x,y,z);
 case 11:return bound_geometry_point_head<NX,NY,NZ,11>(sources,x,y,z);
 default:return __longlong_as_double(0x7ff8000000000000ULL);
 }
}
template<int NX,int NY,int NZ,int KIND>
__device__ __forceinline__ void geometry_transport_body(const GeometrySourceTable& sources,
 const int* rows,const int* channels,const int* offsets,const double* weights,const double* gate,const double* mix,double dt,double h) {
 constexpr int nx=NX,ny=NY,nz=NZ,kind=KIND;
 constexpr Shape s={NX+(KIND==0),NY+(KIND==1),NZ+(KIND==2)};
 long long index=(long long)blockIdx.x*blockDim.x+threadIdx.x;
 if(index>=(long long)s.x*s.y*s.z)return;
 int x=index%s.x,y=(index/s.x)%s.y,z=index/((long long)s.x*s.y);
 int xyz[3]={x,y,z};
 if constexpr(KIND<3) {
  if(xyz[KIND]==0 || xyz[KIND]==(KIND==0?nx:KIND==1?ny:nz)) {sources.output[KIND][index]=0.f;return;}
 }
 double base[3]={double(x),double(y),double(z)},position[3],query[3],middle[3],final[3];
 double upper[3]={double(nx)*h,double(ny)*h,double(nz)*h};
 for(int a=0;a<3;++a)position[a]=(base[a]+(kind==a?0.:.5))*h;
 for(int a=0;a<3;++a) {
  double p=geometry_point<NX,NY,NZ>(sources,kind*3+a,x,y,z,rows,channels,offsets,weights);
  double world=position[a]+(dt*bound_gate(kind*3+a))*p;
  world=world<0.?0.:(world>upper[a]?upper[a]:world);
  query[a]=world/h;
 }
 #pragma unroll
 for(int c=0;c<3;++c)middle[c]=sample(sources.input[c],dims(c,nx,ny,nz),query[0]-(c==0?0.:.5),query[1]-(c==1?0.:.5),query[2]-(c==2?0.:.5),true);
 for(int a=0;a<3;++a) {
  double value=0.;for(int c=0;c<3;++c)value=value+bound_mix((kind*3+a)*3+c)*middle[c];
  double delta=dt*value;delta=delta< -h?-h:delta;delta=delta>h?h:delta;
  double world=position[a]+delta;world=world<0.?0.:(world>upper[a]?upper[a]:world);
  final[a]=world/h-(kind==a?0.:.5);
 }
 bool clip=(s.x%4)!=0 && x>=s.x-(s.x%4);
 sources.output[KIND][index]=float(sample(sources.input[KIND],s,final[0],final[1],final[2],clip));
}
extern "C" __global__ void geometry_transport_9_11_11(const __grid_constant__ GeometrySourceTable sources,
 const int* rows,const int* channels,const int* offsets,const double* weights,const double* gate,const double* mix,double dt,double h) {
 switch(blockIdx.y) {
 case 0: geometry_transport_body<9,11,11,0>(sources,rows,channels,offsets,weights,gate,mix,dt,h);break;
 case 1: geometry_transport_body<9,11,11,1>(sources,rows,channels,offsets,weights,gate,mix,dt,h);break;
 case 2: geometry_transport_body<9,11,11,2>(sources,rows,channels,offsets,weights,gate,mix,dt,h);break;
 case 3: geometry_transport_body<9,11,11,3>(sources,rows,channels,offsets,weights,gate,mix,dt,h);break;
 }
}
extern "C" __global__ void geometry_transport_15_6_8(const __grid_constant__ GeometrySourceTable sources,
 const int* rows,const int* channels,const int* offsets,const double* weights,const double* gate,const double* mix,double dt,double h) {
 switch(blockIdx.y) {
 case 0: geometry_transport_body<15,6,8,0>(sources,rows,channels,offsets,weights,gate,mix,dt,h);break;
 case 1: geometry_transport_body<15,6,8,1>(sources,rows,channels,offsets,weights,gate,mix,dt,h);break;
 case 2: geometry_transport_body<15,6,8,2>(sources,rows,channels,offsets,weights,gate,mix,dt,h);break;
 case 3: geometry_transport_body<15,6,8,3>(sources,rows,channels,offsets,weights,gate,mix,dt,h);break;
 }
}
extern "C" __global__ void geometry_transport_16_16_16(const __grid_constant__ GeometrySourceTable sources,
 const int* rows,const int* channels,const int* offsets,const double* weights,const double* gate,const double* mix,double dt,double h) {
 switch(blockIdx.y) {
 case 0: geometry_transport_body<16,16,16,0>(sources,rows,channels,offsets,weights,gate,mix,dt,h);break;
 case 1: geometry_transport_body<16,16,16,1>(sources,rows,channels,offsets,weights,gate,mix,dt,h);break;
 case 2: geometry_transport_body<16,16,16,2>(sources,rows,channels,offsets,weights,gate,mix,dt,h);break;
 case 3: geometry_transport_body<16,16,16,3>(sources,rows,channels,offsets,weights,gate,mix,dt,h);break;
 }
}
extern "C" __global__ void geometry_transport_16_24_40(const __grid_constant__ GeometrySourceTable sources,
 const int* rows,const int* channels,const int* offsets,const double* weights,const double* gate,const double* mix,double dt,double h) {
 switch(blockIdx.y) {
 case 0: geometry_transport_body<16,24,40,0>(sources,rows,channels,offsets,weights,gate,mix,dt,h);break;
 case 1: geometry_transport_body<16,24,40,1>(sources,rows,channels,offsets,weights,gate,mix,dt,h);break;
 case 2: geometry_transport_body<16,24,40,2>(sources,rows,channels,offsets,weights,gate,mix,dt,h);break;
 case 3: geometry_transport_body<16,24,40,3>(sources,rows,channels,offsets,weights,gate,mix,dt,h);break;
 }
}
extern "C" __global__ void geometry_transport_32_24_48(const __grid_constant__ GeometrySourceTable sources,
 const int* rows,const int* channels,const int* offsets,const double* weights,const double* gate,const double* mix,double dt,double h) {
 switch(blockIdx.y) {
 case 0: geometry_transport_body<32,24,48,0>(sources,rows,channels,offsets,weights,gate,mix,dt,h);break;
 case 1: geometry_transport_body<32,24,48,1>(sources,rows,channels,offsets,weights,gate,mix,dt,h);break;
 case 2: geometry_transport_body<32,24,48,2>(sources,rows,channels,offsets,weights,gate,mix,dt,h);break;
 case 3: geometry_transport_body<32,24,48,3>(sources,rows,channels,offsets,weights,gate,mix,dt,h);break;
 }
}
extern "C" __global__ void geometry_transport_32_32_32(const __grid_constant__ GeometrySourceTable sources,
 const int* rows,const int* channels,const int* offsets,const double* weights,const double* gate,const double* mix,double dt,double h) {
 switch(blockIdx.y) {
 case 0: geometry_transport_body<32,32,32,0>(sources,rows,channels,offsets,weights,gate,mix,dt,h);break;
 case 1: geometry_transport_body<32,32,32,1>(sources,rows,channels,offsets,weights,gate,mix,dt,h);break;
 case 2: geometry_transport_body<32,32,32,2>(sources,rows,channels,offsets,weights,gate,mix,dt,h);break;
 case 3: geometry_transport_body<32,32,32,3>(sources,rows,channels,offsets,weights,gate,mix,dt,h);break;
 }
}
extern "C" __global__ void geometry_transport_64_64_64(const __grid_constant__ GeometrySourceTable sources,
 const int* rows,const int* channels,const int* offsets,const double* weights,const double* gate,const double* mix,double dt,double h) {
 switch(blockIdx.y) {
 case 0: geometry_transport_body<64,64,64,0>(sources,rows,channels,offsets,weights,gate,mix,dt,h);break;
 case 1: geometry_transport_body<64,64,64,1>(sources,rows,channels,offsets,weights,gate,mix,dt,h);break;
 case 2: geometry_transport_body<64,64,64,2>(sources,rows,channels,offsets,weights,gate,mix,dt,h);break;
 case 3: geometry_transport_body<64,64,64,3>(sources,rows,channels,offsets,weights,gate,mix,dt,h);break;
 }
}
extern "C" __global__ void geometry_transport_104_112_120(const __grid_constant__ GeometrySourceTable sources,
 const int* rows,const int* channels,const int* offsets,const double* weights,const double* gate,const double* mix,double dt,double h) {
 switch(blockIdx.y) {
 case 0: geometry_transport_body<104,112,120,0>(sources,rows,channels,offsets,weights,gate,mix,dt,h);break;
 case 1: geometry_transport_body<104,112,120,1>(sources,rows,channels,offsets,weights,gate,mix,dt,h);break;
 case 2: geometry_transport_body<104,112,120,2>(sources,rows,channels,offsets,weights,gate,mix,dt,h);break;
 case 3: geometry_transport_body<104,112,120,3>(sources,rows,channels,offsets,weights,gate,mix,dt,h);break;
 }
}
extern "C" __global__ void geometry_transport_128_128_128(const __grid_constant__ GeometrySourceTable sources,
 const int* rows,const int* channels,const int* offsets,const double* weights,const double* gate,const double* mix,double dt,double h) {
 switch(blockIdx.y) {
 case 0: geometry_transport_body<128,128,128,0>(sources,rows,channels,offsets,weights,gate,mix,dt,h);break;
 case 1: geometry_transport_body<128,128,128,1>(sources,rows,channels,offsets,weights,gate,mix,dt,h);break;
 case 2: geometry_transport_body<128,128,128,2>(sources,rows,channels,offsets,weights,gate,mix,dt,h);break;
 case 3: geometry_transport_body<128,128,128,3>(sources,rows,channels,offsets,weights,gate,mix,dt,h);break;
 }
}

extern "C" __global__ void bound_parameter_bits(unsigned long long* bits,int* metadata) {
 int i=int(blockIdx.x*blockDim.x+threadIdx.x);
 if(i<33 )bits[i]=__double_as_longlong(bound_point_weight(i));
 if(i<12)bits[33+i]=__double_as_longlong(bound_gate(i));
 if(i<36)bits[45+i]=__double_as_longlong(bound_mix(i));
 switch(i) {
 case 0:metadata[i]=0;break;
 case 1:metadata[i]=1;break;
 case 2:metadata[i]=5;break;
 case 3:metadata[i]=9;break;
 case 4:metadata[i]=13;break;
 case 5:metadata[i]=14;break;
 case 6:metadata[i]=18;break;
 case 7:metadata[i]=22;break;
 case 8:metadata[i]=26;break;
 case 9:metadata[i]=27;break;
 case 10:metadata[i]=29;break;
 case 11:metadata[i]=31;break;
 case 12:metadata[i]=33;break;
 case 13:metadata[i]=0;break;
 case 14:metadata[i]=1;break;
 case 15:metadata[i]=1;break;
 case 16:metadata[i]=1;break;
 case 17:metadata[i]=1;break;
 case 18:metadata[i]=2;break;
 case 19:metadata[i]=2;break;
 case 20:metadata[i]=2;break;
 case 21:metadata[i]=2;break;
 case 22:metadata[i]=0;break;
 case 23:metadata[i]=0;break;
 case 24:metadata[i]=0;break;
 case 25:metadata[i]=0;break;
 case 26:metadata[i]=1;break;
 case 27:metadata[i]=2;break;
 case 28:metadata[i]=2;break;
 case 29:metadata[i]=2;break;
 case 30:metadata[i]=2;break;
 case 31:metadata[i]=0;break;
 case 32:metadata[i]=0;break;
 case 33:metadata[i]=0;break;
 case 34:metadata[i]=0;break;
 case 35:metadata[i]=1;break;
 case 36:metadata[i]=1;break;
 case 37:metadata[i]=1;break;
 case 38:metadata[i]=1;break;
 case 39:metadata[i]=2;break;
 case 40:metadata[i]=0;break;
 case 41:metadata[i]=0;break;
 case 42:metadata[i]=1;break;
 case 43:metadata[i]=1;break;
 case 44:metadata[i]=2;break;
 case 45:metadata[i]=2;break;
 case 46:metadata[i]=0;break;
 case 47:metadata[i]=0;break;
 case 48:metadata[i]=0;break;
 case 49:metadata[i]=-1;break;
 case 50:metadata[i]=0;break;
 case 51:metadata[i]=0;break;
 case 52:metadata[i]=0;break;
 case 53:metadata[i]=0;break;
 case 54:metadata[i]=0;break;
 case 55:metadata[i]=-1;break;
 case 56:metadata[i]=1;break;
 case 57:metadata[i]=0;break;
 case 58:metadata[i]=0;break;
 case 59:metadata[i]=1;break;
 case 60:metadata[i]=0;break;
 case 61:metadata[i]=-1;break;
 case 62:metadata[i]=0;break;
 case 63:metadata[i]=0;break;
 case 64:metadata[i]=0;break;
 case 65:metadata[i]=0;break;
 case 66:metadata[i]=0;break;
 case 67:metadata[i]=-1;break;
 case 68:metadata[i]=0;break;
 case 69:metadata[i]=1;break;
 case 70:metadata[i]=0;break;
 case 71:metadata[i]=0;break;
 case 72:metadata[i]=1;break;
 case 73:metadata[i]=0;break;
 case 74:metadata[i]=-1;break;
 case 75:metadata[i]=0;break;
 case 76:metadata[i]=1;break;
 case 77:metadata[i]=-1;break;
 case 78:metadata[i]=0;break;
 case 79:metadata[i]=0;break;
 case 80:metadata[i]=0;break;
 case 81:metadata[i]=0;break;
 case 82:metadata[i]=1;break;
 case 83:metadata[i]=0;break;
 case 84:metadata[i]=0;break;
 case 85:metadata[i]=0;break;
 case 86:metadata[i]=0;break;
 case 87:metadata[i]=0;break;
 case 88:metadata[i]=0;break;
 case 89:metadata[i]=-1;break;
 case 90:metadata[i]=0;break;
 case 91:metadata[i]=0;break;
 case 92:metadata[i]=0;break;
 case 93:metadata[i]=0;break;
 case 94:metadata[i]=0;break;
 case 95:metadata[i]=-1;break;
 case 96:metadata[i]=1;break;
 case 97:metadata[i]=0;break;
 case 98:metadata[i]=0;break;
 case 99:metadata[i]=1;break;
 case 100:metadata[i]=0;break;
 case 101:metadata[i]=0;break;
 case 102:metadata[i]=-1;break;
 case 103:metadata[i]=1;break;
 case 104:metadata[i]=0;break;
 case 105:metadata[i]=-1;break;
 case 106:metadata[i]=0;break;
 case 107:metadata[i]=0;break;
 case 108:metadata[i]=0;break;
 case 109:metadata[i]=1;break;
 case 110:metadata[i]=0;break;
 case 111:metadata[i]=0;break;
 case 112:metadata[i]=0;break;
 case 113:metadata[i]=0;break;
 case 114:metadata[i]=-1;break;
 case 115:metadata[i]=0;break;
 case 116:metadata[i]=1;break;
 case 117:metadata[i]=-1;break;
 case 118:metadata[i]=0;break;
 case 119:metadata[i]=0;break;
 case 120:metadata[i]=0;break;
 case 121:metadata[i]=0;break;
 case 122:metadata[i]=1;break;
 case 123:metadata[i]=0;break;
 case 124:metadata[i]=0;break;
 case 125:metadata[i]=0;break;
 case 126:metadata[i]=0;break;
 case 127:metadata[i]=0;break;
 case 128:metadata[i]=0;break;
 case 129:metadata[i]=0;break;
 case 130:metadata[i]=1;break;
 case 131:metadata[i]=0;break;
 case 132:metadata[i]=0;break;
 case 133:metadata[i]=0;break;
 case 134:metadata[i]=0;break;
 case 135:metadata[i]=0;break;
 case 136:metadata[i]=0;break;
 case 137:metadata[i]=1;break;
 case 138:metadata[i]=0;break;
 case 139:metadata[i]=0;break;
 case 140:metadata[i]=0;break;
 case 141:metadata[i]=0;break;
 case 142:metadata[i]=0;break;
 case 143:metadata[i]=0;break;
 case 144:metadata[i]=1;break;
 }
}

extern "C" __global__ void bound_point_primitive(long long count,const float* vx,const float* vy,const float* vz,
 int nx,int ny,int nz,const int* queries,const int* rows,const int* channels,const int* offsets,const double* weights,
 double* output,double* reference) {
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 const float* fields[3]={vx,vy,vz};Shape shapes[3]={dims(0,nx,ny,nz),dims(1,nx,ny,nz),dims(2,nx,ny,nz)};
 const int* q=queries+4*i;
 output[i]=point_linear(fields,shapes,q[0],q[1],q[2],q[3],rows,channels,offsets,weights);
 reference[i]=dynamic_point_reference(fields,shapes,q[0],q[1],q[2],q[3],rows,channels,offsets,weights);
}
extern "C" __global__ void bound_mix_gate_primitive(long long count,const double* values,const int* heads,
 const double* gates,const double* mixes,double* output,double* reference) {
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 int head=heads[i];const double* v=values+5*i;double a=0.,b=0.;
 for(int c=0;c<3;++c){a=a+bound_mix(head*3+c)*v[c];b=b+mixes[head*3+c]*v[c];}
 output[2*i]=a;reference[2*i]=b;
 output[2*i+1]=(v[3]*bound_gate(head))*v[4];reference[2*i+1]=(v[3]*gates[head])*v[4];
}
