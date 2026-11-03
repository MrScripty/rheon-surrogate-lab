// Exact dyadic midpoint certificates, not floating-point refinement.
// Valid certificates determine the original RN-even F64 quotient; uncertainty
// uses __ddiv_rn. Products fit107bits under the registered normal guards.
__device__ __forceinline__ bool quotient_finite(double x) {
 return (__double_as_longlong(x)&0x7ff0000000000000ULL)!=0x7ff0000000000000ULL;
}
__device__ __forceinline__ bool quotient_candidate(double q) {
 unsigned long long bits=(unsigned long long)__double_as_longlong(q);
 int exponent=int((bits>>52)&2047ULL);
 return !(bits>>63)&&exponent>=2&&exponent<=2045;
}
__device__ __forceinline__ bool quotient_guard(double a,double b,double r) {
 return a>=0x1p-400&&a<=0x1p400&&b>=0x1p-20&&b<=0x1p20&&r>0.&&quotient_finite(r);
}
__device__ __forceinline__ bool quotient_midpoint_compare(double a,double b,double q,bool upper,int& comparison) {
 unsigned long long ab=(unsigned long long)__double_as_longlong(a),bb=(unsigned long long)__double_as_longlong(b);
 unsigned long long qb=(unsigned long long)__double_as_longlong(q);
 unsigned long long ma=(ab&0xfffffffffffffULL)|0x10000000000000ULL;
 unsigned long long mb=(bb&0xfffffffffffffULL)|0x10000000000000ULL;
 unsigned long long mq=(qb&0xfffffffffffffULL)|0x10000000000000ULL;
 int ea=int((ab>>52)&2047ULL)-1075,eb=int((bb>>52)&2047ULL)-1075,eq=int((qb>>52)&2047ULL)-1075;
 unsigned long long mm;int em;
 if(upper){mm=2ULL*mq+1ULL;em=eq-1;}
 else if(mq==0x10000000000000ULL){mm=4ULL*mq-1ULL;em=eq-2;}
 else {mm=2ULL*mq-1ULL;em=eq-1;}
 unsigned long long low=mm*mb,high=__umul64hi(mm,mb);
 if(high==0ULL)return false;
 int product_bits=128-__clzll(high),top_a=ea+52,top_product=em+eb+product_bits-1;
 if(top_a!=top_product){comparison=top_a>top_product?1:-1;return true;}
 int shift=ea-em-eb;if(shift!=53&&shift!=54)return false;
 unsigned long long aligned_low=ma<<shift,aligned_high=ma>>(64-shift);
 comparison=aligned_high<high?-1:aligned_high>high?1:aligned_low<low?-1:aligned_low>low?1:0;
 return true;
}
// mode0=initialcertificate,1=adjacentcertificate,2=zero,3=originaldivision.
__device__ __forceinline__ double integer_midpoint_div(double a,double b,double r,int& mode) {
 mode=3;
 if(a==0.&&b>0.&&quotient_finite(b)){mode=2;return a;}
 if(!quotient_guard(a,b,r))return __ddiv_rn(a,b);
 double q=__dmul_rn(a,r);
 #pragma unroll
 for(int attempt=0;attempt<2;++attempt) {
  if(!quotient_candidate(q))break;
  int lo,hi;
  if(!quotient_midpoint_compare(a,b,q,false,lo)||!quotient_midpoint_compare(a,b,q,true,hi))break;
  unsigned long long qb=(unsigned long long)__double_as_longlong(q);bool even=(qb&1ULL)==0ULL;
  if((lo>0||(lo==0&&even))&&(hi<0||(hi==0&&even))){mode=attempt;return q;}
  qb=(lo<0||(lo==0&&!even))?qb-1ULL:qb+1ULL;
  q=__longlong_as_double((long long)qb);
 }
 return __ddiv_rn(a,b);
}
__device__ __forceinline__ double integer_midpoint_div(double a,double b,double r) {
 int mode;return integer_midpoint_div(a,b,r,mode);
}
extern "C" __global__ void integer_quotient_primitive(long long count,const double* a,const double* b,const double* r,
 double* output,double* reference,int* modes,int* low_compare,int* high_compare,double* initial) {
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 int mode;output[i]=integer_midpoint_div(a[i],b[i],r[i],mode);reference[i]=__ddiv_rn(a[i],b[i]);modes[i]=mode;
 double q=__dmul_rn(a[i],r[i]);initial[i]=q;int lo=9,hi=9;
 if(quotient_guard(a[i],b[i],r[i])&&quotient_candidate(q)) {
  if(!quotient_midpoint_compare(a[i],b[i],q,false,lo))lo=9;
  if(!quotient_midpoint_compare(a[i],b[i],q,true,hi))hi=9;
 }
 low_compare[i]=lo;high_compare[i]=hi;
}

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
__device__ double sample(const float* a,Shape s,double x,double y,double z,bool clip) {
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
__device__ double point_linear(const float*const* a,const Shape* shapes,int head,int x,int y,int z,
    const int* rows,const int* channels,const int* offsets,const double* weights) {
    double total=0.;
    for(int i=rows[head];i<rows[head+1];++i) {
        int c=channels[i];
        double v=read_float(a[c],shapes[c],x+offsets[3*i],y+offsets[3*i+1],z+offsets[3*i+2]);
        total=total+weights[i]*v;
    }
    return total;
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
        double world=position[a]+(dt*gate[kind*3+a])*p;
        world=world<0.?0.:(world>upper[a]?upper[a]:world);
        query[a]=world/h;
    }
    for(int c=0;c<3;++c)middle[c]=sample(input[c],shapes[c],query[0]-(c==0?0.:.5),query[1]-(c==1?0.:.5),query[2]-(c==2?0.:.5),true);
    for(int a=0;a<3;++a) {
        double value=0.;for(int c=0;c<3;++c)value=value+mix[(kind*3+a)*3+c]*middle[c];
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
template<int NX,int NY,int NZ>
__device__ __forceinline__ double geometry_point(const GeometrySourceTable& sources,int head,int x,int y,int z,
 const int* rows,const int* channels,const int* offsets,const double* weights) {
 double total=0.;
 for(int i=rows[head];i<rows[head+1];++i) {
  int c=channels[i];Shape shape={NX+(c==0),NY+(c==1),NZ+(c==2)};
  double v=read_float(sources.input[c],shape,x+offsets[3*i],y+offsets[3*i+1],z+offsets[3*i+2]);
  total=total+weights[i]*v;
 }
 return total;
}
template<int NX,int NY,int NZ,int KIND>
__device__ __forceinline__ void geometry_transport_body(const GeometrySourceTable& sources,
 const int* rows,const int* channels,const int* offsets,const double* weights,const double* gate,const double* mix,double dt,double h,double reciprocal_h) {
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
  double world=position[a]+(dt*gate[kind*3+a])*p;
  world=world<0.?0.:(world>upper[a]?upper[a]:world);
  query[a]=integer_midpoint_div(world,h,reciprocal_h);
 }
 #pragma unroll
 for(int c=0;c<3;++c)middle[c]=sample(sources.input[c],dims(c,nx,ny,nz),query[0]-(c==0?0.:.5),query[1]-(c==1?0.:.5),query[2]-(c==2?0.:.5),true);
 for(int a=0;a<3;++a) {
  double value=0.;for(int c=0;c<3;++c)value=value+mix[(kind*3+a)*3+c]*middle[c];
  double delta=dt*value;delta=delta< -h?-h:delta;delta=delta>h?h:delta;
  double world=position[a]+delta;world=world<0.?0.:(world>upper[a]?upper[a]:world);
  final[a]=integer_midpoint_div(world,h,reciprocal_h)-(kind==a?0.:.5);
 }
 bool clip=(s.x%4)!=0 && x>=s.x-(s.x%4);
 sources.output[KIND][index]=float(sample(sources.input[KIND],s,final[0],final[1],final[2],clip));
}
extern "C" __global__ void geometry_transport_9_11_11(const __grid_constant__ GeometrySourceTable sources,
 const int* rows,const int* channels,const int* offsets,const double* weights,const double* gate,const double* mix,double dt,double h,double reciprocal_h) {
 switch(blockIdx.y) {
 case 0: geometry_transport_body<9,11,11,0>(sources,rows,channels,offsets,weights,gate,mix,dt,h,reciprocal_h);break;
 case 1: geometry_transport_body<9,11,11,1>(sources,rows,channels,offsets,weights,gate,mix,dt,h,reciprocal_h);break;
 case 2: geometry_transport_body<9,11,11,2>(sources,rows,channels,offsets,weights,gate,mix,dt,h,reciprocal_h);break;
 case 3: geometry_transport_body<9,11,11,3>(sources,rows,channels,offsets,weights,gate,mix,dt,h,reciprocal_h);break;
 }
}
extern "C" __global__ void geometry_transport_15_6_8(const __grid_constant__ GeometrySourceTable sources,
 const int* rows,const int* channels,const int* offsets,const double* weights,const double* gate,const double* mix,double dt,double h,double reciprocal_h) {
 switch(blockIdx.y) {
 case 0: geometry_transport_body<15,6,8,0>(sources,rows,channels,offsets,weights,gate,mix,dt,h,reciprocal_h);break;
 case 1: geometry_transport_body<15,6,8,1>(sources,rows,channels,offsets,weights,gate,mix,dt,h,reciprocal_h);break;
 case 2: geometry_transport_body<15,6,8,2>(sources,rows,channels,offsets,weights,gate,mix,dt,h,reciprocal_h);break;
 case 3: geometry_transport_body<15,6,8,3>(sources,rows,channels,offsets,weights,gate,mix,dt,h,reciprocal_h);break;
 }
}
extern "C" __global__ void geometry_transport_16_16_16(const __grid_constant__ GeometrySourceTable sources,
 const int* rows,const int* channels,const int* offsets,const double* weights,const double* gate,const double* mix,double dt,double h,double reciprocal_h) {
 switch(blockIdx.y) {
 case 0: geometry_transport_body<16,16,16,0>(sources,rows,channels,offsets,weights,gate,mix,dt,h,reciprocal_h);break;
 case 1: geometry_transport_body<16,16,16,1>(sources,rows,channels,offsets,weights,gate,mix,dt,h,reciprocal_h);break;
 case 2: geometry_transport_body<16,16,16,2>(sources,rows,channels,offsets,weights,gate,mix,dt,h,reciprocal_h);break;
 case 3: geometry_transport_body<16,16,16,3>(sources,rows,channels,offsets,weights,gate,mix,dt,h,reciprocal_h);break;
 }
}
extern "C" __global__ void geometry_transport_16_24_40(const __grid_constant__ GeometrySourceTable sources,
 const int* rows,const int* channels,const int* offsets,const double* weights,const double* gate,const double* mix,double dt,double h,double reciprocal_h) {
 switch(blockIdx.y) {
 case 0: geometry_transport_body<16,24,40,0>(sources,rows,channels,offsets,weights,gate,mix,dt,h,reciprocal_h);break;
 case 1: geometry_transport_body<16,24,40,1>(sources,rows,channels,offsets,weights,gate,mix,dt,h,reciprocal_h);break;
 case 2: geometry_transport_body<16,24,40,2>(sources,rows,channels,offsets,weights,gate,mix,dt,h,reciprocal_h);break;
 case 3: geometry_transport_body<16,24,40,3>(sources,rows,channels,offsets,weights,gate,mix,dt,h,reciprocal_h);break;
 }
}
extern "C" __global__ void geometry_transport_32_24_48(const __grid_constant__ GeometrySourceTable sources,
 const int* rows,const int* channels,const int* offsets,const double* weights,const double* gate,const double* mix,double dt,double h,double reciprocal_h) {
 switch(blockIdx.y) {
 case 0: geometry_transport_body<32,24,48,0>(sources,rows,channels,offsets,weights,gate,mix,dt,h,reciprocal_h);break;
 case 1: geometry_transport_body<32,24,48,1>(sources,rows,channels,offsets,weights,gate,mix,dt,h,reciprocal_h);break;
 case 2: geometry_transport_body<32,24,48,2>(sources,rows,channels,offsets,weights,gate,mix,dt,h,reciprocal_h);break;
 case 3: geometry_transport_body<32,24,48,3>(sources,rows,channels,offsets,weights,gate,mix,dt,h,reciprocal_h);break;
 }
}
extern "C" __global__ void geometry_transport_32_32_32(const __grid_constant__ GeometrySourceTable sources,
 const int* rows,const int* channels,const int* offsets,const double* weights,const double* gate,const double* mix,double dt,double h,double reciprocal_h) {
 switch(blockIdx.y) {
 case 0: geometry_transport_body<32,32,32,0>(sources,rows,channels,offsets,weights,gate,mix,dt,h,reciprocal_h);break;
 case 1: geometry_transport_body<32,32,32,1>(sources,rows,channels,offsets,weights,gate,mix,dt,h,reciprocal_h);break;
 case 2: geometry_transport_body<32,32,32,2>(sources,rows,channels,offsets,weights,gate,mix,dt,h,reciprocal_h);break;
 case 3: geometry_transport_body<32,32,32,3>(sources,rows,channels,offsets,weights,gate,mix,dt,h,reciprocal_h);break;
 }
}
extern "C" __global__ void geometry_transport_64_64_64(const __grid_constant__ GeometrySourceTable sources,
 const int* rows,const int* channels,const int* offsets,const double* weights,const double* gate,const double* mix,double dt,double h,double reciprocal_h) {
 switch(blockIdx.y) {
 case 0: geometry_transport_body<64,64,64,0>(sources,rows,channels,offsets,weights,gate,mix,dt,h,reciprocal_h);break;
 case 1: geometry_transport_body<64,64,64,1>(sources,rows,channels,offsets,weights,gate,mix,dt,h,reciprocal_h);break;
 case 2: geometry_transport_body<64,64,64,2>(sources,rows,channels,offsets,weights,gate,mix,dt,h,reciprocal_h);break;
 case 3: geometry_transport_body<64,64,64,3>(sources,rows,channels,offsets,weights,gate,mix,dt,h,reciprocal_h);break;
 }
}
extern "C" __global__ void geometry_transport_104_112_120(const __grid_constant__ GeometrySourceTable sources,
 const int* rows,const int* channels,const int* offsets,const double* weights,const double* gate,const double* mix,double dt,double h,double reciprocal_h) {
 switch(blockIdx.y) {
 case 0: geometry_transport_body<104,112,120,0>(sources,rows,channels,offsets,weights,gate,mix,dt,h,reciprocal_h);break;
 case 1: geometry_transport_body<104,112,120,1>(sources,rows,channels,offsets,weights,gate,mix,dt,h,reciprocal_h);break;
 case 2: geometry_transport_body<104,112,120,2>(sources,rows,channels,offsets,weights,gate,mix,dt,h,reciprocal_h);break;
 case 3: geometry_transport_body<104,112,120,3>(sources,rows,channels,offsets,weights,gate,mix,dt,h,reciprocal_h);break;
 }
}
extern "C" __global__ void geometry_transport_128_128_128(const __grid_constant__ GeometrySourceTable sources,
 const int* rows,const int* channels,const int* offsets,const double* weights,const double* gate,const double* mix,double dt,double h,double reciprocal_h) {
 switch(blockIdx.y) {
 case 0: geometry_transport_body<128,128,128,0>(sources,rows,channels,offsets,weights,gate,mix,dt,h,reciprocal_h);break;
 case 1: geometry_transport_body<128,128,128,1>(sources,rows,channels,offsets,weights,gate,mix,dt,h,reciprocal_h);break;
 case 2: geometry_transport_body<128,128,128,2>(sources,rows,channels,offsets,weights,gate,mix,dt,h,reciprocal_h);break;
 case 3: geometry_transport_body<128,128,128,3>(sources,rows,channels,offsets,weights,gate,mix,dt,h,reciprocal_h);break;
 }
}
