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

// Guarded terminal sampling. Only stored F32 output may use paired-F32 math.
// The enclosure derivation is registered separately; all other paths call sample.
struct TerminalPair { float hi,lo; };
__device__ __forceinline__ TerminalPair terminal_two_sum(float a,float b) {
 float x=__fadd_rn(a,b), bv=__fsub_rn(x,a), av=__fsub_rn(x,bv);
 float br=__fsub_rn(b,bv), ar=__fsub_rn(a,av);
 return {x,__fadd_rn(ar,br)};
}
__device__ __forceinline__ TerminalPair terminal_pair_add(TerminalPair a,TerminalPair b) {
 TerminalPair s=terminal_two_sum(a.hi,b.hi);
 float t=__fadd_rn(a.lo,b.lo),v=__fadd_rn(s.lo,t);
 return terminal_two_sum(s.hi,v);
}
__device__ __forceinline__ TerminalPair terminal_pair_mul(TerminalPair a,TerminalPair b) {
 float p=__fmul_rn(a.hi,b.hi), e=__fmaf_rn(a.hi,b.hi,-p);
 float c1=__fmul_rn(a.hi,b.lo),c2=__fmul_rn(a.lo,b.hi);
 float c=__fadd_rn(c1,c2),v=__fadd_rn(e,c);
 return terminal_two_sum(p,v);
}
__device__ __forceinline__ TerminalPair terminal_fraction(double f) {
 float hi=__double2float_rn(f),lo=__double2float_rn(__dsub_rn(f,double(hi)));
 return terminal_two_sum(hi,lo);
}
__device__ __forceinline__ TerminalPair terminal_lerp(TerminalPair a,TerminalPair b,TerminalPair f) {
 TerminalPair negative={-a.hi,-a.lo};
 return terminal_pair_add(a,terminal_pair_mul(terminal_pair_add(b,negative),f));
}
__device__ __forceinline__ bool terminal_try(const float* d,double fx,double fy,double fz,bool clip,
 float& result,double& approximation,double& radius) {
 approximation=0.;radius=0.;
 if(!(fx>=0.&&fx<=1.&&fy>=0.&&fy<=1.&&fz>=0.&&fz<=1.))return false;
 float magnitude=0.f,lower=d[0],upper=d[0];
 #pragma unroll
 for(int i=0;i<8;++i) {
  if((__float_as_int(d[i])&0x7f800000)==0x7f800000)return false;
  float a=d[i]<0.f?-d[i]:d[i];magnitude=a>magnitude?a:magnitude;
  lower=d[i]<lower?d[i]:lower;upper=d[i]>upper?d[i]:upper;
 }
 if(!(magnitude>=0x1p-60f&&magnitude<=0x1p60f))return false;
 TerminalPair x=terminal_fraction(fx),y=terminal_fraction(fy),z=terminal_fraction(fz);
 TerminalPair p0=terminal_lerp({d[0],0.f},{d[1],0.f},x);
 TerminalPair p1=terminal_lerp({d[2],0.f},{d[3],0.f},x);
 TerminalPair p2=terminal_lerp({d[4],0.f},{d[5],0.f},x);
 TerminalPair p3=terminal_lerp({d[6],0.f},{d[7],0.f},x);
 TerminalPair q0=terminal_lerp(p0,p1,y),q1=terminal_lerp(p2,p3,y);
 TerminalPair value=terminal_lerp(q0,q1,z);
 approximation=__dadd_rn(double(value.hi),double(value.lo));
 radius=__dmul_rn(double(magnitude),0x1p-32);
 double lo=__dadd_rd(approximation,-radius),hi=__dadd_ru(approximation,radius);
 if(clip){lo=lo<double(lower)?double(lower):lo;lo=lo>double(upper)?double(upper):lo;
          hi=hi<double(lower)?double(lower):hi;hi=hi>double(upper)?double(upper):hi;}
 float fl=__double2float_rn(lo),fh=__double2float_rn(hi);
 if(fl==0.f||fh==0.f||__float_as_int(fl)!=__float_as_int(fh))return false;
 result=fl;return true;
}
__device__ __forceinline__ float terminal_sample(const float* a,Shape s,double x,double y,double z,bool clip) {
 if(((__double_as_longlong(x)&0x7ff0000000000000ULL)==0x7ff0000000000000ULL)||
    ((__double_as_longlong(y)&0x7ff0000000000000ULL)==0x7ff0000000000000ULL)||
    ((__double_as_longlong(z)&0x7ff0000000000000ULL)==0x7ff0000000000000ULL))return float(sample(a,s,x,y,z,clip));
 double xx=bounded_position(x,s.x),yy=bounded_position(y,s.y),zz=bounded_position(z,s.z);
 int lx=int(xx),ly=int(yy),lz=int(zz),hx=bounded(lx+1,s.x),hy=bounded(ly+1,s.y),hz=bounded(lz+1,s.z);
 float donors[8];
 #pragma unroll
 for(int bz=0;bz<2;++bz) {
  #pragma unroll
  for(int by=0;by<2;++by) {
   #pragma unroll
   for(int bx=0;bx<2;++bx) {
    int dx=bx?hx:lx,dy=by?hy:ly,dz=bz?hz:lz;
    donors[4*bz+2*by+bx]=a[dx+(long long)s.x*(dy+(long long)s.y*dz)];
   }
  }
 }
 float result;double approximation,radius;
 if(terminal_try(donors,xx-lx,yy-ly,zz-lz,clip,result,approximation,radius))return result;
 return float(sample(a,s,x,y,z,clip));
}
__device__ double terminal_reference_donors(const float* d,double fx,double fy,double fz,bool clip) {
 double value=0.,lower=1.7976931348623157e308,upper=-1.7976931348623157e308;
 for(int bz=0;bz<2;++bz)for(int by=0;by<2;++by)for(int bx=0;bx<2;++bx) {
  double donor=double(d[4*bz+2*by+bx]),weight=1.;
  weight=weight*(bx?fx:1.-fx);weight=weight*(by?fy:1.-fy);weight=weight*(bz?fz:1.-fz);
  value=value+weight*donor;if(clip){lower=donor<lower?donor:lower;upper=donor>upper?donor:upper;}
 }
 if(clip){value=value<lower?lower:value;value=value>upper?upper:value;}return value;
}
extern "C" __global__ void terminal_pair_primitive(long long count,const float* donors,const double* fractions,
 const int* clips,float* output,float* reference,int* accepted,double* approximate,double* radius,double* raw_reference) {
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 const float* d=donors+8*i;const double* f=fractions+3*i;float result=0.f;double a,r;
 bool fast=terminal_try(d,f[0],f[1],f[2],clips[i]!=0,result,a,r);
 double ref=terminal_reference_donors(d,f[0],f[1],f[2],clips[i]!=0);
 reference[i]=float(ref);output[i]=fast?result:float(ref);accepted[i]=fast;approximate[i]=a;radius[i]=r;raw_reference[i]=ref;
}
extern "C" __global__ void terminal_grid_primitive(long long count,const float* field,int nx,int ny,int nz,
 const double* coordinates,const int* clips,float* output,float* reference) {
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 Shape s={nx,ny,nz};const double* q=coordinates+3*i;bool clip=clips[i]!=0;
 output[i]=terminal_sample(field,s,q[0],q[1],q[2],clip);reference[i]=float(sample(field,s,q[0],q[1],q[2],clip));
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
  double world=position[a]+(dt*gate[kind*3+a])*p;
  world=world<0.?0.:(world>upper[a]?upper[a]:world);
  query[a]=world/h;
 }
 #pragma unroll
 for(int c=0;c<3;++c)middle[c]=sample(sources.input[c],dims(c,nx,ny,nz),query[0]-(c==0?0.:.5),query[1]-(c==1?0.:.5),query[2]-(c==2?0.:.5),true);
 for(int a=0;a<3;++a) {
  double value=0.;for(int c=0;c<3;++c)value=value+mix[(kind*3+a)*3+c]*middle[c];
  double delta=dt*value;delta=delta< -h?-h:delta;delta=delta>h?h:delta;
  double world=position[a]+delta;world=world<0.?0.:(world>upper[a]?upper[a]:world);
  final[a]=world/h-(kind==a?0.:.5);
 }
 bool clip=(s.x%4)!=0 && x>=s.x-(s.x%4);
 sources.output[KIND][index]=terminal_sample(sources.input[KIND],s,final[0],final[1],final[2],clip);
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
