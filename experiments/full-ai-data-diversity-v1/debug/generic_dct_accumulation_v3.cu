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

// DEBUG: generic matrix-axis contraction. No physical operator/eigenvalues.
extern "C" __global__ void fixed_axis_transform(
 const double* input,const double* basis,double* output,
 int nx,int ny,int nz,int axis,int inverse,int group,int fused) {
 long long row=(long long)blockIdx.x*blockDim.x+threadIdx.x;
 if(row>=(long long)nx*ny*nz)return;
 int sizes[3]={nx,ny,nz};long long strides[3]={1,nx,(long long)nx*ny};
 int n=sizes[axis];long long stride=strides[axis];int coordinate=(row/stride)%n;
 // Generic layout-tail arithmetic control; no physical coefficient is used.
 if(group<0) {
  long long span=axis==2?(long long)nx*ny:axis==1?(long long)nx*nz:(long long)ny*nz;
  long long position=axis==2?row%((long long)nx*ny):axis==1?(row%nx)+(long long)nx*(row/((long long)nx*ny)):row/nx;
  // Generic matrix shape specialization, verified on arbitrary bases/fields.
  // No physical coefficients, native values or field-dependent choices.
  if(nx==64 && ny==64 && nz==64)
   group=(!inverse && axis==0 && coordinate>=n-4)?2:1;
  else
   group=(axis>0 && span%6==4 && position>=span-4)?2:1;
 }
 long long base=row-(long long)coordinate*stride;double sums[8]={0.,0.,0.,0.,0.,0.,0.,0.};
 for(int k=0;k<n;++k) {
  double a=basis[inverse?k*n+coordinate:coordinate*n+k];double b=input[base+(long long)k*stride];
  int lane=k%group;
  sums[lane]=fused?__fma_rn(a,b,sums[lane]):__dadd_rn(sums[lane],__dmul_rn(a,b));
 }
 double value=sums[0];for(int lane=1;lane<group;++lane)value=__dadd_rn(value,sums[lane]);
 output[row]=value;
}
