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
    const double* gate,const double* mix,double alpha) {
    long long index=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    Shape shapes[4];long long length[4],total=0;
    for(int k=0;k<4;++k) {shapes[k]=dims(k,nx,ny,nz);length[k]=volume(shapes[k]);total+=length[k];}
    if(index>=total)return;
    int kind=0;while(index>=length[kind]) {index-=length[kind];++kind;}
    Shape s=shapes[kind];int x=index%s.x,y=(index/s.x)%s.y,z=index/((long long)s.x*s.y);
    float* out[4]={ox,oy,oz,ot};const float* input[4]={vx,vy,vz,tracer};
    int xyz[3]={x,y,z};
    if(kind<3 && (xyz[kind]==0 || xyz[kind]==(kind==0?nx:kind==1?ny:nz))) {out[kind][index]=0.f;return;}
    double base[3]={double(x),double(y),double(z)},query[3],middle[3],final[3];
    for(int a=0;a<3;++a) {
        double p=point_linear(input,shapes,kind*3+a,x,y,z,rows,channels,offsets,weights);
        query[a]=(base[a]+(kind==a?0.:.5))+(alpha*p)*gate[kind*3+a];
    }
    for(int c=0;c<3;++c)middle[c]=sample(input[c],shapes[c],query[0]-(c==0?0.:.5),query[1]-(c==1?0.:.5),query[2]-(c==2?0.:.5),true);
    for(int a=0;a<3;++a) {
        double value=0.;for(int c=0;c<3;++c)value=value+mix[(kind*3+a)*3+c]*middle[c];
        double delta=alpha*value;delta=delta< -1.?-1.:delta;delta=delta>1.?1.:delta;
        final[a]=base[a]+delta;
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

// Known external program only. No physical state, learned input, or solver.
extern "C" __global__ void known_control(int nx,int ny,int nz,int length,
    const double* table,const double* direction,double amplitude,double swirl,double fluctuation,double dt,
    double* fx,double* fy,double* fz) {
    long long index=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    Shape shapes[3];long long sizes[3],total=0;
    for(int d=0;d<3;++d) {shapes[d]=dims(d,nx,ny,nz);sizes[d]=volume(shapes[d]);total+=sizes[d];}
    if(index>=total)return;
    int d=0;while(index>=sizes[d]) {index-=sizes[d];++d;}
    Shape shape=shapes[d];int xyz[3]={int(index%shape.x),int((index/shape.x)%shape.y),int(index/((long long)shape.x*shape.y))};
    double* output[3]={fx,fy,fz};int counts[3]={nx,ny,nz};
    if(xyz[d]==0 || xyz[d]==counts[d]) {output[d][index]=0.;return;}
    double offsets[3],r2=0.,wave=0.;
    for(int a=0;a<3;++a) {
        const double* entry=table+((d*3+a)*length+xyz[a])*3;
        offsets[a]=entry[0];r2=r2+entry[1];wave=wave+entry[2];
    }
    double cutoff=1.-r2/16.;double envelope=r2<16.?exp(-.5*r2)*(cutoff*cutoff):0.;
    int a=(d+1)%3,b=(d+2)%3;
    double rotation=direction[a]*offsets[b]-direction[b]*offsets[a];wave=wave/3.;
    double acceleration=(amplitude*envelope)*((direction[d]+swirl*rotation)+fluctuation*wave);
    output[d][index]=dt*acceleration;
}

// Compact known scalar inputs; GPU evaluates only coordinate/phase tables.
extern "C" __global__ void known_axes(int nx,int ny,int nz,int length,
    double h,double e0,double e1,double e2,double c0,double c1,double c2,
    double radius,double omega,double time,double phase,double* table) {
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index>=9*length)return;
    int d=index/(3*length),a=(index/length)%3,q=index%length;
    int counts[3]={nx,ny,nz};double extents[3]={e0,e1,e2},center[3]={c0,c1,c2};
    double* entry=table+index*3;
    if(q>=counts[a]+int(a==d)) {entry[0]=0.;entry[1]=0.;entry[2]=0.;return;}
    double pos=0.+(double(q)+(a==d?0.:.5))*h;
    double offset=(pos-center[a])/radius;
    double wave=sin(6.283185307179586476925286766559*pos/extents[a]+omega*time*double(a+1)+phase+double(d));
    entry[0]=offset;entry[1]=offset*offset;entry[2]=wave;
}
extern "C" __global__ void known_control_scalar(int nx,int ny,int nz,int length,
    const double* table,double d0,double d1,double d2,double amplitude,double swirl,double fluctuation,double dt,
    double* fx,double* fy,double* fz) {
    const double direction[3]={d0,d1,d2};
    long long index=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    Shape shapes[3];long long sizes[3],total=0;
    for(int d=0;d<3;++d) {shapes[d]=dims(d,nx,ny,nz);sizes[d]=volume(shapes[d]);total+=sizes[d];}
    if(index>=total)return;
    int d=0;while(index>=sizes[d]) {index-=sizes[d];++d;}
    Shape shape=shapes[d];int xyz[3]={int(index%shape.x),int((index/shape.x)%shape.y),int(index/((long long)shape.x*shape.y))};
    double* output[3]={fx,fy,fz};int counts[3]={nx,ny,nz};
    if(xyz[d]==0 || xyz[d]==counts[d]) {output[d][index]=0.;return;}
    double offsets[3],r2=0.,wave=0.;
    for(int a=0;a<3;++a) {
        const double* entry=table+((d*3+a)*length+xyz[a])*3;
        offsets[a]=entry[0];r2=r2+entry[1];wave=wave+entry[2];
    }
    double cutoff=1.-r2/16.;double envelope=r2<16.?exp(-.5*r2)*(cutoff*cutoff):0.;
    int a=(d+1)%3,b=(d+2)%3;
    double rotation=direction[a]*offsets[b]-direction[b]*offsets[a];wave=wave/3.;
    double acceleration=(amplitude*envelope)*((direction[d]+swirl*rotation)+fluctuation*wave);
    output[d][index]=dt*acceleration;
}

extern "C" __global__ void known_axes_dynamic(int nx,int ny,int nz,int length,
    const double* parameters,double* table) {
    double h=parameters[0],e0=parameters[1],e1=parameters[2],e2=parameters[3];
    double c0=parameters[4],c1=parameters[5],c2=parameters[6],radius=parameters[7];
    double omega=parameters[8],time=parameters[9],phase=parameters[10];
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index>=9*length)return;
    int d=index/(3*length),a=(index/length)%3,q=index%length;
    int counts[3]={nx,ny,nz};double extents[3]={e0,e1,e2},center[3]={c0,c1,c2};
    double* entry=table+index*3;
    if(q>=counts[a]+int(a==d)) {entry[0]=0.;entry[1]=0.;entry[2]=0.;return;}
    double pos=0.+(double(q)+(a==d?0.:.5))*h;
    double offset=(pos-center[a])/radius;
    double wave=sin(6.283185307179586476925286766559*pos/extents[a]+omega*time*double(a+1)+phase+double(d));
    entry[0]=offset;entry[1]=offset*offset;entry[2]=wave;
}

extern "C" __global__ void known_control_dynamic(int nx,int ny,int nz,int length,
    const double* table,const double* parameters,
    double* fx,double* fy,double* fz) {
    const double direction[3]={parameters[11],parameters[12],parameters[13]};
    double amplitude=parameters[14],swirl=parameters[15],fluctuation=parameters[16],dt=parameters[17];
    long long index=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    Shape shapes[3];long long sizes[3],total=0;
    for(int d=0;d<3;++d) {shapes[d]=dims(d,nx,ny,nz);sizes[d]=volume(shapes[d]);total+=sizes[d];}
    if(index>=total)return;
    int d=0;while(index>=sizes[d]) {index-=sizes[d];++d;}
    Shape shape=shapes[d];int xyz[3]={int(index%shape.x),int((index/shape.x)%shape.y),int(index/((long long)shape.x*shape.y))};
    double* output[3]={fx,fy,fz};int counts[3]={nx,ny,nz};
    if(xyz[d]==0 || xyz[d]==counts[d]) {output[d][index]=0.;return;}
    double offsets[3],r2=0.,wave=0.;
    for(int a=0;a<3;++a) {
        const double* entry=table+((d*3+a)*length+xyz[a])*3;
        offsets[a]=entry[0];r2=r2+entry[1];wave=wave+entry[2];
    }
    double cutoff=1.-r2/16.;double envelope=r2<16.?exp(-.5*r2)*(cutoff*cutoff):0.;
    int a=(d+1)%3,b=(d+2)%3;
    double rotation=direction[a]*offsets[b]-direction[b]*offsets[a];wave=wave/3.;
    double acceleration=(amplitude*envelope)*((direction[d]+swirl*rotation)+fluctuation*wave);
    output[d][index]=dt*acceleration;
}
