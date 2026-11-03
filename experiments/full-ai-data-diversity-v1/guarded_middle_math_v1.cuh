// Enclose the original RN64 program; uncertain results use complete fallback.
struct MiddleInterval { double lo,hi; };
__device__ __forceinline__ bool middle_finite(double x) {
 return (__double_as_longlong(x)&0x7ff0000000000000ULL)!=0x7ff0000000000000ULL;
}
__device__ __forceinline__ bool middle_ordered(MiddleInterval x) {
 return middle_finite(x.lo)&&middle_finite(x.hi)&&x.lo<=x.hi;
}
__device__ __forceinline__ void middle_donors(const float* field,Shape s,const double* q,float* d) {
 int lx=int(q[0]),ly=int(q[1]),lz=int(q[2]);
 int hx=bounded(lx+1,s.x),hy=bounded(ly+1,s.y),hz=bounded(lz+1,s.z);
 #pragma unroll
 for(int z=0;z<2;++z) {
  #pragma unroll
  for(int y=0;y<2;++y) {
   #pragma unroll
   for(int x=0;x<2;++x)d[4*z+2*y+x]=field[(x?hx:lx)+(long long)s.x*((y?hy:ly)+(long long)s.y*(z?hz:lz))];
  }
 }
}
__device__ __forceinline__ bool middle_sample_interval(const float* field,Shape s,double x,double y,double z,MiddleInterval& out) {
 if(!middle_finite(x)||!middle_finite(y)||!middle_finite(z))return false;
 double q[3]={bounded_position(x,s.x),bounded_position(y,s.y),bounded_position(z,s.z)};
 float d[8],unused;double a,b;
 middle_donors(field,s,q,d);
 terminal_try(d,q[0]-int(q[0]),q[1]-int(q[1]),q[2]-int(q[2]),false,unused,a,b);
 if(b>0.) {
  float lower=d[0],upper=d[0];
  #pragma unroll
  for(int i=1;i<8;++i){lower=d[i]<lower?d[i]:lower;upper=d[i]>upper?d[i]:upper;}
  double lo=__dadd_rd(a,-b),hi=__dadd_ru(a,b);
  lo=lo<double(lower)?double(lower):lo;lo=lo>double(upper)?double(upper):lo;
  hi=hi<double(lower)?double(lower):hi;hi=hi>double(upper)?double(upper):hi;
  out={lo,hi};
 } else {
  double exact=sample(field,s,x,y,z,true);out={exact,exact};
 }
 return middle_ordered(out);
}
__device__ __forceinline__ bool middle_propagate(const MiddleInterval* m,const double* mix,const double* position,
 const double* upper,const double* phase,double dt,double h,MiddleInterval* query) {
 if(!(dt>0.&&h>0.)||!middle_finite(dt)||!middle_finite(h))return false;
 #pragma unroll
 for(int c=0;c<3;++c)if(!middle_ordered(m[c]))return false;
 #pragma unroll
 for(int axis=0;axis<3;++axis) {
  if(!middle_finite(position[axis])||!middle_finite(upper[axis])||!middle_finite(phase[axis])||upper[axis]<0.)return false;
  double lo=0.,hi=0.;
  #pragma unroll
  for(int c=0;c<3;++c) {
   double w=mix[axis*3+c];if(!middle_finite(w))return false;
   lo=__dadd_rn(lo,__dmul_rn(w,w>=0.?m[c].lo:m[c].hi));
   hi=__dadd_rn(hi,__dmul_rn(w,w>=0.?m[c].hi:m[c].lo));
   if(!middle_finite(lo)||!middle_finite(hi))return false;
  }
  lo=__dmul_rn(dt,lo);hi=__dmul_rn(dt,hi);
  if(!middle_finite(lo)||!middle_finite(hi))return false;
  lo=lo< -h?-h:lo;lo=lo>h?h:lo;hi=hi< -h?-h:hi;hi=hi>h?h:hi;
  lo=__dadd_rn(position[axis],lo);hi=__dadd_rn(position[axis],hi);
  if(!middle_finite(lo)||!middle_finite(hi))return false;
  lo=lo<0.?0.:(lo>upper[axis]?upper[axis]:lo);hi=hi<0.?0.:(hi>upper[axis]?upper[axis]:hi);
  query[axis]={__dsub_rn(__ddiv_rn(lo,h),phase[axis]),__dsub_rn(__ddiv_rn(hi,h),phase[axis])};
  if(!middle_ordered(query[axis]))return false;
 }
 return true;
}
// reason: 0 accepted,1 invalid inputs/range,2 zero endpoint,3 differing bits.
__device__ __forceinline__ int middle_terminal_donors(const float* d,const double* low,const double* high,
 bool clip,float& result,double& approximation,double& radius) {
 approximation=0.;radius=0.;
 #pragma unroll
 for(int axis=0;axis<3;++axis)if(!(low[axis]>=0.&&low[axis]<=high[axis]&&high[axis]<=1.))return 1;
 float unused;double base;
 terminal_try(d,low[0],low[1],low[2],false,unused,approximation,base);
 if(!(base>0.))return 1;
 float magnitude=0.f,lower=d[0],upper=d[0];
 #pragma unroll
 for(int i=0;i<8;++i){float a=d[i]<0.f?-d[i]:d[i];magnitude=a>magnitude?a:magnitude;
  lower=d[i]<lower?d[i]:lower;upper=d[i]>upper?d[i]:upper;}
 double width=0.;
 #pragma unroll
 for(int axis=0;axis<3;++axis)width=__dadd_ru(width,__dsub_ru(high[axis],low[axis]));
 radius=__dadd_ru(base,__dmul_ru(2.*double(magnitude),width));
 if(!middle_finite(radius))return 1;
 double lo=__dadd_rd(approximation,-radius),hi=__dadd_ru(approximation,radius);
 if(clip){lo=lo<double(lower)?double(lower):lo;lo=lo>double(upper)?double(upper):lo;
          hi=hi<double(lower)?double(lower):hi;hi=hi>double(upper)?double(upper):hi;}
 float fl=__double2float_rn(lo),fh=__double2float_rn(hi);
 if(fl==0.f||fh==0.f)return 2;
 if(__float_as_int(fl)!=__float_as_int(fh))return 3;
 result=fl;return 0;
}
__device__ __forceinline__ int middle_terminal_sample(const float* field,Shape s,const MiddleInterval* query,
 bool clip,float& result) {
 double lo[3],hi[3];int sizes[3]={s.x,s.y,s.z};
 #pragma unroll
 for(int axis=0;axis<3;++axis) {
  if(!middle_ordered(query[axis]))return 1;
  lo[axis]=bounded_position(query[axis].lo,sizes[axis]);hi[axis]=bounded_position(query[axis].hi,sizes[axis]);
  if(int(lo[axis])!=int(hi[axis]))return 4;
 }
 float donors[8];middle_donors(field,s,lo,donors);
 #pragma unroll
 for(int axis=0;axis<3;++axis){int lower=int(lo[axis]);lo[axis]=__dsub_rn(lo[axis],double(lower));hi[axis]=__dsub_rn(hi[axis],double(lower));}
 double approximation,radius;
 return middle_terminal_donors(donors,lo,hi,clip,result,approximation,radius);
}
extern "C" __global__ void middle_sample_primitive(long long count,const float* field,int nx,int ny,int nz,
 const double* coordinates,double* low,double* high,double* reference,int* valid) {
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 const double* q=coordinates+3*i;Shape s={nx,ny,nz};MiddleInterval out={0.,0.};
 valid[i]=middle_sample_interval(field,s,q[0],q[1],q[2],out);
 low[i]=out.lo;high[i]=out.hi;reference[i]=sample(field,s,q[0],q[1],q[2],true);
}
extern "C" __global__ void middle_interval_primitive(long long count,const float* donors,const double* low,const double* high,
 const double* actual,const int* clips,float* output,float* reference,int* accepted,double* approximate,double* radius,double* raw) {
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 float result=0.f;double a,r;int reason=middle_terminal_donors(donors+8*i,low+3*i,high+3*i,clips[i]!=0,result,a,r);
 double ref=terminal_reference_donors(donors+8*i,actual[3*i],actual[3*i+1],actual[3*i+2],clips[i]!=0);
 output[i]=reason==0?result:float(ref);reference[i]=float(ref);accepted[i]=reason==0;approximate[i]=a;radius[i]=r;raw[i]=ref;
}
extern "C" __global__ void middle_propagation_primitive(long long count,const double* actual,const double* low,const double* high,
 const double* mix,const double* position,const double* upper,const double* phase,const double* dt,const double* h,
 double* qlow,double* qhigh,double* reference,int* valid) {
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 MiddleInterval m[3],q[3];for(int c=0;c<3;++c)m[c]={low[3*i+c],high[3*i+c]};
 valid[i]=middle_propagate(m,mix+9*i,position+3*i,upper+3*i,phase+3*i,dt[i],h[i],q);
 for(int a=0;a<3;++a) {
  qlow[3*i+a]=valid[i]?q[a].lo:0.;qhigh[3*i+a]=valid[i]?q[a].hi:0.;
  double v=0.;for(int c=0;c<3;++c)v=v+mix[9*i+3*a+c]*actual[3*i+c];
  double delta=dt[i]*v;delta=delta< -h[i]?-h[i]:delta;delta=delta>h[i]?h[i]:delta;
  double world=position[3*i+a]+delta;world=world<0.?0.:(world>upper[3*i+a]?upper[3*i+a]:world);
  reference[3*i+a]=world/h[i]-phase[3*i+a];
 }
}
