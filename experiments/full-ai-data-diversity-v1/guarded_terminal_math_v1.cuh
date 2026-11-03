
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
