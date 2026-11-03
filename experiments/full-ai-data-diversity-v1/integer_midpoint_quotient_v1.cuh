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
