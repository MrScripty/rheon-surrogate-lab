"""Exact scalar division certificates; no model, learned weights or timings."""
import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import random
import struct
import time

MASK = (1 << 64) - 1


def bits(x):
    return struct.unpack('Q', struct.pack('d', x))[0]


def value(x):
    return struct.unpack('d', struct.pack('Q', x))[0]


def components(x):
    b = bits(x)
    return (b & ((1 << 52) - 1)) | (1 << 52), ((b >> 52) & 2047) - 1075


def compare_midpoint(a, b, q, upper):
    ma, ea = components(a)
    mb, eb = components(b)
    mq, eq = components(q)
    if upper:
        mm, em = 2 * mq + 1, eq - 1
    elif mq == 1 << 52:
        mm, em = 4 * mq - 1, eq - 2
    else:
        mm, em = 2 * mq - 1, eq - 1
    assert mm.bit_length() <= 54 and mb.bit_length() == 53
    product = mm * mb
    assert product.bit_length() <= 107
    low, high = product & MASK, product >> 64
    product_bits = 64 + high.bit_length()
    top_a, top_product = ea + 52, em + eb + product_bits - 1
    if top_a != top_product:
        return 1 if top_a > top_product else -1
    shift = ea - em - eb
    assert shift in (53, 54)
    aligned_low, aligned_high = (ma << shift) & MASK, ma >> (64 - shift)
    left = (aligned_high, aligned_low)
    right = (high, low)
    comparison = (left > right) - (left < right)
    # Independent exact dyadic comparison validates the limb/exponent method.
    exact_a = Fraction.from_float(a)
    exact_product = Fraction(mm * mb) * Fraction(2) ** (em + eb)
    assert comparison == (exact_a > exact_product) - (exact_a < exact_product)
    return comparison


def native_divide(a, b):
    if b == 0.:
        if a == 0. or math.isnan(a):
            return math.nan
        return math.copysign(math.inf, math.copysign(1., a) * math.copysign(1., b))
    return a / b


def certify(a, b, reciprocal):
    if a == 0. and math.isfinite(b) and b > 0.:
        return a, 'zero'
    if not (2.**-400 <= a <= 2.**400 and 2.**-20 <= b <= 2.**20
            and reciprocal > 0. and math.isfinite(reciprocal)):
        return native_divide(a, b), 'fallback'
    q = a * reciprocal
    for attempt in range(2):
        qb = bits(q)
        exponent = (qb >> 52) & 2047
        if qb >> 63 or not 2 <= exponent <= 2045:
            break
        lo = compare_midpoint(a, b, q, False)
        hi = compare_midpoint(a, b, q, True)
        even = not (qb & 1)
        if (lo > 0 or lo == 0 and even) and (hi < 0 or hi == 0 and even):
            return q, 'initial' if attempt == 0 else 'adjacent'
        q = value(qb - 1 if lo < 0 or lo == 0 and not even else qb + 1)
    return native_divide(a, b), 'fallback'


def run(args):
    if args.output.exists():
        raise ValueError('Fresh arithmetic derivation report required')
    reg = json.loads(args.plan.read_text())
    controls = reg['CPU_math_controls']
    rng = random.Random(controls['seed'])
    counts = {name: 0 for name in ('initial', 'adjacent', 'zero', 'fallback')}
    populations = {}
    comparisons = 0
    exact_ratio_checks = 0
    def check(a, b, r):
        nonlocal comparisons, exact_ratio_checks
        actual, mode = certify(a, b, r)
        reference = native_divide(a, b)
        if math.isfinite(a) and math.isfinite(b) and a != 0. and b != 0. and math.isfinite(reference):
            exact = float(Fraction.from_float(a) / Fraction.from_float(b))
            assert bits(exact) == bits(reference)
            exact_ratio_checks += 1
        assert bits(actual) == bits(reference), (a, b, r, actual, reference)
        counts[mode] += 1
        comparisons += 1
    for _ in range(controls['arbitrary_normal_pairs']):
        a = math.ldexp(rng.uniform(1., 2.), rng.randrange(-399, 400))
        b = math.ldexp(rng.uniform(1., 2.), rng.randrange(-19, 20))
        check(a, b, 1. / b)
    populations['arbitrary_normals'] = comparisons
    cpm_values = (8, 10, 16, 25, 50, 100)
    for i in range(controls['density_pairs']):
        b = 1. / cpm_values[i % 6]
        q = rng.uniform(0., 128.)
        a = q * b
        if i % 3 == 0:
            a = math.nextafter(a, -math.inf if i % 2 else math.inf)
        check(a, b, 1. / b)
    populations['six_spacings'] = comparisons - populations['arbitrary_normals']
    edges = [0., -0., math.nextafter(0., 1.), -math.nextafter(0., 1.), 2.**-400,
             math.nextafter(2.**-400, 0.), 2.**400, math.nextafter(2.**400, math.inf),
             math.inf, -math.inf, math.nan, -1., 1., 2.**-1022, 2.**1023]
    for i in range(controls['guard_or_bad_reciprocal_pairs']):
        if i % 2:
            a, b = rng.uniform(.01, 10.), rng.uniform(.01, 10.)
            r = (1. / b) * (1.01 if i % 4 == 1 else .99)
        else:
            a = edges[(i // 2) % len(edges)]
            b = edges[(i // (2 * len(edges))) % len(edges)]
            r = native_divide(1., b)
        check(a, b, r)
    populations['guards_or_bad_reciprocals'] = comparisons - sum(populations.values())
    assert comparisons == controls['expected_total']
    result = dict(status='passed_CPU_generic_scalar_integer_midpoint_controls_not_GPU_qualification',
        created_unix=time.time(), samples=comparisons, populations=populations, decisions=counts,
        exact_fraction_ratio_checks=exact_ratio_checks, all_bits_match=True,
        exact_midpoint_comparisons_checked_against_Fraction=True,
        proof=['A positive normal F64 is m*2^e with53bit m. Midpoints are exact54bit dyadics, including the lower neighbor at powers of two. Exclude extreme quotient binades.',
               'The product of a midpoint mantissa and denominator mantissa fits107bits. Compare top binary exponents first; equality forces an alignment shift of53or54bits. Two64bit limbs then compare the exact numerator and scaled midpoint.',
               'The RN-even rounding bin is bounded by these exact midpoints with both ties assigned according to candidate mantissa parity. Any accepted certificate therefore proves RN-even(a/b). The adjacent attempt uses the identical certificate. Uncertainty falls back; no estimate accuracy assumption is needed for correctness.',
               'These CPU controls validate generic scalar arithmetic only. They neither exercise a learned model nor establish GPU correctness, throughput, full-call speed or target achievement.'],
        scalar_arithmetic_device='CPU scalar math controls only', cpu_model_forward=False,
        learned_parameters_used=False, learned_model_forward=False, optimizer_updates=0,
        latency_claim=False, GPU_qualification_pass=False,
        plan=str(args.plan),plan_sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        source=str(Path(__file__)),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','samples','decisions','exact_fraction_ratio_checks')}))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
