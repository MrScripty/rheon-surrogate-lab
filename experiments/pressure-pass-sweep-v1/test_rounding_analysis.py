"""Exact one-cell examples distinguish rounding from an imperfect pressure solve."""
from fractions import Fraction
import unittest
from rounding_analysis import cell_divergence, example, face_cast_bound, quantize


class RoundingTests(unittest.TestCase):
    def test_exactly_solenoidal_velocity_can_gain_divergence_on_float32_storage(self):
        observed=example()
        self.assertEqual(observed['ideal_divergence_s_inverse'],0)
        self.assertEqual(observed['rounded_divergence_s_inverse'],-2**-22)
        self.assertLess(abs(observed['rounded_divergence_s_inverse']),observed['cast_divergence_bound_s_inverse'])
        for ideal,stored in zip(observed['ideal_faces'],observed['stored_faces']):
            for v,q in zip(ideal,stored):
                self.assertLessEqual(abs(Fraction(q)-Fraction(v)),face_cast_bound(q))

    def test_representable_flow_has_zero_rounding_error_not_a_fixed_floor(self):
        faces=[(.25,.5),(.5,.25),(.25,.25)]
        stored=[tuple(quantize(v)for v in axis)for axis in faces]
        self.assertEqual(faces,stored)
        self.assertEqual(cell_divergence(faces,[.0625]*3),0)
        self.assertEqual(cell_divergence(stored,[.0625]*3),0)

    def test_half_smallest_subnormal_requires_an_absolute_rounding_term(self):
        ideal=2**-150
        stored=quantize(ideal)
        self.assertEqual(stored,0)
        self.assertLessEqual(Fraction(ideal),face_cast_bound(stored))


if __name__=='__main__':unittest.main()
