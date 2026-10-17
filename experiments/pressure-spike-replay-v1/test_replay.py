"""Target selection preserves physical steps and excludes native warm-up."""
import unittest
from replay import select_steps


class SelectionTests(unittest.TestCase):
    def test_rankings_are_unioned_with_immediate_neighbors_only(self):
        row = dict(peaks=[dict(physical_step=20,excursion=4,absolute_rms_pa=1),
                          dict(physical_step=30,excursion=2,absolute_rms_pa=5)],worst_physical_step=40)
        config = dict(peaks_by_excursion=1,peaks_by_absolute_error=1,
                      neighbor_offsets=[-1,1],include_worst_error_step=True)
        result = select_steps(row,dict(warmup_steps=15,steps=40),config)
        self.assertEqual([r['step'] for r in result],[19,20,21,29,30,31,39,40])
        self.assertEqual([r['step'] for r in result if r['detected_peak']],[20,30])
        self.assertEqual(next(r for r in result if r['step']==40)['roles'],['worst_error_step'])

    def test_selection_deduplicates_peaks_and_excludes_warmup(self):
        row = dict(peaks=[dict(physical_step=16,excursion=4,absolute_rms_pa=5)],worst_physical_step=16)
        config = dict(peaks_by_excursion=1,peaks_by_absolute_error=1,
                      neighbor_offsets=[-1,1],include_worst_error_step=True)
        result = select_steps(row,dict(warmup_steps=15,steps=17),config)
        self.assertEqual([r['step'] for r in result],[16,17])
        self.assertEqual(result[0]['roles'],['selected_peak','worst_error_step'])


if __name__=='__main__':
    unittest.main()
