"""Avoid eight-donor interpolation when the query is exactly one stored sample."""
import sampling_coordinates

def patch(text):
    text=sampling_coordinates.patch(text)
    start=text.index('    fn sample_coordinates(')
    position=text.index('        let mut result = 0.0;',start)
    shortcut='''        // All other donor weights are exactly zero here. Adding positive zero
        // preserves the original accumulator's signed-zero behavior as well.
        if t.iter().all(|value| *value == 0.0) {
            let index = lo[0] + self.dimensions[0] * (lo[1] + self.dimensions[1] * lo[2]);
            return Ok(0.0 + f64::from(self.values[index]));
        }
'''
    text=text[:position]+shortcut+text[position:]
    text+='''
#[cfg(test)]
mod aligned_sample_tests {
    use super::*;
    #[test]
    fn exact_staggered_samples_match_original_including_signed_zero_and_extremes() {
        let choices = [0.0_f32, -0.0, f32::MAX, f32::MIN, f32::MIN_POSITIVE, -f32::MIN_POSITIVE, f32::from_bits(1), -3.25, 7.5];
        for counts in [[1,1,1], [3,5,7], [16,16,16]] {
            let grid = GridGeometry::new(counts,[0.125,0.0625,0.03125],[0.;3]).unwrap();
            for axis in [Axis::X,Axis::Y,Axis::Z] {
                let dimensions = grid.face_counts(axis);
                let values: Vec<f32> = (0..grid.face_len(axis)).map(|i|choices[i%choices.len()]).collect();
                let sampler = ScalarSampler::faces(&grid,axis,&values).unwrap();
                for k in 0..dimensions[2] { for j in 0..dimensions[1] { for i in 0..dimensions[0] {
                    let position = grid.face_position(axis,[i,j,k]).unwrap();
                    assert_eq!(sampler.sample(position).unwrap().to_bits(),sampler.sample_baseline(position).unwrap().to_bits());
                }}}
            }
            let values: Vec<f32> = (0..grid.cell_len()).map(|i|choices[i%choices.len()]).collect();
            let sampler = ScalarSampler::cells(&grid,&values).unwrap();
            for k in 0..counts[2] as usize { for j in 0..counts[1] as usize { for i in 0..counts[0] as usize {
                let position = grid.cell_position([i,j,k]).unwrap();
                assert_eq!(sampler.sample(position).unwrap().to_bits(),sampler.sample_baseline(position).unwrap().to_bits());
            }}}
        }
    }
}
'''
    return text
