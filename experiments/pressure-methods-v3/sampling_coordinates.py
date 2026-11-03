"""Share world-to-grid coordinates across the three staggered velocity samples."""
def patch(text):
    start=text.index('    pub fn sample(&self, position: [f64; 3]) -> Result<f64, SamplingError> {')
    end=text.index('\n}\n\npub struct VelocitySampler',start)
    original=text[start:end]
    before='''        let origin = self.grid.origin();
        let upper = self.grid.upper();
        let spacing = self.grid.spacing();'''
    coordinate='''        let origin = self.grid.origin();
        let upper = self.grid.upper();
        let spacing = self.grid.spacing();
        let coordinates = std::array::from_fn(|d| (position[d].clamp(origin[d], upper[d])-origin[d])/spacing[d]);
        self.sample_coordinates(coordinates)
    }
    fn sample_coordinates(&self, coordinates: [f64; 3]) -> Result<f64, SamplingError> {'''
    changed=original.replace(before,coordinate)
    changed=changed.replace('''            let x = position[d].clamp(origin[d], upper[d]);
            let q = ((x - origin[d]) / spacing[d] - self.offset[d])''','''            let q = (coordinates[d] - self.offset[d])''')
    if changed==original:raise ValueError('Sampling seam changed')
    text=text[:start]+changed+'\n    #[cfg(test)]\n'+original.replace('pub fn sample(', 'fn sample_baseline(',1)+text[end:]
    old='''        Ok([
            self.components[0].sample(position)?,
            self.components[1].sample(position)?,
            self.components[2].sample(position)?,
        ])'''
    new='''        if position.iter().any(|x| !x.is_finite()) { return Err(SamplingError::NonFinitePosition); }
        let origin = self.grid.origin();
        let upper = self.grid.upper();
        let spacing = self.grid.spacing();
        let coordinates = std::array::from_fn(|d| (position[d].clamp(origin[d], upper[d])-origin[d])/spacing[d]);
        Ok([
            self.components[0].sample_coordinates(coordinates)?,
            self.components[1].sample_coordinates(coordinates)?,
            self.components[2].sample_coordinates(coordinates)?,
        ])'''
    if text.count(old)!=1:raise ValueError('Velocity sample seam changed')
    text=text.replace(old,new)
    text+='''
#[cfg(test)]
mod shared_coordinates_tests {
    use super::*;
    #[test]
    fn shared_coordinates_match_original_staggered_samples_bitwise() {
        for counts in [[1,1,1], [3,5,7], [16,16,16]] {
            let grid = GridGeometry::new(counts, [0.125,0.2,0.3], [-0.5,0.2,-0.1]).unwrap();
            let axes = [Axis::X,Axis::Y,Axis::Z];
            let values: [Vec<f32>;3] = std::array::from_fn(|d| (0..grid.face_len(axes[d])).map(|i| ((i*73%997) as f32-498.)/37.).collect());
            let velocity = VelocitySampler::new(&grid, [&values[0],&values[1],&values[2]]).unwrap();
            let cells: Vec<f32> = (0..grid.cell_len()).map(|i| (i*31%127)as f32/17.).collect();
            let scalar = ScalarSampler::cells(&grid,&cells).unwrap();
            for index in 0..1001 {
                let position = std::array::from_fn(|d| grid.origin()[d]+((index*(17+d*4)%1001) as f64/500.-0.5)*(grid.upper()[d]-grid.origin()[d]));
                assert_eq!(scalar.sample(position).unwrap().to_bits(),scalar.sample_baseline(position).unwrap().to_bits());
                let shared = velocity.sample(position).unwrap();
                for (d,value) in shared.iter().enumerate() {
                    assert_eq!(value.to_bits(),velocity.components[d].sample_baseline(position).unwrap().to_bits());
                }
            }
            for position in [[-1e300;3],[1e300;3],[f64::NAN;3],[f64::INFINITY;3]] {
                assert_eq!(scalar.sample(position),scalar.sample_baseline(position));
                let expected = velocity.components[0].sample_baseline(position);
                if expected.is_err() { assert_eq!(velocity.sample(position).unwrap_err(),expected.unwrap_err()); }
            }
        }
    }
}
'''
    return text
