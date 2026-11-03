"""Fine-time pulse/motion inputs with the existing spatial-wave bound retained."""
from continuous_force_program import (
    SUBSTEPS, evaluate, sample_scene, frequency_program, frequency_scene as _frequency_scene,
)


def frequency_scene(*args, **kwargs):
    scene = _frequency_scene(*args, **kwargs)
    # The native spatial-wave base is a separate parameter with an existing 4-Hz cap.
    # Broad temporal forcing is supplied by explicit pulse/motion controls instead.
    scene['frequency'] = min(scene['frequency'], 4.)
    scene['forcing_program']['spatial_wave_temporal_base_hz'] = scene['frequency']
    scene['forcing_program']['spatial_wave_temporal_components_hz'] = [
        scene['frequency'] * v for v in (1, 2, 3)]
    return scene
