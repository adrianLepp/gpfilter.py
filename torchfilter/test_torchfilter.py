from test_filters import _run_filter
import torchfilter
from _linear_system_models import LinearDynamicsModel, LinearParticleFilterMeasurementModel
import torch
from _linear_system_fixtures import generated_data
'''
TODO:
- generate data: Tuple[
        types.StatesTorch, types.ObservationsNoDictTorch, types.ControlsNoDictTorch
    ], 
    Shapes of all inputs should be `(T, N, *)`.
- create dynamics_model equal to LinearDynamicsModel(),
- create measurement_model equal to LinearParticleFilterMeasurementModel(),

'''

def test_particle_filter(generated_data):
    """Smoke test for particle filter."""
    _run_filter(
        torchfilter.filters.ParticleFilter(
            dynamics_model=LinearDynamicsModel(),
            measurement_model=LinearParticleFilterMeasurementModel(),
        ),
        generated_data,
    )


data = generated_data()
test_particle_filter(data)