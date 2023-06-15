from polaris import TestCase
from polaris.ocean.tests.manufactured_solution.analysis import Analysis
from polaris.ocean.tests.manufactured_solution.forward import Forward
from polaris.ocean.tests.manufactured_solution.initial_state import (
    InitialState,
)
from polaris.ocean.tests.manufactured_solution.viz import Viz
from polaris.validate import compare_variables


class TimeConvergence(TestCase):
    """
    The temporal convergence test case for the manufactured solution test group

    Attributes
    ----------
    timesteps : list of floats
        The timesteps of the test case in min. The last is used as the
        reference solution for calculating errors in the analysis and viz steps
    """
    def __init__(self, test_group):
        """
        Create the test case

        Parameters
        ----------
        test_group : polaris.ocean.tests.manufactured_solution.
                     ManufacturedSolution
            The test group that this test case belongs to
        """
        name = 'time_convergence'
        super().__init__(test_group=test_group, name=name)

        self.add_step(InitialState(test_case=self, resolution=200))
        self.timesteps = [20.0, 10.0, 5.0, 2.5, 0.25]
        for dt in self.timesteps:
            self.add_step(Forward(test_case=self,
                                  resolution=200, timestep=dt))

        self.add_step(Analysis(test_case=self, resolutions=[200],
                               timesteps=self.timesteps))
        self.add_step(Viz(test_case=self, resolutions=[200],
                          timesteps=self.timesteps),
                      run_by_default=False)

    def validate(self):
        """
        Compare ``layerThickness`` and
        ``normalVelocity`` in the ``forward`` step with a baseline if one was
        provided.
        """
        super().validate()
        variables = ['layerThickness', 'normalVelocity']
        for dt in self.timesteps:
            compare_variables(test_case=self, variables=variables,
                              filename1=f'dt_{dt}/forward/output.nc')
