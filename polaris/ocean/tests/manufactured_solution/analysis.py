import datetime
import warnings

import cmocean  # noqa: F401
import numpy as np
import xarray as xr

from polaris import Step
from polaris.ocean.tests.manufactured_solution.exact_solution import (
    ExactSolution,
)


class Analysis(Step):
    """
    A step for analysing the output from the manufactured solution
    test case

    Attributes
    ----------
    resolutions : list of int
        The resolutions of the meshes that have been run

    timesteps : list of float
        The timesteps for the forward steps that have been run

    convergence_type : str
        Type of convergece study run
    """
    def __init__(self, test_case, resolutions=None, timesteps=None):
        """
        Create the step

        Parameters
        ----------
        test_case : polaris.TestCase
            The test case this step belongs to

        resolutions : list of int
            The resolutions of the meshes that have been run

        timesteps : list of float
            The timesteps for the forward steps that have been run
        """
        super().__init__(test_case=test_case, name='analysis')
        self.resolutions = resolutions
        self.timesteps = timesteps

        if timesteps:
            # Convergence study in time
            self.convergence_type = 'time'
            for timestep in timesteps:
                self.add_input_file(
                    filename='initial_state.nc',
                    target=f'../{resolutions[0]}km/initial_state/'
                           'initial_state.nc')
                self.add_input_file(
                    filename=f'output_dt_{timestep}.nc',
                    target=f'../dt_{timestep}/forward/output.nc')
        else:
            # Convergence study in space
            self.convergence_type = 'space'
            for resolution in resolutions:
                self.add_input_file(
                    filename=f'init_{resolution}km.nc',
                    target=f'../{resolution}km/initial_state/initial_state.nc')
                self.add_input_file(
                    filename=f'output_{resolution}km.nc',
                    target=f'../{resolution}km/forward/output.nc')

    def run(self):
        """
        Run this step of the test case
        """
        config = self.config
        resolutions = self.resolutions
        timesteps = self.timesteps

        section = config['manufactured_solution']
        conv_thresh = section.getfloat('conv_thresh')
        conv_max = section.getfloat('conv_max')

        if self.convergence_type == 'time':
            runs = timesteps[:-1]
            output_name = 'output_dt_{run}.nc'
            ds_ref = xr.open_dataset(f'output_dt_{timesteps[-1]}.nc')
            init = xr.open_dataset('initial_state.nc')
            ssh_ref = ds_ref.ssh[-1, :]
        else:
            runs = resolutions
            output_name = 'output_{run}km.nc'
            ds = xr.open_dataset(f'output_{runs[0]}km.nc')
            t0 = datetime.datetime.strptime(ds.xtime.values[0].decode(),
                                            '%Y-%m-%d_%H:%M:%S')
            tf = datetime.datetime.strptime(ds.xtime.values[-1].decode(),
                                            '%Y-%m-%d_%H:%M:%S')
            t = (tf - t0).total_seconds()

        rmse = []
        for i, run in enumerate(runs):
            ds = xr.open_dataset(eval(f"f'{output_name}'"))

            if self.convergence_type == 'space':
                init = xr.open_dataset(f'init_{run}km.nc')
                exact = ExactSolution(config, init)
                ssh_ref = exact.ssh(t).values

            ssh_model = ds.ssh.values[-1, :]
            rmse.append(np.sqrt(np.mean((ssh_model - ssh_ref)**2)))

        p = np.polyfit(np.log10(runs), np.log10(rmse), 1)

        conv = p[0]

        if conv < conv_thresh:
            raise ValueError(f'order of convergence '
                             f' {conv} < min tolerence {conv_thresh}')

        if conv > conv_max:
            warnings.warn(f'order of convergence '
                          f'{conv} > max tolerence {conv_max}')
