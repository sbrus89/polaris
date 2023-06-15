import datetime

import cmocean  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from polaris import Step
from polaris.ocean.tests.manufactured_solution.exact_solution import (
    ExactSolution,
)
from polaris.viz import plot_horiz_field


class Viz(Step):
    """
    A step for visualizing the output from the manufactured solution
    test case

    Attributes
    ----------
    resolutions : list of int
        The resolutions of the meshes that have been run

    timesteps : list of float
        The timesteps for the forward steps that have been run

    convergence_type : str
        Type of convergence study run
    """
    def __init__(self, test_case, resolutions=None, timesteps=None):
        """
        Create the step

        Parameters
        ----------
        test_case : polaris.ocean.tests.manufactured_solution.convergence.Convergence # noqa: E501
            The test case this step belongs to

        resolutions : list of int
            The resolutions of the meshes that have been run

        timesteps : list of float
            The timesteps for the forward steps that have been run
        """
        super().__init__(test_case=test_case, name='viz')
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

        self.add_output_file('convergence.png')

    def run(self):
        """
        Run this step of the test case
        """
        plt.switch_backend('Agg')
        config = self.config
        resolutions = self.resolutions
        timesteps = self.timesteps

        section = config['manufactured_solution']
        eta0 = section.getfloat('ssh_amplitude')

        if self.convergence_type == 'time':
            runs = timesteps[:-1]
            ref_orders = {'first': [1, '--'],
                          'second': [2, '-.'],
                          'fourth': [4, '-']}
            conv_xlabel = 'timestep (min)'
            conv_title = 'Temporal Convergence'
            comp_ref_title = 'Reference Solution'
            comp_error_title = 'Error (Numerical - Reference)'
            output_name = 'output_dt_{run}.nc'
            ds_ref = xr.open_dataset(f'output_dt_{timesteps[-1]}.nc')
            init = xr.open_dataset('initial_state.nc')
            ssh_ref = ds_ref.ssh[-1, :]
        else:
            runs = resolutions
            ref_orders = {'first': [1, '--'], 'second': [2, '-.']}
            conv_xlabel = 'resolution (km)'
            conv_title = 'Spatial Error Convergence'
            comp_ref_title = 'Analytical Solution'
            comp_error_title = 'Error (Numerical - Analytical)'
            output_name = 'output_{run}km.nc'
            ds = xr.open_dataset(f'output_{runs[0]}km.nc')

            t0 = datetime.datetime.strptime(ds.xtime.values[0].decode(),
                                            '%Y-%m-%d_%H:%M:%S')
            tf = datetime.datetime.strptime(ds.xtime.values[-1].decode(),
                                            '%Y-%m-%d_%H:%M:%S')
            t = (tf - t0).total_seconds()

        nrows = len(runs)
        fig, axes = plt.subplots(nrows=nrows, ncols=3, figsize=(12, 2 * nrows))
        rmse = []
        for i, run in enumerate(runs):

            # Error computation
            ds = xr.open_dataset(eval(f"f'{output_name}'"))
            ssh_model = ds.ssh.values[-1, :]

            if self.convergence_type == 'space':
                init = xr.open_dataset(f'init_{run}km.nc')
                exact = ExactSolution(config, init)
                ssh_ref = exact.ssh(t)
            rmse.append(np.sqrt(np.mean((ssh_model - ssh_ref.values)**2)))

            # Comparison plots
            ds['ssh_exact'] = ssh_ref
            ds['ssh_error'] = ssh_model - ssh_ref

            if i == 0:
                error_range = np.max(np.abs(ds.ssh_error.values))

            plot_horiz_field(ds, init, 'ssh', ax=axes[i, 0],
                             cmap='cmo.balance', t_index=ds.sizes["Time"] - 1,
                             vmin=-eta0, vmax=eta0, cmap_title="SSH")
            plot_horiz_field(ds, init, 'ssh_exact', ax=axes[i, 1],
                             cmap='cmo.balance',
                             vmin=-eta0, vmax=eta0, cmap_title="SSH")
            plot_horiz_field(ds, init, 'ssh_error', ax=axes[i, 2],
                             cmap='cmo.balance', cmap_title="dSSH",
                             vmin=-error_range, vmax=error_range)

        axes[0, 0].set_title('Numerical solution')
        axes[0, 1].set_title(comp_ref_title)
        axes[0, 2].set_title(comp_error_title)

        pad = 5
        for ax, res in zip(axes[:, 0], resolutions):
            ax.annotate(f'{res}km', xy=(0, 0.5),
                        xytext=(-ax.yaxis.labelpad - pad, 0),
                        xycoords=ax.yaxis.label, textcoords='offset points',
                        size='large', ha='right', va='center')

        fig.savefig('comparison.png', bbox_inches='tight', pad_inches=0.1)

        # Convergence plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        p = np.polyfit(np.log10(runs), np.log10(rmse), 1)
        conv = np.round(p[0], 3)
        ax.loglog(runs, rmse, '-ok', label=f'numerical (order={conv})')

        for order in ref_orders:
            p = ref_orders[order][0]
            marker = ref_orders[order][1]
            c = rmse[0] * 1.5 / runs[0]**p
            ref_error = c * np.power(runs, p)
            ax.loglog(runs, ref_error, f'{marker}k',
                      label=f'{order} order', alpha=0.3)

        ax.set_xlabel(conv_xlabel)
        ax.set_ylabel('RMS error (m)')
        ax.set_title(conv_title)
        ax.invert_xaxis()
        ax.legend(loc='lower left')
        fig.savefig('convergence.png', bbox_inches='tight', pad_inches=0.1)
