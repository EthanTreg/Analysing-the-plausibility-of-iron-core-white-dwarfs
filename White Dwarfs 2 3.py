import os
import pickle
import numpy as np
from time import time
from typing import Tuple, List, Union
from matplotlib import pyplot as plt
from matplotlib.colors import Colormap
from scipy.integrate import solve_ivp


def progress_bar(i: int, total: int):
    """
    Terminal progress bar

    Parameters
    ----------
    i : int
        Current progress
    total : int
        Completion number
    """
    length = 50
    i += 1

    filled = int(i * length / total)
    percent = i * 100 / total
    bar = '█' * filled + '-' * (length - filled)
    print('Progress: |{}| {:.1f}%\t'.format(bar, percent), end='\r')

    if i == total:
        print()


def gamma(y: np.ndarray) -> np.ndarray:
    """
    Gamma function for variable y

    Parameters
    ----------
    y : ndarray
        Input values

    Returns
    -------
    gamma : ndarray
        Output values
    """
    return y ** (2 / 3) / (3 * (1 + y ** (2 / 3)) ** (1 / 2))


def state_equation(x: float, state: Tuple[float, float]) -> Tuple[float, float]:
    """
    Differential equations for y (density) and z (mass) with respect to x (radius)

    Parameters
    ----------
    x : float
        Input value
    state : tuple
        Values for y and z

    Returns
    -------
    dy : float
        Small step in y
    dz : float
        Small step in z
    """
    y, z = state

    # Prevents error for unreal negative y values
    if y > 0:
        dy = -z * y / (gamma(y) * x ** 2)
        dz = 3 * y * x ** 2

        return dy, dz
    else:
        return None, None


def white_dwarfs(resolution: int, A: np.ndarray, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the radii, densities, masses and x_max for a range of white dwarf stars

    Parameters
    ----------
    resolution : int
        Number of white dwarf stars to calculate
    A : ndarray
        Atomic mass of each core
    Z : ndarray
        Atomic number of each core

    Returns
    -------
    radius : ndarray
        Radii for each white dwarf and core type in solar radius
    density : ndarray
        Densities for each white dwarf and core type in solar density
    mass : ndarray
        Masses for each white dwarf and core type in solar mass
    x_max : ndarray
        Maximum x value for each white dwarf
    """
    # Constants
    structure_resolution = 1000
    h_bar = 6.62607015e-34 / (2 * np.pi)
    G = 6.6743e-11
    mp = 1.6726231e-27
    me = 9.1093897e-31
    c = 2.99792458e8

    Ye = Z / A
    r0 = np.sqrt(9 * np.pi * h_bar ** 3 * Ye ** 2 / (4 * G * mp ** 2 * me ** 2 * c))
    p0 = mp * me ** 3 * c ** 3 / (3 * np.pi ** 2 * h_bar ** 3 * Ye)
    m0 = 4 * r0 ** 3 * mp * me ** 3 * c ** 3 / (9 * np.pi * h_bar ** 3 * Ye)
    x_max = np.expand_dims(np.array((2.9)), axis=0)
    radius = np.empty((0, structure_resolution, A.size))
    density = np.empty((0, structure_resolution, A.size))
    mass = np.empty((0, structure_resolution, A.size))
    p_core = np.logspace(-1.7, 11, resolution)
    x_min = np.logspace(-2, -6, resolution)

    # Loop through each white dwarf star
    for i in range(resolution):
        result = white_dwarf_calculation([x_min[i], x_max[-1]], p_core[i], structure_resolution, r0, p0, m0)

        radius = np.append(radius, result[0][None, ...], axis=0)
        density = np.append(density, result[1][None, ...], axis=0)
        mass = np.append(mass, result[2][None, ...], axis=0)
        x_max = np.append(x_max, result[-1])

        progress_bar(i, resolution)

    return radius, density, mass, x_max


def white_dwarf_calculation(x_span: List[float], p_core: float, resolution: int, r0: float, p0: float, m0: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Calculates the radii, densities and masses for a white dwarf star and the maximum x value for integration

    Parameters
    ----------
    x_span : List[float]
        Initial and final values for integration
    p_core : float
        Core density of white dwarf
    resolution : int
        Number of data points to save
    r0 : float
        Radius scale factor for x
    p0 : float
        Density scale factor for y
    m0 : float
        Mass scale factor for z

    Returns
    -------
    radius : ndarray
        Radii of white dwarf in solar radius
    density : ndarray
        Densities of white dwarf for given radii in solar density
    mass : ndarray
        Masses of white dwarf for given radii in solar mass
    x_max : float
        Upper limit of integration for x
    """
    # Constants
    j = 0
    scaler = 0.9
    m_solar = 1.98847e30
    r_solar = 696.34e6
    p_solar = 3 * m_solar / (4 * np.pi * r_solar ** 3)

    x_eval = np.linspace(x_span[0], x_span[1], resolution)

    # First solve attempt
    result = solve_ivp(state_equation, t_span=x_span, y0=[p_core, 1e-6], t_eval=x_eval)

    # Find largets integration upper limit to be successful after j runs
    while not(result.success) or j < 10:
        j += 1
        x_span[1] *= scaler
        x_eval = np.linspace(x_span[0], x_span[1], resolution)
        result = solve_ivp(state_equation, t_span=x_span, y0=[p_core, 1e-6], t_eval=x_eval)

        scaler_change = 0.5 * abs(1 - scaler)

        if result.success:
            scaler = 1 + scaler_change
        else:
            scaler = 1 - scaler_change

    # Calculate radii, densities and masses for all core types in solar units
    radius = np.outer(result.t, r0 / r_solar)
    density = np.outer(result.y[0], p0 / p_solar)
    mass = np.outer(result.y[1], m0 / m_solar)

    return radius, density, mass, x_span[1]


def white_dwarf_analysis(radius: np.ndarray, density: np.ndarray, mass: np.ndarray, wd_radii: np.ndarray, wd_masses: np.ndarray, samples=4, core_type: List[str]=[None]):
    """
    Creates all plots that show individual white dwarf stars
    If core_type is None, then individual plots will be made for each star, else, all white dwarfs will be plotted on one mass-radius graph with overall trend

    Parameters
    ----------
    radius : ndarray
        Radii for each white dwarf and core type in solar radius
    density : ndarray
        Densities for each white dwarf and core type in solar density
    mass : ndarray
        Masses for each white dwarf and core type in solar mass
    wd_radii : ndarray
        Radius for each white dwarf and core type in solar radius
    wd_mass : ndarray
        Mass for each white dwarf and core type in solar mass
    sample : int, default = 4
        Number of white dwarfs to plot
    core_type : List[str], default = [None]
        Names of core types, None for group plot
    """
    # Constants
    minor = 16
    major = 20
    letter = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    sample = np.linspace(1.25 * np.min(wd_masses[:, 0]), 0.975 * np.max(wd_masses[:, 0]), samples)
    sample = np.argmin(np.abs(wd_masses[:, 0] - sample[:, None]), axis=1)
    iron_sample = np.argmin(np.abs(wd_radii[None, :, 1] - wd_radii[sample, None, 0]), axis=1)

    # If individual or group plot depending if core type is specified
    if core_type[0] != None:
        _, ax = plt.subplots(int(samples / 2), 2, figsize=(4 * samples, 16), constrained_layout=True)
        ax = ax.flatten()

        # Plot mass-radius relationship for each sampled white dwarf and core type
        for i, (wd_i, iron_i) in enumerate(zip(sample, iron_sample)):
            radii = np.vstack((radius[wd_i, :, 0], radius[iron_i, :, 1])).swapaxes(0, 1)
            masses = np.vstack((mass[wd_i, :, 0], mass[iron_i, :, 1])).swapaxes(0, 1)

            ax[i].set_title(r'{}) {:.2f} $M_\odot$ White Dwarf'.format(letter[i], np.max(mass[wd_i, :, 0])), fontsize=major)
            structure_plot(ax[i], radii, masses, x_label=r'Radius ($R_\odot$)', y_label=r'Mass ($M_\odot$)', label=core_type)
            fraction_axis(ax[i], np.max(mass[wd_i, :, 0]), 'Mass Fraction')

        plt.savefig(f'Project 2/Figures/WD Structure Mass-Radius Relationship')

        _, ax = plt.subplots(int(samples / 2), 2, figsize=(4 * samples, 16), constrained_layout=True)
        ax = ax.flatten()

        # Plot density-radius relationship for each sampled white dwarf and core type
        for i, (wd_i, iron_i) in enumerate(zip(sample, iron_sample)):
            radii = np.vstack((radius[wd_i, :, 0], radius[iron_i, :, 1])).swapaxes(0, 1)
            densities = np.vstack((density[wd_i, :, 0], density[iron_i, :, 1])).swapaxes(0, 1)

            ax[i].set_title(r'{}) {:.2f} $M_\odot$ White Dwarf'.format(letter[i], np.max(mass[wd_i, :, 0])), fontsize=major)
            structure_plot(ax[i], radii, densities, x_label=r'Radius ($R_\odot$)', y_label=r'Density ($\rho_\odot$)', label=core_type)
            fraction_axis(ax[i], np.max(density[wd_i, :, 0]), 'Density Fraction')

        plt.savefig(f'Project 2/Figures/WD Structure Density-Radius Relationship')
    # Group plot
    else:
        fig = plt.figure(figsize=(16, 9), constrained_layout=True)
        ax = fig.gca()

        ax.plot(np.max(radius[..., 0], axis=1), wd_masses[:, 0], c='k', label='Carbon-Oxygen Model')
        ax.legend(fontsize=minor)

        for i in sample:
            structure_plot(ax, radius[i, :, 0], mass[i, :, 0], x_label=r'Radius ($R_\odot$)', y_label=r'Mass ($M_\odot$)')

        plt.savefig('Project 2/Figures/WD Structure Overview')


def structure_plot(ax: plt.Axes, x: np.ndarray, y: np.ndarray, x_label: str, y_label: str, label: List[str]=None):
    """
    Plots the internal structure of a white dwarf star
    If x data is 2D, labels need to be specified for length of dimension 2

    Parameters
    ----------
    ax : axes object
        Axes to be plotted
    x : ndarray
        x data
    y : ndarray
        y data
    x_label : str
        x-axis label
    y_label : str
        y-axis label
    label : List[str], default = None
        Labels for each white dwarf if x is 2D
    """
    minor = 16

    # Plot each white dwarf with a label if x data is 2D, else, plot without label
    if len(x.shape) != 1:
        for i in range(x.shape[1]):
            ax.plot(x[:, i], y[:, i], label=label[i])
    else:
        ax.plot(x, y)

    # Plot format
    ax.set_xlabel(x_label, fontsize=minor)
    ax.set_ylabel(y_label, fontsize=minor)
    ax.tick_params(axis='both', labelsize=minor)
    ax.yaxis.offsetText.set_fontsize(minor)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    ax.legend(fontsize=minor)


def fraction_axis(ax: plt.Axes, maximum: float, label: str):
    """
    Adds a second y-axis for fractional data

    Parameters
    ----------
    ax : axes object
        Axes to be edited
    maximum : float
        Maximum values of data
    label : str
        y-axis label
    """
    minor = 16

    ax2 = ax.twinx()
    limits = ax.get_ylim()
    ax2.set_ylim(limits[0] / maximum, limits[1] / maximum)
    ax2.set_ylabel(label, fontsize=minor)
    ax2.tick_params(axis='y', labelsize=minor)


def white_dwarf_relationship(radius: np.ndarray, density: np.ndarray, mass: np.ndarray, core_type: List[str]):
    """
    Plots overall relationships for white dwarfs and plots white dwarf data

    radius : ndarray
        Radius for each white dwarf and core type in solar radius
    density : ndarray
        Density for each white dwarf and core type in solar density
    mass : ndarray
        Mass for each white dwarf and core type in solar mass
    core_type : List[str]
        Names of each core type
    """
    # Stellar data file
    data = np.loadtxt('Project 2/Data/Stars.csv', dtype=str, delimiter=',', skiprows=1)

    # Assign file data to arrays
    star_labels = data[:, 0]
    star_radii = data[:, 1:3].astype(float)
    star_masses = data[:, 3:].astype(float)
    no_sdss_count = np.count_nonzero(star_labels != 'SDSS')

    colours = plt.cm.get_cmap('brg', np.ceil(no_sdss_count / 2))

    # Plot graphs
    white_dwarf_plot(radius, mass, core_type, density=density)
    white_dwarf_plot(radius, mass, core_type, star_radii[:no_sdss_count], star_masses[:no_sdss_count], c=colours, labels=star_labels[:no_sdss_count], format=['o', '^'], data_name=' Provencal')
    white_dwarf_plot(radius, mass, core_type, star_radii[no_sdss_count:], star_masses[no_sdss_count:], c='k', labels=['SDSS'], data_name=' SDSS')


def white_dwarf_plot(radius: np.ndarray, mass: np.ndarray, core_type: List[str], star_radii: np.ndarray=None, star_masses: np.ndarray=None, density: np.ndarray=None, c: Union[Colormap, str]=None, labels: List[str]=None, format: str='o', data_name: str=''):
    """
    Plots mass-radius relationship
    If density is not None, plot density-radius relationship
    If labels is not None, plot white dwarf data points
    If labels is singular, only plot one label and one color for all data points

    Parameters
    ----------
    radius : ndarray
        Radius for each white dwarf and core type in solar radius
    mass : ndarray
        Mass for each white dwarf and core type in solar mass
    core_type : List[str]
        Name of each core type
    star_radii : ndarray, default = None
        Radius of each white dwarf data point
    star_masses : ndarray, default = None
        Mass of each white dwarf data point
    density : ndarray, default = None
        Density for each white dwarf and core type in solar density
    c : Union[Colormap, str], default = None
        Color of data point, Colormap type for multiple labels or str for singular label
    labels: List[str], default = None
        Individual Labels or singular label for white dwarf data points
    format : str, default = 'o'
        Data point format, if multiple provided, divides point types evenly
    data_name : str, default = ''
        Name for unique file names
    """
    # Constants
    minor = 16
    major = 20
    ax2 = None

    # If density array is None, only plot mass-radius relationship, else, plot density-radius and mass-radius relationship
    if density is None:
        _, ax = plt.subplots(1, 1, figsize=(16, 9), sharex=True, constrained_layout=True)
    else:
        _, (ax2, ax) = plt.subplots(2, 1, figsize=(16, 9), sharex=True, constrained_layout=True)

    # Plot density-radius, if not None, and mass-radius relationship for each core type
    for i in range(len(core_type)):
        if ax2 is not None:
            ax2.plot(radius[:, i], density[:, i], label=core_type[i] + ' Core')

        ax.plot(radius[:, i], mass[:, i], label=core_type[i])

    # Plot each white dwarf data point if labels is not None
    if labels is not None:
        for i in range(star_radii.shape[0]):
            # If there are multiple labels, assign unique label to each data point, else, only assign label to first data point
            if len(labels) != 1:
                ax.errorbar(star_radii[i, 0], star_masses[i, 0], xerr=star_radii[i, 1], yerr=star_masses[i, 1], color=c(i // 2), label=labels[i], fmt=format[i % 2], capsize=5)
            elif i != 0:
                ax.errorbar(star_radii[i, 0], star_masses[i, 0], xerr=star_radii[i, 1], yerr=star_masses[i, 1], c=c, fmt='o', capsize=5)
            else:
                ax.errorbar(star_radii[0, 0], star_masses[0, 0], xerr=star_radii[0, 1], yerr=star_masses[0, 1], c=c, label=labels[0], fmt='o', capsize=5)

        avg_radius = np.average(star_radii[:, 0], weights=1 / np.sqrt(star_radii[:, 1]))
        radius_error = 1 / np.sqrt(np.sum(1 / star_radii[:, 1] ** 2))
        avg_mass = np.average(star_masses[:, 0], weights=1 / np.sqrt(star_masses[:, 1]))
        mass_error = 1 / np.sqrt(np.sum(1 / star_masses[:, 1] ** 2))
        ax.errorbar(avg_radius, avg_mass, xerr=radius_error, yerr=mass_error, c='#C71DEB', markersize=20, label='Average value', fmt='x', capsize=5)

    # Format second axis and add titles if density-radius relationship is not None
    if ax2 is not None:
        ax2.set_title('a) Density-Radius Relationship', fontsize=major)
        ax.set_title('b) Mass-Radius Relationship', fontsize=major)

        ax2.set_yscale('log')
        ax2.set_ylabel(r'Core Density ($\rho_\odot$)', fontsize=minor)
        ax2.tick_params(axis='both', labelsize=minor)
        ax2.legend(fontsize=minor)

    # Format mass-radius relationship
    ax.set_xlabel(r'Radius ($R_\odot$)', fontsize=minor)
    ax.set_ylabel(r'Mass ($M_\odot$)', fontsize=minor)
    ax.tick_params(axis='both', labelsize=minor)
    ax.legend(fontsize=minor, ncol=3)

    plt.savefig('Project 2/Figures/WD Relationships' + data_name)


def main():
    # Parameters
    load = True
    save = False

    # Constants
    resolution = 200
    file = 'Project 2/Data/Results'
    ti = time()
    core_type = ['Carbon-Oxygen', 'Iron']
    A = np.array((12, 55.845))
    Z = np.array((6, 26))

    file_directory = '/'.join(file.split('/')[:-1])

    # Create directory if it doesn't exist
    if not(os.path.exists(file_directory)):
        os.mkdir(file_directory)

    # Load saved file if load is True, else recalculate white dwarf data
    if load:
        with open(file, 'rb') as f:
            results = pickle.load(f)

        radius = results['Radius']
        density = results['Density']
        mass = results['Mass']
        x_max = results['x max']
    else:
        radius, density, mass, x_max = white_dwarfs(resolution, A, Z)

    # Overall white dwarf properties
    wd_radii = np.max(radius, axis=1)
    wd_densities = np.max(density, axis=1)
    wd_masses = np.max(mass, axis=1)

    # Plot overall relationships
    white_dwarf_relationship(wd_radii, wd_densities, wd_masses, core_type)

    # Plot individual white dwarfs
    white_dwarf_analysis(radius, density, mass, wd_radii, wd_masses, core_type=core_type)
    white_dwarf_analysis(radius, density, mass, wd_radii, wd_masses, samples=100)

    # Save data if save is True
    if save:
        results = {
            'Radius': radius,
            'Density': density,
            'Mass': mass,
            'x max': x_max,
        }

        with open(file, 'wb') as f:
            pickle.dump(results, f)

    print(f'\nTime: {round(time() - ti, 1)} s\nCarbon-Oxygen Chandrasekhar Mass: {round(np.max(wd_masses[:, 0]), 3)} M Solar\nIron Chandrasekhar Mass: {round(np.max(wd_masses[:, 1]), 3)} M Solar')


main()
