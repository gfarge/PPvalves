""" Computes waveforms for valve sources """

# Imports
# =======
import numpy as np


# ----------------------------------------------------------------------------
def multi_source_wvf(xyz_station, xyz_sources, dp_sources, PARAM, Vp=6500, Vs=3500):
    """
    Computes waveforms from the `dp` history at a source (valve), recorded at
    specified station.

    Parameters
    ==========
    xyz_station : array-like
        x, y, z coordinates of station (in m).
    xyz_sources : list of array-likes
        List of x, y, z coordinates of sources (in m).
    dp_sources : list of 1D arrays
        List of histories of pressure differential at source (in Pa). Should
        have the same number of `dp` as there are sources in the coordinates.
    PARAM : dict
        Dictionnary of physical parameters.
    Vp, Vs : floats
        P and S waves velocities.

    Returns
    =======
    u : 1D arrays
        Three components of displacement at station (in physical units),
        contribution from all sources.
    """
    # >> Initialize displacement
    u = np.zeros((3, len(dp_sources[0])))

    # >> Compute and add contribution of each source
    for xyz_s, dp in zip(xyz_sources, dp_sources):
        u += waveforms(xyz_station, xyz_s, dp, PARAM, Vp=Vp, Vs=Vs)

    return u

# ----------------------------------------------------------------------------
def waveforms(xyz_station, xyz_source, dp_source, PARAM, Vp=6500, Vs=3500):
    """
    Computes waveforms from the `dp` history at a source (valve), recorded at
    specified station.

    Parameters
    ==========
    xyz_station, xyz_source : array-like
        x, y, z coordinates of source and station (in m).
    dp_source : 1D array
        History of pressure differential at source (in Pa).
    PARAM : dict
        Dictionnary of physical parameters.
    Vp, Vs : floats
        P and S waves velocities.

    Returns
    =======
    u : 1D arrays
        Three components of displacement at source (in physical units).
    """
    if not isinstance(xyz_station, np.ndarray):
        xyz_station = np.array(xyz_station)
    if not isinstance(xyz_source, np.ndarray):
        xyz_source = np.array(xyz_source)

    # >> Source receiver distance
    r = np.sqrt(np.sum((xyz_source - xyz_station)**2))

    # >> Compute radiation patterns
    rp_P, _, _ = rad_pat_P_xyz(PARAM['alpha'], xyz_station[0], xyz_station[1], -xyz_source[2])
    rp_S, _, _ = rad_pat_S_xyz(PARAM['alpha'], xyz_station[0], xyz_station[1], -xyz_source[2])

    # >> Compute Force history
    F = dp_source * PARAM['A']
    F = F.reshape(1, len(F))

    # >> Compute unshifted waveforms
    uP = 1/(4*np.pi * Vp**2 * PARAM['rho_r']) * 1/r * np.dot(rp_P, F)
    uS = 1/(4*np.pi * Vs**2 * PARAM['rho_r']) * 1/r * np.dot(rp_S, F)

    for ii in range(3):
        uP[ii, :] -= uP[ii, 0]
        uS[ii, :] -= uS[ii, 0]

    # >> Using correct moveouts
    u = np.zeros((3, len(dp_source)))

    trav_time_P = r / Vp ; moveout_P = int(np.round(trav_time_P / (PARAM['dt_']*PARAM['T_scale'])))
    trav_time_S = r / Vs ; moveout_S = int(np.round(trav_time_S / (PARAM['dt_']*PARAM['T_scale'])))

    u[:, moveout_P:] += uP[:, :-moveout_P]
    u[:, moveout_S:] += uS[:, :-moveout_S]

    return u

# ----------------------------------------------------------------------------
def u2v(u, PARAM):
    """Converts displacement into velocity. Adds a zero as first element."""

    v = np.hstack([np.zeros((3, 1)), (u[:, 1:] - u[:, :-1])/(PARAM['dt_']*PARAM['T_scale'])])

    return v

# ----------------------------------------------------------------------------
def rad_pat_S(alpha, theta, phi):
    """
    Computes the S-wave radiation pattern in a given direction for a certain
    dip angle of the source force.

    Parameters
    ----------
    alpha : float
        Dip angle : between horizontal plane and channel. In radians.
    theta : float
        Azimuthal angle of station with regard to source: angle between
        projection of channel on horizontal plane (x,y) --- usually x axis ---
        and projection of source-station vector on horizontal plane. In
        radians, 0 to 2 pi.
    phi : float
        Polar angle of station with regard to source: angle from vertical
        vector z-axis to source-station vector. In radians, 0 to pi.

    Returns
    -------
    R_S : 3d numpy array
        Radiation pattern vector: `u_S_i(t) = factors * 1/r * R_S_i * |F(t)|`.

    """
    # >> Initialize useful vectors
    u_F = np.array([[np.cos(alpha)],  # force direction vector
                    [0],
                    [np.sin(alpha)]])
    gamma_line = np.array([[np.cos(theta)*np.sin(phi),
                           np.sin(theta)*np.sin(phi),
                           np.cos(phi)]])  # source-station direction vector: in line form
    gamma_column = gamma_line.reshape(3, 1)  # source-station direction vector: in column

    # >> Compute matrices
    gamma2_M = np.dot(gamma_column, gamma_line)
    D_F = np.diag(np.ones(3))
    M = D_F - gamma2_M

    # >> Compute radiation pattern
    R_S = np.dot(M, u_F)

    return R_S

# ----------------------------------------------------------------------------
def rad_pat_P(alpha, theta, phi):
    """
    Computes the P-waves radiation pattern in a given direction for a certain
    dip angle of the source force.

    Parameters
    ----------
    alpha : float
        Dip angle : between horizontal plane and channel. In radians.
    theta : float
        Azimuthal angle of station with regard to source: angle between
        projection of channel on horizontal plane (x,y) --- usually x axis ---
        and projection of source-station vector on horizontal plane. In
        radians, 0 to 2 pi.
    phi : float
        Polar angle of station with regard to source: angle from vertical
        vector z-axis to source-station vector. In radians, 0 to pi.

    Returns
    -------
    R_P : 3d numpy array
        Radiation pattern vector: `u_P_i(t) = factors * 1/r * R_P_i * |F(t)|`.

    """
    # >> Initialize useful vectors
    u_F = np.array([[np.cos(alpha)],  # force direction vector
                    [0],
                    [np.sin(alpha)]])
    gamma_line = np.array([[np.cos(theta)*np.sin(phi),
                           np.sin(theta)*np.sin(phi),
                           np.cos(phi)]])  # source-station direction vector: in line form
    gamma_column = gamma_line.reshape(3, 1)  # source-station direction vector: in column

    # >> Compute matrices
    gamma2_M = np.dot(gamma_column, gamma_line)

    # >> Compute radiation pattern
    R_P = np.dot(gamma2_M, u_F)

    return R_P

# ----------------------------------------------------------------------------
def rad_pat_P_xyz(alpha, x, y, z):
    """
    Computes the P-waves radiation pattern in a given direction for a certain
    dip angle of the source force, using cartesian coordinates. Source location
    in (0, 0, z).

    Parameters
    ----------
    alpha : float
        Dip angle : between horizontal plane and channel. In radians.
    x : float
        Horizontal distance of station from source along channel direction.
        Arbitrary units.
    y : float
        Horizontal distance of station from source across channel direction.
        Arbitrary units, same as x.
    z : float
        Source depth. Arbitrary units, same as x.

    Returns
    -------
    R_P : 3d numpy array
        Radiation pattern vector: `u_P_i(t) = factors * 1/r * R_P_i * |F(t)|`.
    theta : float
        Angle between horizontal projection of channel and horizontal
        projection of source-station vector. In radians.
    phi : float
        Angle between vertical vector and source-station vector. In radians.

    """
    d = np.sqrt(x**2 + y**2 + z**2)
    # >> Compute spherical coordinates angles
    r, theta, phi = cart2sph([x,y,z])

    # >> Compute radiation pattern
    R_P = rad_pat_P(alpha, theta, phi)

    return R_P, theta, phi

# ----------------------------------------------------------------------------
def rad_pat_S_xyz(alpha, x, y, z):
    """
    Computes the S-waves radiation pattern in a given direction for a certain
    dip angle of the source force, using cartesian coordinates. Source location
    in (0, 0, z).

    Parameters
    ----------
    alpha : float
        Dip angle : between horizontal plane and channel. In radians.
    x : float
        Horizontal distance of station from source along channel direction.
        Arbitrary units.
    y : float
        Horizontal distance of station from source across channel direction.
        Arbitrary units, same as x.
    z : float
        Source depth. Arbitrary units, same as x. Always positive.

    Returns
    -------
    R_S : 3d numpy array
        Radiation pattern vector: `u_S_i(t) = factors * 1/r * R_S_i * |F(t)|`.
    theta : float
        Angle between horizontal projection of channel and horizontal
        projection of source-station vector. In radians.
    phi : float
        Angle between vertical vector and source-station vector. In radians.

    """
    # >> Convert depth to correct system of coordinates...
    z *= -1

    # >> Compute spherical coordinates angles
    r, theta, phi = cart2sph([x,y,z])

    # >> Compute radiation pattern
    R_S = rad_pat_S(alpha, theta, phi)

    return R_S, theta, phi

# ----------------------------------------------------------------------------
def cart2sph(xyz):
    """
    Converts cartesian coordinates to spherical coordinates.

    Parameter
    ---------
    xyz : array-like
        3D cartesian coordinates, x,y,z, correctly oriented.

    Returns
    -------
    r : float
    theta : float
        Azimuthal angle: angle from x axis to xy, projection of the
        origin-point vector on horizontal plane (x,y).
    phi : float
        Polar angle: angle from vertical z axis to origin-point vector.

    Note
    ----
        Code from: https://stackoverflow.com/questions/4116658/faster-numpy-cartesian-to-spherical-coordinate-conversion
    """
    xy2 = xyz[0]**2 + xyz[1]**2
    r = np.sqrt(xy2 + xyz[2]**2)

    theta = np.arctan2(xyz[1], xyz[0])
    phi = np.arctan2(np.sqrt(xy2), xyz[2])

    return r, theta, phi

