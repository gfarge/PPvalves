""" Computes waveforms for valve sources """

# Imports
# =======
import numpy as np


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
    rp_P = rad_pat_P_xyz(PARAM['alpha'], xyz_station, xyz_source)
    rp_S = rad_pat_S_xyz(PARAM['alpha'], xyz_station, xyz_source)

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
def u2v(u, PARAM):
    """Converts displacement into velocity. Adds a zero as first element."""

    v = np.hstack([np.zeros((3, 1)), (u[:, 1:] - u[:, :-1])/(PARAM['dt_']*PARAM['T_scale'])])

    return v

# ----------------------------------------------------------------------------
def rad_pat_P_xyz(alpha, xyz_station, xyz_source):
    """
    Computes the P-waves radiation pattern in a given direction for a certain
    dip angle of the source force, using cartesian coordinates.

    Parameters
    ----------
    alpha : float
        Dip angle : between horizontal plane and channel. In radians.
    xyz_station : array-like
        x, y, z coordinates of station (in m).
    xyz_sources : list of array-likes
        List of x, y, z coordinates of sources (in m).

    Returns
    -------
    R_S : 3d numpy array
        Radiation pattern vector: `u_S_i(t) = factors * 1/r * R_S_i * |F(t)|`.

    """
    # >> Make coordinates into arrays
    if not isinstance(xyz_station, np.ndarray):
        xyz_station = np.array(xyz_station)
    if not isinstance(xyz_source, np.ndarray):
        xyz_source = np.array(xyz_source)

    # >> Compute source-station vector gamma
    gamma = xyz_station - xyz_source
    gamma /= np.linalg.norm(gamma)

    gamma_line = gamma.reshape((1, 3))
    gamma_colu = gamma.reshape((3, 1))

    # >> Compute the force vector (in the (x,z) plane)
    F = np.array([[np.cos(alpha)], [0], [np.sin(alpha)]])

    # >> Intermediate matrix
    g2 = np.dot(gamma_colu, gamma_line)

    # >> Compute radiation pattern
    R_P = np.dot(g2, F)

    return R_P

# ----------------------------------------------------------------------------
def rad_pat_S_xyz(alpha, xyz_station, xyz_source):
    """
    Computes the S-waves radiation pattern in a given direction for a certain
    dip angle of the source force, using cartesian coordinates.

    Parameters
    ----------
    alpha : float
        Dip angle : between horizontal plane and channel. In radians.
    xyz_station : array-like
        x, y, z coordinates of station (in m).
    xyz_sources : list of array-likes
        List of x, y, z coordinates of sources (in m).

    Returns
    -------
    R_S : 3d numpy array
        Radiation pattern vector: `u_S_i(t) = factors * 1/r * R_S_i * |F(t)|`.

    """
    # >> Make coordinates into arrays
    if not isinstance(xyz_station, np.ndarray):
        xyz_station = np.array(xyz_station)
    if not isinstance(xyz_source, np.ndarray):
        xyz_source = np.array(xyz_source)

    # >> Compute source-station vector gamma
    gamma = xyz_station - xyz_source
    gamma /= np.linalg.norm(gamma)

    gamma_line = gamma.reshape((1, 3))
    gamma_colu = gamma.reshape((3, 1))

    # >> Compute the force vector (in the (x,z) plane)
    F = np.array([[np.cos(alpha)], [0], [np.sin(alpha)]])

    # >> Intermediate matrix
    g2 = np.dot(gamma_colu, gamma_line)
    Ig2 = np.diag(np.ones(3)) - g2

    # >> Compute radiation pattern
    R_S = np.dot(Ig2, F)

    return R_S


# ----------------------------------------------------------------------------
def rad_pat_P_sph(alpha, theta, phi):
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
def rad_pat_S_sph(alpha, theta, phi):
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

# ----------------------------------------------------------------------------
#           Double couple for testing
# ----------------------------------------------------------------------------
def DC_waveforms(xyz_source, xyz_station, m0, PARAM):
    """
    Computes the double couple waveforms. But just for testing, so
    very simplified:
    - dislocation is in the (x, y) plane
    - slip is along x, top of the dislocation goes to the right (positive x)
    """
    Vp = 6500
    Vs = 3500
    if not isinstance(xyz_station, np.ndarray):
        xyz_station = np.array(xyz_station)
    if not isinstance(xyz_source, np.ndarray):
        xyz_source = np.array(xyz_source)

    # >> Source receiver distance
    r = np.sqrt(np.sum((xyz_source - xyz_station)**2))

    # >> Compute radiation patterns
    rp_P = DC_rad_pat_P(xyz_station, xyz_source)
    rp_S = DC_rad_pat_S(xyz_station, xyz_source)

    # >> Compute unshifted waveforms
    uP = 1/(4*np.pi * Vp**3 * PARAM['rho_r']) * 1/r * np.dot(rp_P.reshape((3,1)), m0.reshape((1,len(m0))))
    uS = 1/(4*np.pi * Vs**3 * PARAM['rho_r']) * 1/r * np.dot(rp_S.reshape((3,1)), m0.reshape((1,len(m0))))

    for ii in range(3):
        uP[ii, :] -= uP[ii, 0]
        uS[ii, :] -= uS[ii, 0]

    # >> Using correct moveouts
    u = np.zeros((3, len(m0)))

    trav_time_P = r / Vp ; moveout_P = int(np.round(trav_time_P / (PARAM['dt_']*PARAM['T_scale'])))
    trav_time_S = r / Vs ; moveout_S = int(np.round(trav_time_S / (PARAM['dt_']*PARAM['T_scale'])))

    u[:, moveout_P:] += uP[:, :-moveout_P]
    u[:, moveout_S:] += uS[:, :-moveout_S]

    return u

# ----------------------------------------------------------------------------
def DC_rad_pat_P(xyz_source, xyz_station):
    """
    Computes the radiation of a double couple source at xyz_source.

    Dislocation is in the (x,y) plane, slip in the x direction, top goes
    to the right (positive x).
    """

    # >> Computes the source-station vector
    gamma = xyz_station - xyz_source
    gamma /= np.linalg.norm(gamma)

    r, theta, phi = my_util.cart2sph(gamma)

    vec_r = gamma
    vec_phi = np.array([np.cos(phi)*np.cos(theta), np.cos(phi)*np.sin(theta), -np.sin(phi)])
    vec_theta = np.array([-np.sin(theta), np.cos(theta), 0])

    # >> Computing radiation pattern
    R_P = np.sin(2*phi)*np.cos(theta) * vec_r

    return R_P

# ----------------------------------------------------------------------------

def DC_rad_pat_S(xyz_source, xyz_station):
    """
    Computes the radiation of a double couple source at xyz_source.

    Dislocation is in the (x,y) plane, slip in the x direction, top goes
    to the right (positive x).
    """

    # >> Computes the source-station vector
    gamma = xyz_station - xyz_source
    gamma /= np.linalg.norm(gamma)

    r, theta, phi = my_util.cart2sph(gamma)

    vec_r = gamma
    vec_phi = np.array([np.cos(phi)*np.cos(theta), np.cos(phi)*np.sin(theta), -np.sin(phi)])
    vec_theta = np.array([-np.sin(theta), np.cos(theta), 0])

    # >> Computing radiation pattern
    R_S = np.cos(2*phi)*np.cos(theta) * vec_phi - np.cos(phi)*np.sin(theta) * vec_theta

    return R_S
