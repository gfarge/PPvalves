""" Computes waveforms for valve sources """

# Imports
# =======
import numpy as np


# ----------------------------------------------------------------------------
def SF_waveforms(xyz_station, xyz_source, dp_source, PARAM, Vp=6500, Vs=3500):
    """
    Computes waveforms from the `dp` history at a source (valve), recorded at
    specified station. The canal is oriented along the North (x direction).

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

    # >> Source station distance
    r = np.sqrt(np.sum((xyz_source - xyz_station)**2))

    # >> Compute radiation pattern
    rp_P, rp_S = SF_rad_pat(xyz_source, xyz_station, 0, PARAM['alpha'])
    rp_P = rp_P.reshape(3, 1)
    rp_S = rp_S.reshape(3, 1)

    # >> Compute Force history
    F = dp_source * PARAM['A']
    F = F.reshape(1, len(F))

    # >> Compute unshifted waveforms
    uP = 1/(4*np.pi * Vp**2 * PARAM['rho_r']) * np.dot(rp_P, F)
    uS = 1/(4*np.pi * Vs**2 * PARAM['rho_r']) * np.dot(rp_S, F)

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
def DC_rad_pat(xyz_station, xyz_source, strike, dip, rake, z_is_down=True):
    """
    Computes the radiation pattern of S and P waves (including geometric
    attenuation) for a double couple mechanism.

    x is North, y is East, z is down (default).

    Parameters
    ==========
    xyz_station, xyz_source : array-like
        x, y, z coordinates of source and station (in m).
    strike : float
        Angle in degrees that the fault makes clockwise with the north. From 0
        (North) to 360 (90 = East).
    dip : float
        Angle in degrees that the fault plane makes down from the horizontal
        plane. From 0 (horizontal) to 90 (vertical).
    rake : float
        Angle in degrees that the slip direction makes with the strike (in the
        fault plane). From -180 to 180.
    z_is_down : bool (optional, default `z_is_down=True`)
        Default is a direct coordinate system, with z down. If z has to be
        specified as posivitive at depth, make the option `False`.

    Returns
    =======
    R_P, R_S : 1D arrays
        P-waves and S-waves polarization vectors, including the radiation
        pattern amplitude and geometric attenuation effects on amplitude on a
        "unitary" slip.  (x, y, z), in same coordinates as input.

    Note
    ----
        Normal motion : -180 < rake <= 0
        Reverse motion : 0 < rake <= 180
        Strike-slip : dip = 90, rake = 0 or 180

    All conventions are from Aki and Richards, 2002, Chapter 4.5.

    """
    # >> Make coordinates as arrays
    if not isinstance(xyz_station, np.ndarray):
        xyz_station = np.array(xyz_station)
    if not isinstance(xyz_source, np.ndarray):
        xyz_source = np.array(xyz_source)

    # >> Convert angles in radians
    dip = np.deg2rad(dip)
    rake = np.deg2rad(rake)
    strike = np.deg2rad(strike)

    # >> Reverse z?
    if not z_is_down:
        xyz_station[2] *= -1
        xyz_source[2] *= -1

    # >> Compute vectors
    # --> Unitary source-station vector
    sou_sta= xyz_station - xyz_source
    r = np.linalg.norm(sou_sta)
    sou_sta /= r  # make it unitary

    # --> Normal of the fault (off hanging wall)
    normal = np.array([-np.sin(dip)*np.sin(strike),
                       np.sin(dip)*np.cos(strike),
                       -np.cos(dip)])

    # --> Slip vector
    slip = np.array([
        np.cos(rake)*np.cos(strike) + np.cos(dip)*np.sin(rake)*np.sin(strike),
        np.cos(rake)*np.sin(strike) - np.cos(dip)*np.sin(rake)*np.cos(strike),
        - np.sin(rake)*np.sin(dip)
        ])

    # >> Radiation patterns
    # --> P waves
    R_P = 2 * np.dot(sou_sta, normal) * np.dot(sou_sta, slip) * 1/r * sou_sta

    # --> S waves
    R_S = np.dot(sou_sta, normal)*slip + np.dot(sou_sta, slip)*normal \
          - 2 * np.dot(sou_sta, normal) * np.dot(sou_sta, slip) * sou_sta
    R_S *= 1/r

    # >> Reverse z?
    if not z_is_down:
        R_P[2] *= -1
        R_S[2] *= -1

    return R_P, R_S

# ----------------------------------------------------------------------------
def DC_rad_pat_sph(xyz_source, r, phi, theta, strike, dip, rake):
    """
    Computes the P and S wave radiation pattern of the double couple using
    spherical coordinates.

    x is North, y is East, z is down (default).
    phi is up from z direction (down, 0 to 180), theta is angle of projection
    of station vector from x.

    Parameters
    ----------
    xyz_source: array-like
        x,y,z position of source, as a reference.
    r, phi, theta : floats
        Spherical coordinates of the station, when xyz system is shifted to
        source depth (source shifted to 0, 0, 0). `r` is source station
        distance, `phi` is angle from down in degrees (0 is down, 180 is up),
        `theta` is angle from North in degrees (0 is North (x), 90 is East
        (y)).
    strike : float
        Angle in degrees that the fault makes clockwise with the north. From 0
        (North) to 360 (90 = East).
    dip : float
        Angle in degrees that the fault plane makes down from the horizontal
        plane. From 0 (horizontal) to 90 (vertical).
    rake : float
        Angle in degrees that the slip direction makes with the strike (in the
        fault plane). From -180 to 180.

    Returns
    -------
    R_P, R_S : 1D arrays
        P-waves and S-waves polarization vectors, including the radiation
        pattern amplitude and geometric attenuation effects on amplitude on a
        "unitary" slip.  (x, y, z), in same coordinates as input.

    Note
    ----
        Normal motion : -180 < rake <= 0
        Reverse motion : 0 < rake <= 180
        Strike-slip : dip = 90, rake = 0 or 180

    All conventions are from Aki and Richards, 2002, Chapter 4.5.

    """
    # --> Convert spherical coordinates to cartesian coordinates
    x_station = r * np.sin(np.deg2rad(phi)) * np.cos(np.deg2rad(theta)) + xyz_source[0]
    y_station = r * np.sin(np.deg2rad(phi)) * np.sin(np.deg2rad(theta)) + xyz_source[1]
    z_station = r * np.cos(np.deg2rad(phi)) + xyz_source[2]

    xyz_station = np.array([x_station, y_station, z_station])

    return DC_rad_pat(xyz_station, xyz_source, strike, dip, rake, z_is_down=True)

# ----------------------------------------------------------------------------
def SF_rad_pat(xyz_station, xyz_source, azimuth, elevation, z_is_down=True):
    """
    Computes the radiation pattern of S and P waves (including geometric
    attenuation) for a single force mechanism.

    x is North, y is East, z is down (default).

    Parameters
    ==========
    xyz_station, xyz_source : array-like
        x, y, z coordinates of source and station (in m).
    azimuth : float
        Angle in degrees that the force makes clockwise (anti-trigonometric
        circle) from the North. From 0 (North) to 360 (90 = East).
    elevation : float
        Angle in degrees that the force makes up from the horizontal plane.
        From -90 (straight down) to 90 (straigth up).
    z_is_down : bool (optional, default `z_is_down=True`)
        Default is a direct coordinate system, with z down. If z has to be
        specified as posivitive at depth, make the option `False`.

    Returns
    =======
    R_P, R_S : 1D arrays
        P-waves and S-waves polarization vectors, including the radiation
        pattern amplitude and geometric attenuation effects on amplitude on a
        "unitary" . (x, y, z), in same coordinates as input.

    All conventions are from Aki and Richards, 2002, Chapter 4.2.

    """
    # >> Make coordinates as arrays
    if not isinstance(xyz_station, np.ndarray):
        xyz_station = np.array(xyz_station)
    if not isinstance(xyz_source, np.ndarray):
        xyz_source = np.array(xyz_source)

    # >> Convert angles in radians
    azimuth = np.deg2rad(azimuth)
    elevation = np.deg2rad(elevation)

    # >> Reverse z?
    if not z_is_down:
        xyz_station[2] *= -1
        xyz_source[2] *= -1

    # >> Compute vectors
    # --> Unitary source-station vector
    sou_sta= xyz_station - xyz_source
    r = np.linalg.norm(sou_sta)
    sou_sta /= r  # make it unitary

    sou_sta_col = sou_sta.reshape(3, 1)
    sou_sta_line = sou_sta.reshape(1, 3)

    # --> Force vector
    force = np.array([[np.cos(elevation)*np.cos(azimuth)],
                     [np.cos(elevation)*np.sin(azimuth)],
                     [-np.sin(elevation)]])

    # >> Radiation patterns
    # --> Useful matrices
    Sou_sta2 = np.dot(sou_sta_col, sou_sta_line)
    Identity = np.diag(np.ones(3))

    # --> P waves
    R_P = 1/r * np.dot(Sou_sta2, force)

    # --> S waves
    M = Identity - Sou_sta2
    R_S = 1/r * np.dot(M, force)

    # --> Reshape them
    R_P = R_P.reshape(3)
    R_S = R_S.reshape(3)

    # >> Reverse z?
    if not z_is_down:
        R_P[2] *= -1
        R_S[2] *= -1


    return R_P, R_S

# ----------------------------------------------------------------------------
def SF_rad_pat_sph(xyz_source, r, phi, theta, strike, dip, rake):
    """
    Computes the P and S wave radiation pattern of the double couple using
    spherical coordinates.

    x is North, y is East, z is down (default).
    phi is up from z direction (down, 0 to 180), theta is angle of projection
    of station vector from x.

    Parameters
    ----------
    xyz_source: array-like
        x,y,z position of source, as a reference.
    r, phi, theta : floats
        Spherical coordinates of the station, when xyz system is shifted to
        source depth (source shifted to 0, 0, 0). `r` is source station
        distance, `phi` is angle from down in degrees (0 is down, 180 is up),
        `theta` is angle from North in degrees (0 is North (x), 90 is East
        (y)).
    azimuth : float
        Angle in degrees that the force makes clockwise (anti-trigonometric
        circle) from the North. From 0 (North) to 360 (90 = East).
    elevation : float
        Angle in degrees that the force makes up from the horizontal plane.
        From -90 (straight down) to 90 (straigth up).

    Returns
    -------
    R_P, R_S : 1D arrays
        P-waves and S-waves polarization vectors, including the radiation
        pattern amplitude and geometric attenuation effects on amplitude on a
        "unitary" slip.  (x, y, z), in same coordinates as input.

    Note
    ----
        Normal motion : -180 < rake <= 0
        Reverse motion : 0 < rake <= 180
        Strike-slip : dip = 90, rake = 0 or 180

    All conventions are from Aki and Richards, 2002, Chapter 4.5.

    """
    # --> Convert spherical coordinates to cartesian coordinates
    x_station = r * np.sin(np.deg2rad(phi)) * np.cos(np.deg2rad(theta)) + xyz_source[0]
    y_station = r * np.sin(np.deg2rad(phi)) * np.sin(np.deg2rad(theta)) + xyz_source[1]
    z_station = r * np.cos(np.deg2rad(phi)) + xyz_source[2]

    xyz_station = np.array([x_station, y_station, z_station])

    return SF_rad_pat(xyz_station, xyz_source, azimuth, elevation, z_is_down=True)

## ----------------------------------------------------------------------------
#def rad_pat_P_xyz(alpha, xyz_station, xyz_source):
#    """
#    Computes the P-waves radiation pattern in a given direction for a certain
#    dip angle of the source force, using cartesian coordinates. Geometric 1/r
#    attenuation included.
#
#    Parameters
#    ----------
#    alpha : float
#        Dip angle : between horizontal plane and channel. In radians.
#    xyz_station : array-like
#        x, y, z coordinates of station (in m).
#    xyz_sources : list of array-likes
#        List of x, y, z coordinates of sources (in m).
#
#    Returns
#    -------
#    R_S : 3d numpy array
#        Radiation pattern vector: `u_S_i(t) = factors * 1/r * R_S_i * |F(t)|`.
#
#    """
#    # >> Make coordinates into arrays
#    if not isinstance(xyz_station, np.ndarray):
#        xyz_station = np.array(xyz_station)
#    if not isinstance(xyz_source, np.ndarray):
#        xyz_source = np.array(xyz_source)
#
#    # >> Compute source-station vector gamma
#    gamma = xyz_station - xyz_source
#    r = np.linalg.norm(gamma)  # source-station distance
#    gamma /= r
#
#    gamma_line = gamma.reshape((1, 3))
#    gamma_colu = gamma.reshape((3, 1))
#
#    # >> Compute the force vector (in the (x,z) plane)
#    F = np.array([[np.cos(alpha)], [0], [np.sin(alpha)]])
#
#    # >> Intermediate matrix
#    g2 = np.dot(gamma_colu, gamma_line)
#
#    # >> Compute radiation pattern
#    R_P = 1/r * np.dot(g2, F)
#
#    return R_P
#
## ----------------------------------------------------------------------------
#def rad_pat_S_xyz(alpha, xyz_station, xyz_source):
#    """
#    Computes the S-waves radiation pattern in a given direction for a certain
#    dip angle of the source force, using cartesian coordinates. Geometric 1/r attenuation included.
#
#    Parameters
#    ----------
#    alpha : float
#        Dip angle : between horizontal plane and channel. In radians.
#    xyz_station : array-like
#        x, y, z coordinates of station (in m).
#    xyz_sources : list of array-likes
#        List of x, y, z coordinates of sources (in m).
#
#    Returns
#    -------
#    R_S : 3d numpy array
#        Radiation pattern vector: `u_S_i(t) = factors * 1/r * R_S_i * |F(t)|`.
#
#    """
#    # >> Make coordinates into arrays
#    if not isinstance(xyz_station, np.ndarray):
#        xyz_station = np.array(xyz_station)
#    if not isinstance(xyz_source, np.ndarray):
#        xyz_source = np.array(xyz_source)
#
#    # >> Compute source-station vector gamma
#    gamma = xyz_station - xyz_source
#    r = np.linalg.norm(gamma)  # source-station distance
#    gamma /= r
#
#    gamma_line = gamma.reshape((1, 3))
#    gamma_colu = gamma.reshape((3, 1))
#
#    # >> Compute the force vector (in the (x,z) plane)
#    F = np.array([[np.cos(alpha)], [0], [np.sin(alpha)]])
#
#    # >> Intermediate matrix
#    g2 = np.dot(gamma_colu, gamma_line)
#    Ig2 = np.diag(np.ones(3)) - g2
#
#    # >> Compute radiation pattern
#    R_S = 1/r * np.dot(Ig2, F)
#
#    return R_S
#
#
## ----------------------------------------------------------------------------
#def rad_pat_P_sph(alpha, r, theta, phi):
#    """
#    Computes the P-waves radiation pattern in a given direction for a certain
#    dip angle of the source force. Geometric 1/r attenuation included.
#
#    Parameters
#    ----------
#    alpha : float
#        Dip angle : between horizontal plane and channel. In radians.
#    r : float
#        Source-station distance.
#    theta : float
#        Azimuthal angle of station with regard to source: angle between
#        projection of channel on horizontal plane (x,y) --- usually x axis ---
#        and projection of source-station vector on horizontal plane. In
#        radians, 0 to 2 pi.
#    phi : float
#        Polar angle of station with regard to source: angle from vertical
#        vector z-axis to source-station vector. In radians, 0 to pi.
#
#    Returns
#    -------
#    R_P : 3d numpy array
#        Radiation pattern vector: `u_P_i(t) = factors * 1/r * R_P_i * |F(t)|`.
#
#    """
#    # >> Initialize useful vectors
#    u_F = np.array([[np.cos(alpha)],  # force direction vector
#                    [0],
#                    [np.sin(alpha)]])
#    gamma_line = np.array([[np.cos(theta)*np.sin(phi),
#                           np.sin(theta)*np.sin(phi),
#                           np.cos(phi)]])  # source-station direction vector: in line form
#    gamma_column = gamma_line.reshape(3, 1)  # source-station direction vector: in column
#
#    # >> Compute matrices
#    gamma2_M = np.dot(gamma_column, gamma_line)
#
#    # >> Compute radiation pattern
#    R_P = 1/r * np.dot(gamma2_M, u_F)
#
#    return R_P
## ----------------------------------------------------------------------------
#def rad_pat_S_sph(alpha, r, theta, phi):
#    """
#    Computes the S-wave radiation pattern in a given direction for a certain
#    dip angle of the source force. Geometric 1/r attenuation included.
#
#    Parameters
#    ----------
#    alpha : float
#        Dip angle : between horizontal plane and channel. In radians.
#    r : float
#        Source-station distance.
#    theta : float
#        Azimuthal angle of station with regard to source: angle between
#        projection of channel on horizontal plane (x,y) --- usually x axis ---
#        and projection of source-station vector on horizontal plane. In
#        radians, 0 to 2 pi.
#    phi : float
#        Polar angle of station with regard to source: angle from vertical
#        vector z-axis to source-station vector. In radians, 0 to pi.
#
#    Returns
#    -------
#    R_S : 3d numpy array
#        Radiation pattern vector: `u_S_i(t) = factors * 1/r * R_S_i * |F(t)|`.
#
#    """
#    # >> Initialize useful vectors
#    u_F = np.array([[np.cos(alpha)],  # force direction vector
#                    [0],
#                    [np.sin(alpha)]])
#    gamma_line = np.array([[np.cos(theta)*np.sin(phi),
#                           np.sin(theta)*np.sin(phi),
#                           np.cos(phi)]])  # source-station direction vector: in line form
#    gamma_column = gamma_line.reshape(3, 1)  # source-station direction vector: in column
#
#    # >> Compute matrices
#    gamma2_M = np.dot(gamma_column, gamma_line)
#    D_F = np.diag(np.ones(3))
#    M = D_F - gamma2_M
#
#    # >> Compute radiation pattern
#    R_S = 1/r * np.dot(M, u_F)
#
#    return R_S
#
## ----------------------------------------------------------------------------
#def cart2sph(xyz):
#    """
#    Converts cartesian coordinates to spherical coordinates.
#
#    Parameter
#    ---------
#    xyz : array-like
#        3D cartesian coordinates, x,y,z, correctly oriented.
#
#    Returns
#    -------
#    r : float
#    theta : float
#        Azimuthal angle: angle from x axis to xy, projection of the
#        origin-point vector on horizontal plane (x,y).
#    phi : float
#        Polar angle: angle from vertical z axis to origin-point vector.
#
#    Note
#    ----
#        Code from: https://stackoverflow.com/questions/4116658/faster-numpy-cartesian-to-spherical-coordinate-conversion
#    """
#    xy2 = xyz[0]**2 + xyz[1]**2
#    r = np.sqrt(xy2 + xyz[2]**2)
#
#    theta = np.arctan2(xyz[1], xyz[0])
#    phi = np.arctan2(np.sqrt(xy2), xyz[2])
#
#    return r, theta, phi
#
## ----------------------------------------------------------------------------
##           Double couple for testing
## ----------------------------------------------------------------------------
#def DC_waveforms(xyz_source, xyz_station, m0, PARAM):
#    """
#    Computes the double couple waveforms. But just for testing, so
#    very simplified:
#    - dislocation is in the (x, y) plane
#    - slip is along x, top of the dislocation goes to the right (positive x)
#    """
#    Vp = 6500
#    Vs = 3500
#    if not isinstance(xyz_station, np.ndarray):
#        xyz_station = np.array(xyz_station)
#    if not isinstance(xyz_source, np.ndarray):
#        xyz_source = np.array(xyz_source)
#
#    # >> Source receiver distance
#    r = np.sqrt(np.sum((xyz_source - xyz_station)**2))
#
#    # >> Compute radiation patterns
#    rp_P = DC_rad_pat_P(xyz_station, xyz_source)
#    rp_S = DC_rad_pat_S(xyz_station, xyz_source)
#
#    # >> Compute unshifted waveforms
#    uP = 1/(4*np.pi * Vp**3 * PARAM['rho_r']) * np.dot(rp_P.reshape((3,1)), m0.reshape((1,len(m0))))
#    uS = 1/(4*np.pi * Vs**3 * PARAM['rho_r']) * np.dot(rp_S.reshape((3,1)), m0.reshape((1,len(m0))))
#
#    for ii in range(3):
#        uP[ii, :] -= uP[ii, 0]
#        uS[ii, :] -= uS[ii, 0]
#
#    # >> Using correct moveouts
#    u = np.zeros((3, len(m0)))
#
#    trav_time_P = r / Vp ; moveout_P = int(np.round(trav_time_P / (PARAM['dt_']*PARAM['T_scale'])))
#    trav_time_S = r / Vs ; moveout_S = int(np.round(trav_time_S / (PARAM['dt_']*PARAM['T_scale'])))
#
#    u[:, moveout_P:] += uP[:, :-moveout_P]
#    u[:, moveout_S:] += uS[:, :-moveout_S]
#
#    return u
#
## ----------------------------------------------------------------------------
#def DC_rad_pat_P(xyz_source, xyz_station):
#    """
#    Computes the radiation of a double couple source at xyz_source. Geometric
#    1/r attenuation included.
#
#    Dislocation is in the (x,y) plane, slip in the x direction, top goes
#    to the right (positive x).
#    """
#
#    # >> Computes the source-station vector
#    gamma = xyz_station - xyz_source
#    r, theta, phi = cart2sph(gamma)
#
#    gamma /= np.linalg.norm(gamma)
#
#    vec_r = gamma
#    vec_phi = np.array([np.cos(phi)*np.cos(theta), np.cos(phi)*np.sin(theta), -np.sin(phi)])
#    vec_theta = np.array([-np.sin(theta), np.cos(theta), 0])
#
#    # >> Computing radiation pattern
#    R_P = 1/r * np.sin(2*phi)*np.cos(theta) * vec_r
#
#    return R_P
#
## ----------------------------------------------------------------------------
#
#def DC_rad_pat_S(xyz_source, xyz_station):
#    """
#    Computes the radiation of a double couple source at xyz_source. Geometric
#    1/r attenuation included.
#
#
#    Dislocation is in the (x,y) plane, slip in the x direction, top goes
#    to the right (positive x).
#    """
#
#    # >> Computes the source-station vector
#    gamma = xyz_station - xyz_source
#    r, theta, phi = cart2sph(gamma)
#
#    gamma /= np.linalg.norm(gamma)  # make it unitary
#
#    vec_r = gamma
#    vec_phi = np.array([np.cos(phi)*np.cos(theta), np.cos(phi)*np.sin(theta), -np.sin(phi)])
#    vec_theta = np.array([-np.sin(theta), np.cos(theta), 0])
#
#    # >> Computing radiation pattern
#    R_S = 1/r * (np.cos(2*phi)*np.cos(theta) * vec_phi
#                 - np.cos(phi)*np.sin(theta) * vec_theta)
#
#    return R_S
#
#
