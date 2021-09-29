""" Computes waveforms for valve sources """

# Imports
# =======
import numpy as np


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

