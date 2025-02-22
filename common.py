#! /usr/bin/env python3

import numpy as np, math, sys
import math, os, sys, time

from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import MapMetaData

def numpy_to_occupancy_grid(arr, info=None):
    if not len(arr.shape) == 2:
        raise TypeError('Array must be 2D')
    if not arr.dtype == np.int8:
        raise TypeError('Array must be of int8s')

    grid = OccupancyGrid()
    if isinstance(arr, np.ma.MaskedArray):
        # We assume that the masked value are already -1, for speed
        arr = arr.data
    grid.data = arr.ravel()
    grid.info = info or MapMetaData()
    grid.info.height = arr.shape[0]
    grid.info.width = arr.shape[1]    

    return grid

class Util:
    # constants
    NUM0S = 7
    VID_PREFIX = "vid"
    VID_SUFFIX = "avi"
    IMG_PREFIX = "img"
    IMG_SUFFIX = "jpg"

    # methods
    @staticmethod
    def make_name(path, prefix, n, suffix, extension, numZeros=7):
        """ makes a filename in a certain formatting
        i.e., make_name('./', 'test', 52, 'experiment1', 'png')
        returns ./test_0000052_experiment1.png

        Parameters
        ----------
        path : str
            the folder/path/dir to search in
        prefix : str
            the prefix string
        n: int
            the number between prefix and suffix
        suffix : str
            the suffix string
        extension : str
            the extension string, assumed to be valid
        numZeros : int
            the number of digits in the number between prefix and suffix

        Returns
        -------
        name : str
            the filename except the extension
        fname : str
            the entire filename including extension
        """
        tokens = []
        if (prefix != ''):
            tokens.append(prefix)
        tokens.append(str(n).zfill(numZeros))
        if (suffix != ''):
            tokens.append(suffix)
        name = path + "/" + '_'.join(tokens)
        fname = ".".join([name, extension])
        return name, fname

    @staticmethod
    def get_next_valid_name_increment(path, prefix, n, suffix, extension, numZeros=7):
        """ get the next 'valid' name in a path given a certain formatting
        i.e., make_name('./', 'test', 52, 'experiment1', 'png')
        if ./ contains ./test_0000052_experiment1.png
        will return ./test_0000053_experiment1.png, 53

        Notes
        -----
        'valid' in this sense means the file doesn't already exist
        function does not allow overwriting!

        Parameters
        ----------
        path : str
            the folder/path/dir to search in
        prefix : str
            the prefix string
        n: int
            the number between prefix and suffix
        suffix : str
            the suffix string
        extension : str
            the extension string, assumed to be valid
        numZeros : int
            the number of digits in the number between prefix and suffix

        Returns
        -------
        fname : str
            the entire filename including extension
        n : int
            the count at which the valid file was found
        """

        if (not os.path.isdir(path)):
            raise ValueError('get_next_valid_name:no_such_path', path)

        name, fname = Util.make_name(path, prefix, n, suffix, extension, numZeros)
        while (os.path.isfile(fname)):
            n = n + 1
            name, fname = Util.make_name(path, prefix, n, suffix, extension, numZeros)
        return fname, n

def two_d_make_x_y_theta_hom(x, y, theta):
    hom = np.eye(3)

    theta = theta % (2 * np.pi)
    # 2019-08-02 parentheses!!!

    hom[0, 0] = np.cos(theta)
    hom[0, 1] = -np.sin(theta)
    hom[1, 0] = np.sin(theta)
    hom[1, 1] = np.cos(theta)

    hom[0, 2] = x
    hom[1, 2] = y
    return hom

def two_d_rvec_vec_from_matrix_2d(m):
    x = m[0, 2]
    y = m[1, 2]

    atan2 = np.arctan2(m[1, 0], m[0, 0])
    # SUPER #COOL #IMPORTANT #port
    # this is how we get the original theta 2019-08-02
    # https://stackoverflow.com/a/32549077
    # deal with quadrant logic

    '''
    theta_0 = math.acos(m[0, 0])
    theta_1 = math.asin(m[1, 0])
    theta_2 = math.asin(-1.0 * m[0, 1])
    theta_3 = math.acos(m[1, 1])
    # TODO(someone) finish this up,
    # do not always all match, sometimes negative
    thetas = [theta_0, theta_1, theta_2, theta_3]
    mode_thetas = Util.modes(thetas)
    if len(mode_thetas) > 0:
        # print("more than one theta found", mode_thetas)
        pass
    '''

    return [x, y, atan2], None

def quaternion_from_euler(roll, pitch, yaw):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    q = [0] * 4
    q[0] = cy * cp * cr + sy * sp * sr
    q[1] = cy * cp * sr - sy * sp * cr
    q[2] = sy * cp * sr + cy * sp * cr
    q[3] = sy * cp * cr - cy * sp * sr

    return q

def euler_from_quaternion(quaternion):
    """
    Converts quaternion (w in last place) to euler roll, pitch, yaw
    quaternion = [x, y, z, w]
    Bellow should be replaced when porting for ROS 2 Python tf_conversions is done.
    """
    x = quaternion[0]
    y = quaternion[1]
    z = quaternion[2]
    w = quaternion[3]

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

def hom2d_to_ps(hom_2d, sibling):
    xytheta, _ = two_d_rvec_vec_from_matrix_2d(
        hom_2d)
    return xytheta_to_ps(*xytheta, sibling)

def xytheta_to_ps(x, y, theta, sibling):
    ps = PoseStamped()
    ps.pose.position.x = x
    ps.pose.position.y = y
    ps.pose.position.z = 0.0

    w, x, y, z = quaternion_from_euler(
        0.0, 0.0, theta)
    ps.pose.orientation.x = x
    ps.pose.orientation.y = y
    ps.pose.orientation.z = z
    ps.pose.orientation.w = w

    ps.header = sibling.header
    return ps

def make_path_msg(homs_2d, sibling_msg):
    p = Path()
    p.header = sibling_msg.header
    p.poses = [
        hom2d_to_ps(x, sibling_msg) for x in homs_2d]
    return p

def ps_to_xytheta(ps):
    _, _, y = euler_from_quaternion([
        ps.pose.orientation.x,
        ps.pose.orientation.y,
        ps.pose.orientation.z,
        ps.pose.orientation.w])
    return [ps.pose.position.x, ps.pose.position.y, y]

def path_to_xytheta(path_msg):
    return np.array([ps_to_xytheta(ps) for ps in path_msg.poses])

def interpolate(arr, factor):
    res = []
    i = 0
    while i < len(arr) - 1:
        res.extend(
            np.linspace(arr[i], arr[i+1], factor, endpoint=False)
        )
        i += 1
    res.append(arr[-1])
    return res

def slerp3(arr, xys, factor):
    res = []
    i = 0
    while i < len(arr) - 1:
        diff = arr[i+1] - arr[i]

        if (np.abs(diff) < 1e-8):
            res.extend(
                np.linspace(arr[i], arr[i+1], factor, endpoint=False)
            )
            i += 1
            continue

        # import ipdb; ipdb.set_trace()
        distance = np.linalg.norm(xys[i+1] - xys[i], ord=2)

        interp = []

        min_t = np.abs(diff) / 0.5
        min_d = min_t * 0.2 # target_v

        # print("min_d", min_d)

        # # sinusoidal interpolation
        # for j in range(factor):
        #     k = j / float(factor) # will never hit 1.0
        #     a = 1-np.cos(np.pi*k)
        #     v = arr[i] + diff * a
        #     interp.append(v)

        distance_traveled = 0.0
        delta = np.abs(distance) / factor
        # print("DELTA", delta)
        # print("min_d", min_d)
        for j in range(factor):
            distance_traveled += delta

            if (distance_traveled + min_d >= distance):
                alpha = (distance_traveled - (distance - min_d)) / min_d
                # print("ALPHA", alpha)
                interp.append(arr[i] + alpha * diff)
            else:
                interp.append(arr[i])


            # if j < factor * 0.9:
            #     interp.append(arr[i])

            # else:
            #     v = arr[i+1]
            #     interp.append(v)

        res.extend(
            interp
            # np.linspace(arr[i], arr[i+1], factor, endpoint=False)
        )
        i += 1

    res.append(arr[-1])
    return res