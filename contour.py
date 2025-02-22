#! /usr/bin/env python3


from threading import Thread, Lock, Condition
import rclpy
from rclpy.node import Node
from tf_transformations import euler_from_quaternion

import numpy as np, math, sys

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from rcl_interfaces.msg import SetParametersResult

from nav_msgs.msg import OccupancyGrid
import CMap2D

from collections import deque

import threading

from nav_msgs.msg import MapMetaData

import os, sys, time

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import matplotlib.cm as cm

import argparse

import cv2
from scipy import stats

def plot_line(ax, a, b, mode, color, linewidth):
    if mode == 3:
        ax.plot(
            [a[0], b[0]],
            [a[1], b[1]],
            [a[2], b[2]],
            color=color,
            linewidth=linewidth)
    elif mode == 2:
        ax.plot(
            [a[0], b[0]],
            [a[1], b[1]],
            color=color,
            linewidth=linewidth)

def plot_gnomon(ax, g, mode=3, length=0.1, linewidth=5, c=None, offset = 0.0):
    '''
    mode is dimension of 'canvas'
    '''
    if (mode == 3):
        o = g.dot(np.array([offset, offset, 0.0, 1.0]))
    elif (mode == 2):
        o = g.dot(np.array([offset, offset, 1.0]))

    if (mode == 3):
        x = g.dot(np.array([length*1 + offset, offset, 0.0, 1.0]))
        if c is not None:
            plot_line(ax, o, x, mode, c, linewidth)
        else:
            plot_line(ax, o, x, mode, 'r', linewidth)
    elif (mode == 2):
        x = g.dot(np.array([length*1 + offset, offset, 1.0]))
        if c is not None:
            plot_line(ax, o, x, mode, c, linewidth)
        else:
            plot_line(ax, o, x, mode, 'r', linewidth)

    if (mode == 3):
        y = g.dot(np.array([offset, length*2 + offset, 0.0, 1.0]))
        if c is not None:
            plot_line(ax, o, y, mode, c, linewidth)
        else:
            plot_line(ax, o, y, mode, 'g', linewidth)
    elif (mode == 2):
        y = g.dot(np.array([offset, length*2 + offset, 1.0]))
        if c is not None:
            plot_line(ax, o, y, mode, c, linewidth)
        else:
            plot_line(ax, o, y, mode, 'g', linewidth)

    if (mode == 3):
        z = g.dot(np.array([offset, 0.0, length*3 + offset, 1.0]))
        if c is not None:
            plot_line(ax, o, z, mode, c, linewidth)
        else:
            plot_line(ax, o, z, mode, 'b', linewidth)

def rotate(img, angle):
    (height, width) = img.shape[:2]
    (cent_x, cent_y) = (width // 2, height // 2)

    mat = cv2.getRotationMatrix2D((cent_x, cent_y), -angle, 1.0)
    cos = np.abs(mat[0, 0])
    sin = np.abs(mat[0, 1])

    n_width = int((height * sin) + (width * cos))
    n_height = int((height * cos) + (width * sin))

    mat[0, 2] += (n_width / 2) - cent_x
    mat[1, 2] += (n_height / 2) - cent_y

    mode, count = stats.mode(np.concatenate(img))
    mode = int(mode)
    count = int(count)

    # import ipdb; ipdb.set_trace()

    return cv2.warpAffine(img,
        mat,
        (n_width, n_height),
        borderMode = cv2.BORDER_CONSTANT,
        borderValue=(mode, mode, mode)), mode

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

def sample_sdf(
    g_map_basefootprint,
    radius, samplings,
    A_tile, sdf_map, occupancy):
    xys = []
    sdf_values = []
    invalids = []
    for s_i, s in enumerate(samplings):
        dx = radius * np.cos(s)
        dy = radius * np.sin(s)
        tmp1 = two_d_make_x_y_theta_hom(dx, dy, 0.0)
        tmp2 = np.dot(g_map_basefootprint, tmp1)
        xys.append([tmp2[0, 2], tmp2[1, 2]])

        uv1 = np.dot(A_tile, tmp2[:, 2]).astype(np.int32)
        # print(uv1)
        # l = 5
        # for u in range(uv1[0] - l, uv1[0] + l):
        #     for v in range(uv1[1] - l, uv1[1] + l):
        #         sdf_map[v, u] = 0

        try:
            sdf_values.append([s, sdf_map[uv1[1], uv1[0]]])
        except:
            sdf_values.append([s, -np.inf])

    return np.array(xys), np.array(sdf_values)

def contour_step(
    sdf_data,
    iterations = 50):
    sdf_map = sdf_data["sdf_map"]
    # sdf_map_backup = sdf_data["map"]
    g_map_basefootprint = sdf_data["g_map_basefootprint"]
    # print("g_map_basefootprint", g_map_basefootprint[:, 2])
    last_xytheta = sdf_data["latest_xytheta"]
    data = sdf_data["msg"]
    (m1, m2) = sdf_data["range"]
    occupancy = sdf_data["map"]

    r = data.info.resolution

    # the correct 'matplotlib intrinsic'
    sdf_map = sdf_map.astype(np.uint8)

    func = lambda t, m1=m1, m2=m2: m1 + t / (255) * (m2 - m1)
    vfunc = np.vectorize(func)
    sdf_map = vfunc(sdf_map)

    sdf_map, _ = rotate(sdf_map, 180)
    sdf_map = np.flip(sdf_map, 1)

    rotated_im_w = sdf_map.shape[1] * r
    rotated_im_h = sdf_map.shape[0] * r

    x_min = data.info.origin.position.x
    y_min = data.info.origin.position.y

    rotated_im_w = sdf_map.shape[1] * r
    rotated_im_h = sdf_map.shape[0] * r

    x_max = x_min + rotated_im_w
    y_max = y_min + rotated_im_h

    def func(t):
        if t < 0:
            return 50
        else:
            return t
    vfunc = np.vectorize(func)
    occupancy = vfunc(occupancy)

    # occupancy, _ = rotate(occupancy, 180)
    # occupancy = occupancy.T
    occupancy = np.flip(occupancy, 0)
    occupancy = occupancy.astype(np.int8)

    ################################

    _, _, yaw = euler_from_quaternion([
        data.info.origin.orientation.x,
        data.info.origin.orientation.y,
        data.info.origin.orientation.z,
        data.info.origin.orientation.w])
    g_world_map = two_d_make_x_y_theta_hom(
        data.info.origin.position.x, 
        data.info.origin.position.y,
        yaw)

    A_tile = np.array([
        [1.0 / data.info.resolution, 0.0, -g_world_map[0, 2] / data.info.resolution],
        [0.0, -1.0 / data.info.resolution, g_world_map[1, 2] / data.info.resolution],
        [0.0, 0.0, 1.0]
        ])

    # r1 = 0.5
    # r2 = 1.0
    # r3 = 1.5

    # this is mpc, we only take the 
    # first step, so don't look
    # too far ahead
    r1 = 0.25
    r2 = 0.5
    r3 = 0.75

    max_yaw = 2.0
    samplings = np.arange(-max_yaw, max_yaw, 0.1)

    ################################

    hom = g_map_basefootprint
    last_thetas = [np.pi]
    all_homs = [hom]

    current_vals = [0.0]

    for iteration in range(100):
        uv1 = np.dot(A_tile,
            hom[:, 2]).astype(np.int32)
        current = sdf_map[uv1[1], uv1[0]]

        current_vals.append(current)

        xys_1, sdf_sampling_1 = sample_sdf(
            hom,
            r1, samplings,
            A_tile, sdf_map, occupancy)

        xys_2, sdf_sampling_2 = sample_sdf(
            hom,
            r2, samplings,
            A_tile, sdf_map, occupancy)

        # xys_3, sdf_sampling_3 = sample_sdf(
        #     hom,
        #     r3, samplings,
        #     A_tile, sdf_map, occupancy)

        ################################

        # turn_gain = max(-1.0, -np.abs(last_thetas[-1]))
        turn_gain = -np.abs(last_thetas[-1])
        # turn_gain = -1.0 # -np.abs(last_thetas[-1])
        turn_cost = turn_gain * sdf_sampling_1[:, 0]**2

        '''
        cost_1 = sdf_sampling_1[:, 1] * 1
        cost_2 = sdf_sampling_2[:, 1] * 2
        cost_3 = 0
        # cost_3 = sdf_sampling_3[:, 1] * 3
        # cost_3 = 0.0 # - 1 / np.abs(sdf_sampling_2[:, 1])
        formulation = cost_1 + cost_2 + cost_3 + turn_cost
        '''
        # hunting to the max target => 'cross open-spaces'/ 'tourism'

        # hunting to a specific target => 'wall-hugging'
        target = np.max(sdf_sampling_2[:, 1])
        target = min(target, 1.5)
        target = max(target, 0.8)
        # the 'wall-hugging zone', stay in it
        # if space opens up ahead, take the gap
        # make it clamped and dynamic
        # so in tight spaces you aren't hunting for something unattainable
        # and in open spaces you aren't being a 'tourist'
        # clamp it between these 2

        # to prevent rapid changes, take on step to it
        # complementary / lpf'd
        target = current + (target - current) * 0.1

        cost_1 = 1 / (1 + np.abs(target - sdf_sampling_1[:, 1]))
        # rewarding future distances => smoother / correct paths
        # rewarding closer distances => greediness, 'stuck', failures
        cost_2 = 3 / (1 + np.abs(target - sdf_sampling_2[:, 1]))
        cost_3 = 0
        # cost_3 = sdf_sampling_3[:, 1] * 3
        # cost_3 = 0.0 # - 1 / np.abs(sdf_sampling_2[:, 1])
        formulation = cost_1 + cost_2 + cost_3 + turn_cost

        # # # this is the 'gradient' of sdf, we want +/0 gradient
        # cost_1 = sdf_sampling_2[:, 1] - sdf_sampling_1[:, 1]
        # # cost_2 = - 1.0 / sdf_sampling_2[:, 1]
        # cost_2 = 0.0
        # formulation = cost_1 + cost_2 - turn_cost

        max_idx = np.argmax(formulation)
        best_theta = sdf_sampling_1[max_idx, 0]

        dx = r2 * np.cos(best_theta)
        dy = r2 * np.sin(best_theta)
        tmp1 = two_d_make_x_y_theta_hom(dx, dy, best_theta)
        tmp2 = np.dot(hom, tmp1)
        # print("tmp2", tmp2)
        # print("g_map_basefootprint", g_map_basefootprint)

        xys = np.array([
            [hom[0, 2], hom[1, 2]],
            [tmp2[0, 2], tmp2[1, 2]]
            ])

        unknown_found = False

        # this adds some perturbations
        # keeps it from limit-cycling exactly
        r = 5 # don't make this too large, otherwise it gets stuck
        for u in range(uv1[0] - r, uv1[0] + r):
            for v in range(uv1[1] - r, uv1[1] + r):
                if v >= sdf_map.shape[0] or u >= sdf_map.shape[1]:
                    continue
                # if v < 0 or u < 0:
                #     print("b", v, u)
                #     continue

                try:
                    sdf_map[v, u] -= 1.0 # don't make this too low

                    # print(occupancy[v, u])
                    if occupancy[v, u] == 50:
                        unknown_found = True
                        pass
                except:
                    print("index problem")

        hom = tmp2

        # print("DIST!", np.linalg.norm(hom[:2, 2] - all_homs[-1][:2, 2], ord=2))

        if unknown_found:
            print("UNKNOWN FOUND!", iteration)
            break

        all_homs.append(hom)

        last_thetas.append(best_theta)

    return all_homs, last_thetas, current_vals

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('file', type=str, help='file')
    args = parser.parse_args()

    with open(args.file, 'rb') as f:
        sdf_map = np.load(f, allow_pickle=True)

    all_homs, last_thetas, current_vals = contour_step(sdf_map.tolist())

    tmp = sdf_map.tolist()
    sdf_map = tmp["sdf_map"]
    # sdf_map_backup = tmp["map"]
    g_map_basefootprint = tmp["g_map_basefootprint"]
    # print("g_map_basefootprint", g_map_basefootprint[:, 2])
    last_xytheta = tmp["latest_xytheta"]
    data = tmp["msg"]
    (m1, m2) = tmp["range"]
    occupancy = tmp["map"]

    r = data.info.resolution

    # the correct 'matplotlib intrinsic'
    sdf_map = sdf_map.astype(np.uint8)

    # turn it from uint8 back to meters
    func = lambda t, m1=m1, m2=m2: m1 + t / (255) * (m2 - m1)
    vfunc = np.vectorize(func)
    sdf_map = vfunc(sdf_map)

    sdf_map, _ = rotate(sdf_map, 180)
    sdf_map = np.flip(sdf_map, 1)

    rotated_im_w = sdf_map.shape[1] * r
    rotated_im_h = sdf_map.shape[0] * r

    x_min = data.info.origin.position.x
    y_min = data.info.origin.position.y

    rotated_im_w = sdf_map.shape[1] * r
    rotated_im_h = sdf_map.shape[0] * r

    x_max = x_min + rotated_im_w
    y_max = y_min + rotated_im_h

    ################################

    # turn it from uint8 back to meters
    def func(t):
        if t < 0:
            return 50
        else:
            return t
    vfunc = np.vectorize(func)
    occupancy = vfunc(occupancy)

    # occupancy, _ = rotate(occupancy, 180)
    # occupancy = occupancy.T
    occupancy = np.flip(occupancy, 0)
    occupancy = occupancy.astype(np.int8)

    ################################

    _, _, yaw = euler_from_quaternion([
        data.info.origin.orientation.x,
        data.info.origin.orientation.y,
        data.info.origin.orientation.z,
        data.info.origin.orientation.w])
    g_world_map = two_d_make_x_y_theta_hom(
        data.info.origin.position.x, 
        data.info.origin.position.y,
        yaw)

    A_tile = np.array([
        [1.0 / data.info.resolution, 0.0, -g_world_map[0, 2] / data.info.resolution],
        [0.0, -1.0 / data.info.resolution, g_world_map[1, 2] / data.info.resolution],
        [0.0, 0.0, 1.0]
        ])

    ################################

    fig = plt.figure()

    gs = fig.add_gridspec(4,4)
    ax1 = fig.add_subplot(gs[0, :2])
    ax2 = fig.add_subplot(gs[0, 2:])
    ax3 = fig.add_subplot(gs[1:, :])

    # print("YAW", yaw)
    # print("g_world_map", g_world_map)

    plot_gnomon(ax3, g_world_map, mode=2, length=1)

    ################################

    '''
    x = np.arange(x_min, x_max, r)
    y = np.arange(y_max, y_min, -r)
    # y = np.arange(y_min, y_max, r)
    X, Y = np.meshgrid(x, y)

    # Z = np.abs(sdf_map)
    Z = sdf_map

    # CS = ax.contour(X, Y, Z)
    CS = ax3.contour(X, Y, Z, [-100, 100], alpha=0.5)  # alpha = 0 -> invisible

    # for i in range(len(CS.allsegs)):
    #     the_interesting_one = CS.allsegs[i][0]
    #     plt.plot(
    #         the_interesting_one[:, 0],
    #         the_interesting_one[:, 1], "k--")

    # ax.clabel(CS, fontsize=10)
    # ax.set_title('Simplest default with labels')
    '''

    ################################

    plot_gnomon(ax3, g_map_basefootprint, mode=2, length=1)

    ################################

    all_homs_xs = [t[0, 2] for t in all_homs]
    all_homs_ys = [t[1, 2] for t in all_homs]
    ax3.plot(
        all_homs_xs,
        all_homs_ys,
        c='c')

    # uv1 = np.dot(A_tile, g_map_basefootprint[:, 2]).astype(np.int32)
    # # print(uv1)
    # current = sdf_map[uv1[1], uv1[0]]

    # max_yaw = 2.0
    # samplings = np.arange(-max_yaw, max_yaw, 0.1)

    # hom = g_map_basefootprint

    # last_thetas = []

    # r1 = 0.5
    # r2 = 1.0

    # for iteration in range(50):
    #     xys_1, sdf_sampling_1 = sample_sdf(
    #         hom,
    #         r1, samplings,
    #         A_tile, sdf_map)

    #     xys_2, sdf_sampling_2 = sample_sdf(
    #         hom,
    #         r2, samplings,
    #         A_tile, sdf_map)

    #     ################################

    #     # ax3.plot(xys_1[:, 0], xys_1[:, 1], c='g', alpha=0.5)
    #     # ax3.plot(xys_2[:, 0], xys_2[:, 1], c='r', alpha=0.5)

    #     turn_gain = -1.0
    #     turn_cost = turn_gain * sdf_sampling_1[:, 0]**2
    #     cost_1 = sdf_sampling_1[:, 1] * 1
    #     cost_2 = sdf_sampling_2[:, 1] * 2
    #     cost_3 = 0.0 # - 1 / np.abs(sdf_sampling_2[:, 1])
    #     formulation = cost_1 + cost_2 + cost_3 + turn_cost

    #     # # this is the 'gradient' of sdf, we want +/0 gradient
    #     # cost_1 = sdf_sampling_2[:, 1] - sdf_sampling_1[:, 1]
    #     # cost_2 = - 1.0 / sdf_sampling_2[:, 1]
    #     # formulation = cost_1 + cost_2 - turn_cost

    #     max_idx = np.argmax(formulation)
    #     best_theta = sdf_sampling_1[max_idx, 0]

    #     dx = r2 * np.cos(best_theta)
    #     dy = r2 * np.sin(best_theta)
    #     tmp1 = two_d_make_x_y_theta_hom(dx, dy, best_theta)
    #     tmp2 = np.dot(hom, tmp1)
    #     # print("tmp2", tmp2)
    #     # print("g_map_basefootprint", g_map_basefootprint)

    #     xys = np.array([
    #         [hom[0, 2], hom[1, 2]],
    #         [tmp2[0, 2], tmp2[1, 2]]
    #         ])
    #     ax3.plot(
    #         xys[:, 0],
    #         xys[:, 1],
    #         c='c')

    #     hom = tmp2
    #     last_thetas.append(best_theta)

    #     plot_gnomon(ax3, hom, mode=2, length=0.1, linewidth=1, c='c')

    #     '''
    #     ax1.plot(
    #         sdf_sampling_1[:, 0],
    #         sdf_sampling_1[:, 1],
    #         c='g')
    #     ax1.plot(
    #         sdf_sampling_2[:, 0],
    #         sdf_sampling_2[:, 1],
    #         c='r')

    #     ax1.plot(
    #         sdf_sampling_2[:, 0],
    #         cost_1,
    #         c='k')
    #     ax1.plot(
    #         sdf_sampling_2[:, 0],
    #         cost_2,
    #         c='y')
    #     ax1.plot(
    #         sdf_sampling_2[:, 0],
    #         cost_3,
    #         c='y')
    #     ax1.plot(
    #         sdf_sampling_2[:, 0],
    #         turn_cost,
    #         c='b')

    #     ax1.scatter([0], [current])
    #     ax1.scatter(
    #         [best_theta],
    #         formulation[max_idx], c='c')

    #     # formulation = cost_2
    #     ax1.plot(
    #         sdf_sampling_1[:, 0],
    #         formulation, c='m')
    #     break
    #     '''

    ax1.plot(np.array(last_thetas))

    ax2.plot(np.array(current_vals))

    ################################

    # m1 = np.min(sdf_map)
    # m2 = np.max(sdf_map)
    # print(
    #     "min {}, max {}".format(
    #         m1,
    #         m2
    #     ))

    ################################

    # r = 2
    # uv1 = np.dot(A_tile, np.array([0.0, 0.0, 1.0])).astype(np.int32)
    # print(uv1)
    # for u in range(uv1[0] - r, uv1[0] + r):
    #     for v in range(uv1[1] - r, uv1[1] + r):
    #         sdf_map[v, u] = 0

    # uv1 = np.dot(A_tile, np.array([3.0, 1.0, 1.0])).astype(np.int32)
    # print(uv1)
    # for u in range(uv1[0] - r, uv1[0] + r):
    #     for v in range(uv1[1] - r, uv1[1] + r):
    #         sdf_map[v, u] = 0

    ################################

    im_handle = ax3.imshow(sdf_map, alpha=0.7)
    im_handle.set_extent([
        min([x_min, g_world_map[0, 2]]),
        x_max,
        min([y_min, g_world_map[1, 2]]),
        y_max
    ])

    im_handle2 = ax3.imshow(occupancy, alpha=0.5, cmap='gray')
    im_handle2.set_extent([
        min([x_min, g_world_map[0, 2]]),
        x_max,
        min([y_min, g_world_map[1, 2]]),
        y_max
    ])


    ################################

    plt.show()