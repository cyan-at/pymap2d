#! /usr/bin/env python3

import rclpy
from rclpy.node import Node

import numpy as np, math, sys
import os, sys, time
from collections import deque
import threading

from rcl_interfaces.msg import SetParametersResult

from common import *
from contour import *

import CMap2D

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

class SDFNode(Node):
    def __init__(self, args):
        super().__init__('sdf_node')

        self.sdf_pub = self.create_publisher(
              OccupancyGrid,
              '/sdf',
              rclpy.qos.qos_profile_parameters)

        self.plan_pub = self.create_publisher(
              Path,
              '/contour_plan',
              rclpy.qos.qos_profile_parameters)

        self.mutable_hb = {
            "hb_lock" : threading.Lock(),
            "hb" : True,
        }
        self.running = True

        self.sdf_lock = threading.Lock()
        self.sdf_cv = threading.Condition(self.sdf_lock)
        self.sdf_queue = deque()

        self.last_map = None
        self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_cb,
            rclpy.qos.qos_profile_parameters,
        )

        if len(args.path) > 0:
            self.save_path = os.path.abspath(args.path)
            self.get_logger().warn("save_path: {}".format(
                self.save_path))
        else:
            self.save_path = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.transform_lock = threading.Lock()
        self.latest_xytheta = None
        self.latest_homd2d = None
        self.tf_sem = threading.Semaphore(1)

    def sdf_pub_target(self):
        while self.running:
            self.sdf_lock.acquire()
            # predicate
            while self.running and len(self.sdf_queue) == 0:
                self.sdf_cv.wait()
            self.sdf_lock.release()

            if len(self.sdf_queue) == 0:
                self.get_logger().warn('killed')
                return

            self.get_logger().warn("sdf_queue: {}".format(
                len(self.sdf_queue)))

            # drain the buffer pattern, only do the newest one
            (sdf_msg, plan_msg) = self.sdf_queue.popleft()
            while len(self.sdf_queue) > 0:
                self.sdf_queue.pop()

            # self.get_logger().warn('publishing sdf map')
            self.sdf_pub.publish(sdf_msg)
            self.plan_pub.publish(plan_msg)

    def transform_thread(self):
        while self.running:
            self.tf_sem.acquire()
            try:
                transform = self.tf_buffer.lookup_transform(
                      'map',
                      'base_footprint',
                      rclpy.time.Time(),
                      timeout=rclpy.duration.Duration(seconds=0.1))

                _, _, yaw = euler_from_quaternion([
                      transform.transform.rotation.x,
                      transform.transform.rotation.y,
                      transform.transform.rotation.z,
                      transform.transform.rotation.w,
                ])

                with self.transform_lock:
                    self.latest_xytheta = [
                        transform.transform.translation.x,
                        transform.transform.translation.y,
                        yaw]

                    self.latest_homd2d = two_d_make_x_y_theta_hom(
                        *self.latest_xytheta)

            except TransformException:
                # self.get_logger().warn("Waiting for transformation")
                pass

    def map_cb(self, msg):
        map_ = CMap2D.CMap2D()
        map_.from_msg(msg)

        # put a 'border' on the og
        # so that sdf thinks the border is obstacle
        map_._occupancy[0, :] = 100
        map_._occupancy[-1, :] = 100
        map_._occupancy[:, 0] = 100
        map_._occupancy[:, -1] = 100

        sdf_map = map_.as_sdf().T

        sdf_map[sdf_map == -np.inf] = 0
        sdf_map[sdf_map == np.inf] = 0

        m1 = np.min(sdf_map)
        m2 = np.max(sdf_map)

        self.get_logger().warn(
            "min {}, max {}".format(
                m1,
                m2
            ))

        func = lambda t, m1=m1, m2=m2: (np.abs(t - m1) / np.abs(m2 - m1) * 255)
        vfunc = np.vectorize(func)
        sdf_map = vfunc(sdf_map)

        self.get_logger().warn(
            "min2 {}, max2 {}".format(
                np.min(sdf_map),
                np.max(sdf_map)
            ))

        sdf_map = sdf_map.astype(np.int8)

        self.get_logger().warn(
            "min3 {}, max3 {}".format(
                np.min(sdf_map),
                np.max(sdf_map)
            ))

        sdf_msg = numpy_to_occupancy_grid(sdf_map)
        sdf_msg.header.frame_id = "map"
        sdf_msg.header.stamp = msg.header.stamp
        sdf_msg.info.resolution = msg.info.resolution
        sdf_msg.info.origin = msg.info.origin

        file_name = None
        path_name = None
        if self.save_path is not None:
            file_name, _ = Util.get_next_valid_name_increment(
                self.save_path,
                "sdf_map", 0, "", "npy")

            path_name, _ = Util.get_next_valid_name_increment(
                self.save_path,
                "sdf_path", 0, "", "npy")

        self.tf_sem.release()

        with self.transform_lock:
            if (not np.isnan(m1) and not np.isnan(m2) and (self.latest_homd2d is not None)):

                # import ipdb; ipdb.set_trace()

                payload = {
                        'sdf_map' : sdf_map,
                        'g_map_basefootprint' : self.latest_homd2d,
                        'latest_xytheta' : self.latest_xytheta,
                        'msg' : sdf_msg,
                        'range' : [m1, m2],
                        # 'map' : map_.occupancy()
                        'map' : np.array(msg.data).reshape((msg.info.height, msg.info.width))
                    }

                # print("working...", file_name)
                # with open(file_name, 'wb') as f:
                #     np.save(f, 
                #         payload
                #     )

                all_homs, _, _ = contour_step(
                    payload, iterations=20) # fewer iters to prevent 'degeneracy'

                path_msg = make_path_msg(
                    all_homs,
                    sdf_msg)
                print("len()", len(all_homs))

                if path_name is not None:
                    payload = {
                        "path" : all_homs
                    }
                    with open(path_name, 'wb') as f:
                        np.save(f, 
                            payload
                        )

                self.sdf_queue.append((
                    sdf_msg,
                    path_msg))
                with self.sdf_lock:
                    self.sdf_cv.notify_all()

def main(args=None):
    rclpy.init(args=args)

    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument('--path',
        type=str, help='path', default="")
    args = parser.parse_args()

    node = SDFNode(args)

    # th2 = threading.Thread(target=node.get_static_tf)
    # th2.start()

    th3 = threading.Thread(target=node.transform_thread)
    th3.start()

    th1 = threading.Thread(target=node.sdf_pub_target)
    th1.start()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except rclpy.executors.ExternalShutdownException:
        pass
    finally:
        node.running = False
        with node.sdf_lock:
            node.sdf_cv.notify_all()

        node.tf_sem.release()

        rclpy.try_shutdown()
        node.destroy_node()

    th1.join()
    # th2.join()
    th3.join()

if __name__ == '__main__':
    main()
