#! /usr/bin/env python3

import rclpy
from rclpy.node import Node

import numpy as np, math, sys
import os, sys, time
import threading
from collections import deque

from common import *
from lqr import *

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from rcl_interfaces.msg import SetParametersResult
from geometry_msgs.msg import Twist

target_v = 0.2

from std_srvs.srv import Empty, Trigger
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, String

class LQRNode(Node):
    def __init__(self):
        super().__init__('lqr_node')

        self.path_lock = threading.Lock()
        self.xythetas = []
        self.create_subscription(
            Path,
            '/contour_plan',
            self.path_cb,
            rclpy.qos.qos_profile_parameters,
        )

        self.sem_counter = 0
        self.sem1 = threading.Semaphore(1)

        self.create_subscription(
            Twist,
            '/gen2_base_controller/rtn_vel',
            self.rtn_vel_cb,
            rclpy.qos.qos_profile_parameters,
        )
        self.rtn_lock = threading.Lock()

        self.vel_twist = Twist()
        self.stop_twist = Twist()
        self.vel_pub = self.create_publisher(
              Twist,
              '/cmd_vel',
              rclpy.qos.qos_profile_parameters)

        self.mutable_hb = {
            "hb_lock" : threading.Lock(),
            "hb" : True,
        }
        self.running = True

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.transform_lock = threading.Lock()
        self.latest_xytheta = None
        self.latest_homd2d = None

        self.sdf_lock = threading.Lock()
        self.sdf_cv = threading.Condition(self.sdf_lock)
        self.sdf_queue = deque()

        self.state = State()
        self.sim_state = State()

        self.step_path1 = Path()
        self.step_pub1 = self.create_publisher(
              Path,
              '/lqr_step1',
              rclpy.qos.qos_profile_parameters)

        self.step_path2 = Path()
        self.step_pub2 = self.create_publisher(
              Path,
              '/lqr_step2',
              rclpy.qos.qos_profile_parameters)

        self.active_path = None
        self.active_path_pub = self.create_publisher(
              Path,
              '/active_plan',
              rclpy.qos.qos_profile_parameters)

        self.debug_msg = Float32MultiArray()
        self.debug_msg.layout.dim.append(MultiArrayDimension(
            label='all', stride=11, size=1))
        
        self.debug_msg.layout.data_offset = 0

        self.debug_msg.data = np.array([0.0] * 11)
        self.debug_pub1 = self.create_publisher(
              Float32MultiArray,
              '/debug1',
              rclpy.qos.qos_profile_parameters)

        self.debug_msg2 = String()
        self.debug_pub2 = self.create_publisher(
              String,
              '/debug2',
              rclpy.qos.qos_profile_parameters)

    def rtn_vel_cb(self, msg):
        with self.rtn_lock:
            self.state.v = msg.linear.x

    def path_cb(self, msg):
        with self.path_lock:
            self.xythetas = path_to_xytheta(msg)

        with self.transform_lock:
            local_xytheta = self.latest_xytheta

        # reset
        if self.sem_counter < 1 and local_xytheta is not None:
            self.active_path = msg

            self.sim_state = State(
                x=local_xytheta[0],
                y=local_xytheta[1],
                yaw=local_xytheta[2],
                v=0.0
            )
            self.step_path1.header = msg.header
            self.step_path1.poses = [
                xytheta_to_ps(*local_xytheta, msg)]
            self.step_pub1.publish(self.step_path1)

            self.step_path2.header = msg.header
            self.step_path2.poses = [
                xytheta_to_ps(*local_xytheta, msg)]
            self.step_pub2.publish(self.step_path2)

            self.state.e_th = 0.0
            self.state.e = 0.0

            self.sem1.release()
            self.sem_counter += 1
        # else:
        #     self.get_logger().warn(
        #         "ignoring latest sdf_node path")

        # self.get_logger().warn("_value {}".format(
        #     self.sem1._value))

        # self.get_logger().warn("len(xythetas) {}".format(
        #     len(self.xythetas)))

    def transform_thread(self):
        while self.running:
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
                    self.latest_xytheta = np.array([
                        transform.transform.translation.x,
                        transform.transform.translation.y,
                        yaw])
                    self.latest_homd2d = two_d_make_x_y_theta_hom(
                        *self.latest_xytheta)

                    self.state.x = transform.transform.translation.x
                    self.state.y = transform.transform.translation.y
                    self.state.yaw = yaw

            except TransformException:
                # self.get_logger().warn("Waiting for transformation")
                pass

    def target1(self):
        while self.running:
            self.sem1.acquire()

            # self.get_logger().warn("_value {}".format(
            #     self.sem1._value))

            if self.xythetas is None:
                self.get_logger().warn("killed!")
                return

            if len(self.xythetas) == 0:
                self.get_logger().warn('noop')
                continue

            xythetas = None
            with self.path_lock:
                xythetas = self.xythetas

            self.active_path_pub.publish(self.active_path)

            self.get_logger().warn("xythetas shape: {}".format(
                xythetas.shape))

            if xythetas.shape[0] == 1:
                self.get_logger().warn("injecting dummy wpt to jog slam")
                # dxdy = np.random.rand(1, 2) * 0.1 - 0.05
                dxdy = np.random.rand(1, 2) * 0.05
                newrow = [xythetas[0, 0] + dxdy[0], xythetas[0, 1] + dxdy[1], xythetas[0, 2]]
                xythetas = np.vstack([xythetas, newrow])

            ############################

            xs = interpolate(xythetas[:, 0], 40)
            ys = interpolate(xythetas[:, 1], 40)

            # this is #important, the key is to set the yaw
            # for a wpt to align with the next path segment
            # ayaw = xythetas[:, 2]
            # this, in combination with slerp3
            ayaw = list(xythetas[1:, 2])
            ayaw.append(xythetas[-1, 2]) # note: maintain alignment on last waypoint

            # # INTERPOATION ROTAIONS SUCKS!
            # yaws = [(x + np.pi) % (2 * np.pi) - np.pi for x in xythetas[:, 2]]
            # yaws = rotation_smooth(yaws)
            # # this is key
            # # interpolating between -3.14 and 3.14 through 0 is spatially wrong
            # yaws = interpolate(yaws, 40)

            # INTERPOATION ROTAIONS SUCKS!
            cyaw = [(x + np.pi) % (2 * np.pi) - np.pi for x in ayaw]
            cyaw = rotation_smooth(cyaw)
            # this is key
            # interpolating between -3.14 and 3.14 through 0 is spatially wrong
            # xys = [np.array([ax[i], ay[i]]) for i in range(len(ax))]

            yaws = slerp3(cyaw, xythetas[:, :2], 40)
            # cyaw = do_repeat(cyaw, 40)

            print("len(cx)", len(xs))
            print("len(cy)", len(ys))
            print("len(cyaw)", len(yaws))
            print("################################################")

            ############################

            start = time.time()

            distances = [] # len(cx) - 1
            cumsums = [0.0] # len(cx)
            i = 0
            pt = np.array([xs[i], ys[i]])
            while i < len(xs) - 1:
                new_pt = np.array([xs[i+1], ys[i+1]])
                dist = np.linalg.norm(new_pt - pt, ord=2)
                distances.append(dist)
                cumsums.append(cumsums[-1] + dist)
                i += 1
                pt = new_pt
            cumsums = np.array(cumsums)

            t = 0.0
            distance_traveled = 0.0
            best_dist_estimate = 0.0
            best_idx = 0

            r2 = 0.5 # from contour.py

            if len(cumsums) >= 2:
                terminate_dist = cumsums[-2]
            # else:
            #     print("GOAL: ", xs[0], ys[0], yaws[0])
            #     tmp = np.array([xs[0], ys[0]])

            #     with self.transform_lock:
            #         print("CURRENT: ", self.latest_xytheta[:2])

            #         tmp2 = np.linalg.norm(
            #             tmp - self.latest_xytheta[:2], ord=2
            #             )
            #         print("DIST", tmp2)

            #     terminate_dist = tmp2

            ticks = 0

            with self.transform_lock:
                xys = [self.latest_xytheta[:2]]

            # while distance_traveled < r2 * max(1, len(xythetas) - 1):
            while terminate_dist > best_dist_estimate and (best_idx < len(xs) - 1):
                dl, idx, self.state.e, self.state.e_th, ai, expected_yaw, fb, best_dist_estimate, best_idx = lqr_speed_steering_control(
                    self.state, self.state.e, self.state.e_th,
                    dt,
                    xs, ys, yaws,
                    lqr_Q, lqr_R, L, target_v, t,
                    distance_traveled, cumsums, debug=False)

                # print("best_dist_estimate",
                #     best_dist_estimate,
                #     best_idx,
                #     terminate_dist)

                self.vel_twist.linear.x = self.state.v + (ai * dt)

                # twist.angular.z = (dl - self.latest_xytheta[2]) / dt # not sure about this
                self.vel_twist.angular.z = self.state.v / L * math.tan(dl)

                # clamp it down
                # max_steer = np.deg2rad(180.0)
                # if sim_dl >= max_steer:
                #     sim_dl = max_steer
                # if sim_dl <= - max_steer:
                #     sim_dl = - max_steer

                # twist.linear.x = max(twist.linear.x, -0.5)
                # twist.linear.x = min(twist.linear.x, 0.5)
                self.vel_twist.angular.z = max(self.vel_twist.angular.z, -0.5)
                self.vel_twist.angular.z = min(self.vel_twist.angular.z, 0.5)

                self.vel_pub.publish(self.vel_twist)

                with self.transform_lock:
                    new_xy = self.latest_xytheta[:2]

                dist = np.linalg.norm(
                    new_xy - np.array(xys[-1]),
                    ord=2)
                # print("DIST!!!", new_xy, np.array(xys[-1]), dist, idx)
                distance_traveled += dist

                xys.append(new_xy)

                ############################################

                '''
                # simulation
                sim_dl, sim_idx, self.sim_state.e, self.sim_state.e_th, sim_ai, _, _ =\
                    lqr_speed_steering_control(
                    self.sim_state, self.sim_state.e, self.sim_state.e_th,
                    dt,
                    xs, ys, yaws,
                    lqr_Q, lqr_R, L, target_v, t,
                    distance_traveled, cumsums, debug=False)

                self.sim_state.x = self.sim_state.x + self.sim_state.v * math.cos(self.sim_state.yaw) * dt
                self.sim_state.y = self.sim_state.y + self.sim_state.v * math.sin(self.sim_state.yaw) * dt
                self.sim_state.yaw = self.sim_state.yaw + self.sim_state.v / L * math.tan(sim_dl) * dt
                self.sim_state.v = self.sim_state.v + sim_ai * dt
                '''

                ############################################

                with self.rtn_lock:
                    self.debug_msg.data[0] = self.state.v
                self.debug_msg.data[1] = self.vel_twist.linear.x
                self.debug_msg.data[2] = self.vel_twist.angular.z
                self.debug_msg.data[3] = ai
                self.debug_msg.data[4] = expected_yaw
                with self.transform_lock:
                    self.debug_msg.data[5] = self.latest_xytheta[2]
                self.debug_msg.data[6] = self.state.e_th
                self.debug_msg.data[7] = fb
                # self.debug_msg.data[8] = self.vel_twist.angular.z # redundant
                self.debug_msg.data[9] = dl
                self.debug_msg.data[10] = self.state.e
                self.debug_pub1.publish(self.debug_msg)

                self.debug_msg2.data = "%.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f" % (
                    self.state.v,
                    self.vel_twist.linear.x,
                    self.vel_twist.angular.z,
                    ai,
                    expected_yaw,
                    self.debug_msg.data[5],
                    self.state.e_th,
                    fb,
                    dl,
                    self.state.e
                    )
                self.debug_pub2.publish(self.debug_msg2)

                ############################################

                self.step_path1.poses.append(xytheta_to_ps(
                    # self.sim_state.x,
                    # self.sim_state.y,
                    # self.sim_state.yaw,
                    *new_xy,
                    0.0,
                    self.step_path1.poses[-1]))
                self.step_pub1.publish(self.step_path1)

                self.step_path2.poses.append(xytheta_to_ps(
                    xs[idx],
                    ys[idx],
                    yaws[idx],
                    self.step_path2.poses[-1]))
                self.step_pub2.publish(self.step_path2)

                # distance_traveled += self.sim_state.v * dt

                ############################################

                time.sleep(dt)
                t = t + dt

                ############################################

                ticks += 1
                current = time.time()

            self.vel_pub.publish(self.stop_twist)
            time.sleep(1.0)

            self.get_logger().warn("BAILING!!!!")

            self.sem_counter -= 1

def main(args=None):
    rclpy.init(args=args)

    node = LQRNode()

    # th2 = threading.Thread(target=node.get_static_tf)
    # th2.start()

    th3 = threading.Thread(target=node.transform_thread)
    th3.start()

    th1 = threading.Thread(target=node.target1)
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

        node.xythetas = None
        node.sem1.release()

        rclpy.try_shutdown()
        node.destroy_node()

    th1.join()
    # th2.join()
    th3.join()

if __name__ == '__main__':
    main()
