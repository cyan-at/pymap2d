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

terminate_ticks = 100
target_v = 0.2

from std_srvs.srv import Empty, Trigger

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

    def rtn_vel_cb(self, msg):
        with self.rtn_lock:
            self.state.v = msg.linear.x

    def path_cb(self, msg):
        with self.path_lock:
            self.xythetas = path_to_xytheta(msg)

        # reset
        if self.sem_counter < 1 and self.latest_xytheta is not None:
            self.active_path = msg

            self.sim_state = State(
                x=self.latest_xytheta[0],
                y=self.latest_xytheta[1],
                yaw=self.latest_xytheta[2],
                v=0.0
            )
            self.step_path1.header = msg.header
            self.step_path1.poses = [
                xytheta_to_ps(*self.latest_xytheta, msg)]
            self.step_pub1.publish(self.step_path1)

            self.step_path2.header = msg.header
            self.step_path2.poses = [
                xytheta_to_ps(*self.latest_xytheta, msg)]
            self.step_pub2.publish(self.step_path2)

            self.state.e_th = 0.0
            self.state.e = 0.0

            self.sem1.release()
            self.sem_counter += 1
        else:
            self.get_logger().warn(
                "ignoring latest sdf_node path")

        self.get_logger().warn("_value {}".format(
            self.sem1._value))

        self.get_logger().warn("len(xythetas) {}".format(
            len(self.xythetas)))

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

            self.get_logger().warn("_value {}".format(
                self.sem1._value))

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

            xs = interpolate(xythetas[:, 0], 10)
            ys = interpolate(xythetas[:, 1], 10)
            yaws = interpolate(xythetas[:, 2], 10)

            ticks = 0
            start = time.time()

            while ticks < terminate_ticks:
                dl, idx, self.state.e, self.state.e_th, ai = lqr_speed_steering_control(
                    self.state, self.state.e, self.state.e_th,
                    dt,
                    xs, ys, yaws,
                    lqr_Q, lqr_R, L, target_v)

                dist = np.linalg.norm(
                    self.latest_xytheta[:2] - np.array([xs[idx], ys[idx]]),
                    ord=2)

                # self.get_logger().warn("xy (%.3f, %.3f, %.3f), path_xy (%.3f, %.3f, %.3f), dist=%.3f" % (
                #     self.state.x, self.state.y, self.state.yaw,
                #     xs[idx], ys[idx], yaws[idx],
                #     dist
                #     ))

                # simulation
                sim_dl, sim_idx, self.sim_state.e, self.sim_state.e_th, sim_ai =\
                    lqr_speed_steering_control(
                    self.sim_state, self.sim_state.e, self.sim_state.e_th,
                    dt,
                    xs, ys, yaws,
                    lqr_Q, lqr_R, L, target_v)

                self.sim_state.x = self.sim_state.x + self.sim_state.v * math.cos(self.sim_state.yaw) * dt
                self.sim_state.y = self.sim_state.y + self.sim_state.v * math.sin(self.sim_state.yaw) * dt
                self.sim_state.yaw = self.sim_state.yaw + self.sim_state.v / L * math.tan(sim_dl) * dt
                self.sim_state.v = self.sim_state.v + sim_ai * dt

                self.step_path1.poses.append(xytheta_to_ps(
                    self.sim_state.x,
                    self.sim_state.y,
                    self.sim_state.yaw,
                    self.step_path1.poses[-1]))
                self.step_pub1.publish(self.step_path1)

                self.step_path2.poses.append(xytheta_to_ps(
                    xs[sim_idx],
                    ys[sim_idx],
                    yaws[sim_idx],
                    self.step_path2.poses[-1]))
                self.step_pub2.publish(self.step_path2)

                twist = Twist()
                twist.linear.x = self.sim_state.v + (sim_ai * dt)
                twist.angular.z = sim_dl * dt # not sure about this

                # clamp it down
                # max_steer = np.deg2rad(180.0)
                # if sim_dl >= max_steer:
                #     sim_dl = max_steer
                # if sim_dl <= - max_steer:
                #     sim_dl = - max_steer
                twist.linear.x = max(twist.linear.x, -0.5)
                twist.linear.x = min(twist.linear.x, 0.5)
                twist.angular.z = max(twist.angular.z, -0.5)
                twist.angular.z = min(twist.angular.z, 0.5)

                # self.get_logger().warn("ai {}, dv: {}, v: {}, w: {}".format(
                #     ai,
                #     ai * dt,
                #     twist.linear.x,
                #     twist.angular.z))

                self.vel_pub.publish(twist)

                # time.sleep(dt)

                ticks += 1
                current = time.time()

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
