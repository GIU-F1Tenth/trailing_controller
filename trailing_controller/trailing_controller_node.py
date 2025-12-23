#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import PoseStamped
import numpy as np

from trailing_controller.frenet_converter import FrenetConverter
from trailing_controller.longitudinal_controller import LongitudinalController
from trailing_controller.lateral_controller import LateralController


class TrailingControllerNode(Node):
    def __init__(self):
        super().__init__('trailing_controller_node')
        
        self.declare_parameters(
            namespace='',
            parameters=[
                ('kp', 1.0),
                ('kd', 0.2),
                ('target_gap', 2.0),
                ('v_blind', 1.5),
                ('track_length', 100.0),
                ('wheelbase', 0.33),
                ('lookahead_gain', 0.6),
                ('lookahead_offset', -0.18),
                ('control_rate', 50.0),
            ]
        )
        
        kp = self.get_parameter('kp').value
        kd = self.get_parameter('kd').value
        target_gap = self.get_parameter('target_gap').value
        v_blind = self.get_parameter('v_blind').value
        track_length = self.get_parameter('track_length').value
        wheelbase = self.get_parameter('wheelbase').value
        lookahead_gain = self.get_parameter('lookahead_gain').value
        lookahead_offset = self.get_parameter('lookahead_offset').value
        control_rate = self.get_parameter('control_rate').value
        
        self.frenet_converter = FrenetConverter()
        self.longitudinal_controller = LongitudinalController(
            kp=kp, kd=kd, target_gap=target_gap, v_blind=v_blind, track_length=track_length
        )
        self.lateral_controller = LateralController(
            wheelbase=wheelbase, lookahead_gain=lookahead_gain, lookahead_offset=lookahead_offset
        )
        
        self.ego_x = 0.0
        self.ego_y = 0.0
        self.ego_psi = 0.0
        self.ego_vx = 0.0
        self.ego_vy = 0.0
        self.ego_s = 0.0
        self.ego_vs = 0.0
        
        self.opp_x = 0.0
        self.opp_y = 0.0
        self.opp_s = 0.0
        self.opp_vs = 0.0
        self.opp_vd = 0.0
        self.has_opponent = False
        
        self.raceline_initialized = False
        
        self.ego_odom_sub = self.create_subscription(
            Odometry, '/ego_racecar/odom', self.ego_odom_callback, 10
        )
        
        self.raceline_sub = self.create_subscription(
            Path, '/pp_path', self.raceline_callback, 10
        )
        
        self.opponent_sub = self.create_subscription(
            MarkerArray, '/object_velocities', self.opponent_callback, 10
        )
        
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, '/drive', 10
        )
        
        control_period = 1.0 / control_rate
        self.control_timer = self.create_timer(control_period, self.control_loop)
        
        self.get_logger().info('Trailing Controller Node initialized')
        
    def raceline_callback(self, msg: Path):
        if len(msg.poses) < 2:
            return
            
        s_coords = []
        x_coords = []
        y_coords = []
        
        s = 0.0
        s_coords.append(s)
        x_coords.append(msg.poses[0].pose.position.x)
        y_coords.append(msg.poses[0].pose.position.y)
        
        for i in range(1, len(msg.poses)):
            dx = msg.poses[i].pose.position.x - msg.poses[i-1].pose.position.x
            dy = msg.poses[i].pose.position.y - msg.poses[i-1].pose.position.y
            ds = np.sqrt(dx**2 + dy**2)
            s += ds
            s_coords.append(s)
            x_coords.append(msg.poses[i].pose.position.x)
            y_coords.append(msg.poses[i].pose.position.y)
            
        self.frenet_converter.update_raceline(
            np.array(s_coords), np.array(x_coords), np.array(y_coords)
        )
        
        self.longitudinal_controller.track_length = s
        self.raceline_initialized = True
        
        self.get_logger().info(f'Raceline updated with {len(msg.poses)} points, track length: {s:.2f}m', throttle_duration_sec=5.0)
        
    def ego_odom_callback(self, msg: Odometry):
        self.ego_x = msg.pose.pose.position.x
        self.ego_y = msg.pose.pose.position.y
        
        orientation_q = msg.pose.pose.orientation
        siny_cosp = 2 * (orientation_q.w * orientation_q.z + orientation_q.x * orientation_q.y)
        cosy_cosp = 1 - 2 * (orientation_q.y * orientation_q.y + orientation_q.z * orientation_q.z)
        self.ego_psi = np.arctan2(siny_cosp, cosy_cosp)
        
        self.ego_vx = msg.twist.twist.linear.x
        self.ego_vy = msg.twist.twist.linear.y
        
        if self.raceline_initialized:
            try:
                s, d = self.frenet_converter.cartesian_to_frenet(self.ego_x, self.ego_y)
                self.ego_s = s
                self.ego_vs = np.sqrt(self.ego_vx**2 + self.ego_vy**2)
            except Exception as e:
                self.get_logger().error(f'Failed to convert ego position to Frenet: {e}')
                
    def opponent_callback(self, msg: MarkerArray):
        self.has_opponent = False
        
        for marker in msg.markers:
            if marker.ns == 'velocities' and marker.type == 0:
                try:
                    self.opp_x = marker.points[0].x
                    self.opp_y = marker.points[0].y
                    
                    if self.raceline_initialized:
                        s, d = self.frenet_converter.cartesian_to_frenet(self.opp_x, self.opp_y)
                        self.opp_s = s
                        
                        dx = marker.points[1].x - marker.points[0].x
                        dy = marker.points[1].y - marker.points[0].y
                        speed = np.sqrt(dx**2 + dy**2) * 2.0
                        self.opp_vs = speed
                        
                        self.has_opponent = True
                        break
                except Exception as e:
                    self.get_logger().error(f'Failed to process opponent marker: {e}')
                    
    def control_loop(self):
        if not self.raceline_initialized:
            self.get_logger().warn('Raceline not initialized, skipping control', throttle_duration_sec=2.0)
            return
            
        if not self.has_opponent:
            self.get_logger().warn('No opponent detected, stopping', throttle_duration_sec=2.0)
            drive_msg = AckermannDriveStamped()
            drive_msg.header.stamp = self.get_clock().now().to_msg()
            drive_msg.header.frame_id = 'base_link'
            drive_msg.drive.speed = 0.0
            drive_msg.drive.steering_angle = 0.0
            self.drive_pub.publish(drive_msg)
            return
            
        v_des = self.longitudinal_controller.compute_velocity(
            self.ego_s, self.ego_vs, self.opp_s, self.opp_vs
        )
        
        lookahead_distance = self.lateral_controller.lookahead_gain * self.ego_vx + self.lateral_controller.lookahead_offset
        lookahead_distance = max(0.5, lookahead_distance)
        
        s_lookahead = (self.opp_s + lookahead_distance) % self.longitudinal_controller.track_length
        d_lookahead = 0.0
        
        try:
            target_x, target_y = self.frenet_converter.frenet_to_cartesian(s_lookahead, d_lookahead)
        except Exception as e:
            self.get_logger().error(f'Failed to convert lookahead point: {e}')
            return
            
        steering_angle = self.lateral_controller.compute_steering(
            self.ego_x, self.ego_y, self.ego_psi, self.ego_vx, target_x, target_y
        )
        
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = 'base_link'
        drive_msg.drive.speed = float(v_des)
        drive_msg.drive.steering_angle = float(steering_angle)
        
        self.drive_pub.publish(drive_msg)


def main(args=None):
    rclpy.init(args=args)
    node = TrailingControllerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
