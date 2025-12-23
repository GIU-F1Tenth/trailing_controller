#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
from tf_transformations import euler_from_quaternion

from trailing_controller.target_selector import TargetSelector
from trailing_controller.velocity_controller import VelocityController
from trailing_controller.steering_controller import HybridSteeringController
from trailing_controller.detected_object import DetectedObject



class TrailingControllerNode(Node):
    def __init__(self):
        super().__init__('trailing_controller_node')
        
        self.declare_parameters(
            namespace='',
            parameters=[
                ('max_lateral_distance', 3.0),
                ('max_longitudinal_distance', 20.0),
                ('prefer_ahead', True),
                ('desired_gap', 2.5),
                ('min_gap', 1.5),
                ('max_gap', 4.0),
                ('velocity_kp', 0.8),
                ('velocity_kd', 0.3),
                ('velocity_ki', 0.05),
                ('max_velocity', 7.0),
                ('min_velocity', 0.5),
                ('emergency_brake_distance', 1.0),
                ('wheelbase', 0.33),
                ('max_steer', 0.42),
                ('stanley_k', 1.2),
                ('stanley_ks', 0.5),
                ('pp_lookahead_gain', 0.6),
                ('pp_lookahead_min', 1.0),
                ('pp_lookahead_max', 3.5),
                ('switch_velocity', 2.5),
                ('control_rate', 20.0),
                ('prediction_time', 0.5),
                ('use_prediction', True),
            ]
        )
        
        self.target_selector = TargetSelector(
            max_lateral_distance=self.get_parameter('max_lateral_distance').value,
            max_longitudinal_distance=self.get_parameter('max_longitudinal_distance').value,
            prefer_ahead=self.get_parameter('prefer_ahead').value
        )
        
        self.velocity_controller = VelocityController(
            desired_gap=self.get_parameter('desired_gap').value,
            min_gap=self.get_parameter('min_gap').value,
            max_gap=self.get_parameter('max_gap').value,
            kp=self.get_parameter('velocity_kp').value,
            kd=self.get_parameter('velocity_kd').value,
            ki=self.get_parameter('velocity_ki').value,
            max_velocity=self.get_parameter('max_velocity').value,
            min_velocity=self.get_parameter('min_velocity').value,
            emergency_brake_distance=self.get_parameter('emergency_brake_distance').value
        )
        
        self.steering_controller = HybridSteeringController(
            wheelbase=self.get_parameter('wheelbase').value,
            max_steer=self.get_parameter('max_steer').value,
            stanley_k=self.get_parameter('stanley_k').value,
            stanley_ks=self.get_parameter('stanley_ks').value,
            pp_lookahead_gain=self.get_parameter('pp_lookahead_gain').value,
            pp_lookahead_min=self.get_parameter('pp_lookahead_min').value,
            pp_lookahead_max=self.get_parameter('pp_lookahead_max').value,
            switch_velocity=self.get_parameter('switch_velocity').value
        )
        
        self.prediction_time = self.get_parameter('prediction_time').value
        self.use_prediction = self.get_parameter('use_prediction').value
        
        self.ego_x = 0.0
        self.ego_y = 0.0
        self.ego_yaw = 0.0
        self.ego_velocity = 0.0
        
        self.detected_objects = {}
        self.velocity_info = {}
        self.current_target = None
        self.last_control_time = self.get_clock().now()
        
        self.odom_sub = self.create_subscription(
            Odometry, 'ego_racecar/odom', self.odom_callback, 10
        )
        
        self.bbox_sub = self.create_subscription(
            MarkerArray, 'object_bboxes', self.bbox_callback, 10
        )
        
        self.velocity_sub = self.create_subscription(
            MarkerArray, 'object_velocities', self.velocity_callback, 10
        )
        
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, 'drive', 10
        )
        
        self.target_marker_pub = self.create_publisher(
            Marker, 'target_marker', 10
        )
        
        control_period = 1.0 / self.get_parameter('control_rate').value
        self.control_timer = self.create_timer(control_period, self.control_callback)
        
        self.get_logger().info('Trailing Controller Node initialized')
        
    def odom_callback(self, msg: Odometry):
        self.ego_x = msg.pose.pose.position.x
        self.ego_y = msg.pose.pose.position.y
        
        orientation = msg.pose.pose.orientation
        _, _, self.ego_yaw = euler_from_quaternion([
            orientation.x, orientation.y, orientation.z, orientation.w
        ])
        
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.ego_velocity = np.sqrt(vx**2 + vy**2)
    
    def bbox_callback(self, msg: MarkerArray):
        current_ids = set()
        
        for marker in msg.markers:
            if marker.action == marker.DELETEALL:
                continue
                
            if len(marker.points) < 4:
                continue
            
            center_x = sum(p.x for p in marker.points[:4]) / 4.0
            center_y = sum(p.y for p in marker.points[:4]) / 4.0
            
            obj_id = marker.id
            current_ids.add(obj_id)
            
            if obj_id in self.detected_objects:
                self.detected_objects[obj_id].x = center_x
                self.detected_objects[obj_id].y = center_y
            else:
                self.detected_objects[obj_id] = DetectedObject(obj_id, center_x, center_y)
        
        for obj_id in list(self.detected_objects.keys()):
            if obj_id not in current_ids:
                del self.detected_objects[obj_id]
                if obj_id in self.velocity_info:
                    del self.velocity_info[obj_id]
                    
    def velocity_callback(self, msg: MarkerArray):
        for marker in msg.markers:
            if marker.action == marker.DELETEALL:
                continue
                
            if len(marker.points) < 2:
                continue
            
            obj_id = marker.id
            
            start = marker.points[0]
            end = marker.points[1]
            
            vx = (end.x - start.x) * 2.0
            vy = (end.y - start.y) * 2.0
            
            velocity = np.sqrt(vx**2 + vy**2)
            
            is_static = (marker.color.r > 0.9 and marker.color.g > 0.9)
            
            self.velocity_info[obj_id] = {
                'vx': vx,
                'vy': vy,
                'velocity': velocity,
                'is_static': is_static
            }
            
            if obj_id in self.detected_objects:
                self.detected_objects[obj_id].vs = velocity
                self.detected_objects[obj_id].is_static = is_static
        
    def objects_callback(self, msg):
        self.detected_objects = msg.objects
        
    def predict_target_position(self, target, dt: float):
        predicted_x = target.x + target.vd * np.cos(np.arctan2(target.vd, target.vs)) * dt
        predicted_y = target.y + target.vd * np.sin(np.arctan2(target.vd, target.vs)) * dt
        
        return predicted_x, predicted_y
        
    def control_callback(self):
        current_time = self.get_clock().now()
        dt = (current_time - self.last_control_time).nanoseconds / 1e9
        self.last_control_time = current_time
        
        if dt <= 0:
            dt = 0.05
        
        objects_list = list(self.detected_objects.values())
        
        target_id = self.target_selector.select_target(objects_list)
        
        if target_id is None:
            self.publish_stop_command()
            self.current_target = None
            return
        
        self.current_target = self.target_selector.get_target_object(
            objects_list, target_id
        )
        
        if self.current_target is None:
            self.publish_stop_command()
            return
        
        target_x = self.current_target.x
        target_y = self.current_target.y
        
        if self.use_prediction:
            target_x, target_y = self.predict_target_position(
                self.current_target, self.prediction_time
            )
        
        distance_to_target = np.sqrt(
            (target_x - self.ego_x)**2 + (target_y - self.ego_y)**2
        )
        
        target_velocity_magnitude = np.sqrt(
            self.current_target.vs**2 + self.current_target.vd**2
        )
        
        desired_velocity = self.velocity_controller.compute_velocity(
            distance_to_target, target_velocity_magnitude, dt
        )
        
        steering_angle = self.steering_controller.compute_steering(
            target_x, target_y,
            self.ego_x, self.ego_y, self.ego_yaw,
            max(self.ego_velocity, 0.5)
        )
        
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = current_time.to_msg()
        drive_msg.header.frame_id = 'base_link'
        drive_msg.drive.speed = float(desired_velocity)
        drive_msg.drive.steering_angle = float(steering_angle)
        
        self.drive_pub.publish(drive_msg)
        
        self.publish_target_marker(target_x, target_y)
        
    def publish_stop_command(self):
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = 'base_link'
        drive_msg.drive.speed = 0.0
        drive_msg.drive.steering_angle = 0.0
        self.drive_pub.publish(drive_msg)
        
    def publish_target_marker(self, target_x: float, target_y: float):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'trailing_target'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        marker.pose.position.x = float(target_x)
        marker.pose.position.y = float(target_y)
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 0.8
        
        self.target_marker_pub.publish(marker)


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
