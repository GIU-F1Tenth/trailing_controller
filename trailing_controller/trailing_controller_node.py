#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from visualization_msgs.msg import MarkerArray
import numpy as np

from std_msgs.msg import Float64
from trailing_controller.frenet_converter import FrenetConverter

class TrailingControllerNode(Node):
    def __init__(self):
        super().__init__('trailing_controller_node')
        
        self.declare_parameters(
            namespace='',
            parameters=[
                ('kp', 1.0),
                ('kd', 0.2),
                ('target_gap', 2.0),
                ('control_rate', 50.0),
                ('path_topic', '/pp_path'),
                ('object_velocity_topic', '/object_velocities'),
                ('speed_cap_topic', '/speed_cap'),
                ('max_speed', 8.0),
                ('min_speed', 0.0),
            ]
        )
        
        self.kp = self.get_parameter('kp').value
        self.kd = self.get_parameter('kd').value
        self.target_gap = self.get_parameter('target_gap').value
        control_rate = self.get_parameter('control_rate').value
        path_topic = self.get_parameter('path_topic').value
        object_velocity_topic = self.get_parameter('object_velocity_topic').value
        speed_cap_topic = self.get_parameter('speed_cap_topic').value
        self.max_speed = self.get_parameter('max_speed').value
        self.min_speed = self.get_parameter('min_speed').value
        
        self.frenet_converter = FrenetConverter()
    
        self.opp_s = 0.0
        self.opp_d = 0.0
        self.has_opponent = False
        
        self.raceline_initialized = False
        self.prev_error = 0.0
        self.prev_time = self.get_clock().now()
        
        self.raceline_sub = self.create_subscription(
            Path, path_topic, self.raceline_callback, 10
        )
        
        self.opponent_sub = self.create_subscription(
            MarkerArray, object_velocity_topic, self.opponent_callback, 10
        )
        
        self.velocity_pub = self.create_publisher(
            Float64, speed_cap_topic, 10
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
        
        self.raceline_initialized = True
        
        self.get_logger().info(f'Raceline updated with {len(msg.poses)} points, track length: {s:.2f}m', throttle_duration_sec=5.0)
                
    def opponent_callback(self, msg: MarkerArray):
        self.has_opponent = False
        
        for marker in msg.markers:
            if marker.ns == 'velocities' and marker.type == 0:
                try:
                    if self.raceline_initialized:
                        opp_x = marker.points[0].x
                        opp_y = marker.points[0].y
                        s, d = self.frenet_converter.cartesian_to_frenet(opp_x, opp_y)
                        self.opp_s = s
                        self.opp_d = d
                        
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
            self.velocity_pub.publish(Float64(data=-1.0))
            return
            
        opp_x, opp_y = self.frenet_converter.frenet_to_cartesian(self.opp_s, self.opp_d)
        distance = np.sqrt(opp_x**2 +  opp_y**2)
        
        self.get_logger().info(f'Distance to opponent: {distance:.2f}m', throttle_duration_sec=1.0)
       
        current_time = self.get_clock().now()
        error, v_des = self.control_distance(distance, current_time)
        self.prev_time = current_time
        self.prev_error = error
       
        velocity_msg = Float64()
        velocity_msg.data = float(v_des)
        self.velocity_pub.publish(velocity_msg)

    def control_distance(self, distance, current_time):
        dt = (current_time - self.prev_time).nanoseconds / 1e9
        
        error = distance - self.target_gap
        
        if dt > 0:
            error_derivative = (error - self.prev_error) / dt
        else:
            error_derivative = 0.0
        
        v_des = self.kp * error + self.kd * error_derivative
        
        v_des = max(self.min_speed, min(self.max_speed, v_des))
        
        return error, v_des
    
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
