# Trailing Controller

A ROS2 implementation of the trailing controller from the ForzaETH Race Stack paper for autonomous head-to-head racing.

## Overview

This package implements a trailing controller that follows an opponent vehicle by maintaining a constant gap distance. The controller uses:

- **Longitudinal Control**: PD controller with feedforward term to maintain constant gap
- **Lateral Control**: MAP (Model- and Acceleration-based Pursuit) controller for steering
- **Frenet Frame**: Coordinate transformation for racing line following

## Architecture

### Components

1. **FrenetConverter** (`frenet_converter.py`)
   - Converts between Cartesian and Frenet coordinates
   - Uses cubic splines for smooth interpolation
   - Handles circular track wrapping

2. **LongitudinalController** (`longitudinal_controller.py`)
   - PD controller with opponent velocity feedforward
   - Gap error computation with track wrapping
   - Minimum velocity enforcement for blind spots

3. **LateralController** (`lateral_controller.py`)
   - MAP controller implementation
   - Lookup table for steering angle computation
   - L1 guidance law for lookahead point tracking

4. **TrailingControllerNode** (`trailing_controller_node.py`)
   - Main ROS2 node integrating all components
   - Subscribes to odometry, raceline, and opponent detection
   - Publishes Ackermann drive commands

## Usage

### Launch the Controller

```bash
ros2 launch trailing_controller trailing_controller_launch.py
```

### Topics

**Subscribed:**
- `/ego_racecar/odom` (nav_msgs/Odometry) - Ego vehicle state
- `/pp_path` (nav_msgs/Path) - Racing line waypoints
- `/object_velocities` (visualization_msgs/MarkerArray) - Detected opponent

**Published:**
- `/drive` (ackermann_msgs/AckermannDriveStamped) - Control commands

### Parameters

Configuration in `config/trailing_controller_params.yaml`:

```yaml
kp: 1.0                    # Proportional gain for gap control
kd: 0.2                    # Derivative gain for gap control
target_gap: 2.0            # Target following distance (m)
v_blind: 1.5               # Minimum velocity when no opponent visible (m/s)
track_length: 100.0        # Track length for wrapping (m)
wheelbase: 0.33            # Vehicle wheelbase (m)
lookahead_gain: 0.6        # Lookahead distance proportional gain
lookahead_offset: -0.18    # Lookahead distance offset
control_rate: 50.0         # Control loop frequency (Hz)
```

## Controller Details

### Longitudinal Control (Equation 12 from paper)

```
v_des = v_s,opp - (kp * e_gap + kd * Δv_s)
```

Where:
- `e_gap = target_gap - measured_gap`
- `Δv_s = ego_vs - opp_vs`

### Lateral Control (MAP Controller)

1. Compute lookahead distance: `L_d = m * v_x + q`
2. Calculate required lateral acceleration using L1 guidance
3. Lookup steering angle from pre-computed table

## Testing

The controller is designed to:
1. Detect opponent from `/object_velocities` topic
2. Track opponent position in Frenet coordinates
3. Maintain constant gap distance behind opponent
4. Follow opponent's trajectory smoothly