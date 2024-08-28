# kalman_filter.py

## Overview
`kalman_filter.py` implements a Kalman Filter in Python using the `numpy` library. This Kalman Filter is designed to estimate the position and velocity of an object given noisy measurements. It is particularly useful in applications like object tracking where both prediction and correction of state variables are required.

## Key Components

### `KalmanFilter` Class
The `KalmanFilter` class encapsulates all the operations required to perform prediction and correction on a state vector, which includes position and velocity.

#### Initialization
The constructor (`__init__`) initializes the Kalman Filter with the following parameters:
- **position_vector**: Initial position of the object.
- **dt**: Time interval between measurements.
- **std_acceleration**: Standard deviation of the acceleration noise (process noise).
- **std_measurement**: Standard deviation of the measurement noise.

During initialization, the following matrices are set up:
- **State Transition Matrix (`state_transition_matrix`)**: Models the relationship between position and velocity over time.
- **Measurement Matrix (`measurement_matrix`)**: Maps the true state space into the observed space.
- **Process Covariance Matrix (`state_covariance_matrix`)**: Reflects the uncertainty in the system's state.
- **Measurement Covariance Matrix (`measurement_covariance_matrix`)**: Represents the noise in the measurements.
- **State Covariance Matrix (`P`)**: Represents the uncertainty in the initial state.

#### Methods

- **`predict()`**:
    - Predicts the next state (position and velocity) and updates the state covariance matrix.
  
- **`update(z)`**:
    - Updates the state estimate using a new measurement `z`.
    - Calculates the Kalman Gain, which determines how much the predictions should be adjusted based on the new measurement.
    - Updates both the state vector and the state covariance matrix.

- **`get_position()`**:
    - Returns the current position and velocity estimates from the state

 ## Applications
This Kalman Filter implementation can be used in various applications, including but not limited to:

- **Object Tracking in Video Processing**: Accurately track the position and velocity of moving objects in video sequences, such as vehicles, people, or particles.
  
- **Robotics for Sensor Fusion**: Combine data from multiple sensors (e.g., GPS, accelerometers, gyroscopes) to estimate a robot's position and velocity with higher accuracy.
  
- **Navigation Systems**: Estimate the position and velocity of vehicles, drones, or other moving objects in real-time, even with noisy sensor data.
