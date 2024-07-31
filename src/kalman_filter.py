import numpy as np

class KalmanFilter():
    def __init__(self, position_vector, dt, std_acceleration, std_measurement):
        self.dt = dt
        self.std_acceleration = std_acceleration
        
        # State transition matrix
        self.state_transition_matrix = np.matrix([[1, self.dt],
                                                  [0, 1]])
        # Measurement matrix
        self.measurement_matrix = np.matrix([[1, 0]])
        
        # Process covariance matrix
        self.state_covariance_matrix = np.matrix([[(self.dt**4)/4, (self.dt**3)/2],
                                                  [(self.dt**3)/2, self.dt**2]]) * self.std_acceleration
        
        # Measurement covariance matrix
        self.measurement_covariance_matrix = std_measurement**2
        
        # Initial state covariance
        self.P = np.eye(2)
        
        # Initial state vector (position and velocity)
        self.X = np.matrix([[position_vector[0]], [0]])  # Initialize with position and zero velocity

    def predict(self):
        # Predict the state and covariance
        self.X = self.state_transition_matrix * self.X
        self.P = self.state_transition_matrix * self.P * self.state_transition_matrix.T + self.state_covariance_matrix

    def update(self, z):
        # Kalman Gain
        S = self.measurement_matrix * self.P * self.measurement_matrix.T + self.measurement_covariance_matrix
        K = self.P * self.measurement_matrix.T * np.linalg.inv(S)
        
        # Update the estimate
        Y = np.matrix(z).T - self.measurement_matrix * self.X
        self.X = self.X + K * Y
        
        # Update the state covariance
        I = np.eye(self.P.shape[0])
        self.P = (I - K * self.measurement_matrix) * self.P

    def get_position(self):
        return self.X[0, 0], self.X[1, 0]
