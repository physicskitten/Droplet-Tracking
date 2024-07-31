import numpy as np

class KalmanFilter():
    def __init__(self, position, dt, std_acceleration, std_measurement):
        self.dt = dt
        self.std_acceleration = std_acceleration
        self.std_measurement = std_measurement
        
        # State transition matrix
        self.A = np.matrix([[1, self.dt],
                            [0, 1]])
        
        # Control matrix (not used in this case, but here for completeness)
        self.B = np.matrix([[0.5 * (self.dt**2)],
                            [self.dt]])
        
        # Measurement matrix
        self.H = np.matrix([[1, 0]])
        
        # Process noise covariance matrix
        self.Q = np.matrix([[self.dt**4 / 4, self.dt**3 / 2],
                            [self.dt**3 / 2, self.dt**2]]) * self.std_acceleration**2
        
        # Measurement noise covariance matrix
        self.R = np.matrix([[self.std_measurement**2]])
        
        # Initial state estimate
        self.x = np.matrix([[position],
                            [0]])  # Initial velocity is assumed to be zero
        
        # Initial covariance estimate
        self.P = np.eye(2)

    def predict(self):
        # Predict the next state
        self.x = self.A * self.x
        self.P = self.A * self.P * self.A.T + self.Q

    def update(self, z):
        # Compute the Kalman gain
        S = self.H * self.P * self.H.T + self.R
        K = self.P * self.H.T * np.linalg.inv(S)
        
        # Update the estimate via measurement z
        y = z - self.H * self.x  # Innovation or residual
        self.x = self.x + K * y
        
        # Update the error covariance
        I = np.eye(self.A.shape[0])
        self.P = (I - K * self.H) * self.P

    def get_position(self):
        return self.x[0, 0]

    def get_velocity(self):
        return self.x[1, 0]
