import numpy as np

class KalmanFilter():
    def __init__(self, position_vector, dt, std_acceleration, std_measurement):
        self.dt = dt
        self.std_acceleration = std_acceleration
        # Matrix
        self.state_transition_matrix = np.matrix([[1, self.dt],
                                                  [0, 1]])
        self.measurement_matrix = np.matrix([[1, 0]])
        self.state_covariance_matrix = np.matrix([[(self.dt**4)/4, (self.dt**3)/2],
                                                  [(self.dt**3)/2, self.dt**2]])*self.std_acceleration
        self.measurement_covariance_matrix = std_measurement**2

        # Priori
        self.P = np.matrix([[1, 0],
                            [0, 1]])
        self.X = np.matrix([position_vector, [ 0, 0]])

    def predict(self):
        # Neglecting control vector
        self.X = np.dot(self.state_transition_matrix, self.X)
        self.P = np.dot(np.dot(self.state_transition_matrix, self.P), self.state_transition_matrix.T) + self.state_covariance_matrix

    def update(self, z):
        # covariance innovation
        S = np.dot(self.measurement_matrix, np.dot(self.P, self.measurement_matrix.T)) + self.measurement_covariance_matrix
        # Optimal Kalman gain
        K = np.dot(np.dot(self.P, self.measurement_matrix.T), np.linalg.inv(S))
        # State innovation
        Y = z - np.dot(self.measurement_matrix, self.X)

        # Posteori
        self.X = np.round(self.X + np.dot(K, Y))

        I = np.eye(self.measurement_matrix.shape[1])
        self.P = (I - (K * self.measurement_matrix)) * self.P

    def get_position(self):
        return [self.X[0, 0], self.X[0, 1]]