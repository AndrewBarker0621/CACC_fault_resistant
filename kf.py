import numpy as np

class KalmanFilter1D:
    """
    Constant-velocity KF for leader state: [position; velocity].
    Passive FTC: if measurement missing, skip update (prediction-only).
    """
    def __init__(self, dt, process_var=1e-2, measurement_var=1e-1):
        self.dt = dt
        self.A = np.array([[1, dt],
                           [0, 1 ]])          # x_k+1 = A x_k
        self.H = np.eye(2)                    # measure [pos, vel]
        self.Q = process_var * np.eye(2)
        self.R = measurement_var * np.eye(2)
        self.x = np.zeros((2, 1))             # [pos; vel]
        self.P = np.eye(2)

    def initialize(self, position, velocity):
        self.x = np.array([[position], [velocity]])
        self.P = np.eye(2)

    def get_state(self):
        return self.x[0, 0], self.x[1, 0]

    def predict(self):
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x.flatten()

    def update(self, position, velocity):
        z = np.array([[position], [velocity]])
        y = z - self.H @ self.x                                # innovation
        S = self.H @ self.P @ self.H.T + self.R                # innovation cov
        K = self.P @ self.H.T @ np.linalg.inv(S)               # Kalman gain
        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ self.H) @ self.P
        return self.x.flatten()

    def update_optional(self, position=None, velocity=None, available=True):
        """
        If available=False, skip update (prediction-only).
        """
        if not available:
            return self.x.flatten()
        assert position is not None and velocity is not None
        return self.update(position, velocity)