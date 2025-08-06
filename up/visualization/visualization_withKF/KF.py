import numpy as np
from numpy import ndarray

class KF:
    def __init__(self, A, B, H, Q, R):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.H = H
        self.dt = 0.0333
        self.last_P = None
        self.last_x = None
        self.first_flag = 1

    def _predict(self,u=np.zeros(8)):
        x_pre = self.A @ self.last_x + self.B @ u
        P_pre = self.A @ self.last_P @ self.A.T + self.Q
        return  x_pre, P_pre

    def predict(self):
        self.last_x, self.last_P = self._predict()
        return self.last_x, self.last_P

    def estimate(self,z):
        if self.first_flag == 1:
            self.last_x = np.hstack([z,np.zeros(4)])
            self.last_P = np.eye(self.A.shape[1]) * 10
            self.first_flag = 0
            return self.last_x[:4]

        x_pre, P_pre = self._predict()
        K = P_pre @ self.H.T @ np.linalg.inv(self.H @ P_pre @ self.H.T + self.R)
        x_est = x_pre + K @ (z - self.H @ x_pre)
        P = P_pre - K @ self.H @ P_pre

        self.last_x = x_est
        self.last_P = P

        return x_est
    
    def reload(self, **kwargs):
        self.A = kwargs.get('A', self.A)
        self.B = kwargs.get('B', self.B)
        self.H = kwargs.get('H', self.H)