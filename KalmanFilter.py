import numpy as np

class KalmanFilter:
    def __init__(self, dt, u_x, u_y, std_acc, x_sdt_meas, y_sdt_meas):
        self.dt = dt 
        
        self.u = np.array([[u_x], [u_y]])
        
        self.x = np.array([[0], [0], [0], [0]])
        
        self.A = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        self.B = np.array([
            [0.5 * self.dt**2, 0],
            [0, 0.5 * self.dt**2],
            [self.dt, 0],
            [0, self.dt]
        ])
        
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        self.P = np.eye(self.A.shape[0])
        
        G = np.array([
            [0.5 * self.dt**2],
            [0.5 * self.dt**2],
            [self.dt],
            [self.dt]
        ])
        self.Q = np.dot(G, G.T) * std_acc**2
        
        self.R = np.array([
            [x_sdt_meas**2, 0],
            [0, y_sdt_meas**2]
        ])

    def predict(self):
        
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)
        
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        
        return self.x[0:2]

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        self.x = self.x + np.dot(K, y)
        
        I = np.eye(self.A.shape[0])
        self.P = np.dot(I - np.dot(K, self.H), self.P)
        
        return self.x[0:2]