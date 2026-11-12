import numpy as np

class HybridTurbulenceGapDetector:
    def __init__(self, embedding_dim=5, delay_tau=6, window_size=150, dt=0.004, reg_alpha=1e-3):
        self.d = embedding_dim
        self.tau = delay_tau
        self.M = window_size
        self.dt = dt
        self.alpha = reg_alpha

    def embed_time_series(self, signal):
        n_samples = len(signal)
        n_vectors = n_samples - (self.d - 1) * self.tau
        if n_vectors <= 0:
            return np.empty((self.d, 0))
        embedded = np.zeros((self.d, n_vectors))
        for i in range(self.d):
            start_idx = (self.d - 1 - i) * self.tau
            end_idx = start_idx + n_vectors
            embedded[i, :] = signal[start_idx:end_idx]
        return embedded

    def analyze_stream(self, signal):
        embedded = self.embed_time_series(signal)
        n_points = embedded.shape[1]
        
        omega1_series = np.full(n_points, np.nan)
        theta_series = np.full(n_points, np.nan)
        
        if n_points < self.M:
            return omega1_series, theta_series

        for k in range(self.M, n_points):
            Z = embedded[:, k - self.M : k - 1]
            Y = embedded[:, k - self.M + 1 : k]
            
            ZZT = np.dot(Z, Z.T) + self.alpha * np.eye(self.d)
            YZT = np.dot(Y, Z.T)
            
            try:
                Ak = np.linalg.solve(ZZT.T, YZT.T).T
                eigenvals = np.linalg.eigvals(Ak)
                magnitudes = np.abs(eigenvals)
                
                max_mag = np.max(magnitudes)
                if max_mag > 1e-12:
                    omega_val = - (1.0 / self.dt) * np.log(max_mag)
                else:
                    omega_val = 10.0
                
                det_Ak = np.abs(np.linalg.det(Ak))
                if det_Ak > 1e-24:
                    theta_val = (1.0 / self.dt) * np.log(det_Ak)
                else:
                    theta_val = -100.0
                
                omega1_series[k] = omega_val
                theta_series[k] = theta_val
            except Exception:
                continue

        return omega1_series, theta_series
