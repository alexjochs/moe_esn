import numpy as np

# ---------------- WOx RRAM device model ---------------- #
class WOxRRAM:
    """
    Simplified WOx RRAM model adapted for ESN integration.
    - Each neuron holds one device state w in [w_min, w_max].
    - Drive v ("write" voltage) updates w with a decaying memory term.
    - Read current is obtained at a fixed V_read using a conductance that
      interpolates between G_off and G_on by the normalized state x_w.

    This is written as an init/reset + step so it drops into the ESN loop.
    """
    def __init__(self, N, *,
                 w_min=0.0, w_max=1.0, w_init=0.1,
                 tau_decay=50.0,          # larger = slower natural relaxation
                 eta1=0.02, eta2=2.0,     # nonlinearity gains for sinh
                 p_win=2.0,               # window exponent (smooth saturation)
                 V_read=0.1,              # read voltage (Volts)
                 G_off=5e-6, G_on=200e-6  # conductance bounds (Siemens)
                 ):
        self.N = N
        self.w_min = float(w_min)
        self.w_max = float(w_max)
        self.tau_decay = float(tau_decay)
        self.eta1 = float(eta1)
        self.eta2 = float(eta2)
        self.p_win = float(p_win)
        self.V_read = float(V_read)  # Volts (V)
        self.G_off = float(G_off)    # Siemens (S)
        self.G_on = float(G_on)      # Siemens (S)
        # internal state
        self.w = np.full(N, float(w_init), dtype=float)  # dimensionless, internal state (0–1)

    def reset(self, w_init=None):
        if w_init is None:
            self.w[:] = max(self.w_min, min(self.w_max, np.mean(self.w)))
        else:
            self.w[:] = np.clip(float(w_init), self.w_min, self.w_max)
        return self.w.copy()

    def _window(self, v):
        """Smooth window; slows motion near bounds; depends on v sign."""
        # normalize w into [0,1]
        x = (self.w - self.w_min) / (self.w_max - self.w_min + 1e-12)
        x = np.clip(x, 0.0, 1.0)
        w_plus  = (1.0 - x)**self.p_win   # motion toward w_max under +v
        w_minus = (x)**self.p_win         # motion toward w_min under -v
        return np.where(v >= 0.0, w_plus, w_minus)

    def step(self, v, dt=1.0):
        """
        Advance device states by one ESN tick given drive v (shape: (N,)).
        Returns the read current vector i_read (shape: (N,)).
        """
        v = np.asarray(v, dtype=float)
        assert v.shape == (self.N,), f"Expected v shape ({self.N},), got {v.shape}"

        # 1) natural relaxation toward w_min (leak)
        if self.tau_decay > 0:
            self.w -= (self.w - self.w_min) * (dt / (self.tau_decay + 1e-12))

        # 2) state change due to applied drive
        dw = dt * (self.eta1 * np.sinh(self.eta2 * v)) * self._window(v)  # dimensionless change per timestep
        self.w += dw

        # 3) clamp to physical bounds
        self.w = np.clip(self.w, self.w_min, self.w_max)

        # 4) read current at fixed V_read using interpolated conductance
        x = (self.w - self.w_min) / (self.w_max - self.w_min + 1e-12)
        G = self.G_off + x * (self.G_on - self.G_off)
        i_read = G * self.V_read  # Amperes (A)
        return i_read
