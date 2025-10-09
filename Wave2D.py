import numpy as np
import sympy as sp
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from matplotlib import cm

x, y, t = sp.symbols('x,y,t')

class Wave2D:
    def __init__(self):
        self.L = 1.0
        self._mx = None
        self._my = None
        self._c = 1.0


    def create_mesh(self, N, sparse=False):
        """Create 2D mesh and store in self.xij and self.yij"""
        self.N = int(N)
        self.h = self.L / self.N
        self.x = np.linspace(0.0, self.L, self.N + 1)
        self.y = np.linspace(0.0, self.L, self.N + 1)
        self.xij, self.yij = np.meshgrid(self.x, self.y, indexing="ij")

        T = self.D2(self.N)               
        I = sparse.identity(self.N + 1, format="csr")
        self.Lap = (sparse.kron(I, T) + sparse.kron(T, I)) / (self.h**2)
        self._bmask = self.boundary_mask() 

    def D2(self, N):
        """Return second order differentiation matrix"""
        n = N + 1
        main = -2.0 * np.ones(n)
        off  =  1.0 * np.ones(n - 1)
        T = sparse.diags([off, main, off], [-1, 0, 1], format="lil")
        T[0, :] = 0.0; T[0, 0] = 1.0
        T[-1, :] = 0.0; T[-1, -1] = 1.0
        return T.tocsr()

    def boundary_mask(self):
        N = self.N
        mask = np.zeros((N+1, N+1), dtype=bool)
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
        mask[:, -1] = True
        return mask


    @property
    def w(self):
        """Dispersion coefficient"""
        if self._mx is None or self._my is None:
            raise RuntimeError("Call initialize(...) first so mx,my are known.")
        return (self._c * np.pi * np.sqrt(self._mx**2 + self._my**2) / self.L)

    
    def ue(self, mx, my):
        return sp.sin(mx*sp.pi*x) * sp.sin(my*sp.pi*y) * sp.cos(self.w*t)

    
    def initialize(self, N, mx, my):
        """Set U^n and U^{n-1} at t=0."""
        self.create_mesh(N)
        self._mx, self._my = int(mx), int(my)
        ue0 = sp.lambdify((x, y, t), self.ue(mx, my), "numpy")
        U0 = ue0(self.xij, self.yij, 0.0).astype(float)
        self.U_nm1 = U0.copy()  # U^{-1}
        self.U_n   = U0.copy()  # U^{0}
        self.apply_bcs()

    @property
    def dt(self):
        return self._dt

    def l2_error(self, u, t0):
        ue_fun = sp.lambdify((x, y, t), self.ue(self._mx, self._my), "numpy")
        Ue = ue_fun(self.xij, self.yij, float(t0)).astype(float)
        e = u - Ue
        return np.sqrt(self.h**2 * np.sum(e**2))

    def apply_bcs(self):
        self.U_nm1[self._bmask] = 0.0
        self.U_n[self._bmask]   = 0.0

   
    def __call__(self, N, Nt, cfl=0.5, c=1.0, mx=3, my=3, store_data=-1):
        
      
        self._c = float(c)
        self._dt = float(cfl * (self.L / N) / c)

        self.initialize(N, mx, my)

    
        lap = self.Lap
        cdt2 = (c * self.dt)**2
        errors = []
        snapshots = {}

      
        ue_fun = sp.lambdify((x, y, t), self.ue(mx, my), "numpy")

        for n in range(1, Nt+1):
            
            Uvec = self.U_n.flatten()
            Lu = lap.dot(Uvec).reshape(self.N+1, self.N+1)

            U_np1 = 2.0*self.U_n - self.U_nm1 + cdt2 * Lu

            self.U_nm1, self.U_n = self.U_n, U_np1

            self.apply_bcs()

            if store_data > 0 and (n % store_data == 0):
                snapshots[n] = self.U_n.copy()

            if store_data == -1:
                tn = n * self.dt
                errors.append(self.l2_error(self.U_n, tn))

        if store_data > 0:
            return snapshots
        else:
            return self.h, np.array(errors)

    def convergence_rates(self, m=4, cfl=0.1, Nt=10, mx=3, my=3):
        E, h = [], []
        N0 = 8
        for _ in range(m):
            dx, err = self(N0, Nt, cfl=cfl, mx=mx, my=my, store_data=-1)
            E.append(err[-1])
            h.append(dx)
            N0 *= 2
            Nt *= 2
        r = [np.log(E[i-1]/E[i])/np.log(h[i-1]/h[i]) for i in range(1, m)]
        return r, np.array(E), np.array(h)

# -------- Neumann variant --------
class Wave2D_Neumann(Wave2D):
    def D2(self, N):
        """Second derivative with homogeneous Neumann at boundaries."""

        n = N + 1
        main = -2.0 * np.ones(n)
        off  =  1.0 * np.ones(n - 1)
        T = sparse.diags([off, main, off], [-1, 0, 1], format="lil")
        
        T[0, :] = 0.0;     T[0, 0] = -2.0; T[0, 1] = 2.0
        T[-1, :] = 0.0;    T[-1, -1] = -2.0; T[-1, -2] = 2.0
        return T.tocsr()

    def ue(self, mx, my):
        # Neumann standing wave
        return sp.cos(mx*sp.pi*x) * sp.cos(my*sp.pi*y) * sp.cos(self.w*t)

    def apply_bcs(self):
        # For homogeneous Neumann, no values are imposed directly.
        
        pass
