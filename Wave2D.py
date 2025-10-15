import numpy as np
import sympy as sp
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from matplotlib import cm

# Symbolic variables for the manufactured solution
x, y, t = sp.symbols('x, y, t')


class Wave2D:
    """2D wave equation solver with a manufactured standing-wave solution."""

 
    # Grid & discrete operators

    def build_grid(self, N: int, use_sparse_mesh: bool = False) -> None:
        """Create a uniform [0,1]x[0,1] grid with N intervals per axis."""
        self.N = N
        self.h = 1.0 / N
        self.x_nodes = np.linspace(0.0, 1.0, N + 1)
        self.y_nodes = np.linspace(0.0, 1.0, N + 1)
        self.X, self.Y = np.meshgrid(
            self.x_nodes, self.y_nodes, indexing="ij", sparse=use_sparse_mesh
        )

    def second_diff_matrix(self, N: int) -> sparse.spmatrix:
        """Centered second-derivative matrix ."""
        main = -2.0 * np.ones(N + 1)
        off = 1.0 * np.ones(N)
        D = sparse.diags([off, main, off], offsets=[-1, 0, 1],
                         shape=(N + 1, N + 1), format="lil")
        return D


    # Exact (manufactured) solution
    
    @property
    def omega(self) -> float:
        """Dispersion relation ω = c * sqrt((m_x π)^2 + (m_y π)^2)."""
        kx = self.m_x * np.pi
        ky = self.m_y * np.pi
        return self.c * np.sqrt(kx**2 + ky**2)

    def exact_mode(self, m_x: int, m_y: int) -> sp.Expr:
        """Exact standing wave for Dirichlet boundaries."""
        return sp.sin(m_x * sp.pi * x) * sp.sin(m_y * sp.pi * y) * sp.cos(self.omega * t)

   
    # Initialization & utilities
  
    def _initialize_state(self, N: int, m_x: int, m_y: int) -> None:
        """Initialize U^{n}, U^{n-1} using the exact solution and a Taylor step."""
        self.N = N
        self.m_x = m_x
        self.m_y = m_y

        # Operators & grid
        self.D = self.second_diff_matrix(N)
        D = self.D
        self.build_grid(N, use_sparse_mesh=False)

        # Exact solution (symbolic -> numeric)
        self.uex = self.exact_mode(m_x, m_y)
        ue_fun = sp.lambdify((t, x, y), self.uex, "numpy")

        # Allocate timestep states
        self.u_next, self.u_now, self.u_prev = np.zeros((3, N + 1, N + 1))

        # t = 0 displacement (and velocity = 0 for standing wave)
        self.u_prev[:] = ue_fun(0.0, self.X, self.Y)
        self.u_now[:] = self.u_prev
        self.apply_bcs()

        # One Taylor step to get U^1
        self.u_prev[:] = self.u_now
        lap_u_prev = (D @ self.u_prev + self.u_prev @ D.T) / (self.h**2)
        self.u_now[:] = self.u_prev + 0.5 * (self.c * self.dt)**2 * lap_u_prev
        self.apply_bcs()

    @property
    def dt(self) -> float:
        """Time step from CFL condition."""
        return self.cfl * self.h / self.c

    def l2_error(self, U: np.ndarray, t_eval: float) -> float:
        """Grid L2 error at a given time."""
        ue_fun = sp.lambdify((t, x, y), self.uex, "numpy")
        Ue = ue_fun(t_eval, self.X, self.Y)
        diff = (U - Ue).ravel()
        return self.h * np.linalg.norm(diff, 2)

    def apply_bcs(self) -> None:
        """Homogeneous Dirichlet BCs on all boundaries."""
        U = self.u_now
        U[0, :] = 0.0
        U[-1, :] = 0.0
        U[:, 0] = 0.0
        U[:, -1] = 0.0


    # Main solver entry point

    def __call__(self, N: int, Nt: int, cfl: float = 0.5, c: float = 1.0,
                 mx: int = 3, my: int = 3, store_data: int = -1):
        """
        Solve the 2D wave equation with leapfrog.
        - If store_data > 0: returns {timestep: solution}
        - If store_data == -1: returns (h, l2_error_array)
        """
        self.cfl = cfl
        self.c = c
        self.Nt = Nt

       
        self._initialize_state(N, mx, my)

        D = self.D
        dt = self.dt

        def laplacian(U: np.ndarray) -> np.ndarray:
            return (D @ U + U @ D.T) / (self.h**2)

        if store_data > 0:
            out = {0: self.u_prev.copy()}
        else:
            errs = [self.l2_error(self.u_prev, 0.0)]

        # Leapfrog time stepping
        for n in range(2, Nt + 1):
            self.u_next[:] = 2 * self.u_now - self.u_prev + (self.c * dt)**2 * laplacian(self.u_now)
            self.u_prev, self.u_now, self.u_next = self.u_now, self.u_next, self.u_prev
            self.apply_bcs()

            if store_data > 0:
                if n % store_data == 0:
                    out[n] = self.u_now.copy()
            else:
                tn = n * dt
                errs.append(self.l2_error(self.u_now, tn))

        return out if store_data > 0 else (self.h, np.array(errs))

  
    # Convergence
   
    def convergence_rates(self, m: int = 4, cfl: float = 0.1, Nt: int = 10,
                          mx: int = 3, my: int = 3):
        """
        Refine grid & time together; report observed order, errors, mesh sizes.
        """
        errors = []
        hs = []
        N_coarse = 8
        Nt_local = Nt

        for i in range(m):
            dx, err_hist = self(N_coarse, Nt_local, cfl=cfl, mx=mx, my=my, store_data=-1)
            errors.append(err_hist[-1])
            hs.append(dx)
            N_coarse *= 2
            Nt_local *= 2

        orders = [np.log(errors[i - 1] / errors[i]) / np.log(hs[i - 1] / hs[i])
                  for i in range(1, m)]
        return orders, np.array(errors), np.array(hs)


class Wave2D_Neumann(Wave2D):
    

    def second_diff_matrix(self, N: int) -> sparse.spmatrix:
        # Standard interior stencil, with one-sided second derivative on boundaries
        main = -2.0 * np.ones(N + 1)
        off = 1.0 * np.ones(N)
        D = sparse.diags([off, main, off], offsets=[-1, 0, 1],
                         shape=(N + 1, N + 1), format="lil")
        # Neumann at i=0 and i=N: d^2/dx^2 with mirrored first-derivative
        D[0, :] = 0.0
        D[0, 0] = -2.0
        D[0, 1] = 2.0
        D[N, :] = 0.0
        D[N, N] = -2.0
        D[N, N - 1] = 2.0
        return D

    def exact_mode(self, m_x: int, m_y: int) -> sp.Expr:
        return sp.cos(m_x * sp.pi * x) * sp.cos(m_y * sp.pi * y) * sp.cos(self.omega * t)

    def apply_bcs(self) -> None:
        # For homogeneous Neumann, the stencil takes care of boundary updates,
        # so no explicit enforcement needed here.
        return


# Tests
def test_convergence_wave2d():
    sol = Wave2D()
    r, E, h_vals = sol.convergence_rates(mx=2, my=3)
    assert abs(r[-1] - 2) < 1e-2


def test_convergence_wave2d_neumann():
    solN = Wave2D_Neumann()
    r, E, h_vals = solN.convergence_rates(mx=2, my=3)
    assert abs(r[-1] - 2) < 0.05


def test_exact_wave2d():
    sol = Wave2D()
    N = 64
    Nt = 50
    c = 1.0
    cfl = 0.3
    mx, my = 2, 3
    h, errs = sol(N, Nt, cfl=cfl, c=c, mx=mx, my=my, store_data=-1)
    dt = sol.dt

    tol = 20.0 * (h**2 + dt**2)
    print("\ntest_exact_wave2d() Result:")
    print(f"errs[-1] = {errs[-1]:.2g}")
    assert errs[-1] < tol, f"L2 error {errs[-1]:.3e} not below tolerance {tol:.3e}"
