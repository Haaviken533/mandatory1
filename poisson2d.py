import numpy as np
import sympy as sp
import scipy.sparse as sparse

x, y = sp.symbols('x,y')

class Poisson2D:
    r"""Solve Poisson's equation in 2D::

        \nabla^2 u(x, y) = f(x, y), in [0, L]^2

    where L is the length of the domain in both x and y directions.
    Dirichlet boundary  conditions are used for the entire boundary.
    The Dirichlet values depend on the chosen manufactured solution.
    """

    def __init__(self, L, ue):
        self.L = float(L)
        self.ue = ue
        self.f = sp.diff(self.ue, x, 2) + sp.diff(self.ue, y, 2)

        # Numpy-callables of ue and f
        self.ue_fun = sp.lambdify((x, y), self.ue, 'numpy')
        self.f_fun  = sp.lambdify((x, y), self.f,  'numpy')

        # Placeholders filled in create_mesh()
        self.N = None
        self.h = None
        self.xg = None
        self.yg = None
        self.xij = None
        self.yij = None
        self.U = None

    def create_mesh(self, N):
        """Create 2D mesh and store in self.xij and self.yij"""
        self.N = int(N)
        self.h = self.L / self.N
        self.xg = np.linspace(0.0, self.L, self.N + 1)
        self.yg = np.linspace(0.0, self.L, self.N + 1)
        self.xij, self.yij = np.meshgrid(self.xg, self.yg, indexing='ij')

    def D2(self):
        """Return second order differentiation matrix"""
        n = self.N + 1
        main = -2.0 * np.ones(n)
        off  =  1.0 * np.ones(n - 1)
        return sparse.diags([off, main, off], offsets=[-1, 0, 1], format='csr')

    def laplace(self):
        """Return vectorized Laplace operator (Kronecker sum)"""
        n = self.N + 1
        I = sparse.identity(n, format='csr')
        T = self.D2()
        L = sparse.kron(I, T) + sparse.kron(T, I)  
        return (1.0 / self.h**2) * L.tocsr()

    def get_boundary_indices(self):
        """Return indices (flattened) that belong to the boundary"""
        N = self.N
        n = N + 1
        bnd = []

        # rows i=0 and i=N
        for j in range(n):
            bnd.append(0 * n + j)   
            bnd.append(N * n + j)    

        for i in range(1, N):
            bnd.append(i * n + 0)    
            bnd.append(i * n + N)     

        return np.array(sorted(set(bnd)), dtype=int)

    def assemble(self):
        """Return assembled matrix A and right hand side vector b"""
        Lop = self.laplace()
        F = self.f_fun(self.xij, self.yij).astype(float)
        b = F.flatten()

        A = Lop.tolil()
        Ue = self.ue_fun(self.xij, self.yij).astype(float)
        bnd = self.get_boundary_indices()
        for k in bnd:
            A.rows[k] = [k]
            A.data[k] = [1.0]
            b[k] = Ue.flatten()[k]

        return A.tocsr(), b

    def l2_error(self, U):
        """Return L2 error norm"""
        Ue = self.ue_fun(self.xij, self.yij).astype(float)
        e = U - Ue
        return np.sqrt(self.h**2 * np.sum(e**2))

    def __call__(self, N):
        """Solve Poisson's equation"""
        self.create_mesh(N)
        A, b = self.assemble()
        self.U = sparse.linalg.spsolve(A, b).reshape((self.N + 1, self.N + 1))
        return self.U

    def convergence_rates(self, m=6):
        E = []
        h = []
        N0 = 8
        for _ in range(m):
            u = self(N0)
            E.append(self.l2_error(u))
            h.append(self.h)
            N0 *= 2
        r = [np.log(E[i-1] / E[i]) / np.log(h[i-1] / h[i]) for i in range(1, m)]
        return r, np.array(E), np.array(h)

    def eval(self, xq, yq):
        
        xq = float(np.clip(xq, 0.0, self.L))
        yq = float(np.clip(yq, 0.0, self.L))

        i = min(int(np.floor(xq / self.h)), self.N - 1)
        j = min(int(np.floor(yq / self.h)), self.N - 1)

        x0, x1 = self.xg[i], self.xg[i+1]
        y0, y1 = self.yg[j], self.yg[j+1]
        tx = 0.0 if x1 == x0 else (xq - x0) / (x1 - x0)
        ty = 0.0 if y1 == y0 else (yq - y0) / (y1 - y0)

        u00 = self.U[i, j]
        u10 = self.U[i + 1, j]
        u01 = self.U[i, j + 1]
        u11 = self.U[i + 1, j + 1]

        return (1-tx)*(1-ty)*u00 + tx*(1-ty)*u10 + (1-tx)*ty*u01 + tx*ty*u11
