from Wave2D import Wave2D_Neumann
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
from IPython.display import HTML, display

#Simulation setup (Neumann, mx=my=2, CFL = 1/sqrt(2))
solver = Wave2D_Neumann()
N, Nt = 64, 200
c, cfl = 1.0, 1/np.sqrt(2)
mx = my = 2
sample_every = 3  # store every k-th step to keep frames reasonable

snapshots = solver(N, Nt, cfl=cfl, c=c, mx=mx, my=my, store_data=sample_every)
X, Y = solver.X, solver.Y
dt = solver.dt


z_amp = max(np.abs(U).max() for U in snapshots.values())
zmin, zmax = -z_amp, z_amp

fig = plt.figure(figsize=(6.0, 4.5))
ax = fig.add_subplot(111, projection="3d")
ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 1.0)
ax.set_zlim(zmin, zmax)
ax.set_title("2D Wave (Neumann, mx=my=2, C=1/√2)", fontsize=13)


ax.view_init(elev=25, azim=-60)

artists = []
for n, U in sorted(snapshots.items()):
    wire = ax.plot_wireframe(
        X, Y, U, rstride=3, cstride=3, color="k", linewidth=0.6
    )
    t_label = ax.text2D(
        0.02, 0.95, f"t = {n*dt:.3f}", transform=ax.transAxes, fontsize=10
    )
    artists.append([wire, t_label])


ani = animation.ArtistAnimation(
    fig, artists, interval=60, blit=True, repeat_delay=800
)

writer = animation.PillowWriter(fps=10)
ani.save("neumannwave.gif", writer=writer)

try:
    display(HTML(ani.to_jshtml()))
except Exception:
    pass
