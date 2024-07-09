import numpy as np
import matplotlib.pyplot as plt

def plot_lp_contours(p, ax, title, range_lim=2.0, num_points=100):
    x = np.linspace(-range_lim, range_lim, num_points)
    y = np.linspace(-range_lim, range_lim, num_points)
    X, Y = np.meshgrid(x, y)
    
    Z = (np.abs(X)**p + np.abs(Y)**p)**(1/p)
    
    contours = ax.contour(X, Y, Z, levels=10, cmap='plasma')
    ax.clabel(contours, inline=True, fontsize=8)
    ax.set_title(f'Contours of ${title}$ norm')
    ax.set_xlabel('w1')
    ax.set_ylabel('w2')
    ax.axis('equal')

fig, axs = plt.subplots(1, 3, figsize=(18, 5))
plot_lp_contours(0.5, axs[0], 'l_{0.5}')
plot_lp_contours(1, axs[1], 'l_1')
plot_lp_contours(2, axs[2], 'l_2')
plt.show()
