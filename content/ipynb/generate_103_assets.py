import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

# Create directory for images
output_dir = "static/images/calculus-103"
os.makedirs(output_dir, exist_ok=True)

# Function definitions
def f_example_3(x,y):
    return (85+ 0.1*(- 1/9*(x-6)*x**2*y**3 + 2/3*(x-6)*x**2*y**2))

def dfdx_example_3(x,y):
    return 0.1/3*x*y**2*(2-y/3)*(3*x-12)

def dfdy_example_3(x,y):
    return 0.1/3*(x-6)*x**2*y*(4-y)

def f_example_4(x,y):
    return -(10/(3+3*(x-.5)**2+3*(y-.5)**2) + \
            2/(1+2*((x-3)**2)+2*(y-1.5)**2) + \
            3/(1+.5*((x-3.5)**2)+0.5*(y-4)**2))+10

def dfdx_example_4(x,y):
    return  -(-2*3*(x-0.5)*10/(3+3*(x-0.5)**2+3*(y-0.5)**2)**2 + \
            -2*2*(x-3)*2/(1+2*((x-3)**2)+2*(y-1.5)**2)**2 +\
            -2*0.5*(x-3.5)*3/(1+.5*((x-3.5)**2)+0.5*(y-4)**2)**2)

def dfdy_example_4(x,y):
    return -(-2*3*(y-0.5)*10/(3+3*(x-0.5)**2+3*(y-0.5)**2)**2 + \
            -2*2*(y-1.5)*2/(1+2*((x-3)**2)+2*(y-1.5)**2)**2 +\
            -0.5*2*(y-4)*3/(1+.5*((x-3.5)**2)+0.5*(y-4)**2)**2)

# Gradient Descent Implementation
def gradient_descent(dfdx, dfdy, x, y, learning_rate, num_iterations):
    x_history = [x]
    y_history = [y]
    for _ in range(num_iterations):
        grad_x = dfdx(x, y)
        grad_y = dfdy(x, y)
        x = x - learning_rate * grad_x
        y = y - learning_rate * grad_y
        x_history.append(x)
        y_history.append(y)
    return x_history, y_history

# Plotting Function
def plot_surface_and_contour(x_range, y_range, f, title, filename, path=None, paths=None):
    x = np.linspace(*x_range, 100)
    y = np.linspace(*y_range, 100)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    fig = plt.figure(figsize=(14, 6))
    
    # Contour Plot
    ax1 = fig.add_subplot(1, 2, 1)
    cont = ax1.contour(X, Y, Z, levels=20, cmap='viridis')
    ax1.clabel(cont, inline=1, fontsize=10)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title(f'{title} - Contour')
    
    if path:
        ax1.plot(path[0], path[1], 'r.-', markersize=5, label='Gradient Descent')
        ax1.plot(path[0][0], path[1][0], 'bo', label='Start')
        ax1.plot(path[0][-1], path[1][-1], 'go', label='End')
        ax1.legend()
    
    if paths:
        colors = ['r', 'b', 'g', 'm', 'c']
        for i, p in enumerate(paths):
            ax1.plot(p[0], p[1], f'{colors[i%len(colors)]}.-', markersize=5, label=f'Path {i+1}')
            ax1.plot(p[0][0], p[1][0], 'ko', markersize=3) # Start point
    
    # Surface Plot
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    surf = ax2.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('f(x, y)')
    ax2.set_title(f'{title} - Surface')
    fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=10)

    if path:
         # Need to calculate Z for the path
         path_z = [f(px, py) for px, py in zip(path[0], path[1])]
         ax2.plot(path[0], path[1], path_z, 'r.-', markersize=5, zorder=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"Generated {filename}")

# Generate Plots

# Example 3: One Global Minimum
plot_surface_and_contour([0, 10], [0, 10], f_example_3, 
                         'Function with One Global Minimum', 'example_3_plot.png')

# Example 4: Multiple Minima
plot_surface_and_contour([0, 5], [0, 5], f_example_4, 
                         'Function with Multiple Minima', 'example_4_plot.png')

# Example 4 with Gradient Descent paths
# Path 1: Converges to global minimum
x_hist1, y_hist1 = gradient_descent(dfdx_example_4, dfdy_example_4, 0.5, 0.5, 0.5, 20) # Start near global min
# Path 2: Converges to local minimum
x_hist2, y_hist2 = gradient_descent(dfdx_example_4, dfdy_example_4, 3.0, 3.0, 0.5, 20) # Start near local min
# Path 3: Saddle point/Another local min area
x_hist3, y_hist3 = gradient_descent(dfdx_example_4, dfdy_example_4, 3.0, 1.0, 0.5, 20) 

# Run GD for plot 3
paths = [(x_hist1, y_hist1), (x_hist2, y_hist2), (x_hist3, y_hist3)]

plot_surface_and_contour([0, 5], [0, 5], f_example_4, 
                         'Gradient Descent Paths on Function with Multiple Minima', 
                         'example_4_gd_paths.png', paths=paths)
