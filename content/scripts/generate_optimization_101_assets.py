"""
Generate assets for Optimization 101: Metric Spaces and Normed Vector Spaces
Based on Gallier & Quaintance - Fundamentals of Optimization Theory (2025)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
from pathlib import Path

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent.parent / "static" / "images" / "opt"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14


def plot_norm_balls_2d():
    """
    Figure 2.1 from the book: Unit balls for L1, L2, and L∞ norms in 2D.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    # L1 norm (diamond shape)
    ax = axes[0]
    diamond = np.array([[1, 0], [0, 1], [-1, 0], [0, -1], [1, 0]])
    ax.fill(diamond[:, 0], diamond[:, 1], alpha=0.3, color='steelblue', edgecolor='steelblue', linewidth=2)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.axvline(x=0, color='gray', linewidth=0.5)
    ax.set_title(r'$\ell_1$ norm: $\|x\|_1 = |x_1| + |x_2|$')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.grid(True, alpha=0.3)
    
    # L2 norm (circle)
    ax = axes[1]
    theta = np.linspace(0, 2*np.pi, 100)
    x = np.cos(theta)
    y = np.sin(theta)
    ax.fill(x, y, alpha=0.3, color='coral', edgecolor='coral', linewidth=2)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.axvline(x=0, color='gray', linewidth=0.5)
    ax.set_title(r'$\ell_2$ norm: $\|x\|_2 = \sqrt{x_1^2 + x_2^2}$')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.grid(True, alpha=0.3)
    
    # L∞ norm (square)
    ax = axes[2]
    square = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]])
    ax.fill(square[:, 0], square[:, 1], alpha=0.3, color='forestgreen', edgecolor='forestgreen', linewidth=2)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.axvline(x=0, color='gray', linewidth=0.5)
    ax.set_title(r'$\ell_\infty$ norm: $\|x\|_\infty = \max(|x_1|, |x_2|)$')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'unit_balls_2d.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'unit_balls_2d.png'}")


def plot_norm_balls_comparison():
    """
    Figure 2.2 from the book: Overlay of L1, L2, L∞ unit balls showing containment.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # L∞ norm (square) - largest
    square = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]])
    ax.fill(square[:, 0], square[:, 1], alpha=0.2, color='forestgreen', 
            edgecolor='forestgreen', linewidth=2.5, label=r'$\ell_\infty$ ball')
    
    # L2 norm (circle) - middle
    theta = np.linspace(0, 2*np.pi, 100)
    x = np.cos(theta)
    y = np.sin(theta)
    ax.fill(x, y, alpha=0.3, color='coral', edgecolor='coral', linewidth=2.5, label=r'$\ell_2$ ball')
    
    # L1 norm (diamond) - smallest in some directions
    diamond = np.array([[1, 0], [0, 1], [-1, 0], [0, -1], [1, 0]])
    ax.fill(diamond[:, 0], diamond[:, 1], alpha=0.4, color='steelblue', 
            edgecolor='steelblue', linewidth=2.5, label=r'$\ell_1$ ball')
    
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.axvline(x=0, color='gray', linewidth=0.5)
    ax.set_title('Comparison of Unit Balls in 2D', fontsize=16)
    ax.set_xlabel('$x_1$', fontsize=14)
    ax.set_ylabel('$x_2$', fontsize=14)
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'unit_balls_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'unit_balls_comparison.png'}")


def plot_norm_balls_3d():
    """
    Figure 2.3 from the book: Unit balls for L1, L2, L∞ norms in 3D.
    """
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    
    fig = plt.figure(figsize=(15, 5))
    
    # L1 norm (octahedron)
    ax1 = fig.add_subplot(131, projection='3d')
    vertices = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], 
                         [0, -1, 0], [0, 0, 1], [0, 0, -1]])
    faces = [[0, 2, 4], [0, 4, 3], [0, 3, 5], [0, 5, 2],
             [1, 2, 4], [1, 4, 3], [1, 3, 5], [1, 5, 2]]
    poly = Poly3DCollection([vertices[face] for face in faces], alpha=0.3, 
                             facecolor='steelblue', edgecolor='steelblue', linewidth=1)
    ax1.add_collection3d(poly)
    ax1.set_xlim([-1.2, 1.2])
    ax1.set_ylim([-1.2, 1.2])
    ax1.set_zlim([-1.2, 1.2])
    ax1.set_xlabel('$x_1$')
    ax1.set_ylabel('$x_2$')
    ax1.set_zlabel('$x_3$')
    ax1.set_title(r'$\ell_1$ ball (Octahedron)')
    
    # L2 norm (sphere)
    ax2 = fig.add_subplot(132, projection='3d')
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax2.plot_surface(x, y, z, alpha=0.3, color='coral', edgecolor='coral', linewidth=0.1)
    ax2.set_xlim([-1.2, 1.2])
    ax2.set_ylim([-1.2, 1.2])
    ax2.set_zlim([-1.2, 1.2])
    ax2.set_xlabel('$x_1$')
    ax2.set_ylabel('$x_2$')
    ax2.set_zlabel('$x_3$')
    ax2.set_title(r'$\ell_2$ ball (Sphere)')
    
    # L∞ norm (cube)
    ax3 = fig.add_subplot(133, projection='3d')
    # Define the vertices of a cube
    r = [-1, 1]
    vertices = np.array([[x, y, z] for x in r for y in r for z in r])
    faces = [
        [vertices[0], vertices[1], vertices[3], vertices[2]],  # bottom
        [vertices[4], vertices[5], vertices[7], vertices[6]],  # top
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
        [vertices[0], vertices[2], vertices[6], vertices[4]],  # left
        [vertices[1], vertices[3], vertices[7], vertices[5]],  # right
    ]
    poly = Poly3DCollection(faces, alpha=0.3, facecolor='forestgreen', 
                             edgecolor='forestgreen', linewidth=1)
    ax3.add_collection3d(poly)
    ax3.set_xlim([-1.2, 1.2])
    ax3.set_ylim([-1.2, 1.2])
    ax3.set_zlim([-1.2, 1.2])
    ax3.set_xlabel('$x_1$')
    ax3.set_ylabel('$x_2$')
    ax3.set_zlabel('$x_3$')
    ax3.set_title(r'$\ell_\infty$ ball (Cube)')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'unit_balls_3d.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'unit_balls_3d.png'}")


def plot_open_set_illustration():
    """
    Figure 2.5 from the book: Open set with an open ball around a point.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create an irregular "blob" shape for the open set U
    theta = np.linspace(0, 2*np.pi, 100)
    r = 2 + 0.5*np.sin(3*theta) + 0.3*np.cos(5*theta)
    x_blob = r * np.cos(theta)
    y_blob = r * np.sin(theta)
    
    # Fill the open set
    ax.fill(x_blob, y_blob, alpha=0.3, color='peachpuff', edgecolor='darkorange', linewidth=2)
    ax.text(0, 2.5, r'$U$ (open set)', fontsize=14, ha='center')
    
    # Point a inside U
    a = np.array([0.5, 0.3])
    ax.plot(a[0], a[1], 'ko', markersize=8)
    ax.text(a[0] + 0.15, a[1] + 0.15, r'$a$', fontsize=14)
    
    # Open ball around a
    rho = 0.8
    circle = plt.Circle(a, rho, fill=True, alpha=0.4, color='lightcoral', 
                         edgecolor='darkred', linewidth=2, linestyle='--')
    ax.add_patch(circle)
    ax.text(a[0], a[1] - rho - 0.2, r'$B_0(a, \rho)$', fontsize=12, ha='center')
    
    # Draw radius line
    ax.annotate('', xy=(a[0] + rho, a[1]), xytext=a,
                arrowprops=dict(arrowstyle='<->', color='darkred', lw=1.5))
    ax.text(a[0] + rho/2, a[1] + 0.1, r'$\rho$', fontsize=12, ha='center')
    
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_aspect('equal')
    ax.set_xlabel('$x$', fontsize=14)
    ax.set_ylabel('$y$', fontsize=14)
    ax.set_title('Open Set: Every point has an open ball contained in U', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'open_set_illustration.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'open_set_illustration.png'}")


def plot_hausdorff_separation():
    """
    Figure 2.6 from the book: Hausdorff separation axiom illustration.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Two distinct points
    a = np.array([-1.5, 0])
    b = np.array([1.5, 0])
    
    # Draw the space E (background rectangle)
    rect = plt.Rectangle((-4, -2.5), 8, 5, fill=True, alpha=0.1, 
                          color='lightblue', edgecolor='blue', linewidth=2)
    ax.add_patch(rect)
    ax.text(3.5, 2, r'$E$', fontsize=16, fontweight='bold')
    
    # Open neighborhoods
    rho = 1.2
    circle_a = plt.Circle(a, rho, fill=True, alpha=0.3, color='lightcoral', 
                           edgecolor='darkred', linewidth=2)
    circle_b = plt.Circle(b, rho, fill=True, alpha=0.3, color='lightgreen', 
                           edgecolor='darkgreen', linewidth=2)
    ax.add_patch(circle_a)
    ax.add_patch(circle_b)
    
    # Points
    ax.plot(a[0], a[1], 'ko', markersize=10)
    ax.plot(b[0], b[1], 'ko', markersize=10)
    ax.text(a[0], a[1] + 0.25, r'$a$', fontsize=14, ha='center', fontweight='bold')
    ax.text(b[0], b[1] + 0.25, r'$b$', fontsize=14, ha='center', fontweight='bold')
    
    # Labels for neighborhoods
    ax.text(a[0], a[1] - rho - 0.3, r'$U_a$', fontsize=14, ha='center', color='darkred')
    ax.text(b[0], b[1] - rho - 0.3, r'$U_b$', fontsize=14, ha='center', color='darkgreen')
    
    # Annotation for disjoint sets
    ax.text(0, -2, r'$U_a \cap U_b = \emptyset$', fontsize=14, ha='center', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.set_title('Hausdorff Separation: Distinct points have disjoint neighborhoods', fontsize=14)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'hausdorff_separation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'hausdorff_separation.png'}")


def plot_triangle_inequality():
    """
    Illustration of the triangle inequality: d(x,z) ≤ d(x,y) + d(y,z)
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Three points forming a triangle
    x = np.array([0, 0])
    y = np.array([3, 2])
    z = np.array([5, 0])
    
    # Draw triangle
    triangle = plt.Polygon([x, y, z], fill=False, edgecolor='gray', linewidth=1, linestyle='--')
    ax.add_patch(triangle)
    
    # Draw sides with different colors
    ax.plot([x[0], z[0]], [x[1], z[1]], 'b-', linewidth=3, label=r'$d(x, z)$')
    ax.plot([x[0], y[0]], [x[1], y[1]], 'r-', linewidth=2, label=r'$d(x, y)$')
    ax.plot([y[0], z[0]], [y[1], z[1]], 'g-', linewidth=2, label=r'$d(y, z)$')
    
    # Points
    for pt, name, offset in [(x, 'x', (-0.3, -0.3)), (y, 'y', (0, 0.3)), (z, 'z', (0.2, -0.3))]:
        ax.plot(pt[0], pt[1], 'ko', markersize=12)
        ax.text(pt[0] + offset[0], pt[1] + offset[1], f'${name}$', fontsize=16, fontweight='bold')
    
    # Distance labels
    ax.text(2.5, -0.5, r'$d(x,z)$', fontsize=14, color='blue', ha='center')
    ax.text(1.2, 1.3, r'$d(x,y)$', fontsize=14, color='red', ha='center')
    ax.text(4.2, 1.3, r'$d(y,z)$', fontsize=14, color='green', ha='center')
    
    # Formula
    ax.text(2.5, 2.8, r'Triangle Inequality: $d(x, z) \leq d(x, y) + d(y, z)$', 
            fontsize=16, ha='center', 
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='orange'))
    
    ax.set_xlim(-1, 6.5)
    ax.set_ylim(-1, 3.5)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_title('The Triangle Inequality in Metric Spaces', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'triangle_inequality.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'triangle_inequality.png'}")


def plot_lp_norms_family():
    """
    Show how Lp balls change as p varies from 0.5 to infinity.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    theta = np.linspace(0, 2*np.pi, 1000)
    
    # Different p values
    p_values = [0.5, 1, 1.5, 2, 3, 4, 10, 100]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(p_values)))
    
    for p, color in zip(p_values, colors):
        # Parametric form of Lp ball boundary
        x = np.sign(np.cos(theta)) * np.abs(np.cos(theta))**(2/p)
        y = np.sign(np.sin(theta)) * np.abs(np.sin(theta))**(2/p)
        label = rf'$p = {p}$' if p < 100 else r'$p \to \infty$'
        ax.plot(x, y, color=color, linewidth=2, label=label)
    
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.axvline(x=0, color='gray', linewidth=0.5)
    ax.set_xlabel('$x_1$', fontsize=14)
    ax.set_ylabel('$x_2$', fontsize=14)
    ax.set_title(r'Family of $\ell_p$ Unit Balls as $p$ Varies', fontsize=16)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'lp_norms_family.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'lp_norms_family.png'}")


def plot_open_closed_balls():
    """
    Illustrate open vs closed balls.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    center = np.array([0, 0])
    rho = 1.5
    
    # Open ball
    ax = axes[0]
    theta = np.linspace(0, 2*np.pi, 100)
    x = center[0] + rho * np.cos(theta)
    y = center[1] + rho * np.sin(theta)
    ax.fill(x, y, alpha=0.3, color='lightblue')
    ax.plot(x, y, 'b--', linewidth=2, label='Boundary (not included)')
    ax.plot(center[0], center[1], 'ko', markersize=8)
    ax.text(center[0] + 0.1, center[1] + 0.1, r'$a$', fontsize=14)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.set_title(r'Open Ball $B_0(a, \rho) = \{x : d(a,x) < \rho\}$', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Closed ball
    ax = axes[1]
    ax.fill(x, y, alpha=0.3, color='lightcoral')
    ax.plot(x, y, 'r-', linewidth=3, label='Boundary (included)')
    ax.plot(center[0], center[1], 'ko', markersize=8)
    ax.text(center[0] + 0.1, center[1] + 0.1, r'$a$', fontsize=14)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.set_title(r'Closed Ball $B(a, \rho) = \{x : d(a,x) \leq \rho\}$', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'open_closed_balls.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'open_closed_balls.png'}")


def plot_discrete_metric():
    """
    Illustrate the discrete metric where d(x,y) = 1 if x ≠ y.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Points in discrete space
    np.random.seed(42)
    points = np.random.randn(10, 2) * 2
    
    # Case 1: Ball with radius < 1 (only contains center)
    ax = axes[0]
    ax.scatter(points[:, 0], points[:, 1], s=100, c='gray', alpha=0.6, edgecolors='black')
    center = points[5]
    ax.scatter([center[0]], [center[1]], s=200, c='red', edgecolors='black', zorder=5, 
               label='Center (only point in ball)')
    circle = plt.Circle(center, 0.3, fill=False, edgecolor='red', linestyle='--', linewidth=2)
    ax.add_patch(circle)
    ax.set_title(r'Discrete Metric: $B(a, \rho)$ with $\rho < 1$' + '\n(Ball contains only center)', fontsize=13)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Case 2: Ball with radius ≥ 1 (contains entire space)
    ax = axes[1]
    ax.scatter(points[:, 0], points[:, 1], s=100, c='blue', alpha=0.6, edgecolors='black',
               label='All points in ball')
    center = points[5]
    ax.scatter([center[0]], [center[1]], s=200, c='red', edgecolors='black', zorder=5)
    # Draw a large circle to indicate "entire space"
    rect = plt.Rectangle((-3.8, -3.8), 7.6, 7.6, fill=True, alpha=0.15, color='blue', 
                          edgecolor='blue', linestyle='-', linewidth=2)
    ax.add_patch(rect)
    ax.set_title(r'Discrete Metric: $B(a, \rho)$ with $\rho \geq 1$' + '\n(Ball contains entire space)', fontsize=13)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'discrete_metric.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'discrete_metric.png'}")


def main():
    """Generate all figures."""
    print("Generating Optimization 101 assets...")
    print(f"Output directory: {OUTPUT_DIR}")
    
    plot_norm_balls_2d()
    plot_norm_balls_comparison()
    plot_norm_balls_3d()
    plot_open_set_illustration()
    plot_hausdorff_separation()
    plot_triangle_inequality()
    plot_lp_norms_family()
    plot_open_closed_balls()
    plot_discrete_metric()
    
    print("\nAll figures generated successfully!")


if __name__ == "__main__":
    main()
