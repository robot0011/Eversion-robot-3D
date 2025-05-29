import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

# Define cube size and position
origin = np.array([0, 0, 0])
size = 1  # length of cube sides

# Define vertices relative to origin
vertices = np.array([
    origin,
    origin + [size, 0, 0],
    origin + [size, size, 0],
    origin + [0, size, 0],
    origin + [0, 0, size],
    origin + [size, 0, size],
    origin + [size, size, size],
    origin + [0, size, size]
])

# Define cube faces
faces = [
    [vertices[j] for j in [0, 1, 2, 3]],  # bottom
    [vertices[j] for j in [4, 5, 6, 7]],  # top
    [vertices[j] for j in [0, 1, 5, 4]],  # front
    [vertices[j] for j in [2, 3, 7, 6]],  # back
    [vertices[j] for j in [1, 2, 6, 5]],  # right
    [vertices[j] for j in [0, 3, 7, 4]],  # left
]

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create 3D cube
ax.add_collection3d(Poly3DCollection(faces, facecolors='cyan', linewidths=1, edgecolors='black', alpha=0.3))

# Set the aspect ratio and limits
ax.set_xlim([0, 2])
ax.set_ylim([0, 2])
ax.set_zlim([0, 2])
ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
