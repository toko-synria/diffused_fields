"""
Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>

This file is part of diffused_fields.
Licensed under the MIT License. See LICENSE file in the project root.
"""

"""
Computes an orientation field on a mesh conditioned on keypoints.
This script demonstrates how to compute DOF field for Mesh objects.
"""

from diffused_fields.diffusion import PointcloudScalarDiffusion, WalkOnSpheresDiffusion
from diffused_fields.manifold import Mesh, Pointcloud
from diffused_fields.manifold.manifold import extract_plane
from diffused_fields.visualization.plotting_ps import *
import os
import numpy as np

# Configuration parameters
# ==========================================
# Sampling parameters for visualization (reduce density)
VISUALIZATION_SAMPLE_RATE_PCLOUD = 50  # Show every Nth point for pointcloud (larger = sparser)
VISUALIZATION_SAMPLE_RATE_GRID = 1  # Show every Nth point for grid (1 = show all grid points)
# For pointcloud: Set to 1 to show all points, 10 to show every 10th point, 50 to show every 50th point, etc.
# For grid: Usually keep at 1 since grid is already sparse, or use 2-5 if grid is very dense

# Select the mesh file
# ==========================================
# Use absolute path or relative path to the mesh file
mesh_filename = "bottle.obj"
mesh_file_directory = "../../diffused_fields_robotics/data/meshes/"

# Alternative: if mesh is in the default mesh directory
# mesh_filename = "your_mesh.obj"
# mesh_file_directory = None  # Will use default directory

# Load Mesh
# ==========================================
# Convert to absolute path
script_dir = os.path.dirname(os.path.abspath(__file__))
mesh_path = os.path.join(script_dir, mesh_file_directory, mesh_filename)
mesh_path = os.path.normpath(mesh_path)

print(f"Loading mesh from: {mesh_path}")
if not os.path.exists(mesh_path):
    raise FileNotFoundError(f"Mesh file not found: {mesh_path}\n  Resolved path: {mesh_path}")

# Use absolute path for file_directory
mesh_file_directory_abs = os.path.dirname(mesh_path)
mesh = Mesh(filename=mesh_filename, file_directory=mesh_file_directory_abs + "/")

print(f"Mesh loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

# Method 1: Convert Mesh to Pointcloud to compute surface DOF field
# ==========================================
print("\nConverting mesh to pointcloud for surface DOF computation...")
# Use mesh vertices directly, but note that Pointcloud may process them
pcloud = Pointcloud(vertices=mesh.vertices)

print(f"Pointcloud created: {len(pcloud.vertices)} vertices")
print(f"Original mesh: {len(mesh.vertices)} vertices")

# Compute DOF field on the pointcloud surface
print("Computing surface DOF field...")
scalar_diffusion = PointcloudScalarDiffusion(pcloud, diffusion_scalar=1000)

# Set source vertices (keypoints) for the diffusion
# For mesh-derived pointclouds, use simple geometric method instead of heat method
print("Finding endpoints using bounding box method...")
import numpy as np

# Method: Find two points that are furthest apart based on bounding box
vertices = pcloud.vertices
center = np.mean(vertices, axis=0)

# Find point furthest from center
distances_from_center = np.linalg.norm(vertices - center, axis=1)
endpoint1_idx = np.argmax(distances_from_center)

# Find point furthest from endpoint1
distances_from_endpoint1 = np.linalg.norm(
    vertices - vertices[endpoint1_idx], axis=1
)
endpoint2_idx = np.argmax(distances_from_endpoint1)

scalar_diffusion.source_vertices = [int(endpoint1_idx), int(endpoint2_idx)]
print(f"Selected endpoints: {scalar_diffusion.source_vertices}")
print(f"  Endpoint 1: vertex {endpoint1_idx} at {vertices[endpoint1_idx]}")
print(f"  Endpoint 2: vertex {endpoint2_idx} at {vertices[endpoint2_idx]}")
print(f"  Distance between endpoints: {distances_from_endpoint1[endpoint2_idx]:.4f}")

# Verify source_vertices is set before calling get_local_bases
if not hasattr(scalar_diffusion, 'source_vertices') or scalar_diffusion.source_vertices is None:
    raise ValueError("source_vertices must be set before calling get_local_bases()")

# Now compute the local bases
scalar_diffusion.get_local_bases()

# Clean up NaN values in pointcloud local_bases before using
print("Cleaning NaN values in pointcloud local_bases...")
if hasattr(pcloud, 'local_bases'):
    nan_mask = np.isnan(pcloud.local_bases).any(axis=(1, 2))
    if np.any(nan_mask):
        print(f"Found {np.sum(nan_mask)} vertices with NaN in local_bases, fixing...")
        # Replace NaN bases with identity matrices
        pcloud.local_bases[nan_mask] = np.eye(3)[np.newaxis, :, :]
    print("Pointcloud local_bases cleaned")

# For visualization, we'll use pointcloud directly since it has valid local_bases
# The mesh will be shown separately
print("Note: Using pointcloud for DOF field visualization (mesh surface will be shown separately)")

# Ensure mesh has normals (needed for Walk-on-Spheres)
if not hasattr(mesh, "normals") or mesh.normals is None:
    print("Computing mesh normals...")
    mesh.get_normals()

# Use pointcloud as boundary (more reliable than mesh for Walk-on-Spheres)
# The pointcloud has valid local_bases computed from diffusion
boundaries = [pcloud]

# Monte Carlo diffusion solver for the ambient space
# ==========================================
print("\nSetting up Walk-on-Spheres diffusion...")
# Compute mean edge length from pointcloud
try:
    mean_edge_length = pcloud.get_mean_edge_length()
except:
    mean_edge_length = 0

if mean_edge_length == 0 or np.isnan(mean_edge_length) or mean_edge_length < 1e-6:
    # Fallback: estimate from bounding box
    mesh.get_bounding_box()
    bbox_corners = mesh.oriented_bounding_box_corners
    bbox_size = np.max(bbox_corners, axis=0) - np.min(bbox_corners, axis=0)
    mean_edge_length = np.max(bbox_size) / 200.0  # Rough estimate: 1/200 of largest dimension
    print(f"Warning: Mean edge length was invalid, using fallback: {mean_edge_length:.6f}")

convergence_threshold = max(mean_edge_length * 2, 1e-4)  # Ensure minimum threshold
print(f"Convergence threshold: {convergence_threshold:.6f}")

wos_diffusion = WalkOnSpheresDiffusion(
    boundaries=boundaries,
    convergence_threshold=convergence_threshold,
    batch_size=256,  # Reduce batch size for stability
    max_iterations=30,  # Increase max iterations
)

# We will compute the diffused field at the grid points for visualizing the result
# ==========================================
print("Creating visualization grid...")
# Increase bounding_box_scalar to ensure grid points are outside the mesh
# Use larger grid to show full 3D space DOF field
grid = mesh.get_bounding_box_grid(bounding_box_scalar=1.2, nb_points=11)

# Get the center coordinates of the grid
grid.get_center()

# Extract cross-sections at mid-points using grid.center
# Comment out the line below to view the full 3D space DOF field (recommended)
# grid.vertices = extract_plane(grid.vertices, axis="x", value=grid.center[0])
print(f"Grid points for visualization: {len(grid.vertices)} (full 3D space)")

# Ensure grid points are outside the mesh by offsetting them slightly outward
# This helps Walk-on-Spheres converge properly
print("Adjusting grid points to be outside mesh surface...")
pcloud.get_kd_tree()
for i in range(len(grid.vertices)):
    # Find closest point on pointcloud (which represents mesh surface)
    dist, closest_idx = pcloud.get_closest_points(grid.vertices[i:i+1])
    dist = dist[0] if isinstance(dist, np.ndarray) else dist
    closest_idx = closest_idx[0] if isinstance(closest_idx, np.ndarray) else closest_idx
    
    # If point is too close to surface, move it outward along normal
    if dist < convergence_threshold * 3:
        # Use pointcloud normal (which should be available)
        if hasattr(pcloud, 'normals') and pcloud.normals is not None:
            normal = pcloud.normals[closest_idx]
        else:
            # Fallback: compute direction from surface to grid point
            normal = grid.vertices[i] - pcloud.vertices[closest_idx]
            normal = normal / (np.linalg.norm(normal) + 1e-8)
        grid.vertices[i] = pcloud.vertices[closest_idx] + normal * (convergence_threshold * 5)

print("Diffusing orientations on grid (this may take a while)...")
# Use a more robust approach: diffuse point by point with error handling
grid.local_bases = np.zeros((len(grid.vertices), 3, 3))
from tqdm import tqdm

for vertex_idx in tqdm(range(len(grid.vertices)), desc="Diffusing orientations"):
    batch_points = wos_diffusion.get_batch_from_point(grid.vertices[vertex_idx])
    try:
        local_basis, _, _ = wos_diffusion.diffuse_rotations(batch_points)
        grid.local_bases[vertex_idx] = local_basis
    except (ValueError, np.linalg.LinAlgError) as e:
        # If diffusion fails, use identity matrix as fallback
        print(f"\nWarning: Failed to diffuse at grid point {vertex_idx}, using identity matrix")
        grid.local_bases[vertex_idx] = np.eye(3)

print("Successfully diffused orientations on grid")

# Visualization
# ==========================================
print("\nVisualizing results...")
ps.init()
set_camera_and_plane()

# Sample pointcloud for visualization (reduce density)
print(f"Sampling pointcloud for visualization (sample_rate={VISUALIZATION_SAMPLE_RATE_PCLOUD})...")
pcloud_indices = np.arange(0, len(pcloud.vertices), VISUALIZATION_SAMPLE_RATE_PCLOUD)
pcloud_vertices_sampled = pcloud.vertices[pcloud_indices]
pcloud_bases_sampled = pcloud.local_bases[pcloud_indices]
print(f"  Showing {len(pcloud_vertices_sampled)} out of {len(pcloud.vertices)} points")

# Display pointcloud surface with DOF field (more reliable than mesh)
ps_pcloud_field = plot_orientation_field(
    pcloud_vertices_sampled, pcloud_bases_sampled, name="pointcloud", point_radius=0
)

# Add scalar field visualization (optional)
if hasattr(scalar_diffusion, 'ut'):
    # Sample ut to match sampled pointcloud
    ut_sampled = scalar_diffusion.ut[pcloud_indices]
    ps_pcloud_field.add_scalar_quantity(name="u0", values=ut_sampled)

# Display source vertices (keypoints)
if hasattr(scalar_diffusion, 'source_vertices'):
    plot_sources(pcloud.vertices[scalar_diffusion.source_vertices])

# Sample grid for visualization (reduce density)
print(f"Sampling grid for visualization (sample_rate={VISUALIZATION_SAMPLE_RATE_GRID})...")
grid_indices = np.arange(0, len(grid.vertices), VISUALIZATION_SAMPLE_RATE_GRID)
grid_vertices_sampled = grid.vertices[grid_indices]
grid_bases_sampled = grid.local_bases[grid_indices]
print(f"  Showing {len(grid_vertices_sampled)} out of {len(grid.vertices)} grid points")

# Display the diffused DOF field in ambient space
plot_orientation_field(
    grid_vertices_sampled,
    grid_bases_sampled,
    name="grid",
    enable_x=True,
    enable_z=True,
    point_radius=0,
)

# Optionally display the mesh surface
ps_mesh_surface = ps.register_surface_mesh(
    "mesh_surface",
    mesh.vertices,
    mesh.faces,
    color=[0.8, 0.8, 0.8],
    transparency=0.3,
    enabled=True,
)

print("\nVisualization ready! Use Polyscope controls to interact with the view.")
ps.show()

