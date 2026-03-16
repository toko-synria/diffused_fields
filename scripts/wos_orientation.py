"""
Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>

This file is part of diffused_fields.
Licensed under the MIT License. See LICENSE file in the project root.
"""

"""
Computes an orientation field on a pointcloud conditioned on
keypoints.
"""

from diffused_fields.diffusion import PointcloudScalarDiffusion, WalkOnSpheresDiffusion
from diffused_fields.manifold import *
from diffused_fields.manifold.manifold import extract_plane
from diffused_fields.visualization.plotting_ps import *

# Select the object
# ==========================================
filename = "spot.ply"


pcloud = Pointcloud(filename=filename)
scalar_diffusion = PointcloudScalarDiffusion(pcloud, diffusion_scalar=1000)
scalar_diffusion.get_local_bases()
# the object itself is the boundary condition for the diffusion
# on the ambient space (robot's workspace)
boundaries = [pcloud]


# Monte Carlo diffusion solver for the ambient space
wos_diffusion = WalkOnSpheresDiffusion(
    boundaries=boundaries,
    convergence_threshold=pcloud.get_mean_edge_length() * 2,
)

# We will compute the diffused field at the grid points for visualizing the result
grid = pcloud.get_bounding_box_grid(bounding_box_scalar=1, nb_points=11)

# Get the center coordinates of the grid
grid.get_center()

# Extract cross-sections at mid-points using grid.center
# grid.vertices = extract_plane(grid.vertices, axis="x", value=grid.center[0])
wos_diffusion.diffuse_orientations_on_grid(grid)


ps.init()
set_camera_and_plane()
ps_field = plot_orientation_field(
    pcloud.vertices, pcloud.local_bases, name="pcloud", point_radius=0
)
ps_field.add_scalar_quantity(name="u0", values=scalar_diffusion.ut)
plot_sources(pcloud.vertices[scalar_diffusion.source_vertices])
plot_orientation_field(
    grid.vertices,
    grid.local_bases,
    name="grid",
    enable_x=True,
    enable_z=True,
    point_radius=0,
)

ps.show()
