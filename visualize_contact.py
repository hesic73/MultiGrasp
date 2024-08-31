import trimesh
import numpy as np
import pyrender

from typing import Sequence


def visualize_colored_vertices(mesh_file: str, vertex_indices: Sequence[int]):
    # Load the mesh from the file
    mesh = trimesh.load(mesh_file)

    # Get the vertex positions
    vertices = mesh.vertices

    # Create a Pyrender scene
    scene = pyrender.Scene()

    # Convert the trimesh mesh to a Pyrender mesh and add it to the scene
    trimesh_mesh = pyrender.Mesh.from_trimesh(mesh)
    scene.add(trimesh_mesh)

    # Highlight the specific vertices by adding small spheres at those positions
    highlight_color = [1.0, 0.0, 0.0, 1.0]  # Red color
    sphere_radius = 0.001

    for idx in vertex_indices:
        sphere = trimesh.creation.icosphere(
            radius=sphere_radius, subdivisions=2)
        sphere.apply_translation(vertices[idx])
        sphere.visual.vertex_colors = highlight_color
        mesh_sphere = pyrender.Mesh.from_trimesh(sphere)
        scene.add(mesh_sphere)

    # Add a light source to the scene
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light)

    # Setup the camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = np.eye(4)
    camera_pose[2, 3] = 0.2
    scene.add(camera, pose=camera_pose)

    # Render the scene
    viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)


if __name__ == "__main__":

    # Example usage
    mesh_file = '/home/sichengh/24fall/MultiGrasp/data/urdf/shadow_hand_description/meshes/F3.obj'
    vertex_indices = [417, 536, 104, 4]
    visualize_colored_vertices(mesh_file, vertex_indices)
