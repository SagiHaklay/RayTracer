import argparse
from PIL import Image
import numpy as np

from camera import Camera
from light import Light
from material import Material
from scene_settings import SceneSettings
from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane
from surfaces.sphere import Sphere


def parse_scene_file(file_path):
    objects = []
    camera = None
    scene_settings = None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            obj_type = parts[0]
            params = [float(p) for p in parts[1:]]
            if obj_type == "cam":
                camera = Camera(params[:3], params[3:6], params[6:9], params[9], params[10])
            elif obj_type == "set":
                scene_settings = SceneSettings(params[:3], params[3], params[4])
            elif obj_type == "mtl":
                material = Material(params[:3], params[3:6], params[6:9], params[9], params[10])
                objects.append(material)
            elif obj_type == "sph":
                sphere = Sphere(params[:3], params[3], int(params[4]))
                objects.append(sphere)
            elif obj_type == "pln":
                plane = InfinitePlane(params[:3], params[3], int(params[4]))
                objects.append(plane)
            elif obj_type == "box":
                cube = Cube(params[:3], params[3], int(params[4]))
                objects.append(cube)
            elif obj_type == "lgt":
                light = Light(params[:3], params[3:6], params[6], params[7], params[8])
                objects.append(light)
            else:
                raise ValueError("Unknown object type: {}".format(obj_type))
    return camera, scene_settings, objects


def save_image(image_array):
    image = Image.fromarray(np.uint8(image_array))

    # Save the image to a file
    image.save("scenes/Spheres.png")

def calc_soft_shadows(light: Light, point: np.ndarray, scene_settings: SceneSettings):
    # find a plane which is perpendicular to the ray between the point and the light
    light_vec = light.position - point
    vx = np.random.randn(3)
    vx -= vx.dot(light_vec) * light_vec / np.linalg.norm(light_vec)**2
    vx /= np.linalg.norm(vx)
    vy = np.cross(light_vec, vx)
    # construct a rectangle on the plane centered in the light position and as wide as the light radius
    rec_origin = light.position - light.radius * (vx + vy)
    # divide rectangle into a grid of N X N cells, when N = scene_settings.root_number_shadow_rays
    N = scene_settings.root_number_shadow_rays
    cell_width = 2 * light.radius / N
    cell_x = cell_width * vx
    cell_y = cell_width * vy
    # sample a random point from each cell and create a ray from it to the point
    # calculate the fraction of rays that hit the point (AKA didn't intersect with any surface before hitting the point)
    hit_count = 0
    for i in range(N):
        for j in range(N):
            p = rec_origin + i * cell_y + j * cell_x
            cell_sample = p + np.random.rand(2) * cell_width
            # create ray from cell_sample to point and check if there is intersection
    hit_fraction = hit_count / (N**2)
    # calculate light intesity
    light_intensity = 1 - light.shadow_intensity + light.shadow_intensity * hit_fraction
    return light_intensity

def calc_color_by_light(point: np.ndarray, norm: np.ndarray, material: Material, light: Light, camera: Camera , scene_settings: SceneSettings):
    light_vec = light.position - point
    light_vec /= np.linalg.norm(light_vec)
    diffuse = material.diffuse_color * np.dot(light_vec, norm) * light.color
    view_vec = camera.position - point
    view_vec /= np.linalg.norm(view_vec)
    refl_light_vec = 2 * np.dot(light_vec, norm) * norm - light_vec
    specular = material.specular_color * (np.dot(refl_light_vec, view_vec) ** material.shininess) * light.color * light.specular_intensity
    light_intensity = calc_soft_shadows(light, point, scene_settings)
    return (diffuse + specular) * light_intensity


def calc_color(point: np.ndarray, norm: np.ndarray, material: Material, bg_color, refl_color: np.ndarray, lights, camera: Camera, scene_settings: SceneSettings):
    color = np.zeros(3)
    for light in lights:
        color += calc_color_by_light(point, norm, material, light, camera, scene_settings)
    color *= (1 - material.transparency)
    color += material.transparency * bg_color + refl_color

    return color


def main():
    parser = argparse.ArgumentParser(description='Python Ray Tracer')
    parser.add_argument('scene_file', type=str, help='Path to the scene file')
    parser.add_argument('output_image', type=str, help='Name of the output image file')
    parser.add_argument('--width', type=int, default=500, help='Image width')
    parser.add_argument('--height', type=int, default=500, help='Image height')
    args = parser.parse_args()

    # Parse the scene file
    camera, scene_settings, objects = parse_scene_file(args.scene_file)

    # TODO: Implement the ray tracer

    # Dummy result
    image_array = np.zeros((500, 500, 3))

    # Save the output image
    save_image(image_array)


if __name__ == '__main__':
    main()
