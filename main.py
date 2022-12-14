import numpy as np
from PIL import Image

from perlin_numpy import generate_fractal_noise_2d

def perlin_noise (
        shape = (512,512),
        res = (8,8),
        octaves = 5,
        persistence = 0.5,
        lacunarity = 2,
        seed = None,
    ):
    if seed:
        np.random.seed(seed)

    noise = generate_fractal_noise_2d(
            shape=shape,
            res=res,
            octaves=octaves,
            persistence=persistence,
            lacunarity=lacunarity,
        )

    clamp_noise = (noise < -1)*-1+\
                  (noise < 1)*(noise >= -1)*noise+\
                  (noise >= 1)*1
    return clamp_noise

def pythagoras(x1, x2, y1, y2):
    return np.sqrt(np.abs(x1-x2)**2+np.abs(y1-y2)**2)

def linear(value, x1, x2, y1, y2):
    return (y1-y2)*(value-x1)/(x1-x2) + y1

def sigmoid(value, bias=0):
    return np.tanh(np.tan(np.pi*value/2))

def semicircle(value):
    return np.sqrt(4-(value-1)**2)-1

def add_border(world):
    center_x, center_y = world.shape[1] // 2, world.shape[0] // 2

    # perlin = linear(world, -.5, .5, -1, 1)
    perlin = world
    xx, yy = np.meshgrid(np.arange(world.shape[1]), np.arange(world.shape[0]))
    dist = pythagoras(xx, center_x, yy, center_y)
    dist = linear(dist, min(center_x, center_y), 0, -1, 1)
    dist = np.maximum(dist, -1)
    dist = sigmoid(semicircle(semicircle(dist)))
    dist = linear(dist, -1, 1, 0, 1)
    perlin = linear(perlin, -1, 1, 0, 1)
    perlin_in_sphere = linear(dist*perlin, 0, 1, -1, 1)

    return perlin_in_sphere

deep_blue = np.array([0, 95, 215])
blue = np.array([0, 135, 255])
green = np.array([0, 135, 0])
beach = np.array([255, 215, 175])
snow = np.array([255, 255, 255])
mountain = np.array([138, 138, 138])

def add_color(world):
    world = world[..., np.newaxis]

    color_world = (world < -0.2)*deep_blue+\
                  (world < 0.25)*(world >= -0.2)*blue+\
                  (world < 0.35)*(world >= 0.25)*beach+\
                  (world < 0.65)*(world >= 0.35)*green+\
                  (world < 0.85)*(world >= 0.65)*mountain+\
                  (world <= 1.0)*(world >= 0.85)*snow
    return color_world.astype(np.uint8)


generated_noise = perlin_noise()
world = add_border(generated_noise)
color_world = add_color(world)

Image.fromarray(linear(generated_noise, -1, 1, 0, 255).astype(np.uint8), mode="L").show()
Image.fromarray(color_world).show()
