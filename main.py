import noise
import numpy as np
from PIL import Image
import math

from utils import *

shape = (512,512)
scale = 0.1
octaves = 5
persistence = 0.5
lacunarity = 2.0
seed = np.random.randint(0,100)

world = np.zeros(shape)

# make coordinate grid on [0,1]^2
x_idx = np.linspace(0, 1, shape[0])
y_idx = np.linspace(0, 1, shape[1])
world_x, world_y = np.meshgrid(x_idx, y_idx)

# apply perlin noise, instead of np.vectorize, consider using itertools.starmap()
world = np.vectorize(noise.pnoise2)(world_x/scale,
                        world_y/scale,
                        octaves=octaves,
                        persistence=persistence,
                        lacunarity=lacunarity,
                        repeatx=1024,
                        repeaty=1024,
                        base=seed)

deep_blue = [0, 95, 215]
blue = [0, 135, 255]
green = [0, 135, 0]
beach = [255, 215, 175]
snow = [255, 255, 255]
mountain = [138, 138, 138]

def add_color(world):
    color_world = np.zeros(world.shape+(3,), dtype=np.uint8)
    center_x, center_y = shape[1] // 2, shape[0] // 2

    for y in range(shape[0]):
        for x in range(shape[1]):

            perlin = linear(world[x][y], -.5, .5, -1, 1)

            # distance from centre, measured in pixels
            dist = pythagoras(x, center_x, y, center_y)

            # edge -> -1, centre -> 1
            dist = linear(dist, min(center_x, center_y), 0, -1, 1)

            # never go below -1 (important for corners)
            dist = max(-1, dist)

            # change the distribution to make the circle more consistent
            dist = sigmoid(semicircle(semicircle(dist)))

            # overlay distance and perlin, using [0, 1] for multiplication
            dist = linear(dist, -1, 1, 0, 1)
            perlin = linear(perlin, -1, 1, 0, 1)
            perlin_in_sphere = linear(dist*perlin, 0, 1, -1, 1)

            # edge -> 0, centre -> 255
            # perlin_in_sphere = linear(perlin_in_sphere, -1, 1, 0, 255)

            # color_world[x][y] = greyscale(perlin_in_sphere)

            if perlin_in_sphere < -0.2:
                color_world[x][y] = deep_blue
            elif perlin_in_sphere < 0.25:
                color_world[x][y] = blue
            elif perlin_in_sphere < 0.35:
                color_world[x][y] = beach
            elif perlin_in_sphere < 0.65:
                color_world[x][y] = green
            elif perlin_in_sphere < 0.85:
                color_world[x][y] = mountain
            elif perlin_in_sphere < 1.0:
                color_world[x][y] = snow

    return color_world


# img = np.floor((world + .5) * 255).astype(np.uint8) # <- Normalize world first
color_world = add_color(world)

Image.fromarray(color_world).show()
