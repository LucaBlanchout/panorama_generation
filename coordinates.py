import math


def spherical_from_latlong(u, v, width, height):
    phi = math.pi * (2 * u / width - 1)
    theta = math.pi * (-v / height + 0.5)

    return theta, phi


def latlong_from_spherical(phi, theta, width, height):
    u = (width / 2) * ((phi / math.pi) + 1)
    v = height * (0.5 - (theta / math.pi))
    return u, v
