
import numpy as np

def fibonacci_sphere(samples=10, radius=1, randomize=False):
    rnd = 1.
    if randomize:
        rnd = np.random.random() * samples

    points = []
    offset = 2./samples
    print(offset) 
    print(samples)  
    increment = np.pi * (3. - np.sqrt(5.))

    for i in range(samples+1):
        y = ((i * offset) - 1) 
        print(y)
        r = np.sqrt(1 - y*y) * radius  # Scale the radius
        phi = ((i + rnd) % samples) * increment
        x = np.cos(phi) * r
        z = np.sin(phi) * r

        # Adjust to only keep points on the upper hemisphere
        points.append((x, y * radius, z))  # Apply radius scaling to y

    return np.array(points)

print(fibonacci_sphere())