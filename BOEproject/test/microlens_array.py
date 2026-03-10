import numpy as np


def create_microlens_array_with_sampling(array_size, lens_pitch, radius_of_curvature, sampling_points,
                                         lens_type="parabolic"):
    """
    Create a microlens array with specified sampling points.

    Parameters:
        array_size (tuple): Size of the array (rows, cols) as (m, n).
        lens_pitch (float): Distance between the centers of adjacent microlenses.
        radius_of_curvature (float): Radius of curvature of the microlenses.
        sampling_points (tuple): Sampling points per microlens (points_x, points_y).
        lens_type (str): Type of lens profile. Options are "parabolic" or "spherical".

    Returns:
        numpy.ndarray: 2D array representing the microlens surface sag.
    """
    # Extract parameters
    rows, cols = array_size
    points_x, points_y = sampling_points

    # Total sampling dimensions
    total_x = cols * points_x
    total_y = rows * points_y

    # Generate grid for sampling
    x = np.linspace(-lens_pitch * cols / 2, lens_pitch * cols / 2, total_x)
    y = np.linspace(-lens_pitch * rows / 2, lens_pitch * rows / 2, total_y)
    xx, yy = np.meshgrid(x, y)

    # Initialize the sag array
    sag = np.zeros_like(xx)

    # Compute the sag for each microlens
    for i in range(rows):
        for j in range(cols):
            # Center of the current microlens
            x_center = j * lens_pitch - (cols // 2) * lens_pitch
            y_center = i * lens_pitch - (rows // 2) * lens_pitch

            # Distance from the microlens center
            distance = np.sqrt((xx - x_center) ** 2 + (yy - y_center) ** 2)

            # Apply the sag profile
            mask = distance <= lens_pitch / 2
            if lens_type == "parabolic":
                # Parabolic profile: sag = -(x^2 + y^2) / (2R)+rmax^2/(2R)
                sag[mask] += -(distance[mask] ** 2) / (2 * radius_of_curvature) + (lens_pitch/2)**2/ (2 * radius_of_curvature)
            elif lens_type == "spherical":
                # Spherical profile: sag = R - sqrt(R^2 - r^2), r <= lens_radius
                sag[mask] += radius_of_curvature - np.sqrt(radius_of_curvature ** 2 - distance[mask] ** 2)
            else:
                raise ValueError("Unsupported lens_type. Choose 'parabolic' or 'spherical'.")

    return (sag)




