import numpy as np

def polar_to_cartesian(r, theta):
    return r * np.cos(theta), r * np.sin(theta)

def logarithmic_spiral(a, b, theta):
    r = a * np.exp(b * theta)
    return polar_to_cartesian(r, (1/b) * np.log(r/a))

def custom_transform(data):
    """
    Transform the `spiral.csv` data such that it can be more easily classified.

    To pass test_custom_transform_hard, your transformation should create at
    most three features and should allow a LogisticRegression model to achieve
    at least 90% accuracy.

    You can use free_response.q2.visualize_spiral() to visualize the spiral
    as we give it to you, and free_response.q2.visualize_transform() to
    visualize the 3D data transformation you implement here.

    Args:
        data: a Nx2 matrix from the `spiral.csv` dataset.

    Returns:
        A transformed data matrix that is (more) easily classified.
    """
    """
    #48-53%
    transformed_data = np.zeros((data.shape[0], 3))
    a, b = 1, 0.01
    for i in range(data.shape[0]):
        x, y = data[i]
        r, theta = np.sqrt(x**2 + y**2), np.arctan2(y, x)
        r_new, theta_new = logarithmic_spiral(a, b, theta)
        transformed_data[i, 0] = r_new
        transformed_data[i, 1] = theta_new
        transformed_data[i, 2] = abs(x - y)
    return transformed_data
    """
    #81.2%
    transformed_data = np.zeros((data.shape[0], 2))
    transformed_data[:, 0] = np.sin(np.pi*data[:, 0])
    transformed_data[:, 1] = np.sin(np.pi*data[:, 1])
    return transformed_data