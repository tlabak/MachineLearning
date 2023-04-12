import numpy as np
import src.random


def generate_random_numbers(degree, N, amount_of_noise):
    """
    Args:
        degree (int): degree of Polynomial that relates the output x and y
        N (int): number of points to generate
        amount_of_noise (float): amount of random noise to add to the relationship 
            between x and y.
    
    Returns:     
        Use src.random to generate `explanatory variable x`: an array of shape (N, 1) that contains 
        floats chosen uniformly at random between -1 and 1.
        Use src.random to generate `coefficient value coefs`: an array of shape (degree+1, ) that contains 
        floats chosen uniformly at random between -10 and 10.
        Use src.random to generate `noise variable n`: an array of shape (N, 1) that contains 
        floats chosen in the normal distribution. The mean is 0 and the standard deviation is `amount_of_noise`.

    Note that noise should have std `amount_of_noise`
        which we'll later multiply by `np.std(y)`
    """
    x = src.random.uniform(low=-1, high=1, size=(N, 1))
    coefs = src.random.uniform(low=-10, high=10, size=(degree+1,))
    noise = src.random.normal(loc=0, scale=amount_of_noise, size=(N, 1))
    return x, coefs, noise


def generate_regression_data(degree, N, amount_of_noise=1.0):
    """

    1. Call `generate_random_numbers` to generate the x values, the
       coefficients of our Polynomial, and the noise.

    2. Use the coefficients to construct a Polynomial function f()
       with the given coefficients.
       If coefficients is array([1, -2, 3]), f(x) = 1 - 2 x + 3 x^2

    3. Compute y0 = f(x) as the output of the regression *without noise*

    4. Create our noisy data `y` as `y0 + noise * np.std(y0)`

    Do not import or use these packages: scipy, sklearn, sys, importlib.
    Do not use these numpy or internal functions: polynomial, polyfit, polyval, getattr, globals

    Args:
        degree (int): degree of Polynomial that relates the output x and y
        N (int): number of points to generate
        amount_of_noise (float): scale of random noise to add to the relationship 
            between x and y.
    Returns:
        x (np.ndarray): explanatory variable of size (N, 1), ranges between -1 and 1.
        y (np.ndarray): response variable of size (N, 1), which responds to x as a
                        Polynomial of degree 'degree'.

    """
    x, coefs, noise = generate_random_numbers(degree, N, amount_of_noise)
    y0 = np.zeros_like(x)
    for i in range(degree, -1, -1):
        y0 = y0 * x + coefs[i]
    y = y0 + noise * np.std(y0)
    return x, y