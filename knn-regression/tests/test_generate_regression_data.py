import numpy as np

from src.random import rng


def test_generate_regression_data():
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error

    from src.generate_regression_data import generate_regression_data

    degrees = range(2, 5)
    n_examples = [10, 100, 1000, 10000]
    noise_amounts = [0, 1, 2]

    for degree in degrees:
        good_transform = PolynomialFeatures(degree)
        bad_transform = PolynomialFeatures(degree - 1)
        model = LinearRegression()

        for n in n_examples:
            prev_good_mse = -1
            for amount_of_noise in noise_amounts:
                rng.seed()
                x, y = generate_regression_data(degree, n, amount_of_noise=amount_of_noise)
                assert (len(x) == n and len(y) == n), "Incorrect amount of data"
                assert (x.shape == (n, 1) and y.shape == (n, 1)), "Data arrays should be of shape (n, 1)"
                assert (x.min() >= -1 and x.max() <= 1), "X data outside of [-1, 1] range"

                model.fit(good_transform.fit_transform(x), y)
                good_mse = mean_squared_error(y, model.predict(good_transform.fit_transform(x)))
                msg = f"With degree {degree} and {n} examples, expected {good_mse:.3f} > {prev_good_mse:.3f}"
                assert good_mse > prev_good_mse, msg
                prev_good_mse = good_mse

                model.fit(bad_transform.fit_transform(x), y)
                bad_mse = mean_squared_error(y, model.predict(bad_transform.fit_transform(x)))
                msg = f"With {degree} and {n} examples, expected {bad_mse:.3f} > {good_mse:.3f}"
                assert bad_mse > good_mse, msg

                if amount_of_noise == 0:
                    assert np.isclose(good_mse, 0), "With no noise, {good_mse:.3f} should be 0"
                    

def test_generate_random_numbers():
    from src.generate_regression_data import generate_random_numbers

    degrees = range(2, 4)
    n_examples = [100, 10000]
    noise_amounts = [1, 2]
    expected = [[0.14039354083575928, 3.556330735924602, -1.4543656745987648],
                [0.14039354083575928, 3.556330735924602, -2.9087313491975295],
                [-0.26643956089597864, 4.965359625793974, -1.6196845653727006],
                [-0.26643956089597864, 4.965359625793974, -3.239369130745401],
                [0.14039354083575928, -4.599840536156703, -1.4543656745987648],
                [0.14039354083575928, -4.599840536156703, -2.9087313491975295],
                [-0.26643956089597864, -6.395945853674199, -1.6196845653727006],
                [-0.26643956089597864, -6.395945853674199, -3.239369130745401]]
    row = 0
    for degree in degrees:
        for n in n_examples:
            for amount_of_noise in noise_amounts:
                rng.seed()
                x, coefs, noise = generate_random_numbers(degree, n, amount_of_noise)

                x_right = (x[n // 2][0] == expected[row][0])
                coefs_right = (coefs[degree // 3] == expected[row][1])
                noise_right = (noise[n // 4][0] == expected[row][2])

                message = "Your random number generation is off; ask for help!"
                assert x_right and coefs_right and noise_right, message
                row += 1
