import gym
import numpy as np
import warnings
from src import MultiArmedBandit, QLearning
from src import matplotlib, plt
from src.random import rng
    

def q3a():
    # https://www.gymlibrary.dev/environments/toy_text/frozen_lake/
    env = gym.make('FrozenLake-v1', map_name='8x8', is_slippery=True)

    epsilons = [0.008, 0.08, 0.8]
    n_eps = len(epsilons)
    trials = 10
    steps = 50000
    gamma = 0.9
    alpha = 0.2
    num_bins = 50

    results = np.zeros((n_eps, trials, num_bins))

    # For each value of epsilon, for `trials` number of trials,
    #   train a QLearning Agent with the given `epsilon` and
    #   the other above hyperparameters.
    for i, epsilon in enumerate(epsilons):
        rng.seed()
        for j in range(trials):
            # TODO: write your code here and delete "pass"
            pass

    average_results = []
    for i in range(n_eps):
        # TODO: replace this with a calculation of your results for each
        #       epsilon, averaged over all trials
        average_results.append((i + 1) * np.arange(num_bins))

    plt.title(fr"FRQ3a Q-Learning Comparison, $\gamma={gamma}$, $\alpha={alpha}$")
    plt.xlabel(f"Steps ({steps // num_bins} per point)")
    plt.ylabel("Reward")
    for i, eps in enumerate(epsilons):
        label = fr"$\epsilon={eps}$"
        plt.plot(average_results[i], label=label)

    plt.legend()
    fn = f"free_response/3a_g{gamma}_a{alpha}.png"
    plt.savefig(fn)
    plt.clf()
    print(f"Plot saved to {fn}")


if __name__ == "__main__":
    print("Running...")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        q3a()
