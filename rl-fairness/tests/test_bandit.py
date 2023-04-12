import gym
import numpy as np
import pytest


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_bandit_slots():
    """
    Tests that the MultiArmedBandit implementation successfully finds the slot
    machine with the largest expected reward.
    """
    from src import MultiArmedBandit
    from src.random import rng
    rng.seed()

    env = gym.make('SlotMachines-v0', n_machines=10, mean_range=(-10, 10), std_range=(5, 10))
    env.seed(0)
    means = np.array([m.mean for m in env.machines])

    agent = MultiArmedBandit(epsilon=0.2)
    state_action_values, rewards = agent.fit(env, steps=10000, num_bins=100)

    assert state_action_values.shape == (1, 10)
    assert len(rewards) == 100
    assert np.argmax(means) == np.argmax(state_action_values)

    _, rewards = agent.fit(env, steps=1000, num_bins=42)
    assert len(rewards) == 42
    _, rewards = agent.fit(env, steps=777, num_bins=100)
    assert len(rewards) == 100

    states, actions, rewards = agent.predict(env, state_action_values)
    assert len(actions) == 1 and actions[0] == np.argmax(means)
    assert len(states) == 1
    assert len(rewards) == 1


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_bandit_random_argmax():
    """
    When choosing to exploit the best action, do not use np.argmax: it will
    deterministically break ties by choosing the action with the lowest index.
    Instead, please *randomly choose* one of those tied-for-the-largest values.
    """

    from src import MultiArmedBandit
    from src.random import rng
    rng.seed()

    n_machines = 10
    env = gym.make('SlotMachines-v0', n_machines=n_machines,
                   mean_range=(-10, 10), std_range=(5, 10))
    env.seed(0)

    agent = MultiArmedBandit(epsilon=0.2)
    state_action_values = np.zeros([1, n_machines])

    actions = []
    for _ in range(1000):
        _, a, _ = agent.predict(env, state_action_values)
        actions.append(a[0])

    assert np.unique(actions).shape[0] == n_machines


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_bandit_frozen_lake():
    """
    Tests the MultiArmedBandit implementation on the FrozenLake-v1 environment.
    """
    from src import MultiArmedBandit
    from src.random import rng
    rng.seed()

    # https://www.gymlibrary.dev/environments/toy_text/frozen_lake/
    env = gym.make('FrozenLake-v1')
    env.reset()

    agent = MultiArmedBandit(epsilon=0.2)
    state_action_values, rewards = agent.fit(env, steps=1000)

    assert state_action_values.shape == (16, 4)
    assert len(rewards) == 100, "Rewards should have 100 elements regardless of the number of steps"
