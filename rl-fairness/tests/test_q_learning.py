import gym
import pytest
import numpy as np


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_q_learning_slots():
    """
    Tests that the Qlearning implementation successfully finds the slot
    machine with the largest expected reward.
    """
    from src import QLearning
    from src.random import rng
    rng.seed()

    env = gym.make('SlotMachines-v0', n_machines=10, mean_range=(-10, 10), std_range=(1, 5))
    env.seed(0)
    means = np.array([m.mean for m in env.machines])

    agent = QLearning(epsilon=0.2, gamma=0)
    state_action_values, rewards = agent.fit(env, steps=1000)

    assert state_action_values.shape == (1, 10)
    assert len(rewards) == 100
    assert np.argmax(means) == np.argmax(state_action_values)

    states, actions, rewards = agent.predict(env, state_action_values)
    assert len(actions) == 1 and actions[0] == np.argmax(means)
    assert len(states) == 1 and states[0] == 0
    assert len(rewards) == 1


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_q_learning_frozen_lake():
    """
    Tests that the QLearning implementation successfully learns the
    FrozenLake-v1 environment.
    """
    from src import QLearning
    from src.random import rng
    rng.seed()

    # https://www.gymlibrary.dev/environments/toy_text/frozen_lake/
    env = gym.make('FrozenLake-v1')
    env.reset()

    agent = QLearning(epsilon=0.4, gamma=0.9, alpha=0.5)
    state_action_values, rewards = agent.fit(env, steps=10000)

    state_values = np.max(state_action_values, axis=1)

    assert state_action_values.shape == (16, 4)
    assert len(rewards) == 100

    assert np.allclose(state_values[np.array([5, 7, 11, 12, 15])], np.zeros(5))
    assert np.all(state_values[np.array([0, 1, 2, 3, 4, 6, 8, 9, 10, 13, 14])] > 0)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_q_learning_random_argmax():
    """
    When choosing to exploit the best action, do not use np.argmax: it will
    deterministically break ties by choosing the action with the lowest index.
    Instead, please *randomly choose* one of those tied-for-the-largest values.
    """
    from src import QLearning
    from src.random import rng
    rng.seed()

    n_machines = 10
    env = gym.make('SlotMachines-v0', n_machines=n_machines,
                   mean_range=(-10, 10), std_range=(5, 10))
    env.seed(0)

    agent = QLearning()
    state_action_values = np.zeros([1, n_machines])

    actions = []
    for _ in range(1000):
        _, a, _ = agent.predict(env, state_action_values)
        actions.append(a[0])

    assert np.unique(actions).shape[0] == n_machines


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_q_learning_deterministic():
    """
    Tests that the QLearning implementation successfully navigates a
    deterministic environment with provided state-action-values.
    """
    from src import QLearning

    np.random.seed(0)

    env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False)

    agent = QLearning(epsilon=0.5)
    state_action_values = np.array([
        [0.0, 0.7, 0.3, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.51, 0.49, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.5, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.2, 0.8, 0.0],
        [0.0, 0.2, 0.8, 0.0],
        [0.0, 0.6, 0.4, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0]
    ])

    states, actions, rewards = agent.predict(env, state_action_values)
    assert np.all(states == np.array([4, 8, 9, 10, 14, 15]))
    assert np.all(actions == np.array([1, 1, 2, 2, 1, 2]))
    assert np.all(rewards == np.array([0, 0, 0, 0, 0, 1]))
