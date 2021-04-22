import numpy as np
import matplotlib.pyplot as plt
import random
import time


class GaussianBandit:
    def __init__(self):
        self._arm_means = np.random.uniform(0., 1., 10)  # Sample some means
        self.n_arms = len(self._arm_means)
        self.rewards = []
        self.total_played = 0

    def reset(self):
        self.rewards = []
        self.total_played = 0

    def play_arm(self, a):
        reward = np.random.normal(self._arm_means[a], 1.)  # Use sampled mean and covariance of 1.
        self.total_played += 1
        self.rewards.append(reward)
        return reward


def greedy(bandit, timesteps):
    rewards = np.zeros(bandit.n_arms)
    n_plays = np.zeros(bandit.n_arms)
    Q = np.zeros(bandit.n_arms)
    possible_arms = range(bandit.n_arms)

    # TODO: init variables (rewards, n_plays, Q) by playing each arm once
    for i in possible_arms:
        rewards[i] = bandit.play_arm(i)     # play each arm
        n_plays[i] += 1

    Q[np.argmax(rewards)] += np.max(rewards)

    # Main loop
    while bandit.total_played < timesteps:
        # TODO: instead do greedy action selection
        # TODO: update the variables (rewards, n_plays, Q)for the selected arm
        a = np.argmax(Q)                    # greedy select the best arm
        rewards[a] = bandit.play_arm(a)     # play this arm
        n_plays[a] += 1
        prev_reward = Q[np.argmax(rewards)]
        Q[np.argmax(rewards)] = prev_reward + ((np.max(rewards) - prev_reward) / n_plays[a])  # Book, page 54 - Meaning?



def epsilon_greedy(bandit, timesteps):
    # TODO: epsilon greedy action selection (you can copy your code for greedy as a starting point)
    rewards = np.zeros(bandit.n_arms)
    n_plays = np.zeros(bandit.n_arms)
    Q = np.zeros(bandit.n_arms)
    possible_arms = range(bandit.n_arms)
    epsilon = 0.01

    for i in possible_arms:
        rewards[i] = bandit.play_arm(i)
        n_plays[i] += 1
    Q[np.argmax(rewards)] += np.max(rewards)

    # Main loop
    while bandit.total_played < timesteps:
        r = random.random()  # number between 0.0 and 1.0
        # Prob == r ? Do random action
        if r <= epsilon:
            a = random.choice(possible_arms)
            rewards[a] = bandit.play_arm(a)
        # do greedy action
        else:
            a = np.argmax(Q)
            rewards[a] = bandit.play_arm(a)
        n_plays[a] += 1
        prev_reward = Q[np.argmax(rewards)]
        Q[np.argmax(rewards)] = prev_reward + ((np.max(rewards) - prev_reward) / n_plays[a])  # Book, page 54 - Meaning?


def main():
    n_episodes = 500  # TODO: set to 10000 to decrease noise in plot
    n_timesteps = 1000
    rewards_greedy = np.zeros(n_timesteps)
    rewards_egreedy = np.zeros(n_timesteps)

    for i in range(n_episodes):
        if i % 100 == 0:
            print ("current episode: " + str(i))

        b = GaussianBandit()  # initializes a random bandit
        greedy(b, n_timesteps)
        rewards_greedy += b.rewards

        b.reset()  # reset the bandit before running epsilon_greedy
        epsilon_greedy(b, n_timesteps)
        rewards_egreedy += b.rewards

    rewards_greedy /= n_episodes
    rewards_egreedy /= n_episodes
    plt.plot(rewards_greedy, label="greedy")
    print("Total reward of greedy strategy averaged over " + str(n_episodes) + " episodes: " + str(np.sum(rewards_greedy)))
    plt.plot(rewards_egreedy, label="e-greedy")
    print("Total reward of epsilon greedy strategy averaged over " + str(n_episodes) + " episodes: " + str(np.sum(rewards_egreedy)))
    plt.legend()
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.savefig('bandit_strategies.eps')
    plt.show()


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))