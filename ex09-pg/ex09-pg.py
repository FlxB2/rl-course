import gym
import numpy as np
import matplotlib.pyplot as plt


def policy(state, theta):
    """ TODO: return probabilities for actions under softmax action selection """
    res = 1 / (1 + np.exp(-np.dot(state, theta)))
    return [float(res[0]), float(1 - res[0])]


def generate_episode(env, theta, display=False):
    """ generates one episode and returns the list of states, the list of rewards and the list of actions of that
    episode """
    state = env.reset()
    states = [state]
    actions = []
    rewards = []
    for t in range(500):
        if display:
            env.render()
        p = policy(state, theta)
        action = np.random.choice(len(p), p=p)

        state, reward, done, info = env.step(action)
        rewards.append(reward)
        actions.append(action)
        if done:
            break
        states.append(state)

    return states, rewards, actions


def REINFORCE(env):
    theta = np.random.rand(4, 2)  # policy parameters
    T = 100
    gamma = 0.5
    alpha = 0.1
    length = []
    averages = []

    for e in range(10000):
        if e % 300 == 0:
            states, rewards, actions = generate_episode(env, theta, True)  # display the policy every 300 episodes
        else:
            states, rewards, actions = generate_episode(env, theta, False)

        print("episode: " + str(e) + " length: " + str(len(states)))
        # TODO: keep track of previous 100 episode lengths and compute mean
        length.append(len(states))

        if e > 100:
            a = np.mean(length[-100:])
            averages.append(a)
        else:
            a = 0

        if a > 498:
            break

        # TODO: implement the reinforce algorithm to improve the policy weights
        for t in range(0, T - 1):
            G = 0
            for k in range(t + 1, len(states)):
                G += gamma ** (k - t - 1) * rewards[k]
            theta += alpha * gamma ** t * G * np.dot(np.array(states).T, (1 / (1 + np.exp(-np.dot(states, theta)))) - 1)
    return averages


def main():
    env = gym.make('CartPole-v1')
    averages = REINFORCE(env)
    plt.plot(range(0, len(averages)), averages)
    plt.xlabel('Epochs')
    plt.ylabel('Average length')
    plt.show()
    env.close()
    env.close()


if __name__ == "__main__":
    main()
