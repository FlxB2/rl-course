#!/usr/bin/env python3
import gym
import numpy as np
import matplotlib.pyplot as plt


EPISODE_LOG = 50

BUCKET_AMOUNT = [20, 20]
env = gym.make('MountainCar-v0')
BUCKET_SIZE = (env.observation_space.high - env.observation_space.low) / BUCKET_AMOUNT


def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / BUCKET_SIZE
    return tuple(discrete_state.astype(np.int32))


# tune learning rate
def qlearning(env, Q, alpha=0.001, gamma=0.9, epsilon=0.1, num_ep=int(5000)):

    episode_rewards = []
    episode_lengths = []
    episode_hits = []
    aggr_ep_rewards = {'ep': [], 'avg': [], 'max': [], 'min': []}
    aggr_ep_goal = {'ep': [], 'goal': [], 'length': []}

    for episode in range(num_ep):
        episode_reward = 0
        episode_length = 0
        reached_goal = 0

        state = env.reset()
        discrete_state = get_discrete_state(state)
        done = False


        while not done:
            if np.random.uniform(0, 1) > epsilon:
                action = np.argmax(Q[discrete_state])
            else:
                action = env.action_space.sample()

            new_state, reward, done, _ = env.step(action)
            new_discrete_state = get_discrete_state(new_state)

            '''render if wanted
            if (episode + 1) % 50 == 0:
                env.render()
            '''
            Q[discrete_state + (action,)] += alpha * (reward + gamma *
                                                                    np.max(Q[new_discrete_state]) -
                                                                    Q[discrete_state + (action,)])

            if new_state[0] >= env.goal_position:
                reached_goal += 1

            discrete_state = new_discrete_state

            episode_reward += reward
            episode_length += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_hits.append(reached_goal)

        if not episode % EPISODE_LOG:
            if episode == 0:
                average_reward = episode_rewards[0]
            else:
                average_reward = sum(episode_rewards[-EPISODE_LOG:]) / EPISODE_LOG
            aggr_ep_rewards['ep'].append(episode)
            aggr_ep_rewards['avg'].append(average_reward)
            aggr_ep_rewards['max'].append(max(episode_rewards[-EPISODE_LOG:]))
            aggr_ep_rewards['min'].append(min(episode_rewards[-EPISODE_LOG:]))
            if episode == 0:
                average_goal = episode_hits[0]
            else:
                average_goal = sum(episode_hits[-EPISODE_LOG:]) / EPISODE_LOG
            if episode == 0:
                average_length = episode_lengths[0]
            else:
                average_length = sum(episode_lengths[-EPISODE_LOG:]) / EPISODE_LOG
            aggr_ep_goal['ep'].append(episode)
            aggr_ep_goal['goal'].append(average_goal)
            aggr_ep_goal['length'].append(average_length)

    env.close()

    plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'],
             label="aggregated average rewards of " + str(EPISODE_LOG) + " episodes")
    plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'],
             label="aggregated max rewards of " + str(EPISODE_LOG) + " episodes")
    plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'],
             label="aggregated min rewards of " + str(EPISODE_LOG) + " episodes")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig('reward.png')
    # plt.show()
    plt.clf()

    return episode_hits, episode_lengths


def main():
    reached_goals = []
    episode_lengths = []
    for i in range(10):
        Q = np.random.uniform(low=-2, high=0, size=(BUCKET_AMOUNT + [env.action_space.n]))
        reached_goal, episode_length = qlearning(env, Q)
        reached_goals.append(reached_goal)
        episode_lengths.append(episode_length)
    env.close()

    episodes = [i for i in range(np.mean(reached_goals, axis=0).shape[0])]

    plt.plot(episodes, np.mean(reached_goals, axis=0), label="reaching goal")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig('goal.png')
    # plt.show()

    plt.clf()
    plt.plot(episodes, np.mean(episode_lengths, axis=0), label="episode length")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig('length.png')
    # plt.show()


if __name__ == "__main__":
    main()
