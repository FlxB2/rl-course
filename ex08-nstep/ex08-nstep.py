import gym
import numpy as np
import matplotlib.pyplot as plt


def sarsa(env, alpha=0.1, gamma=0.9, epsilon=0.1, num_ep=int(1e4)):
    # Initalize Q arbitrarily, but make terminal states = 0
    Q = np.random.random((env.observation_space.n, env.action_space.n))
    env_t_states = ((env.desc == b'H') | (env.desc == b'G')).flatten()
    Q[env_t_states, :] = 0

    for _ in range(num_ep):
        state = env.reset()
        done = False
        action = pick_action(state, Q, epsilon)

        while not done:
            next_state, reward, done, _ = env.step(action)
            next_action = pick_action(state, Q, epsilon)

            Q[state, action] += alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])

            state = next_state
            action = next_action
    return Q


def nstep_sarsa(env, n=1, alpha=0.1, gamma=0.9, epsilon=0.1, num_ep=int(1e4)):
    Q = np.zeros([64, 4])  # 16 states and 4 actions

    for e in range(0, num_ep):
        initial_state = env.reset()
        T = np.inf
        t = 0
        action = pick_action(initial_state, Q, epsilon)
        rewards = [0]
        actions = [action]
        states = [initial_state]
        tau = 0
        while True:
            if t < T:
                (observation, reward, done, info) = env.step(actions[t])
                rewards.append(reward)
                states.append(observation)
                if done:
                    T = t + 1
                else:
                    action = pick_action(states[t + 1], Q, epsilon)
                    actions.append(action)
            tau = t - n + 1

            if tau >= 0:
                G = 0
                for i in range(tau + 1, min(tau + n, T + 1)):
                    G += np.power(gamma, i - tau - 1) * rewards[i]
                if tau + n < T:
                    G += np.power(gamma, n) * Q[states[tau + n]][actions[tau + n]]
                    Q[states[tau]][actions[tau]] += alpha * (G - Q[states[tau]][actions[tau]])

            t += 1
            if tau == T - 1:
                break
    return Q


# eps - greedy policy
def pick_action(state, Q, epsilon):
    eps = epsilon
    rnd = np.random.random()
    best_action = np.random.randint(0, 3)

    if rnd <= eps:
        return best_action

    for i in range(0, 3):
        if Q[state][best_action] is not None and Q[state][i] is not None:
            if Q[state][best_action] < Q[state][i]:
                best_action = i
    return best_action


environment = gym.make('FrozenLake-v0', map_name="8x8")

different_ns = [1, 5, 10, 100]
different_alphas = np.arange(0.01, 0.9, 0.1)
result = np.zeros((len(different_ns) + 1, len(different_alphas)))
sarsa_q = sarsa(environment)
alpha_count = 0
for n in range(0, len(different_ns)):
    for alpha in range(0, len(different_alphas)):
        print("n " + str(different_ns[n]) + " alpha " + str(different_alphas[alpha]))
        nstep_sarsa_q = nstep_sarsa(environment, different_ns[n], different_alphas[alpha])
        result[n][alpha] = np.sqrt(((nstep_sarsa_q - sarsa_q) ** 2).mean())
    plt.plot(different_alphas, result[n, :], label="n:"+str(n))
print("done")
print(str(result))
plt.show()
