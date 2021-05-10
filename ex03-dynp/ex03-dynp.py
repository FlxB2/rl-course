import gym
import numpy as np

# Init environment
env = gym.make("FrozenLake-v0")
# you can set it to deterministic with:
# env = gym.make("FrozenLake-v0", is_slippery=False)

# If you want to try larger maps you can do this using:
# random_map = gym.envs.toy_text.frozen_lake.generate_random_map(size=5, p=0.8)
# env = gym.make("FrozenLake-v0", desc=random_map)


# Init some useful variables:
n_states = env.observation_space.n
n_actions = env.action_space.n


def value_iteration():
    V_states = np.zeros(n_states)  # init values as zero
    theta = 1e-8
    gamma = 0.8
    # TODO: implement the value iteration algorithm and return the policy Hint: env.P[state][action] gives you tuples
    #  (p, n_state, r, is_terminal), which tell you the probability p that you end up in the next state n_state and
    #  receive reward r
    delta = 1

    policy = {}
    # init policy for the first states possible after t=0 (during t=1 n_action states are possible)
    for s in range(0, n_actions):
        policy[s] = np.random.choice(range(0, n_actions))

    # initial value function
    V = np.zeros(n_states)  # init values as zero

    steps = 0
    while delta >= theta:
        delta = 0
        for s in range(0, n_states):
            old_v = V[s]
            mx = -1
            # test all possible actions
            for a in range(0, n_actions):
                add = 0
                # test all possible outcomes
                for x in range(0, len(env.P[s][a])):
                    (p, n_state, r, is_terminal) = env.P[s][a][x]
                    add += p * (r + gamma * V[n_state])
                if add > mx:
                    mx = add
                    policy[s] = a
            V[s] = mx
            delta = max(delta, abs(old_v - mx))
        steps += 1
    print("# steps " + str(steps))
    print("optimal value function " + str(V))
    return policy


def main():
    # print the environment
    print("current environment: ")
    env.render()
    print("")

    # run the value iteration
    policy = value_iteration()
    print("Computed policy:")
    print(policy)

    # This code can be used to "rollout" a policy in the environment:
    print("rollout policy:")
    maxiter = 100
    state = env.reset()
    for i in range(maxiter):
        new_state, reward, done, info = env.step(policy[state])
        env.render()
        state = new_state
        if done:
            print("Finished episode")
            break


if __name__ == "__main__":
    main()
