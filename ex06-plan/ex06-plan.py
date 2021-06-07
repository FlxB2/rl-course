import gym
import copy
import random
import numpy as np
import matplotlib.pyplot as plt


class Node:
    def __init__(self, parent=None, action=None):
        self.parent = parent  # parent of this node
        self.action = action  # action leading from parent to this node
        self.children = []
        self.sum_value = 0.  # sum of values observed for this node, use sum_value/visits for the mean
        self.visits = 1


def rollout(env, maxsteps=100):
    """ Random policy for rollouts """
    G = 0
    for i in range(maxsteps):
        action = env.action_space.sample()
        _, reward, terminal, _ = env.step(action)
        G += reward
        if terminal:
            return G
    return G


def mcts(env, root, maxiter=500):
    """ TODO: Use this function as a starting point for implementing Monte Carlo Tree Search
    """

    # this is an example of how to add nodes to the root for all possible actions:
    root.children = [Node(root, a) for a in range(env.action_space.n)]

    eps = 0.01

    path_array = []
    for i in range(maxiter):
        state = copy.deepcopy(env)
        G = 0.

        # TODO: traverse the tree using an epsilon greedy tree policy
        terminal = True
        current_path_length = 0
        while len(root.children) > 0:
            rand = random.uniform(0.0, 1.0)
            new_node = random.choice(root.children)
            if rand > eps:
                # take greedy action
                curr_max = -1000000
                for n in root.children:
                    if n.sum_value > curr_max:
                        new_node = n
                        curr_max = n.sum_value

            # step down to the next node
            current_path_length += 1
            root = new_node
            _, reward, terminal, _ = state.step(root.action)
            G += reward

        path_array.append(current_path_length)

        # TODO: Expansion of tree
        if not terminal:
            expanded_nodes = [Node(root, a) for a in range(state.action_space.n)]
            root.children = expanded_nodes

        # This performs a rollout (Simulation):
        if not terminal:
            G += rollout(state)

        # TODO: update all visited nodes in the tree
        # This updates values for the current node:
        while True:
            root.visits += 1
            root.sum_value += G
            if root.parent is not None:
                root = root.parent
            else:
                break

    plt.plot(range(len(path_array)), path_array, color='red')
    plt.show()

def main():
    env = gym.make("Taxi-v3")
    env.seed(0)  # use seed to make results better comparable
    # run the algorithm 10 times:
    rewards = []
    for i in range(10):
        env.reset()
        terminal = False
        root = Node()  # Initialize empty tree
        sum_reward = 0.
        while not terminal:
            #env.render()
            mcts(env, root)  # expand tree from root node using mcts
            values = [c.sum_value / c.visits for c in root.children]  # calculate values for child actions
            bestchild = root.children[np.argmax(values)]  # select the best child
            _, reward, terminal, _ = env.step(bestchild.action)  # perform action for child
            root = bestchild  # use the best child as next root
            root.parent = None
            sum_reward += reward
        rewards.append(sum_reward)
        print("finished run " + str(i + 1) + " with reward: " + str(sum_reward))
    print("mean reward: ", np.mean(rewards))


if __name__ == "__main__":
    main()
