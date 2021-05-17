import gym
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def main():
    # This example shows how to perform a single run with the policy that hits for player_sum >= 20
    env = gym.make('Blackjack-v0')
    obs = env.reset()  # obs is a tuple: (player_sum, dealer_card, useable_ace)
    done = False
    count = {}
    # current sum (12-21); dealer showing card 2-A; usable ace?; reward
    result = np.zeros((10, 10))
    episodes = 0
    while episodes < 10000:
        env.reset()
        done = False
        while not done:
            # no more cards
            if obs[0] >= 20:
                obs, reward, done, _ = env.step(0)
            # receive another card
            else:
                obs, reward, done, _ = env.step(1)
            # represent state 3 tuple as string
            r = reward
            usable_ace = int(obs[2])
            if obs[0] <= 21 and usable_ace == 1:
                sum_index = obs[0] - 12
                showing_card_index = obs[1] - 1
                key = str(obs[2]) + " " + str(showing_card_index) + " " + str(sum_index)
                if key not in count:
                    count[key] = [r]
                    result[showing_card_index][sum_index] = r
                else:
                    count[key].append(r)
                    result[showing_card_index][sum_index] = sum(count[key]) / len(count[key])

        episodes += 1
    fig = plt.figure()
    print(result)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(10, 200)
    x = range(0, len(result[0]), 1)
    y = range(0, len(result[0]), 1)
    X, Y = np.meshgrid(x, y)
    print(x)
    ax.plot_wireframe(X, Y, result, color='black')
    plt.show()

if __name__ == "__main__":
    main()
