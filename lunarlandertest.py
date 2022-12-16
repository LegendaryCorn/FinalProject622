import gym
import numpy as np
import random
import multiprocessing
import matplotlib.pyplot as plt
import lunarlandertrain as llt
import sys
import os

SEED = 256
NUM_EPISODES = 1000


def get_state_from_observation(observation):

    # Calculate the state
    state = np.zeros(len(observation))
    
    # Handles values that are too large/small
    for i in range(len(state)):
        if observation[i] < llt.FOCUSED_OS_MIN[i]:
            state[i] = 0
        elif observation[i] > llt.FOCUSED_OS_MAX[i]:
            state[i] = llt.DISCRETE_OS_SIZE[i] - 1
        else:
            state[i] = ((llt.DISCRETE_OS_SIZE[i] - 2) * (observation[i] - llt.FOCUSED_OS_MIN[i]) / (llt.FOCUSED_OS_MAX[i] - llt.FOCUSED_OS_MIN[i])) + 1

    state = state.astype(int)

    return tuple(state)



def get_action_from_state(q, state):
    return np.argmax(q[state])



def test_model(input):

    seed = SEED
    env = gym.make('LunarLander-v2')

    np.random.seed(seed)
    random.seed(seed)
    env.seed(seed)
    env.action_space.seed(seed)

    env.reset()

    success_count = 0
    total_rewards = np.zeros(NUM_EPISODES)

    q = input.q_table
    id = input.id

    for e in range(NUM_EPISODES):
        state = llt.get_state_from_observation(env.reset())
        total_reward = 0
        done = False

        while not done:

            action = llt.get_action_from_state(q, state)
            
            # Get state
            observation, reward, done, _ = env.step(action)
            state = llt.get_state_from_observation(observation)

            # Total reward
            total_reward += reward

        # Record data
        total_rewards[e] = total_reward
        success_count += 1 if total_reward >= 200 else 0
    
    print("Q Table", id, "- Average Reward:", np.average(total_rewards), "Success Count:", success_count)
    return total_rewards



def main():
    try:
        inpList = []
        num_processes = 0
        dir = sys.argv[1]

        for file in os.listdir(dir):
            npArr = np.load(dir + "/" + file)
            inpList.append(Input(npArr, num_processes))
            num_processes += 1

        pool = multiprocessing.Pool(processes=num_processes)
        outputs = pool.map(test_model, inpList)

    except:
        print("Please make sure that the 'qmodels' folder is in your command line input!")



class Input:
  def __init__(self, q_table, id):
    self.q_table = q_table
    self.id = id



if __name__=="__main__":
    main()