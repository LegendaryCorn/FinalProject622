import gym
import numpy as np
import random
import multiprocessing
import matplotlib.pyplot as plt

SEED = 128
NUM_PROCESSES = 10

LEARNING_RATE = 0.4

EPSILON_INIT = 0.5
EPSILON_REDUCTION = 0.000004
EPSILON_MIN = 0.001

DISCOUNT = 0.999
EPISODES = 10000
EPSIODE_CHECK = 500

DISCRETE_OS_SIZE = [6, 6, 4, 4, 4, 4, 2, 2]
DISCRETE_OS_MIN = [-1, -0.5, -2,   -2, -1.5, -2, 0, 0]
DISCRETE_OS_MAX = [ 1,  1.5,  2,  0.5,  1.5,  2, 1, 1]
FOCUSED_OS_MIN = [-0.1, -0.1, -0.5, -0.5, -0.2, -0.2, 0.5, 0.5]
FOCUSED_OS_MAX = [ 0.1,  0.1,  0.5,  0.1,  0.2,  0.2, 0.5, 0.5]

Q_INIT_MIN = -100
Q_INIT_MAX = -99

def get_state_from_observation(observation):

    # Calculate the state
    state = np.zeros(len(observation))
    
    # Handles values that are too large/small
    for i in range(len(state)):
        if observation[i] < FOCUSED_OS_MIN[i]:
            state[i] = 0
        elif observation[i] > FOCUSED_OS_MAX[i]:
            state[i] = DISCRETE_OS_SIZE[i] - 1
        else:
            state[i] = ((DISCRETE_OS_SIZE[i] - 2) * (observation[i] - FOCUSED_OS_MIN[i]) / (FOCUSED_OS_MAX[i] - FOCUSED_OS_MIN[i])) + 1

    state = state.astype(int)

    return tuple(state)

def get_action_from_state(q, state):
    return np.argmax(q[state])

def q_learning(seed):
    env = gym.make('LunarLander-v2')

    np.random.seed(seed)
    random.seed(seed)
    env.seed(seed)
    env.action_space.seed(seed)
    
    env.reset()

    num_actions = env.action_space.n
    q = np.random.uniform(low=Q_INIT_MIN, high=Q_INIT_MAX, size=(DISCRETE_OS_SIZE + [num_actions]))
    epsilon = EPSILON_INIT

    reward_table = np.zeros(EPISODES)

    for e in range(EPISODES):

        state = get_state_from_observation(env.reset())
        total_reward = 0
        done = False
        while not done:

            # Choose a random action (explore)
            if(random.random() < epsilon):
                action = random.randint(0, num_actions - 1)
            
            # Choose the most effective action(exploit)
            else:
                action = get_action_from_state(q, state)
            
            # Get state
            observation, reward, done, _ = env.step(action)
            new_state = get_state_from_observation(observation)


            # Calculate new Q
            optimal_future_value = np.max(q[new_state])
            new_q = q[state + (action,)] + LEARNING_RATE * (reward + (DISCOUNT) * optimal_future_value - q[state + (action,)])
            q[state + (action,)] = new_q

            # Total reward
            total_reward += reward

            # Recalculate Epsilon
            epsilon = max(epsilon * (1 - EPSILON_REDUCTION), EPSILON_MIN)
            state = new_state
        
        # Record data
        reward_table[e] = total_reward

        # Progress Check
        if(e % EPSIODE_CHECK == 0):
            print("Seed", seed, "is", "%f%% done."%(100.0 * e / EPISODES), epsilon)

    env.close()
    print("Seed", seed, "has finished.")
    return reward_table

def main():

    # Seeds
    random.seed(SEED)
    np.random.seed(SEED)
    np_seeds = np.random.randint(0, 10000000, NUM_PROCESSES)
    seeds = np_seeds.tolist()
    
    # Run
    pool = multiprocessing.Pool(processes=NUM_PROCESSES)
    outputs = pool.map(q_learning, seeds)
    np_outputs = np.array(outputs)
    np_avg_outputs = np.average(np_outputs, 0)

    # Plot
    plt.plot(range(EPISODES), np_avg_outputs)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.show()

if __name__=="__main__":
    main()