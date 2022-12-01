import gym
import time
import numpy as np
import random
import multiprocessing

SEED = 128
NUM_PROCESSES = 10

LEARNING_RATE = 0.20

EPSILON_INIT = 0.9
EPSILON_REDUCTION = 0.00000010
EPSILON_MIN = 0.005

DISCOUNT = 0.999
EPISODES = 5000
AVG_EPISODE = 500

DISCRETE_OS_SIZE = [20, 20, 12, 12, 12, 12, 4, 4]
DISCRETE_OS_MIN = [-1, -0.5, -3, -3, -1.5, -3, 0, 0]
DISCRETE_OS_MAX = [ 1,  1.5,  3,  3,  1.5,  3, 1, 1]

Q_INIT_MIN = -0.01
Q_INIT_MAX = 0

def get_state_from_observation(observation):

    # Calculate the state
    state = np.array(observation)
    state = np.array(DISCRETE_OS_SIZE) * (state - np.array(DISCRETE_OS_MIN)) / (np.array(DISCRETE_OS_MAX) - np.array(DISCRETE_OS_MIN))
    state = state.astype(int)
    # Handles values that are too large/small
    for i in range(len(state)):
        if state[i] < 0:
            state[i] = 0
        if state[i] >= DISCRETE_OS_SIZE[i]:
            state[i] = DISCRETE_OS_SIZE[i] - 1

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
    sum_avg_reward = 0.0
    sum_success_count = 0
    q = np.random.uniform(low=Q_INIT_MIN, high=Q_INIT_MAX, size=(DISCRETE_OS_SIZE + [num_actions]))
    epsilon = EPSILON_INIT

    for e in range(EPISODES):

        state = get_state_from_observation(env.reset())

        done = False
        total_reward = 0
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
            new_q = q[state + (action,)] + LEARNING_RATE * (reward + DISCOUNT * optimal_future_value - q[state + (action,)])
            q[state + (action,)] = new_q

            # Total reward
            total_reward += reward

            # Recalculate Epsilon
            epsilon = max(epsilon * (1 - EPSILON_REDUCTION), EPSILON_MIN)

            if done:
                sum_avg_reward += total_reward

                sum_success_count += 1 if total_reward > 200 else 0

                if e % AVG_EPISODE == AVG_EPISODE - 1:
                    print(seed, e + 1, sum_avg_reward / AVG_EPISODE, epsilon, sum_success_count)
                    sum_avg_reward = 0
                    sum_success_count = 0

            
            state = new_state

    env.close()
    print(seed, " done")
    return seed

def main():
    pool = multiprocessing.Pool(processes=NUM_PROCESSES)

    random.seed(SEED)
    np.random.seed(SEED)
    np_seeds = np.random.randint(0, 10000000, NUM_PROCESSES)
    seeds = np_seeds.tolist()
    
    outputs = pool.map(q_learning, seeds)

if __name__=="__main__":
    main()