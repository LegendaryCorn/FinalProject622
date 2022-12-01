import gym
import time
import numpy as np
import random

LEARNING_RATE = 0.15
EPSILON = 0.05
DISCOUNT = 0.95
EPISODES = 100000
RENDER_EPISODE = 1000000
AVG_EPISODE = 200

DISCRETE_OS_SIZE = [20, 40, 20, 40]
DISCRETE_OS_MIN = [-4.8, -20.0, -0.418, -2.0]
DISCRETE_OS_MAX = [4.8, 20.0, 0.418, 2.0]

Q_INIT_MIN = -2
Q_INIT_MAX = 0

env = gym.make('CartPole-v1')
env.reset()

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



num_actions = env.action_space.n
sum_avg_reward = 0.0
q = np.random.uniform(low=Q_INIT_MIN, high=Q_INIT_MAX, size=(DISCRETE_OS_SIZE + [num_actions]))

for e in range(EPISODES):

    state = get_state_from_observation(env.reset())

    if e % RENDER_EPISODE == RENDER_EPISODE - 1:
        render = True
    else:
        render = False

    done = False
    total_reward = 0
    while not done:

        # Choose a random action (explore)
        if(random.random() < EPSILON):
            action = random.randint(0, num_actions - 1)
        
        # Choose the most effective action(exploit)
        else:
            action = get_action_from_state(q, state)
        
        # Get state
        observation, reward, done, _ = env.step(action)
        new_state = get_state_from_observation(observation)
            
        # Render
        if render:
            env.render()
            time.sleep(1 / 60)

        if not done:

            # Calculate new Q
            optimal_future_value = np.max(q[new_state])
            new_q = q[state + (action,)] + LEARNING_RATE * (reward + DISCOUNT * optimal_future_value - q[state + (action,)])
            q[state + (action,)] = new_q

            # Total reward
            total_reward += reward

        else:
            sum_avg_reward += total_reward
            if e % AVG_EPISODE == AVG_EPISODE - 1:
                print(e + 1, sum_avg_reward / AVG_EPISODE)
                sum_avg_reward = 0

        
        state = new_state

env.close()