import gym
import numpy as np

env = gym.make('MountainCar-v0', render_mode = 'human')
env.reset()

# #Highest value for the observation space.
# print(env.observation_space.high) 

# #Lowest value for the observation space.
# print(env.observation_space.low) 

# #Number of actions the agent can perform.
# print(env.action_space.n)


LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25000
SHOW_EVERY = 2000

epsilon = 0.5 #The extent to which the agent explores the environment.
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

DISCRETE_OBSERVATION_SIZE = [20] * len(env.observation_space.high)
DISCRETE_OBSERVATION_WINDOW_SIZE = (env.observation_space.high - env.observation_space.low) / DISCRETE_OBSERVATION_SIZE

q_table = np.random.uniform(low = -2, high = 0, size = (DISCRETE_OBSERVATION_SIZE + [env.action_space.n])) #(20, 20, 3)

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / DISCRETE_OBSERVATION_WINDOW_SIZE
    return tuple(discrete_state.astype(np.int))


#Iterate over episodes.
for episode in range(EPISODES): 
    if episode % SHOW_EVERY == 0:
        print(episode)
        render = True
    else:
        render = False

    discrete_state = get_discrete_state(env.reset())
    done = False  

    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, done, _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)

        if render:
            env.render()

        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]

            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

            #Update the q table.
            q_table[discrete_state + (action, )] = new_q
        
        elif new_state[0] >= env.goal_position:
            q_table[discrete_state + (action, )] = 0
        
        discrete_state = new_discrete_state
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value
env.close()