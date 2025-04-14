
from __future__ import division, print_function
from env import CartPole, Physics
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import lfilter
import random
from scipy.io import savemat
import copy
import gym
import time as t_o
import imageio

def initialize_mdp_data(num_states):

    transition_counts = np.zeros((num_states, num_states, 2))
    transition_probs = np.ones((num_states, num_states, 2)) / num_states
    reward_counts = np.zeros((num_states, 2)) #Index zero is count of rewards being -1 , index 1 is count of total num state is reached
    reward = np.zeros(num_states)
    value = np.random.rand(num_states) * 0.1

    return {
        'transition_counts': transition_counts,
        'transition_probs': transition_probs,
        'reward_counts': reward_counts,
        'reward': reward,
        'value': value,
        'num_states': num_states,
    }

def choose_action(state, mdp_data,GAMMA): 

    update=mdp_data['reward'] + GAMMA*mdp_data['value']

    vs_0= np.dot(mdp_data['transition_probs'][state,:,0],update)
    vs_1=np.dot(mdp_data['transition_probs'][state,:,1],update)

    if vs_0 > vs_1:
        result = 0
    elif vs_1 > vs_0:
        result = 1
    else:
        result = random.choice([0, 1])
    return result

def update_mdp_transition_counts_reward_counts(mdp_data, state, action, new_state, reward):

    mdp_data['transition_counts'][state,new_state,action]= mdp_data['transition_counts'][state,new_state,action]+1
    if reward==-1:
        mdp_data['reward_counts'][new_state,0]=mdp_data['reward_counts'][new_state,0]-1
    mdp_data['reward_counts'][new_state,1]=mdp_data['reward_counts'][new_state,1]+1
    return

def update_mdp_transition_probs_reward(mdp_data):

    '''
    mdp_data['transition_probs'][:,:,0] = mdp_data['transition_probs'][:,:,0] + mdp_data['transition_counts'][:,:,0]/(np.sum(mdp_data['transition_counts'][:,:,0], axis=1, keepdims=True))
    
    mdp_data['transition_probs'][:,:,0] = mdp_data['transition_probs'][:,:,0] / np.sum(mdp_data['transition_probs'][:,:,0], axis=1, keepdims=True)

    mdp_data['transition_probs'][:,:,1] = mdp_data['transition_probs'][:,:,1] + mdp_data['transition_counts'][:,:,1]/(np.sum(mdp_data['transition_counts'][:,:,1], axis=1, keepdims=True))

    mdp_data['transition_probs'][:,:,1] = mdp_data['transition_probs'][:,:,1] / np.sum(mdp_data['transition_probs'][:,:,1], axis=1, keepdims=True)
    '''
    for i in range(mdp_data['transition_counts'].shape[0]):
    
        has_non_zero_0 = np.any(mdp_data['transition_counts'][i,:,0] != 0)

        if has_non_zero_0:
            mdp_data['transition_probs'][i,:,0]= mdp_data['transition_counts'][i,:,0]/(np.sum(mdp_data['transition_counts'][i,:,0]))
        
        has_non_zero_1 = np.any(mdp_data['transition_counts'][i,:,1] != 0)
      
        if has_non_zero_1:
            mdp_data['transition_probs'][i,:,1]= mdp_data['transition_counts'][i,:,1]/(np.sum(mdp_data['transition_counts'][i,:,1]))

        if mdp_data['reward_counts'][i, 1]!=0:
            mdp_data['reward'][i] = mdp_data['reward_counts'][i, 0]/mdp_data['reward_counts'][i, 1]

    return

def update_mdp_value(mdp_data, tolerance, gamma):

    updates = mdp_data['reward'] + gamma * mdp_data['value']
    value_0 = np.dot(mdp_data['transition_probs'][:, :, 0], updates)
    value_1 = np.dot(mdp_data['transition_probs'][:, :, 1], updates)
    value_previous=copy.deepcopy(mdp_data['value'])
    mdp_data['value'] = np.maximum(value_0, value_1)

    error = np.max(np.abs(mdp_data['value'] - value_previous))

    print(error)

    converged = error < tolerance

    #print(converged)
    return converged,error

def main(plot=True):
    
    np.random.seed(0) # Seed the randomness of the simulation so this outputs the same thing each time
    
    # Simulation parameters
    min_trial_length_to_start_display = 100
   

    NUM_STATES = 163
    GAMMA = 0.995
    TOLERANCE = 0.01
    NO_LEARNING_THRESHOLD = 20

    # Time cycle of the simulation
    time = 0

    # These variables perform bookkeeping (how many cycles was the pole
    # balanced for before it fell). Useful for plotting learning curves.
    time_steps_to_failure = []
    num_failures = 0
    time_at_start_of_current_trial = 0
    max_failures = 500

    # Initialize a cart pole
    cart_pole = CartPole(Physics())
    x, x_dot, theta, theta_dot = 0.0, 0.0, 0.0, 0.0
    state_tuple = (x, x_dot, theta, theta_dot)
    state = cart_pole.get_state(state_tuple)
    mdp_data = initialize_mdp_data(NUM_STATES)
    errors=[]
    consecutive_no_learning_trials = 0
    while consecutive_no_learning_trials < NO_LEARNING_THRESHOLD:

        action = choose_action(state, mdp_data,GAMMA)
        state_tuple = cart_pole.simulate(action, state_tuple)
        time = time + 1

        new_state = cart_pole.get_state(state_tuple)

        if new_state == NUM_STATES - 1:
            R = -1
        else:
            R = 0

        update_mdp_transition_counts_reward_counts(mdp_data, state, action, new_state, R)

        # Recompute MDP model whenever pole falls
        # Compute the value function V for the new model
        if new_state == NUM_STATES - 1:

            update_mdp_transition_probs_reward(mdp_data)

            converged_in_one_iteration,error = update_mdp_value(mdp_data, TOLERANCE, GAMMA)
            errors.append(error)

            if converged_in_one_iteration:
                consecutive_no_learning_trials = consecutive_no_learning_trials + 1

            else:
                consecutive_no_learning_trials = 0

        # Do NOT change this code: Controls the simulation, and handles the case
        # when the pole fell and the state must be reinitialized.
        if new_state == NUM_STATES - 1:
            num_failures += 1
            if num_failures >= max_failures:
                break
            print('[INFO] Failure number {}'.format(num_failures))
            time_steps_to_failure.append(time - time_at_start_of_current_trial)
            # time_steps_to_failure[num_failures] = time - time_at_start_of_current_trial
            time_at_start_of_current_trial = time
            
            # Reinitialize state
            # x = 0.0
            x = -1.1 + np.random.uniform() * 2.2
            x_dot, theta, theta_dot = 0.0, 0.0, 0.0
            state_tuple = (x, x_dot, theta, theta_dot)
            state = cart_pole.get_state(state_tuple)
        else:
            state = new_state

    if plot:
        # plot the learning curve (time balanced vs. trial)
        log_tstf = np.log(np.array(time_steps_to_failure))
        plt.plot(np.arange(len(time_steps_to_failure)), log_tstf, 'k')
        window = 30
        w = np.array([1/window for _ in range(window)])
        weights = lfilter(w, 1, log_tstf)
        x = np.arange(window//2, len(log_tstf) - window//2)
        plt.plot(x, weights[window:len(log_tstf)], 'r--')
        plt.xlabel('Num failures')
        plt.ylabel('Log of num steps to failure')
        plt.savefig('./control.pdf')
    to_be=weights[window:len(log_tstf)]
    savemat('data.mat', mdp_data)
    savemat('log_num_of_Steps.mat',{'log_failure_steps': to_be})

    plt.figure()
    plt.plot(np.arange(len(errors)),errors)
    plt.savefig('./residual.pdf')
    ## Simulate the cart using open AI gymnasium
    x, x_dot, theta, theta_dot = 0.0, 0.0, 0.0, 0.0
    state_tuple = (x, x_dot, theta, theta_dot)
    state = cart_pole.get_state(state_tuple)
    env_o = gym.make("CartPole-v1", render_mode="rgb_array") #change render_mode to human to have live simulation.\
    obs,_= env_o.reset()
    frames=[]
    filename="cartpole_converged.gif"
    for step in range(300):
        action = choose_action(state, mdp_data, GAMMA)
        state_tuple = cart_pole.simulate(action, state_tuple)
        env_o.unwrapped.state = np.array(state_tuple)
        frame=env_o.render()
        frames.append(frame)
        t_o.sleep(0.02)
        state = cart_pole.get_state(state_tuple)
    imageio.mimsave(filename, frames, fps=30)

    return np.array(time_steps_to_failure)
    
if __name__ == '__main__':
    main()
