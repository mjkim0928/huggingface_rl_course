import numpy as np
import gymnasium as gym
import random
import imageio
import os


import pickle5 as pickle


#Q 테이블 생성 및 초기화
def init_q_table(state_space, action_space):
    Qtable = np.zeros((state_space, action_space))
    return Qtable


#greedy policy 구현
def greedy_policy(Qtable, state):
    action = np.argmax(Qtable[state][:])
    return action

#epsilon greedy policy 구현
def epsilon_greedy_policy(Qtable, state, epsilon, actions):
    #0과 1사이의 무작위 수 하나 뽑기
    random_num = random.uniform(0,1)

    #random_num이 epsilon보다 크면 exploitation(활용)
    if random_num>epsilon:
        action = greedy_policy(Qtable, state)
    #아니면 exploration(탐험)
    else:
        action = random.choice(actions)
    return action

#train 정의;반환값은 훈련된 Qtable
def train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, Qtable, max_steps):
    for episode in range(n_training_episodes):
        #에피소드마다 epsilon decay
        epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode) #max_epsilon이 min_epsilon을 향해 근사되는 의미를 가진 수식
        state, info = env.reset()
        step = 0
        terminated = False
        truncated = False
        actions = np.arange(env.action_space.n - 1)

        for step in range(max_steps):
            
            #inference 시엔 epsilon_greedy_policy로 진행
            action = epsilon_greedy_policy(Qtable, state, epsilon, actions)
            next_state, reward, terminated, truncated, info = env.step(action)
            

            #train 시엔 greed_policy(np.max(Qtable[next_state][:]))로 진행
            Qtable[state][action] += learning_rate*(reward+gamma*np.max(Qtable[next_state][:]) - Qtable[state][action])
            #둘 중 하나일 때 에피소드 종료
            if terminated or truncated:
                break

            state = next_state

    return Qtable
#evaluation 정의;반환값은 mean,std episode reward
def evaluate(env, max_steps, n_eval_episodes, Q, seed):
    episode_rewards = []
    for episode in range(n_eval_episodes):
        #eval_seed가 비어있지 않으면 그 episode의 seed로 env 설정
        if seed:
            state, info = env.reset(seed=seed[episode])
        else:
            state, info = env.reset()
        step = 0
        truncated = False
        terminated = False
        total_reward_episode = 0 #한 episode의 reward들을 모두 더한 값
        
        for step in range(max_steps):
            #evaluate할 때는 inference 시에도 greedy_policy로 진행
            action = greedy_policy(Q, state)
            next_state, reward, terminated, truncated, info = env.step(action)
            total_reward_episode += reward

            if terminated or truncated:
                break
            state = next_state
        episode_rewards.append(total_reward_episode) #에피소드 끝나고 해당 에피소드의 총 reward를 저장
    
    mean_episode_reward = np.mean(episode_rewards) #episode reward의 평균
    std_episode_reward = np.std(episode_rewards) #episode reward의 표준편차
    
    return mean_episode_reward, std_episode_reward


#hyperparameter 정의
n_training_episodes = 10000 #총 훈련할 에피소드 수
learning_rate = 0.7
n_eval_episodes = 100       #평가할 에피소드 수
env_id = "FrozenLake-v1"    #환경의 이름
max_steps = 99              #에피소드 당 최대 타임스텝
gamma = 0.95                #감가율 감마
eval_seed = []              #평가할 때 고정할 환경의 시드
max_epsilon = 1.0           #시작할 때 탐험 확률->최대 입실론
min_epsilon = 0.05          #최소 탐험 확률
decay_rate = 0.0005         #에피소드를 거듭할수록 epsilon을 이만큼 exponential하게 떨어뜨림


if __name__=='__main__':
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="rgb_array")

    #observation space 확인
    print("-----------------OBSERVATION SPACE-----------------")
    print("Observation Space: ", env.observation_space)
    print("Observation Space Shape: ", env.observation_space.n)
    print("Observation Sample: ", env.observation_space.sample())
    #action space 확인
    print("\n-----------------ACTION SPACE-----------------")
    print("Action Space: ", env.action_space)
    print("Action Space Shape: ", env. action_space.n)
    print("Action Sample: ", env.action_space.sample())
    #state와 action space int형식으로 저장
    state_space = env.observation_space.n
    action_space = env.action_space.n
    Qtable_frozenlake = init_q_table(state_space, action_space)
    #train 하기
    Qtable_frozenlake = train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, Qtable_frozenlake, max_steps)

    #train된 Qtable 확인
    print(Qtable_frozenlake)

    #evaluate 하기
    mean, std = evaluate(env, max_steps, n_eval_episodes, Qtable_frozenlake, eval_seed)
    #episode reward의 평균과 표준편차 확인
    print(f"episode reward의 평균: {mean:.2f} +/- {std:.2f}")




