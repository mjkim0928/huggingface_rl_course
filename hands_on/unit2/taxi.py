import numpy as np
import gymnasium as gym
import random
import imageio
import os


import pickle5 as pickle
from frozenlake_non_slippery import init_q_table, train, evaluate #frozenlake 파일에서 구현한 함수들 가져와 사용
#from a import b: a.py 전체가 실행된다.


#hyperparameter 정의(frozenlake와 사용하는 hyperparameter동일)
n_training_episodes = 25000   
learning_rate = 0.7         
n_eval_episodes = 100      
# DO NOT MODIFY EVAL_SEED
# eval_seed는 evaluate할 에피소드 100개에 대해서 택시 시작 위치를 지정해 모든 코드 실행자의 에이전트가 동일한 택시 시작위치에서 
eval_seed = [16,54,165,177,191,191,120,80,149,178,48,38,6,125,174,73,50,172,100,148,146,6,25,40,68,148,49,167,9,97,164,176,61,7,54,55,
 161,131,184,51,170,12,120,113,95,126,51,98,36,135,54,82,45,95,89,59,95,124,9,113,58,85,51,134,121,169,105,21,30,11,50,65,12,43,82,145,152,97,106,55,31,85,38,
 112,102,168,123,97,21,83,158,26,80,63,5,81,32,11,28,148] # Evaluation seed, this ensures that all classmates agents are trained on the same taxi starting position
                                                          # Each seed has a specific starting state

env_id = "Taxi-v3"     
max_steps = 99            
gamma = 0.95               
max_epsilon = 1.0         
min_epsilon = 0.05          
decay_rate = 0.005         

if __name__=='__main__':
    env = gym.make("Taxi-v3", render_mode="rgb_array")

    #observation space 확인
    state_space = env.observation_space.n
    print("observation space 수: ", state_space)

    #action space 확인
    action_space = env.action_space.n
    print("action space의 수", action_space)

    #Qtable 생성 및 초기화
    Qtable_taxi = init_q_table(state_space, action_space)

    #초기 Qtable과 shape 확인
    print(Qtable_taxi)
    print(Qtable_taxi.shape)
    #train 하기
    Qtable_taxi = train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, Qtable_taxi, max_steps)

    #train된 Qtable 확인
    print(Qtable_taxi)

    #evaluate 하기
    mean, std = evaluate(env, max_steps, n_eval_episodes, Qtable_taxi, eval_seed)
    #episode reward의 평균과 표준편차 확인
    print(f"episode reward의 평균: {mean:.2f} +/- {std:.2f}")