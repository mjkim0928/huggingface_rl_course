import gymnasium as gym
from huggingface_sb3 import load_from_hub, package_to_hub
from huggingface_hub import notebook_login 

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

#gymnasium 환경 생성
env = gym.make("LunarLander-v2")
env.reset()
#observation space 확인
print("\n=========OBSERVATION SPACE============")
print("Observation Space Shape:", env.observation_space.shape)
print("Sample observation", env.observation_space.sample())
#action space 확인
print("\n=========ACTION SPACE============")
print("Observation Space Shape:", env.action_space.n)
print("Sample observation", env.action_space.sample())

#환경 벡터화: 하나의 환경에 여러 독립 환경 스태킹
env = make_vec_env('LunarLander-v2', n_envs=16)

#심층강화학습(DRL) PPO 모델 생성
model = PPO(
    policy = 'MlpPolicy',
    env = env,
    n_steps = 1024,
    batch_size = 64,
    n_epochs = 4,
    gamma = 0.999,
    gae_lambda = 0.98,
    ent_coef = 0.01,
    verbose=1)

#총(maximum) 타임스텝 100만번으로 학습
model.learn(total_timesteps=1000000)
print("모델 학습 완료!\n")
#학습된 모델 저장
model_name="ppo-LunarLander-v2-hands_on"
model.save(model_name)
print("모델 저장 완료!\n")

#모델 평가
eval_env = Monitor(gym.make("LunarLander-v2", render_mode='rgb_array'))
#에피소드 10회분 가져와 에피소드 리워드 평균 및 표준편차 구해 모델 평가 진행
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

#모델 인퍼런스하여 lunarlander가 두 발이 땅에 닿았을 때 보상을 얼마나 받는지 확인

#학습된 모델에서 parameter가져옴
param_dict=model.get_parameters()
print("학습된 모델에서 parameter가져옴")
#인퍼런스할 새 모델 생성
new_model = PPO("MlpPolicy", env, verbose=1)
#새 모델에 학습된 모델 parameter를 설정
new_model.set_parameters(param_dict)
print("새 모델에 학습된 모델 parameter를 설정")
print("새 모델 인퍼런스 시작")
obs = env.reset()
flag=False
for i in range(1000):
    action, _states = new_model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    for reward in rewards:
        if reward==100:
            print("두 발이 땅에 닿았다! 그때의 보상은 ", reward)
            flag=True
            break
    if flag:
        break

