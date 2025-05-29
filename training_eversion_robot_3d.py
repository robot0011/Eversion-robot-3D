import gymnasium as gym

from stable_baselines3 import PPO

from eversion_robot_3d import EversionRobot3D


env = EversionRobot3D(obs_use=True)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=5000)
model.save("ppo_eversion")
print("Done")