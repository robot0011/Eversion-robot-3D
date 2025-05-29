from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from eversion_robot_3d import EversionRobot3D

# 1. Create and wrap the environment
env = DummyVecEnv([lambda: EversionRobot3D(obs_use=True)])

# 3. Test the model
obs = env.reset()
real_env = env.envs[0]  # Access the underlying environment
model = PPO.load("ppo_eversion", env=env)  #
for _ in range(200):
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    real_env.render()  # Render the environment
    # Access custom attributes from the real environment
    print(f"State: {real_env.state}, Action: {real_env.act}, Reward: {reward}")
    
    if done:
        obs = env.reset()