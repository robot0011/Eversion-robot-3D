import gymnasium as gym
from stable_baselines3 import PPO
from eversion_robot_3d import EversionRobot3D

# Create environment
env = EversionRobot3D(obs_use=True)

# Set up PPO with TensorBoard logging
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./ppo_eversion_tensorboard/"
)

# Train the model and log to TensorBoard under the "eversion_run" subdirectory
model.learn(total_timesteps=3000000, tb_log_name="eversion_run")

# Save the model
model.save("ppo_eversion")
print("Done")
