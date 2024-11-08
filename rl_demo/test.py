from stable_baselines3 import PPO
from env import RK4Env
from config import config
import numpy as np

# Load the trained model
model_path = "policy/mpm7zixo/model.zip"  # Replace with your model's path
model = PPO.load(model_path, RK4Env(1, config=config, test=True, gif=True))

# Create the environment for testing



# Perform a single test run
obs = model.env.reset()
done = False
total_reward = 0
step = 0

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = model.env.step(action)
    total_reward += reward
    
    
    # Log step information to wandb
    
    step += 1


print(f"Test run completed. Total reward: {total_reward}, Steps: {step}")
