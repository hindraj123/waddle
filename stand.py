import gymnasium as gym
from sb3_contrib import TRPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy

def main():
    # 1. Create the environment
    env_id = "Humanoid-v4"
    
    # We wrap it in a DummyVecEnv to allow for normalization
    env = DummyVecEnv([lambda: gym.make(env_id)])
    
    # 2. Normalize observations and rewards (CRITICAL for Humanoid)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # 3. Initialize TRPO
    # TRPO is stable but sample-inefficient compared to PPO. 
    # The default hyperparameters in sb3-contrib are a good starting point.
    print("Initializing TRPO Model...")
    model = TRPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=1e-3,
        n_steps=2048,
        batch_size=128,
        gamma=0.99,
        cg_max_steps=15,
        tensorboard_log="./trpo_humanoid_tensorboard/"
    )

    # 4. Train the Agent
    # Note: 1,000,000 timesteps is a bare minimum for Humanoid. 
    # For a polished walk, you may need 5M to 10M timesteps.
    training_steps = 1_000_000
    print(f"Training for {training_steps} timesteps...")
    model.learn(total_timesteps=training_steps, progress_bar=True)

    # Save the model and the normalization statistics
    model.save("trpo_humanoid")
    env.save("vec_normalize.pkl")
    print("Model saved!")

    # 5. Evaluate the trained policy
    print("Evaluating policy...")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")

    # 6. Render the trained agent
    print("Rendering the trained agent...")
    
    # Create a new environment for rendering
    render_env = gym.make(env_id, render_mode="human")
    render_env = DummyVecEnv([lambda: render_env])
    
    # Load the normalization stats (do not update them during testing)
    render_env = VecNormalize.load("vec_normalize.pkl", render_env)
    render_env.training = False
    render_env.norm_reward = False

    obs = render_env.reset()
    for _ in range(1000):
        # Predict the action using the trained model
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = render_env.step(action)
        render_env.render()
        
        if dones:
            obs = render_env.reset()

    render_env.close()

if __name__ == "__main__":
    main()