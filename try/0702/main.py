from src.enviroments import MiniGameEnv, get_user_input
import gymnasium as gym


if __name__ == "__main__":
    env = MiniGameEnv(
        field_size=(10, 10),
        flag_num=10,
        whole_num=10,
        limit_frame=1000,
        # capture_path="learning.mp4",
    )

    env.rendering = True

    for i in range(100):
        obs, _ = env.reset()
        while True:
            env.render(update_interval=10)
            action = env.action_space.sample()
            obs, rwd, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
