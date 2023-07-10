import gymnasium as gym
import numpy as np
import cv2
from .games import make_stage, Layer, Layer1TileID, Layer2TileID


def get_user_input(env):
    while True:
        event = env.pygame["pygame"].event.wait()
        if event.type == env.pygame["pygame"].QUIT or (
            event.type == env.pygame["pygame"].KEYDOWN
            and event.key == env.pygame["pygame"].K_ESCAPE
        ):
            env.pygame["pygame"].quit()
            exit()

        if event.type == env.pygame["pygame"].KEYDOWN:
            if event.key == env.pygame["pygame"].K_LEFT:
                return 1
            elif event.key == env.pygame["pygame"].K_RIGHT:
                return 0
            elif event.key == env.pygame["pygame"].K_UP:
                return 3
            elif event.key == env.pygame["pygame"].K_DOWN:
                return 2


class MiniGameEnv(gym.Env):
    def __init__(
        self,
        limit_frame=10000,
        render_mode="human",
        field_size=(8, 6),
        flag_num=5,
        whole_num=5,
        action_size=4,
        capture_path=None,
    ):
        super().__init__()
        self.field_size = field_size
        self.flag_num = flag_num
        self.whole_num = whole_num
        self.limit_frame = limit_frame

        self.start_coord = (0, 0)
        self.current_coord = (0, 0)
        self.goal_coord = (0, 0)

        self.action_size = action_size
        self.action_space = gym.spaces.Discrete(action_size)
        self.rendering = False

        self.observation_size = (2, *self.field_size)
        self.observation_space = gym.spaces.Box(
            low=-1, high=255, shape=self.observation_size, dtype=np.uint8
        )

        self.layers = []
        self.score = 0
        self.prev_score = 0
        self.flags = 0
        self.frame = 0
        self.render_mode = render_mode
        self.pygame = {}
        if self.render_mode == "human":
            self.render_init(capture_path)

    def from_data(self, start_coord, goal_coord, layer_arrays):
        if self.layers == []:
            self.reset()

        self.start_coord = start_coord
        self.current_coord = start_coord
        self.goal_coord = goal_coord

        for i, layer_array in enumerate(layer_arrays):
            self.layers[i].field = layer_array

    def render_init(self, capture_path=None):
        import pygame

        self.pygame["pygame"] = __import__("pygame")
        self.pygame["pygame"].init()
        self.pygame["pygame"].display.set_caption("mini game")
        self.pygame["screen"] = self.pygame["pygame"].display.set_mode(
            (self.field_size[0] * 32, self.field_size[1] * 32 + 32 * 3)
        )
        self.pygame["font"] = self.pygame["pygame"].font.SysFont(
            "TimesNewRoman", 16, bold=True
        )
        self.pygame["display"] = self.pygame["pygame"].display

        if capture_path is not None:
            screen: pygame.Surface = self.pygame["screen"]
            frame_rate = 10.0  # フレームレート
            fmt = cv2.VideoWriter_fourcc("m", "p", "4", "v")  # ファイル形式(ここではmp4)
            self.pygame["writer"] = cv2.VideoWriter(
                capture_path, fmt, frame_rate, screen.get_size()
            )

    def reset(self, *args, **kwargs):
        layers, start_coord, goal_coord = make_stage(
            self.field_size, self.flag_num, self.whole_num
        )
        if self.score != 0:
            self.prev_score = self.score

        self.start_coord = start_coord
        self.current_coord = start_coord
        self.goal_coord = goal_coord
        self.layers = layers
        self.score = 0
        self.flags = 0
        self.frame = 0
        return self.observation, {}

    @property
    def observation(self):
        obs = np.stack([layer.field for layer in self.layers])
        return (obs - obs.min()) / (obs.max() - obs.min())

    def step(self, action_index):
        terminated = False
        truncated = False
        info = {}
        self.frame += 1
        if self.limit_frame < self.frame:
            terminated = True

        next_coord_options = [
            (self.current_coord[0] + 1, self.current_coord[1])
            if self.current_coord[0] < self.field_size[0] - 1
            else self.current_coord,
            (self.current_coord[0] - 1, self.current_coord[1])
            if self.current_coord[0] > 0
            else self.current_coord,
            (self.current_coord[0], self.current_coord[1] + 1)
            if self.current_coord[1] < self.field_size[1] - 1
            else self.current_coord,
            (self.current_coord[0], self.current_coord[1] - 1)
            if self.current_coord[1] > 0
            else self.current_coord,
        ]
        next_coord = next_coord_options[action_index]

        if next_coord == self.current_coord:
            self.score -= 1

        self.current_coord = self.layers[1].move(self.current_coord, next_coord, 0)
        next_tile = Layer1TileID(self.layers[0].field[self.current_coord])

        if next_tile == Layer1TileID.GOAL:
            terminated = True
            self.score += 100

        elif next_tile == Layer1TileID.WHOLE:
            terminated = True

        elif next_tile == Layer1TileID.FLAG:
            self.flags += 1
            self.score += 1

            self.layers[0].field[self.current_coord] = Layer1TileID.NORMAL.value
            self.score += self.flags

        self.score -= self.frame / self.limit_frame
        return self.observation, self.score, terminated, truncated, info

    def render(self, update_interval=100):
        if self.render_mode != "human":
            return
        if self.rendering is False:
            return

        if self.pygame == {}:
            self.render_init()

        pygame = self.pygame["pygame"]
        screen = self.pygame["screen"]
        font = self.pygame["font"]
        display = self.pygame["display"]

        screen.fill((255, 255, 255))

        information_text = [
            font.render(
                f"Frame: {self.frame:4d} / {self.limit_frame}", True, (0, 0, 0)
            ),
            font.render(
                f"Current: {self.current_coord}, {self.layers[0].field[self.current_coord]}",
                True,
                (0, 0, 0),
            ),
            font.render(
                f"Flags: {self.flags:2d} / {self.flag_num:2d}", True, (0, 0, 0)
            ),
            font.render(f"Score: {self.score:.4f}", True, (0, 0, 0)),
            font.render(f"Prev_Score: {self.prev_score:.4f}", True, (0, 0, 0)),
        ]

        for i, text in enumerate(information_text):
            screen.blit(text, (32, self.field_size[1] * 32 + 16 * i + 8))

        for layer in self.layers:
            layer.update(screen)

        if self.pygame.get("writer", None) is not None and self.pygame["writer"].isOpened():
            writer: cv2.VideoWriter = self.pygame["writer"]
            writer.write(
                cv2.cvtColor(
                    pygame.surfarray.array3d(screen).swapaxes(0, 1), cv2.COLOR_RGB2BGR
                )
            )
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                pygame.quit()
                exit()

        if update_interval > 0:
            pygame.time.wait(update_interval)
        display.update()


if __name__ == "__main__":
    env = MiniGameEnv()
    obs, info = env.reset()
    for i in range(10000):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        env.render()

        if terminated or truncated:
            observation, info = env.reset()
