import pygame
import pygame.locals as pgl
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, Union
import matplotlib.pyplot as plt
from functools import lru_cache
import random
from enum import Enum

__src__ = Path(__file__).parent
__assets__ = __src__ / "assets"


class Layer1TileID(Enum):
    NORMAL = 0
    FLAG = 1
    WHOLE = 2
    START = 3
    GOAL = 4


class Layer2TileID(Enum):
    EMPTY = -1
    AGENT = 0


class Tiles:
    def __init__(
        self,
        path: Path,
        tile_size: int = 32,
        mode="RGBA",
        animation_length: int = 1,
    ):
        self._path = path
        self.tile_size = tile_size
        self.animation_length = animation_length

        t = np.array(Image.open(path).convert(mode))
        t = np.stack(np.split(t, animation_length, axis=1))
        self.rows, self.cols = t.shape[1:3]
        self.rows, self.cols = self.rows // tile_size, self.cols // tile_size
        t = np.stack(np.split(t, t.shape[1] // tile_size, axis=1), axis=1)
        t = np.stack(np.split(t, t.shape[3] // tile_size, axis=3), axis=1)
        self.tiles = t.transpose(0, 2, 3, 1, 4, 5)

    def __len__(self) -> int:
        return self.rows * self.cols

    def __getitem__(self, pos: Union[tuple[int, int], int]) -> list[Image.Image]:
        # return (animation_length, tile_size, tile_size, channels)
        if isinstance(pos, tuple):
            x, y = pos
        else:
            x, y = pos // self.cols, pos % self.cols

        return [
            Image.fromarray(a.squeeze(0))
            for a in np.split(self.tiles[:, x, :, y, :, :], self.tiles.shape[0], axis=0)
        ]

    @lru_cache(maxsize=1024)
    def as_pygame_surface(
        self, pos: Union[tuple[int, int], int]
    ) -> list[pygame.Surface]:
        return [pygame.image.fromstring(im.tobytes(), im.size, im.mode) for im in self[pos]]  # type: ignore


class Layer:
    def __init__(self, tiles: Tiles, field: Optional[np.ndarray] = None):
        self.tiles: Tiles = tiles
        if field is None:
            field = np.zeros((tiles.rows, tiles.cols), dtype=np.int32)
        self.field: np.ndarray = field

    def update(self, screen: pygame.Surface, animation_step: int = 0):
        for row in range(self.field.shape[0]):
            for col in range(self.field.shape[1]):
                if self.field[row, col] == -1:
                    continue
                screen.blit(
                    self.tiles.as_pygame_surface(self.field[row, col])[
                        animation_step % self.tiles.animation_length
                    ],
                    (row * self.tiles.tile_size, col * self.tiles.tile_size),
                )
        return screen

    def move(self, current_coord=(0, 0), next_coord=(0, 1), target_id=0):
        self.field[current_coord] = -1
        self.field[next_coord] = target_id
        return next_coord


def make_stage(field_size=(16, 12), flag_num=8, whole_num=16, tile_size=32):
    coords = [(i, j) for i in range(field_size[0]) for j in range(field_size[1])]
    random.shuffle(coords)
    start, goal = coords.pop(), coords.pop()
    map_chip_dir = __assets__ / "mapchip"

    tile1 = Tiles(map_chip_dir / "tile1.png", tile_size, animation_length=1)
    tile2 = Tiles(map_chip_dir / "chara_tiles.png", tile_size, animation_length=1)

    data = [
        np.zeros(field_size, dtype=np.int32),
        np.full(field_size, -1, dtype=np.int32),
    ]

    for coord in coords[:flag_num]:
        data[0][coord] = 1
    for coord in coords[flag_num : flag_num + whole_num]:
        data[0][coord] = 2

    data[0][start] = 3
    data[0][goal] = 4
    data[1][start] = 0

    layers = [Layer(tile1, data[0]), Layer(tile2, data[1])]
    return layers, start, goal


def make_game(
    max_step, field_size, flag_num, whole_num, tile_size=32, next_coord_policy_fn=None
):
    layers, start_coord, goal_coord = make_stage(field_size, flag_num, whole_num)
    if next_coord_policy_fn is None:
        next_coord_policy_fn = lambda: random.randint(0, 3)

    pygame.init()
    pygame.display.set_caption("mini game")
    screen = pygame.display.set_mode(
        (field_size[0] * tile_size, field_size[1] * tile_size + 64)
    )
    font = pygame.font.SysFont("TimesNewRoman", 16, bold=True)
    WHITE = (255, 255, 255)
    current_coord: tuple[int, int] = start_coord
    score = 0
    flags = 0
    step = 0

    while True:
        screen.fill(WHITE)
        step_text = font.render(
            f"Step: {step:4d} / {max_step}        Flags: {flags:2d} / {flag_num:2d}        Score: {score:.4f}",
            True,
            (0, 0, 0),
        )
        screen.blit(step_text, (16, 400))

        next_coord_options = [
            (current_coord[0] + 1, current_coord[1])
            if current_coord[0] < field_size[0] - 1
            else current_coord,
            (current_coord[0] - 1, current_coord[1])
            if current_coord[0] > 0
            else current_coord,
            (current_coord[0], current_coord[1] + 1)
            if current_coord[1] < field_size[1] - 1
            else current_coord,
            (current_coord[0], current_coord[1] - 1)
            if current_coord[1] > 0
            else current_coord,
        ]

        next_coord = current_coord

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pgl.KEYDOWN and event.key == pgl.K_ESCAPE
            ):
                pygame.quit()
                exit()
            if next_coord_policy_fn == "human":
                if event.type == pgl.KEYDOWN:
                    if event.key == pgl.K_LEFT:
                        next_coord = next_coord_options[1]
                    elif event.key == pgl.K_RIGHT:
                        next_coord = next_coord_options[0]
                    elif event.key == pgl.K_UP:
                        next_coord = next_coord_options[3]
                    elif event.key == pgl.K_DOWN:
                        next_coord = next_coord_options[2]
                    else:
                        continue
                    step += 1
            else:
                next_coord = next_coord_options[next_coord_policy_fn()]
                step += 1

        current_coord = layers[1].move(current_coord, next_coord, 0)

        info_text = font.render(
            f"Start: {start_coord} => Current: {current_coord} => Goal: {goal_coord}",
            True,
            (0, 0, 0),
        )
        screen.blit(info_text, (16, 416))

        for layer in layers:
            layer.update(screen)

        if current_coord == goal_coord:
            pygame.quit()
            return score

        elif layers[0].field[current_coord] == 2:
            score -= 10
            pygame.quit()
            return score

        elif layers[0].field[current_coord] == 1:
            flags += 1
            layers[0].field[current_coord] = 0

        score = flags / ((step + 1) ** 0.5)

        pygame.display.update()


if __name__ == "__main__":
    make_game(1000, (16, 12), 8, 16, next_coord_policy_fn="human")
    # print(make_stage((16, 12), 8, 16))
