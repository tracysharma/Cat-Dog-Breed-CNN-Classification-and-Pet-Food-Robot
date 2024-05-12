from typing import Any, Dict

import numpy as np

from panda_gym.envs.core import PyBulletRobot, Task
from panda_gym.utils import distance

from random import randint, choice


class Slide(Task):
    def __init__(
        self,
        sim,
        reward_type="dense",
        distance_threshold=0.05,
        goal_xy_range=0.3,
        goal_x_offset=0.4,
        obj_xy_range=0.3,
    ) -> None:
        super().__init__(sim)

        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.object_size = 0.15
        self.goal_range_low = np.array([-goal_xy_range / 2 + goal_x_offset, -goal_xy_range / 2, 0])
        self.goal_range_high = np.array([goal_xy_range / 2 + goal_x_offset, goal_xy_range / 2, 0])
        self.obj_range_low = np.array([-obj_xy_range / 2, -obj_xy_range / 2, 0])
        self.obj_range_high = np.array([obj_xy_range / 2, obj_xy_range / 2, 0])
        with self.sim.no_rendering():
            self._create_scene()

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.9, width=1.4, height=0.4, x_offset=-0.1)
        self.sim.create_cylinder(
            body_name="objectCat",
            mass=0.2,
            radius=self.object_size / 2,
            height=self.object_size / 2,
            position=np.array([0.0, 0.2, self.object_size / 2]),
            rgba_color=np.array([0.1, 0.9, 0.1, 1.0]),
            lateral_friction=0.1,
        )
        self.sim.create_cylinder(
            body_name="objectDog",
            mass=0.2,
            radius=self.object_size / 2,
            height=self.object_size / 2,
            position=np.array([0.0, -0.2, self.object_size / 2]),
            rgba_color=np.array([0.9, 0.1, 0.1, 1.0]),
            lateral_friction=0.1,
        )
        self.sim.create_cylinder(
            body_name="target",
            mass=0.0,
            ghost=True,
            radius=self.object_size / 2,
            height=self.object_size / 2,
            position=np.array([0.0, 0.0, self.object_size / 2]),
            rgba_color=np.array([0.9, 0.9, 0.1, 0.3]),
        )

    def get_obs(self) -> np.ndarray:
        objectCat_position = np.array(self.sim.get_base_position("objectCat"))
        objectCat_position_xy = objectCat_position[:2]
        objectDog_position = np.array(self.sim.get_base_position("objectDog"))
        objectDog_position_xy = objectDog_position[:2]

        observation = np.concatenate(
            [
                objectCat_position_xy,
                objectDog_position_xy,
                np.array([self.animal]) # Provided to supply which animal has been chosen
            ]
        )
        return observation

    def get_achieved_goal(self) -> np.ndarray:
        # print("Goal selected: ", self.animal)
        if self.animal == 0:
            # print("Cat")
            good_object_position = np.array(self.sim.get_base_position("objectCat"))
        else:
            # print("Dog")
            good_object_position = np.array(self.sim.get_base_position("objectDog"))
        return good_object_position.copy()

    def reset(self) -> None:
        # THIS SELECTS THE GOAL (0 = cat, 1 = dog)
        self.animal = randint(0, 1) # For training, choose randomly
        # self.animal = 1.0 # DOG
        # self.animal = 0.0 # CAT

        self.goal = self._sample_goal()
        # object_position = self._sample_object()
        self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("objectCat", self._sample_object(0), np.array([0.0, 0.2, 0.0, 1.0]))
        self.sim.set_base_pose("objectDog", self._sample_object(1), np.array([0.0, -0.2, 0.0, 1.0]))

    def _sample_goal(self) -> np.ndarray:
        """Randomize goal."""
        goal = np.array([0.0, 0.0, 0.0])  # z offset for the cube center
        # noise = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        # goal += noise
        return goal.copy()

    def _sample_object(self, animal) -> np.ndarray:
        """Randomize start position of object."""
        if animal == 0:
            object_position = np.array([-0.3, 0.4, self.object_size / 2])
        else:
            object_position = np.array([-0.3, -0.4, self.object_size / 2])
        # noise = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        # object_position += noise
        return object_position

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=bool)

    def compute_reward(self, observation, achieved_goal, desired_goal, info: Dict[str, Any]) -> np.ndarray:
        # get bad object position
        if self.animal == 0:
            bad_object_position = np.array(self.sim.get_base_position("objectDog"))
        else:
            bad_object_position = np.array(self.sim.get_base_position("objectCat"))
        # d = 10*distance(achieved_goal, desired_goal)                      # Heavily reward moving correct can to goal
        # d = d - 3 * distance(bad_object_position, desired_goal)           # Penalize moving the incorrect object towards goal
        # d = d + distance(observation[:2], achieved_goal[:2])              # Reward moving ee close to correct can
        d = distance(observation[:2], achieved_goal[:2])

        return -d.astype(np.float32)
