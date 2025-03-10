import typing
from typing import Callable, Dict, List

import torch
from torch import Tensor

from vmas import render_interactively
from vmas.simulator.core import Agent, Entity, Landmark, Sphere, World
from vmas.simulator.dynamics.holonomic import Holonomic
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar
from vmas.simulator.utils import Color, ScenarioUtils, X, Y

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom
from vmas.scenarios.navigation import Scenario as NavigationScenario

class Scenario(NavigationScenario):

    def observation(self, agent: Agent):
        goal_poses = []
        if self.observe_all_goals:
            for a in self.world.agents:
                goal_poses.append(agent.state.pos - a.goal.state.pos)
        else:
            goal_poses.append(agent.state.pos - agent.goal.state.pos)
        obs = {
            "obs": torch.cat(
                goal_poses  # Relative position to goal (fundamental)
                # + [
                #     agent.state.pos - obstacle.state.pos for obstacle in self.obstacles
                # ]  # Relative position to obstacles (fundamental)
                + [
                    agent.sensors[0]._max_range - agent.sensors[0].measure()
                ] if self.collisions else [],  # LIDAR to avoid other agents
                dim=-1,
                ),
            "pos": agent.state.pos,
            "vel": agent.state.vel,
        }
        if not isinstance(agent.dynamics, Holonomic):
            # Non hoonomic agents need to know angular states
            obs.update(
                {
                    "rot": agent.state.rot,
                    "ang_vel": agent.state.ang_vel,
                }
            )
        return obs