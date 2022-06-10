#  Copyright (c) 2022. Matteo Bettini
#  All rights reserved.

import torch

from maps.simulator.core import Agent, Landmark, Sphere, World
from maps.simulator.scenario import BaseScenario
from maps.simulator.utils import Color


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        num_agents = kwargs.get("n_agents", 3)
        obs_agents = kwargs.get("obs_agents", True)

        self.obs_agents = obs_agents

        world = World(batch_dim=batch_dim, device=device)
        # set any world properties first
        num_landmarks = num_agents
        # Add agents
        for i in range(num_agents):
            agent = Agent(
                name=f"agent {i}",
                collide=True,
                shape=Sphere(radius=0.15),
                color=Color.BLUE,
            )
            world.add_agent(agent)
        # Add landmarks
        for i in range(num_landmarks):
            landmark = Landmark(
                name=f"landmark {i}",
                collide=False,
                color=Color.BLACK,
            )
            world.add_landmark(landmark)

        return world

    def reset_world_at(self, env_index: int = None):
        for agent in self.world.agents:
            agent.set_pos(
                2
                * torch.rand(
                    self.world.dim_p, device=self.world.device, dtype=torch.float32
                )
                - 1,
                batch_index=env_index,
            )

        for landmark in self.world.landmarks:
            landmark.set_pos(
                2
                * torch.rand(
                    self.world.dim_p, device=self.world.device, dtype=torch.float32
                )
                - 1,
                batch_index=env_index,
            )

    def is_collision(self, agent1: Agent, agent2: Agent):
        delta_pos = agent1.state.pos - agent2.state.pos
        dist = delta_pos.square().sum(-1).sqrt()
        dist_min = agent1.shape.radius + agent2.shape.radius
        return dist < dist_min

    def reward(self, agent: Agent):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = torch.zeros(
            self.world.batch_dim, device=self.world.device, dtype=torch.float32
        )
        self._done = torch.full(self.world.batch_dim, True, device=self.world.device)
        for landmark in self.world.landmarks:
            closest = torch.min(
                torch.stack(
                    [
                        (a.state.pos - landmark.state.pos).square().sum(-1).sqrt()
                        for a in self.world.agents
                    ],
                    dim=1,
                ),
                dim=-1,
            )[0]
            goal_reached = closest < landmark.shape.radius
            self._done *= goal_reached
            rew -= closest

        if agent.collide:
            for a in self.world.agents:
                rew[self.is_collision(a, agent)] -= 1
        return rew

    def observation(self, agent: Agent):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in self.world.landmarks:  # world.entities:
            entity_pos.append(entity.state.pos - agent.state.pos)
        # communication of all other agents
        other_pos = []
        for other in self.world.agents:
            if other is agent:
                continue
            other_pos.append(other.state.pos - agent.state.pos)
        return torch.cat(
            [
                agent.state.pos,
                agent.state.vel,
                *entity_pos,
                *(other_pos if self.obs_agents else []),
            ],
            dim=-1,
        )

    def done(self):
        return self._done
