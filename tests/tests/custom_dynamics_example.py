import numpy as np
import torch
import logging
import math
import pytorch_mppi as mppi

logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


class ToyEnv:
    def __init__(self, dt=0.05, render_mode=None):
        self.dt = dt
        self.render_mode = render_mode
        self.unwrapped = self
        self.state = None

    def reset(self):
        # state: [x, y, vx, vy]
        self.state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return self.state

    def step(self, action):
        # action: [ax, ay]
        ax, ay = float(action[0]), float(action[1])

        # clamp action
        ax = max(min(ax, 2.0), -2.0)
        ay = max(min(ay, 2.0), -2.0)

        x, y, vx, vy = self.state
        vx = vx + ax * self.dt
        vy = vy + ay * self.dt
        # optional velocity clamp
        vx = max(min(vx, 5.0), -5.0)
        vy = max(min(vy, 5.0), -5.0)
        x = x + vx * self.dt
        y = y + vy * self.dt

        self.state = np.array([x, y, vx, vy], dtype=np.float64)

        # goal at (5, 5); reward = -running_cost
        goal = np.array([5.0, 5.0], dtype=np.float64)
        pos_cost = np.sum((self.state[:2] - goal) ** 2)
        vel_cost = np.sum(self.state[2:] ** 2)
        act_cost = ax ** 2 + ay ** 2
        cost = pos_cost + 0.1 * vel_cost + 0.05 * act_cost
        reward = -float(cost)

        return self.state, reward

    def render(self):
        if self.render_mode == "human":
            logger.info("state: %s", self.state)


def make_controller(device, dtype):
    # dimensions
    nx = 4
    nu = 2

    # bounds
    u_min = torch.tensor([-2.0, -2.0], dtype=dtype, device=device)
    u_max = torch.tensor([2.0, 2.0], dtype=dtype, device=device)

    dt = 0.05
    goal = torch.tensor([5.0, 5.0], dtype=dtype, device=device)

    def dynamics(state, perturbed_action):
        u = torch.max(torch.min(perturbed_action, u_max), u_min)
        if state.dim() == 1:
            state = state.view(1, -1)
        if u.dim() == 1:
            u = u.view(1, -1)

        x = state[:, 0]
        y = state[:, 1]
        vx = state[:, 2]
        vy = state[:, 3]

        ax = u[:, 0]
        ay = u[:, 1]

        vx_next = torch.clamp(vx + ax * dt, -5.0, 5.0)
        vy_next = torch.clamp(vy + ay * dt, -5.0, 5.0)
        x_next = x + vx_next * dt
        y_next = y + vy_next * dt

        next_state = torch.stack((x_next, y_next, vx_next, vy_next), dim=1)
        return next_state

    def running_cost(state, action):
        pos = state[:, 0:2]
        vel = state[:, 2:4]
        pos_cost = torch.sum((pos - goal) ** 2, dim=1)
        vel_cost = torch.sum(vel ** 2, dim=1)
        act_cost = torch.sum(action ** 2, dim=1)
        return pos_cost + 0.1 * vel_cost + 0.05 * act_cost

    noise_sigma = torch.diag(torch.tensor([1.0, 1.0], dtype=dtype, device=device))
    lambda_ = 1.0
    num_samples = 500
    horizon = 20

    ctrl = mppi.MPPI(
        dynamics,
        running_cost,
        nx,
        noise_sigma,
        num_samples=num_samples,
        horizon=horizon,
        lambda_=lambda_,
        device=device,
        u_min=u_min,
        u_max=u_max,
    )

    return ctrl


def do_nothing(_):
    pass


if __name__ == "__main__":
    d = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.double

    env = ToyEnv(render_mode=None)
    env.reset()
    ctrl = make_controller(d, dtype)

    total_reward, data = mppi.run_mppi(ctrl, env, do_nothing, retrain_after_iter=50, iter=300, render=False)
    logger.info("Total reward %f", total_reward)