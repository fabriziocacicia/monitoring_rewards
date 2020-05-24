from typing import List, Callable

from monitoring_rewards.core import TraceStep, Reward
from monitoring_rewards.monitoring_specification import MonitoringSpecification
from monitoring_rewards.reward_monitor import RewardMonitor


class MultiRewardMonitor:
    def __init__(
            self,
            monitoring_specifications: List[MonitoringSpecification],
            obs_to_trace_step: Callable[[any], TraceStep]
    ):
        self.monitors: List[RewardMonitor] = []

        for specification in monitoring_specifications:
            self.monitors.append(RewardMonitor(specification, obs_to_trace_step))

    def __call__(self, observation) -> (Reward, bool):
        total_reward: Reward = 0
        total_is_perm = False

        for monitor in self.monitors:
            reward, is_perm = monitor(observation)
            total_reward += reward
            total_is_perm = total_is_perm or is_perm
        return total_reward, total_is_perm

    def reset(self):
        """
        Reset the monitor to its initial state.
        """
        for monitor in self.monitors:
            monitor.reset()
