from typing import Callable

from monitoring_rewards.core import TraceStep, Reward
from monitoring_rewards.monitoring_specification import MonitoringSpecification
from monitoring_rewards.reward_transducer import RewardTransducer


class RewardMonitor:
    """
    A component to insert into a RL scenario that takes the observations of the environment,
    interprets it, updates its state and produces a reward signal that is then given to the agent
    """

    def __init__(
            self,
            monitoring_specification: MonitoringSpecification,
            obs_to_trace_step: Callable[[any], TraceStep]
    ):
        """
        :param monitoring_specification: A MonitoringSpecification 5-tuple
        :param obs_to_trace_step: A map from observation to trace step.
        """
        self.reward_transducer = RewardTransducer(
            ltlf_formula=monitoring_specification.ltlf_formula,
            r=monitoring_specification.r,
            c=monitoring_specification.c,
            s=monitoring_specification.s,
            f=monitoring_specification.f,
        )

        self.obs_to_trace_step: Callable[[any], TraceStep] = obs_to_trace_step

    def __call__(self, observation) -> (Reward, bool):
        trace_step = self.map_obs_to_trace_step(observation)

        return self.reward_transducer(trace_step)

    def map_obs_to_trace_step(self, observation) -> TraceStep:
        """
        Maps a given observation coming from an environment to a transducer-compatible trace step
        :param observation: an observation coming from an environment
        :return: a transducer-compatible trace step
        """
        return self.obs_to_trace_step(observation)

    def reset(self):
        """
        Resets the monitor to its initial state.
        """
        self.reward_transducer.reset()
