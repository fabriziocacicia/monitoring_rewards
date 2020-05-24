from monitoring_rewards.core import Reward


class MonitoringSpecification:
    """
    A 5-tuple Monitoring Rewards specification as defined in:

    De Giacomo, G.; Favorito, M.; Iocchi, L.; Patrizi, F.; Ronca, A.
    2020. Temporal Logic Monitoring Rewards via Transducers
    """

    def __init__(
            self,
            ltlf_formula: str,
            r: Reward,
            c: Reward,
            s: Reward,
            f: Reward
    ):
        """
        :param ltlf_formula: A flloat-compatible LTLf formula
        :param r: (reward) the reward value given to the agent when the formula is temporarily true in the current
        partial trace
        :param c: (cost) the reward value given to the agent when the formula is temporarily false in the current
        partial trace
        :param s: (success) the reward value given to the agent when the formula is permanently true in the current
        partial trace
        :param f: (failure) the reward value given to the agent when the formula is permanently false in the current
        partial trace
        """
        self.ltlf_formula: str = ltlf_formula
        self.r: Reward = r
        self.c: Reward = c
        self.s: Reward = s
        self.f: Reward = f
