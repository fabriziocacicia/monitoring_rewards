from monitoring_rewards.core import TraceStep
from monitoring_rewards.monitoring_specification import MonitoringSpecification
from monitoring_rewards.reward_monitor import RewardMonitor


def obs_to_trace_step(observation) -> TraceStep:
    position = observation

    if position >= 0.5:
        return {'goal': True}
    else:
        return {'goal': False}


def main():
    lflf_formula = "F goal"

    monitoring_specification = MonitoringSpecification(
        ltlf_formula=lflf_formula,
        r=0,
        c=-1,
        s=0,
        f=0
    )

    reward_monitor = RewardMonitor(
        monitoring_specification=monitoring_specification,
        obs_to_trace_step=obs_to_trace_step
    )

    successful_trace = [0, -1, 0.2, 0, 0.3, 0.6, 0.4]
    total_reward = 0
    print("First attempt")
    for step in successful_trace:
        reward, is_perm = reward_monitor(step)
        print(reward, is_perm)
        total_reward += reward
    print("Total reward", total_reward)

    reward_monitor.reset()

    failing_trace = [0, -1, 0.2, 0, 0.3, 0.1, 0.4]
    total_reward = 0
    print("Second attempt")
    for step in failing_trace:
        reward, is_perm = reward_monitor(step)
        print(reward, is_perm)
        total_reward += reward
    print("Total reward", total_reward)


if __name__ == '__main__':
    main()