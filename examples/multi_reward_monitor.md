# Multi Reward Monitor example
`multi_reward_monitor.py`
```python
from monitoring_rewards import TraceStep
from monitoring_rewards import MonitoringSpecification
from monitoring_rewards import MultiRewardMonitor


def obs_to_trace_step(observation) -> TraceStep:
    position = observation
    trace_step = {
        'goal': False,
        'not_neg': False,
    }

    if position >= 0.5:
        trace_step['goal'] = True

    if position < 0:
        trace_step['neg'] = True

    return trace_step

def main():
    lflf_formula_1 = "F goal"

    monitoring_specification_1 = MonitoringSpecification(
        ltlf_formula=lflf_formula_1,
        r=0,
        c=-1,
        s=0,
        f=0
    )

    lflf_formula_2 = "G !neg"
    monitoring_specification_2 = MonitoringSpecification(
        ltlf_formula=lflf_formula_2,
        r=0,
        c=-1,
        s=0,
        f=0
    )

    monitoring_specifications = [monitoring_specification_1, monitoring_specification_2]

    multi_reward_monitor = MultiRewardMonitor(
        monitoring_specifications=monitoring_specifications,
        obs_to_trace_step=obs_to_trace_step
    )

    neg_trace = [0, -1, 0.2, 0, 0.3, 0.6, 0.4]
    total_reward = 0
    print("Negative trace attempt")
    for step in neg_trace:
        reward, is_perm = multi_reward_monitor(step)
        print(reward, is_perm)
        total_reward += reward
    print("Total reward", total_reward)

    multi_reward_monitor.reset()

    failing_trace = [0, 0.1, 0.2, 0, 0.3, 0.1, 0.4]
    total_reward = 0
    print("Goal not reached attempt")
    for step in failing_trace:
        reward, is_perm = multi_reward_monitor(step)
        print(reward, is_perm)
        total_reward += reward
    print("Total reward", total_reward)

    multi_reward_monitor.reset()

    goal_trace = [0, 0.1, 0.2, 0, 0.6, 0.1, 0.4]
    total_reward = 0
    print("Goal reached attempt")
    for step in goal_trace:
        reward, is_perm = multi_reward_monitor(step)
        print(reward, is_perm)
        total_reward += reward
    print("Total reward", total_reward)


if __name__ == '__main__':
    main()
```