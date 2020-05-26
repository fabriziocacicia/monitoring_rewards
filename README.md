# Monitoring Rewards via Transducers

An implementation of Monitoring Rewards based on the paper
[Temporal Logic Monitoring Rewards via Transducers](http://www.diag.uniroma1.it/degiacom/papers/2020draft/kr2020dfipr.pdf)

It is developed to be easily integrated with [OpenAI Gym](https://gym.openai.com/) environments.

## Install 
You need to install the python [flloat](https://github.com/whitemech/flloat) library in order to use this library.

* First install flloat
    ```shell script
    pip install flloat
    ```
* Then clone this repo
    ```shell script
    git clone htts://github.com/fabriziocacicia/monitoring_rewards.git
    ```
## Overview
### Monitoring Specification
A Monitoring Specification is composed by a LTLf formula and 4 numeric reward values.

```python
MonitoringSpecification(ltlf_formula, r, c, s, f)
```
### Monitors
A monitor takes observations from an environment and gives rewards to an agent acting on it.
#### Reward Monitor
A Reward Monitor is built from a single Monitoring Specification and a mapping from environment's observations to transducer's 
traces.

It takes a single environment's observation and gives a reward according to its specification together with a flag that
tells if the underlying automata went to an inescapable state.
```python
reward_monitor = RewardMonitor(
    monitoring_specification,
    obs_to_trace_step
)

# ...

reward, is_perm = reward_monitor(observation)
```

#### Multi Reward Monitor
A Multi Reward Monitor is built from a list of Monitoring Specifications and a mapping from environment's observations 
to transducer's traces and gives rewards.

It takes a single environment's observation and gives as reward the sum of the rewards of the specifications from which
is made  together with a flag that tells if the underlying automata went to an inescapable state.
```python
multi_monitor = MultiRewardMonitor(
    monitoring_specifications=[
        monitoring_specification1,
        monitoring_specification2,
        #...
        monitoring_specificationN
    ],
    obs_to_trace_step=obs_to_trace
)

# ...

reward, is_perm = multi_monitor(observation)
```
  
## How to use
This library provides either single or multiple monitoring reward specifications.

### Single Monitoring Reward Specification
Let's consider as example the Mountain Car environment. 

As stated in the paper:
>The state space is the set of
pair <position, velocity>. A reward of −1 is given at each
timestep. The goal state is when position ≥ 0.5. We model
the reward function with a monitoring temporal specification (♦goal, 0, −1, 0, 0), where goal is a fluent that is true
when (position ≥ 0.5), false otherwise.

```python
# We define the LTLf formula
ltlf_formula="F goal" # ♦goal

# We define the Monitoring Specification
monitoring_specification = MonitoringSpecification(
    ltlf_formula=ltlf_formula,
    r=0,
    c=-1,
    s=0,
    f=0
)

# We define a function that map environment's observations into trace steps
def obs_to_trace_step(observation) -> TraceStep:
    position, velocity = observation

    if position >= 0.5:
        return {'goal': True}
    else:
        return {'goal': False}

# We create a Reward Monitor
reward_monitor = RewardMonitor(
    monitoring_specification=monitoring_specification,
    obs_to_trace_step=obs_to_trace_step
)

# Import and initialize Mountain Car Environment
env = gym.make('MountainCar-v0')
env.reset()

while not done:
    action = agent.act()

    observation, _, done, info = env.step(action) 

    reward, is_perm = reward_monitor(observation)

    # To stop the episode earlier
    done = done | is_perm
```

### Multiple Monitoring Reward Specification
Let's consider as example the Taxi domain. 

As stated in the paper:
>There are 4 locations and the goal is to pick up the passenger at
one location (the taxi itself is a possible location) and drop
him off in another. The reward is +20 points for a successful drop-off, and -1 point for every timestep it takes. There
is also a -10 reward signal for illegal pick-up and drop-off
actions. The goal state is when the passenger is dropped
off at the right place. We can model the Taxi problem as a
sequence task: (♦(p ∧ ♦q), 0, −1, +20, 0), where p means
“pick up the passenger” and q means “drop-off the passenger to the right location”. The bad action penalty is another temporal specification (♦bad action, −10, 0, 0, 0). 

```python
# We define the LTLf formulas
ltlf_formula_1="F(p & F(q))" # ♦(p ∧ ♦q)
ltlf_formula_2="F(bad_action)" # ♦bad_action

# We define the Monitoring Specifications
monitoring_specification_1 = MonitoringSpecification(
    ltlf_formula=ltlf_formula_1,
    r=0,
    c=-1,
    s=20,
    f=0
)

monitoring_specification_2 = MonitoringSpecification(
    ltlf_formula=ltlf_formula_2,
    r=-10,
    c=0,
    s=0,
    f=0
)

monitoring_specifications = [monitoring_specification_1, monitoring_specification_2]

# We define a function that map environment's observations into trace steps
def obs_to_trace_step(observation) -> TraceStep:
    # ...

# We create a Multi Reward Monitor
multi_monitor = MultiRewardMonitor(
    monitoring_specifications=monitoring_specifications,
    obs_to_trace_step=obs_to_trace_step
)

# Import and initialize Taxi domain Environment
env = gym.make('Taxi-v2')
env.reset()

while not done:
    action = agent.act()

    observation, _, done, info = env.step(action) 

    reward, is_perm = multi_monitor(observation)
    
    # To stop the episode earlier
    done = done | is_perm

```

### Train on multiple episodes
When you train an agent on multiple episodes you will reset the environment every time an episode end.

You can do the same with a Monitor:
```python
for episode in range(num_episodes):
    while not done:
        action = agent.act()
    
        observation, _, done, info = env.step(action) 
    
        reward, is_perm = monitor(observation)
        
        # To stop the episode earlier
        done = done | is_perm
    
    env.reset()
    monitor.reset()
```