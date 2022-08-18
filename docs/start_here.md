
# Quickstart

A2RL is a tool that recommends the best action to take when making a decision. It uses state of
the art technologies to make this happen.

## Concepts

The workflow is straightforward and simple.

```{mermaid}
flowchart LR

    subgraph Learning -Step 1
    A[Offline data] -->|Label context, actions <div></div> and rewards| B{Simulator}
    end
    B <-.->|Whatif API| C(Planner)

    subgraph Planning -Step 2
    D[New question to ask] -->|New context| C
    C -->|Action 1| E(Result 1)
    C -->|Action 2| F(Result 2)
    C -->|Action 3| G(Result 3)
    end
```

The rest of this document shows the end-to-end workflow in a nutshell, then dive deep into the data inspection, building a simulator and getting recommendation.

## Usage in a Nutshell

A typical [Reinforcement Learning](https://en.wikipedia.org/wiki/Reinforcement_learning) approach
requires you to first train an RL agent (e.g.,
[SAC](https://paperswithcode.com/method/soft-actor-critic),
[PPO](https://paperswithcode.com/method/ppo)) with a simulator, then only the agent can recommend an
action.

The way `whatif` simulator provides recommendations is different. First, it calculates for you a
[Q-value](https://en.wikipedia.org/wiki/Reinforcement_learning#Value_function) internally when you
load your data. Then the Simulator is trained with tokenized dataframe's states,
actions, rewards, and Q-value in row major order flatten sequences. In order to choose an action, you just need to sample multiple one step trajectory based on the current context, and pick the best action (i.e. with best Q-value).

Here's what you need to do in three steps:

1. Load data and perform inspection.

    Below example shows how to load a built-in dataset. You can choose to import your own dataset as
    well --- refer to this [example on creating a new dataset](auto-notebooks/create_dataset).

    ```python
    import a2rl as wi
    from a2rl.utils import plot_information
    # Load data
    wi_df = wi.read_csv_dataset(wi.sample_dataset_path("chiller"))
    wi_df.add_value()
    # Reduce data size for demo purpose
    wi_df = wi_df.iloc[:1000]
    # Show data properties
    plot_information(wi_df)

    ```

2. Step 2 - Train a simulator.

    ```python
    # Setup tokenizer
    tokenizer = wi.AutoTokenizer(wi_df, block_size_row=2)
    # Setup trainer and indicate where to save the model, and tokenizer to be used
    model_dir = "my-model-dir/"
    builder = wi.GPTBuilder(tokenizer, model_dir)
    # Start training
    model = builder.fit()
    ```

3. Step 3 - Get a recommendation.

    ```python
    # Setup simulator with tokenizer and trained model
    simulator = wi.Simulator(tokenizer, model, max_steps=100, reset_coldstart=2)

    # Select some context in the form of (s,a,r,v,...s) from tokenized dataframe.
    # Context must end with states.
    custom_context = simulator.tokenizer.df_tokenized.sequence[:7]

    # Get recommendation
    recommendation_df = simulator.sample(custom_context, 3)
    recommendation_df
    ```

You can get more information for each step in the following notebooks.

- [Data Properties](auto-notebooks/data_properties)
- [Simulator](auto-notebooks/simulator)

## Data Properties

For many sequential decision making problems, we look for some key patterns in the data. More
precisely, the data should exhibit the [MDP (Markov Decision
Process)](https://en.wikipedia.org/wiki/Markov_decision_process) property for
[offline RL](https://paperswithcode.com/paper/offline-reinforcement-learning-tutorial) techniques to
be effective.

```{mermaid}
flowchart LR
    A[S<sub>t-1</sub>] -.-> B{A<sub>t-1</sub>}
    C[S<sub>t</sub>] -.-> D{A<sub> t</sub>}
    E[S<sub>t+1</sub>] -.-> F{A<sub>t+1</sub>}
    A --> G((r<sub>t-1</sub>))
    B --> G
    C --> H((r<sub> t</sub>))
    D --> H
    E --> J[S<sub>t+2</sub>]
    F --> J
    E --> I((r<sub>t+1</sub>))
    F --> I
    A --> C
    B --> C
    C --> E
    D --> E
```

`whatif` offers convenience functions to rapidly qualify the MDP-ness of your data. With these, you
can rapidly assess your data to make an informed decision whether you're ready to proceed with
offline RL, or whether you should first look at how to improve your data collection process.

```python
# Run all the tests with different lags in time
normalized_markovian_matrix(df: wi.WiDataFrame) -> pd.DataFrame

# Return the entropy of a sequence
entropy(Y: np.ndarray) -> float

# Return the entropy of a column of a series conditioned on another series
conditional_entropy(Y: np.ndarray, X: np.ndarray) -> float

# Return a True/False flag to say if the knowing the conditioning series reduces the entropy
better_than_random(Y: np.ndarray, X: np.ndarray) -> bool

# Test if there is a reward function
reward_function(df: wi.WiDataFrame, lag: int, mask: bool = False) -> float

# Test if there is a Markov property
is_markovian(df: wi.WiDataFrame, lag: int, mask: bool = False) -> float
```

Combine the test functions with the following rubrics to qualify your data.

### Can we predict the future by knowing the past?

There should be a real benefit to us in knowing what happened in the past. Data from a real process
(e.g., temperature changing in a heated room) should be quite different to something that is random
(e.g., rolling 4 dice).

Mathematically speaking, this is the [Markov
Property](https://en.wikipedia.org/wiki/Markov_property). This reads as probability of the future
state conditioned on the previous state and actions.

 $$P \left( S^{t+1} \, \middle| \, S^{t},A^{t}\right) = P \left( S^{t+1} \, \middle| \, S^{t},A^{t}, S^{t-1},A^{t-1}...\right)$$

Rather than proving this directly, which is actually quite difficult, `whatif` tests for
[information gain](https://en.wikipedia.org/wiki/Entropy_(information_theory)#Use_in_machine_learning)
between different measurements which is a lot easier to do.

 $$I \left( S^{t+1} \, \middle| \, S^{t},A^{t}\right) = H(S^{t+1}) - H \left( S^{t+1} \, \middle| \, S^{t},A^{t}\right)$$

We maginalise by shuffling the data

$$H(S^{t+1}) ≈ H  \left( S^{t+1} \, \middle| \, rand(S^{t}),A^{t}\right)$$

That leaves us with this True/False test that tells us the utility of predicting something by knowing something else.

$$H \left( S^{t+1} \, \middle| \, S^{t},A^{t}\right) < H  \left( S^{t+1} \, \middle| \, rand(S^{t}),A^{t}\right)$$

We can also add lags to the data we are conditioning on to see how far back past measurements and actions influence the system (mathematically known as the order of the [Markov Process](https://en.wikipedia.org/wiki/Markov_chain)).

### Can we distinguish between good and bad actions?

We need to assign credit to good actions in different situations. E.g., choosing the right ads to
show a user signed in from a certain location and time.

Rewards are captured in historic data and we can test if there is any information gained between the reward and the previous states and actions.

 $$I \left( r^{t+1} \, \middle| \, S^{t},A^{t}\right) = H  \left( r^{t+1} \, \middle| \, S^{t},rand(A^{t})\right) - H \left( r^{t+1} \, \middle| \, S^{t},A^{t}\right)$$

We can also add lags to the previous states and actions to see how delayed the reward could be.

These tests can determine if we can actually optimise our process.

### Can we influence our states through our actions?

We need to see if we can influence our states with our actions. Ideally we would want to push
ourselves to the best state possible given that our data has the Markov property.

 $$I \left( S^{t+1} \, \middle| \,A^{t}\right) = H  \left( S^{t+1} \, \middle| \, S^{t},rand(A^{t})\right) - H \left( r^{t+1} \, \middle| \, S^{t},A^{t}\right)$$

This is important to make sure that our states are influencable and not exogenous.

## Global Simulator

`whatif` puts state of the art Generative Pre-trained Transformer (GPT) networks to work to predict outcomes for sequential tasks.

Take multiple timeseries and label them are either states, actions, or rewards.

```{mermaid}
flowchart LR
    A[States] -->|Information <div></div> measurements| C{Decision}
    C -->|Action 1| D((Reward 1))
    C -->|Action 2| E((Reward 2))
    C -->|Action 3| F((Reward 3))
```

### States

These measurement or information that are used to inform a decision. The column names of the states need to be labelled in the YAML file.

You can have any number of $N$ state dimentions.

### Actions

These are outcomes of a decision or policy. The column names of the actions need to be labelled in the YAML file.

You can have any number of $M$ action dimentions.

### Rewards

These are a measurement of how well you are doing. The column name for the reward needs be labelled in the YAML file.

The reward comes from a single column.

### Expected Returns or Value

A estimation of the value (or expectation of all future returns) is added to the `whatif` data frame
after the data is loaded. The [Bellman equations](https://en.wikipedia.org/wiki/Bellman_equation)
are used to estimate this. `whatif` offers either
[SARSA](https://en.wikipedia.org/wiki/State-action-reward-state-action) which is a conservative on
policy algorithm, or [Q-Learning](https://en.wikipedia.org/wiki/Q-learning) which is a more
aggresive off policy value estimate.

$ V(s) = \sum\limits_{a} π(a|s) \sum\limits_{s^{t+1},r} π[s^{t+1},r|^{t},a](r + γV(s^{t+1}))$

Here the Q value function is updated iteratively. This is the Q-learning algorithm:

$ Q^{t+1}(s,a) = Q^{t}(s,a) + α[r + max_{a^{next}} γQ^{t}(s^{next},a^{next})  - Q^{t}(s,a)] $

This is SARSA:

$ Q^{t+1}(s,a) = Q^{t}(s,a) + α[r +  γQ^{t}(s^{next},a^{next})  - Q^{t}(s,a)] $

### Everything becomes a Token

Data needs be transformed into tokens to work with GPT. Continuous data needs to be converted to quantiles using a quantile discretizer. Categorical data is converted to tokens.

Each column being used is given a unique token range so that there are no collisions. `whatif`
enforces that the next token be picked from the range that it is expecting.

```{mermaid}
flowchart LR
    A[S<sub>t-1</sub>x N tokens] -.-> B{A<sub>t-1</sub> x M tokens}
    B -.-> C((r<sub>t-1</sub>))
    C -.-> D((V<sub>t-1</sub>))
    D -.-> E[S<sub>t</sub>x N tokens]
    E -.-> F{A<sub>t</sub> x M tokens}
    F -.-> ...
```

## Asking Counterfactuals or What if

A what-if analysis starts with deciding a *context* which could just be the current state or the
current state with a history of previous states, actions, rewards and returns.

```{mermaid}
flowchart LR
    A[Context] -->|S<sub>t-1</sub>,A<sub>t-1</sub>,r<sub>t-1</sub>,R<sub>t-1</sub>S<sub>t</sub>| C{Simulator}
    C -.-> |A<sup>option 1</sup>| Z[r,V]
    subgraph Recommend Action
    Z -->  D[highest V]
    end
    C -->|A<sup>option 2</sup>| E[r,V]
    C -->|A<sup>option 3</sup>| F[r,V]
    C -->|A<sup>option 4</sup>| G[r,V]
```

The context is passed to the simulator is used to generate many outcomes that have been taken in the
past in similar context. The outcome with the highest reward-to-go is recommended as the best action
to take.

Please refer to the [simulator example](auto-notebooks/simulator) to learn more about this.

## Problem Formulation

Here is how `whatif` thinks about making decisions. You will get the hang of it.

### Example 1: Traveling

Lets say that we wanted to make the shortest trip from Singapore to Kuala Lumpur.

You have the choice to fly, drive or take a train.

```{mermaid}
flowchart LR
    A[Information] -->|Get travel time| C{Decision}
    C -->|Train| D(Time taken-hours 7.5)
    C -->|Plane| E(Time taken-hours 4.3)
    C -->|Car| F(Time taken-hours 5.5)
```

Let us further assume that some historic data has been collected on the weekday and on the weekend.
Take a look at the following tables.

Weekday travel (note: `o` denotes the median time):

| Transport | Travel Time        |
| --------- | ------------------ |
| Train     | `-------\|--o--\|` |
| Car       | `----\|---o---\|`  |
| Plane     | `--\|--o--\|`      |

Weekend travel (note: `o` denotes the median time):

| Transport | Travel Time       |
| --------- | ----------------- |
| Train     | `------\|--o--\|` |
| Car       | `-\|--o--\|`      |
| Plane     | `------\|--o--\|` |

These data shows us that if you travelled on the weekday then the plane is the quickest. If you
travelled on the weekend then driving is quicker.

**So, what could be the travelling problem formulation? To `whatif` this problem can be represented like this.**

The thing that measures how well we have done is called the reward, which could *time* in this
example.

```{mermaid}
flowchart LR
    D(Time taken-hours 7.5)
```

There was the context in which the decision was made. For this example, it was *time* we wanted to
travel.

```{mermaid}
flowchart LR
    D[Weekend/Weekday]
```

Finally there were the actions that we could choose. For this example, this was the *mode of
transport*.

```{mermaid}
flowchart LR
    C{Decision}
    C -->|Train| D[Reward]
    C -->|Car| E[Reward]
```

### Example 2: Taking a bath

Lets say that we wanted to take a bath. We like our bath at 27 C, we have a hot and cold tap the we
can open. Any excess water is drained.

You have the choice to:

- keep both taps off [cold:0, hot:0]
- add cold water [cold:100, hot:0]
- add hot water [cold:0, hot:100]
- add warm water [cold:50, hot:50]

You make your choice every 10 minutes while you have been sitting in the tub.

```{mermaid}
flowchart LR
    A[Information] -->|Bath temp, Cold/Hot day| C{Decision}
    C -->|cold:0, hot:0| D[Information] -->|New Bath temp| z(temperature difference from Bath Temp - 27)
    C -->|cold:100, hot:0| E[Information]-->|New Bath temp| x(temperature difference from Bath Temp - 27)
    C -->|cold:0, hot:100| F[Information]-->|New Bath temp| y(temperature difference from Bath Temp - 27)
    C -->|cold:50, hot:50| G[Information]-->|New Bath temp| s(temperature difference from Bath Temp - 27)
```

Winter Day:

| Intial Bath Temp | Taps Open         | Bath Temp (+10m) | Reward -abs(Temp - 27) |
| ---------------- | ----------------- | ---------------- | ---------------------- |
| 22               | [cold:0, hot:100] | 28               | -1                     |
| 28               | [cold:0, hot:0]   | 25               | -2                     |
| 25               | [cold:0, hot:100] | 30               | -2                     |
| 30               | [cold:100, hot:0] | 18               | -10                    |
| 18               | [cold:0, hot:100] | 23               | -4                     |

Summer Day:

| Intial Bath Temp | Taps Open           | Bath Temp (+10m) | Reward -abs(Temp - 27) |
| ---------------- | ------------------- | ---------------- | ---------------------- |
| 24               | [cold:0, hot:100]   | 28               | -1                     |
| 28               | [cold:0, hot:0]     | 29               | -2                     |
| 29               | [cold:100, hot:100] | 25               | -2                     |
| 25               | [cold:100, hot:0]   | 20               | -7                     |
| 20               | [cold:0, hot:100]   | 30               | -3                     |

**So, what could be the bath problem formulation? To `whatif` this problem can be represented like this.**

The thing that measures how well we have done is called the *reward*. For this example, it was the *difference in the bath temperature to 27 C*. Any number higher or lower would be negative.

```{mermaid}
flowchart LR
    D(negative abs temp - 27)
```

There was the context in which the decision was made. For this example, it was the current *bath temperature* and how *hot or cold it was outside*.

```{mermaid}
flowchart LR
    D[Bath temp, Cold/Hot day]
```

Finally there were the actions that we could choose. For this example, this was *how much to open
both the hot and cold taps*.

```{mermaid}
flowchart LR
    C{Decision}
    C -->|cold:0, hot:0| D[Reward]
    C -->|cold:100, hot:0| E[Reward]
```

Once data is formulated into that format, `whatif` handles the rest.
