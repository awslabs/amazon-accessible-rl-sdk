# Amazon Accessible RL SDK <!-- omit from toc -->

[ [Documentation](https://awslabs.github.io/amazon-accessible-rl-sdk/) |
[PyPI](https://pypi.org/project/a2rl/) |
[Blog-01](https://medium.com/@yapweiyih/underfloor-heating-optimisation-using-offline-reinforcement-learning-44f7747f4d6f)
| Blog-02 (coming soon) ]

Amazon Accessible RL (A2RL) is an open-source Python package for [sequential decision making
problem](https://en.wikipedia.org/wiki/Sequential_decision_making) using offline time-series data.
It focuses on offline RL using state-of-the-art generative transformer technology – the same
technology behind [GATO](https://www.deepmind.com/publications/a-generalist-agent), [trajectory
transformer](https://trajectory-transformer.github.io/) and [decision
transformer](https://arxiv.org/abs/2106.01345).

A2RL guides you through [problem formulation](https://awslabs.github.io/amazon-accessible-rl-sdk/)
via [data frames
API](https://awslabs.github.io/amazon-accessible-rl-sdk/example.html#historical-data), conduct
[initial data
analysis](https://awslabs.github.io/amazon-accessible-rl-sdk/auto-notebooks/data_properties.html) to
see if a solution is possible, use the data to train a
[simulator](https://awslabs.github.io/amazon-accessible-rl-sdk/auto-notebooks/simulator.html) (aka
*digital twin*) based on the data, and providing [recommended
actions](https://awslabs.github.io/amazon-accessible-rl-sdk/auto-notebooks/planner_byo_example.html).

## Installation

```bash
pip install a2rl
```

## Usage

You should start by formulating your problem into *states*, *actions*, and *rewards* (see the
[online documentation](https://awslabs.github.io/amazon-accessible-rl-sdk/)). Then, prepare a
dataset that reflects the formulation, using A2RL's [Pandas](https://pandas.pydata.org/)-like API.

A synthetic dataset is included to help you quickly jump into the end-to-end workflow:

```python
import a2rl as wi
from a2rl.utils import plot_information

# Load a sample dataset which contains historical states, actions, and rewards.
wi_df = wi.read_csv_dataset(wi.sample_dataset_path("chiller")).trim().add_value()
wi_df = wi_df.iloc[:1000]  # Reduce data size for demo purpose

# Checks and analysis
plot_information(wi_df)

# Train a simulator
tokenizer = wi.AutoTokenizer(wi_df, block_size_row=2)
builder = wi.GPTBuilder(tokenizer, model_dir="my-model", )
model = builder.fit()
simulator = wi.Simulator(tokenizer, model, max_steps=100, reset_coldstart=2)

# Get recommended actions given an input context (s,a,r,v,...s).
# Context must end with states, and its members must be tokenized.
custom_context = simulator.tokenizer.df_tokenized.sequence[:7]
recommendation_df = simulator.sample(custom_context, 3)

# Show recommendations (i.e., trajectory)
recommendation_df
```

For more examples, see `notebooks/` (pre-rendered versions
[here](https://awslabs.github.io/amazon-accessible-rl-sdk/example.html)), and the A2RL blog series:
[part-1](https://medium.com/@yapweiyih/underfloor-heating-optimisation-using-offline-reinforcement-learning-44f7747f4d6f)
and part-2 (coming soon).

## Help and Support

* [Contributing](CONTRIBUTING.md)
* Apache-2.0 [License](LICENSE)
