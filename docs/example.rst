Examples
========

To optimize sequential decision makings purely from historical data, A2RL adopts the following key
concepts:

1. Historical data
2. Problem formulation: Markovian test
3. Simulator
4. Agents

Historical Data
---------------

Load your historical data into a :class:`a2rl.WiDataFrame` of *states*, *actions*, and *rewards*
columns.

.. nbgallery::

    auto-notebooks/dataframe
    auto-notebooks/create_dataset

Markovian Test
--------------

Quantify the MDP-ness of the historical data, to determine whether your problem is best formulated
as an MDP problem where a state can be predicted from its previous states and actions, or a
multi-bandit problem where the simulator shows past decision are not good predictors for future
states.

.. nbgallery::

    Data Properties <auto-notebooks/data_properties>

Simulator
---------

Train a :class:`a2rl.Simulator` that's able to predict what happens when an action is taken on a
certain state: what's the reward for taking the action, and what's the next state once the action is
performed.

.. nbgallery::

    auto-notebooks/simulator
    auto-notebooks/backtest
    auto-notebooks/planner_byo_example
    auto-notebooks/planner_loon_example

Blog Series
-----------

- Part-1: `Underfloor Heating Optimisation using Offline Reinforcement Learning <https://medium.com/@yapweiyih/underfloor-heating-optimisation-using-offline-reinforcement-learning-44f7747f4d6f>`_

- Part-2: Offline Reinforcement Learning with A2RL on Amazon SageMaker (coming soon)
