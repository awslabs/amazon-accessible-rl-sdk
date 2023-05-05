# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
import gym
import numpy as np
import pandas as pd
from gym import spaces

import a2rl as wi


def fsigmoid(x, a, b, c):
    """This is the sigmoid function that we use for the propensity to buy.

    We use this to sample from later.

    Args:

    x = price
    c = max conversion

    a and b are sigmoid shape parameters
    a = smoothness of transition
    b = transition or mid price

    Returns:

    Conversion probability
    """
    x = [i * -1 for i in x]
    return c / (1.0 + np.exp(-a * (x + b)))


def parameters(days):
    """This function takes the simulation day and converts it into seasonal parameters
    that are used with the sigmoid function

    Args:
    simulation day

    Returns:
    [sigmoid_smoothness, mid_price, max_conversion_probability]
    """
    days = np.asarray(days)

    conversion = 0.2 + (np.cos(days * (2.0 * np.pi / 365)) + 2) / 20
    smoothness = 0.3 + (np.cos(days * (2.0 * np.pi / 365)) + 2) / 50
    mid_price = 10 * np.ones_like(days)

    if len(days) > 1:
        return np.array(list(zip(smoothness, mid_price, conversion)))
    else:
        return np.array((smoothness, mid_price, conversion)).reshape(
            -1,
        )


config = {
    "freight_price": 0.2,
    "max_fare": 20,
    "seats": 20,
    "visitors": 1000,
    "max_weight": 5600,
    "max_time": 1000,
}


class flight_sales_gym(gym.Env):
    def __init__(self):
        super().__init__()

        self.config = config
        self.day = 1
        self.max_time = self.config["max_time"]
        self.params = parameters([self.day])
        self.availableSeats = self.config["seats"]
        self.visitors = self.config["visitors"]
        self.priceScale = self.config["max_fare"]
        self.freight_price = self.config["freight_price"] + np.random.random()
        self.max_weight = self.config["max_weight"]
        self.action_space = spaces.Box(0, 1, (1,), dtype=np.float32)
        self.observation_space = spaces.Box(np.array([1, 0.1]), np.array([30, 5]), dtype=np.float32)
        self.reward_history = []

    def render(self):
        pass

    def step(self, action):
        self.freight_price = self.config["freight_price"] + np.random.random()
        self.freight_price = np.round(self.freight_price, decimals=1)

        done = False

        scaled_action = action
        action = action * self.priceScale
        reward = 0
        tickets = 0
        seats_left = self.availableSeats

        def fsigmoid(x, a, b, c):
            x = [i * -1 for i in x]
            return c / (1.0 + np.exp(-a * (x + b)))

        for i in range(self.visitors):
            if seats_left > 0:
                if np.random.random() < fsigmoid([action], *self.params)[0]:
                    seats_left -= 1
                    tickets += 1

        season = (np.cos(np.asarray(self.day) * (2.0 * np.pi / 365)) + 1) / 2
        season = np.round(season, decimals=1)

        state = np.array([season, self.freight_price])

        revenue = float(tickets * action + (self.max_weight - tickets * 100) * self.freight_price)
        revenue = np.round(revenue, decimals=1)

        self.day += 1
        self.params = parameters([self.day])

        if self.day > self.max_time:
            done = True

        reward = revenue
        self.reward_history.append(reward)

        self.history.iloc[-1, self.history.columns.get_loc("ticket_price")] = scaled_action
        self.history.iloc[-1, self.history.columns.get_loc("revenue")] = revenue

        dictionary = {
            "season": season,
            "freight_price": self.freight_price,
            "ticket_price": np.nan,
            "revenue": np.nan,
        }
        self.history = self.history.append(dictionary, ignore_index=True)
        return state, reward, done, {}

    def context(self):
        return wi.WiDataFrame(
            self.history.fillna(method="ffill"),
            states=["season", "freight_price"],
            actions=["ticket_price"],
            rewards=["revenue"],
        )

    def reset(self):
        self.config = config
        self.day = 1
        self.max_time = self.config["max_time"]
        self.params = parameters([self.day])
        self.availableSeats = self.config["seats"]
        self.visitors = self.config["visitors"]
        self.priceScale = self.config["max_fare"]
        self.freight_price = self.config["freight_price"] + np.random.random()
        self.max_weight = self.config["max_weight"]
        self.action_space = spaces.Box(0, 1, (1,), dtype=np.float32)
        self.observation_space = spaces.Box(np.array([1, 0.1]), np.array([30, 5]), dtype=np.float32)
        self.reward_history = []

        self.freight_price = np.round(self.freight_price, decimals=1)

        season = (np.cos(np.asarray(self.day) * (2.0 * np.pi / 365)) + 1) / 2
        season = np.round(season, decimals=1)

        state = np.array([season, self.freight_price])

        self.history = pd.DataFrame()

        dictionary = {
            "season": season,
            "freight_price": self.freight_price,
            "ticket_price": np.nan,
            "revenue": np.nan,
        }
        self.history = self.history.append(dictionary, ignore_index=True)

        return state
