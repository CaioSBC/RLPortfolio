from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

try:
    import quantstats as qs
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        """QuantStats module not found, environment can't plot results and calculate indicators.
        This module is not installed with rl_portfolio. Install by running one of the options:
        pip install quantstats --upgrade --no-cache-dir
        conda install -c ranaroussi quantstats
        """
    )


class PortfolioOptimizationEnv(gym.Env):
    """A portfolio allocation environment for Gymnasium.

    This environment simulates the interactions between an agent and the financial market
    based on data provided by a dataframe. The dataframe contains the time series of
    features defined by the user (such as closing, high and low prices) and must have
    a time and a tic column with a list of datetimes and ticker symbols respectively.
    An example of dataframe is shown below::

            date        high            low             close           tic
        0   2020-12-23  0.157414        0.127420        0.136394        ADA-USD
        1   2020-12-23  34.381519       30.074295       31.097898       BNB-USD
        2   2020-12-23  24024.490234    22802.646484    23241.345703    BTC-USD
        3   2020-12-23  0.004735        0.003640        0.003768        DOGE-USD
        4   2020-12-23  637.122803      560.364258      583.714600      ETH-USD
        ... ...         ...             ...             ...             ...

    Based on this dataframe, the environment will create an observation space that can
    be a Dict or a Box. The Box observation space is a three-dimensional array of shape
    (f, n, t), where f is the number of features, n is the number of stocks in the
    portfolio and t is the user-defined time window. If the environment is created with
    the parameter return_last_action set to True, the observation space is a Dict with
    the following keys::

        {
        "state": three-dimensional Box (f, n, t) representing the time series,
        "last_action": one-dimensional Box (n+1,) representing the portfolio weights
        }

    Note that the action space of this environment is an one-dimensional Box with size
    n + 1 because the portfolio weights must contains the weights related to all the
    stocks in the portfolio and to the remaining cash.

    Attributes:
        action_space: Action space.
        observation_space: Observation space.
        episode_length: Number of timesteps of an episode.
        portfolio_size: Number of stocks in the portfolio.
    """

    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(
        self,
        df,
        initial_amount,
        order_df=True,
        return_last_action=False,
        data_normalization=None,
        state_normalization=None,
        reward_scaling=1,
        comission_fee_model="trf",
        comission_fee_pct=0,
        features=["close", "high", "low"],
        valuation_feature="close",
        time_column="date",
        time_format=None,
        tic_column="tic",
        tics_in_portfolio="all",
        time_window=1,
        print_metrics=True,
        plot_graphs=True,
        cwd="./",
    ):
        """Initializes environment's instance.

        Args:
            df: Dataframe with market information over a period of time.
            initial_amount: Initial amount of cash available to be invested.
            order_df: If True input dataframe is ordered by time.
            return_last_action: If True, observations also return the last performed
                action. Note that, in that case, the observation space is a Dict.
            data_normalization: Defines the normalization method applied to input
                dataframe. Possible values are "by_previous_time", "by_COLUMN_NAME"
                (where COLUMN_NAME must be changed to a real column name) and a
                custom function. If None, no normalization is done.
            state_normalization: Defines the normalization method applied to the state
                output during simulation. Possible values are "by_initial_value",
                "by_last_value", "by_initial_FEATURE_NAME", "by_last_FEATURE_NAME"
                (where FEATURE_NAME must be change to the name of the feature used
                as normalizer) and a custgom function. If None, no normalization is
                done.
            reward_scaling: A scaling factor to multiply the reward function. This
                factor can help training.
            comission_fee_model: Model used to simulate comission fee. Possible values
                are "trf" (for transaction remainder factor model), "trf_approx" (for
                a faster approximate version of "trf") and "wvm" (for weights vector
                modifier model). If None, commission fees are not considered.
            comission_fee_pct: Percentage to be used in comission fee. It must be a
                value between 0 and 1.
            features: List of features to be considered in the observation space. The
                items of the list must be names of columns of the input dataframe.
            valuation_feature: Feature to be considered in the portfolio value calculation.
            time_column: Name of the dataframe's column that contain the datetimes that
                index the dataframe.
            time_format: Formatting string of time column (if format string is invalid,
                an error will be raised). If None, time column will not be transformed
                to datetime.
            tic_name: Name of the dataframe's column that contain ticker symbols.
            tics_in_portfolio: List of ticker symbols to be considered as part of the
                portfolio. If "all", all tickers of input data are considered.
            time_window: Size of time window.
            print_metrics: If True, performance metrics will be printed at the end of
                episode.
            plot_graphs: If True, graphs will be ploted and saved in the specified folder.
            cwd: Local repository in which resulting graphs will be saved.
        """
        self._time_window = time_window
        self._time_index = time_window - 1
        self._time_column = time_column
        self._time_format = time_format
        self._tic_column = tic_column
        self._df = df
        self._initial_amount = initial_amount
        self._state_normalization = state_normalization
        self._return_last_action = return_last_action
        self._reward_scaling = reward_scaling
        self._comission_fee_pct = comission_fee_pct
        self._comission_fee_model = comission_fee_model
        self._features = features
        self._valuation_feature = valuation_feature
        self._print_metrics = print_metrics
        self._plot_graphs = plot_graphs
        self._cwd = Path(cwd)

        # results file
        self._results_file = self._cwd / "results" / "rl"
        self._results_file.mkdir(parents=True, exist_ok=True)

        # initialize price variation
        self._df_price_variation = None

        # preprocess data
        self._preprocess_data(order_df, data_normalization, tics_in_portfolio)

        # dims and spaces
        self._tic_list = self._df[self._tic_column].unique()
        self.portfolio_size = (
            len(self._tic_list)
            if tics_in_portfolio == "all"
            else len(tics_in_portfolio)
        )
        action_space = 1 + self.portfolio_size

        # sort datetimes and define episode length
        self._sorted_times = sorted(set(self._df[time_column]))
        self.episode_length = len(self._sorted_times) - time_window + 1

        # define action space
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(action_space,))

        # define observation space
        if self._return_last_action:
            # if  last action must be returned, a dict observation
            # is defined
            self.observation_space = gym.spaces.Dict(
                {
                    "state": gym.spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(
                            len(self._features),
                            len(self._tic_list),
                            self._time_window,
                        ),
                    ),
                    "last_action": gym.spaces.Box(low=0, high=1, shape=(action_space,)),
                }
            )
        else:
            # if information about last action is not relevant,
            # a 3D observation space is defined
            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(len(self._features), len(self._tic_list), self._time_window),
            )

        self._reset_memory()

        self._portfolio_value = self._initial_amount
        self._terminal = False

    def reset(self, seed=None, options=None):
        """Resets the environment and returns it to its initial state (the
        fist date of the dataframe).

        Args:
            seed: A seeding number to configure random values.
            options: A dictionary with reset options. It's not used.

        Note:
            If the environment was created with "return_last_action" set to
            True, the initial observation returned will be a Dict. If it's set
            to False, the initial observation will be a Box. You can check the
            observation space through the attribute "observation_space".

        Returns:
            A tuple (observation, info) with the initial observation and the
            initial info dictionary.
        """
        super().reset(seed=seed)

        # time_index must start a little bit in the future to implement lookback
        self._time_index = self._time_window - 1
        self._reset_memory()

        self._observation, self._info = self._generate_observation_and_info(
            self._time_index
        )
        self._portfolio_value = self._initial_amount
        self._terminal = False

        return self._observation, self._info

    def step(self, action):
        """Performs a simulation step.

        Args:
            action: A unidimensional array containing portfolio weights to be
                considered in the simulation.

        Note:
            If the environment was created with "return_last_action" set to
            True, the observation returned will be a Dict. If it's set to False,
            the observation will be a Box. You can check the observation space
            through the attribute "observation_space".

        Returns:
            The following tuple is returned: (observation, reward, terminal,
            truncated, info).

            observation: Current simulation observation.
            reward: Reward related to the last performed action.
            terminal: If True, the environment is in a terminal state.
            truncated: If True, the environment has passed it's simulation
                time limit. Currently, it's always False.
            info: A dictionary containing informations about the last
                simulation step.
        """
        self._terminal = self._time_index >= len(self._sorted_times) - 1

        if self._terminal:
            self._terminal_routine()
        else:
            # transform action to numpy array (if it's a list)
            action = np.array(action, dtype=np.float32)

            # if necessary, normalize portfolio weights
            if math.isclose(np.sum(action), 1, abs_tol=1e-6) and np.min(action) >= 0:
                weights = action
            else:
                weights = self._softmax_normalization(action)

            # save initial portfolio weights for this time step
            self._actions_memory.append(weights)

            # get last step final weights and portfolio_value
            last_weights = self._final_weights[-1]

            # load next observation
            self._time_index += 1
            self._observation, self._info = self._generate_observation_and_info(
                self._time_index
            )

            # if using weights vector modifier, we need to modify weights vector
            if self._comission_fee_model == "wvm":
                weights, self._portfolio_value = self._apply_wvm_fee(
                    weights, last_weights
                )
            elif self._comission_fee_model == "trf":
                self._info["trf_mu"], self._portfolio_value = self._apply_trf_fee(
                    weights, last_weights
                )
            elif self._comission_fee_model == "trf_approx":
                self._info["trf_mu"], self._portfolio_value = (
                    self._apply_trf_approx_fee(weights, last_weights)
                )

            # save initial portfolio value of this time step
            self._asset_memory["initial"].append(self._portfolio_value)

            # time passes and time variation changes the portfolio distribution
            portfolio = self._portfolio_value * (weights * self._price_variation)

            # calculate new portfolio value and weights
            self._portfolio_value = np.sum(portfolio)
            weights = portfolio / self._portfolio_value

            # save final portfolio value and weights of this time step
            self._asset_memory["final"].append(self._portfolio_value)
            self._final_weights.append(weights)

            # save date memory
            self._date_memory.append(self._info["end_time"])

            # define portfolio return
            rate_of_return = (
                self._asset_memory["final"][-1] / self._asset_memory["final"][-2]
            )
            portfolio_return = rate_of_return - 1
            portfolio_reward = np.log(rate_of_return)

            # save portfolio return memory
            self._portfolio_return_memory.append(portfolio_return)
            self._portfolio_reward_memory.append(portfolio_reward)

            # Define portfolio return
            self._reward = portfolio_reward
            self._reward = self._reward * self._reward_scaling

        return self._observation, self._reward, self._terminal, False, self._info

    def render(self, mode="human"):
        """Renders the environment.

        Returns:
            Observation of current simulation step.
        """
        return self._observation

    def enumerate_portfolio(self):
        """Enumerates the current porfolio by showing the ticker symbols
        of all the investments considered in the portfolio.
        """
        print("Index: 0. Tic: Cash")
        for index, tic in enumerate(self._tic_list):
            print(f"Index: {index + 1}. Tic: {tic}")

    def _temporal_variation_df(self, periods=1):
        """Calculates the temporal variation dataframe. For each feature, this
        dataframe contains the rate of the current feature's value and the last
        feature's value given a period. It's used to normalize the dataframe.

        Args:
            periods: Periods (in time indexes) to calculate temporal variation.

        Returns:
            Temporal variation dataframe.
        """
        df_temporal_variation = self._df.copy()
        prev_columns = []
        for column in self._features:
            prev_column = f"prev_{column}"
            prev_columns.append(prev_column)
            df_temporal_variation[prev_column] = df_temporal_variation.groupby(
                self._tic_column
            )[column].shift(periods=periods)
            df_temporal_variation[column] = (
                df_temporal_variation[column] / df_temporal_variation[prev_column]
            )
        df_temporal_variation = (
            df_temporal_variation.drop(columns=prev_columns)
            .fillna(1)
            .reset_index(drop=True)
        )
        return df_temporal_variation

    def _normalize_dataframe(self, normalize):
        """ "Normalizes the environment's dataframe.

        Args:
            normalize: Defines the normalization method applied to the dataframe.
                Possible values are "by_previous_time", "by_fist_time_window_value",
                "by_COLUMN_NAME" (where COLUMN_NAME must be changed to a real column
                name) and a custom function. If None no normalization is done.

        Note:
            If a custom function is used in the normalization, it must have an
            argument representing the environment's dataframe.
        """
        if type(normalize) == str:
            if normalize == "by_previous_time":
                print(f"Normalizing {self._features} by previous time...")
                self._df = self._temporal_variation_df()
            elif normalize.startswith("by_"):
                normalizer_column = normalize[3:]
                print(f"Normalizing {self._features} by {normalizer_column}")
                for column in self._features:
                    self._df[column] = self._df[column] / self._df[normalizer_column]
        elif callable(normalize):
            print("Applying custom normalization function...")
            self._df = normalize(self._df)
        else:
            print("No normalization was performed.")

    def _preprocess_data(self, order, normalize, tics_in_portfolio):
        """Orders and normalizes the environment's dataframe.

        Args:
            order: If true, the dataframe will be ordered by ticker list
                and datetime.
            normalize: Defines the normalization method applied to the dataframe.
                Possible values are "by_previous_time", "by_fist_time_window_value",
                "by_COLUMN_NAME" (where COLUMN_NAME must be changed to a real column
                name) and a custom function. If None no normalization is done.
            tics_in_portfolio: List of ticker symbols to be considered as part of the
                portfolio. If "all", all tickers of input data are considered.
        """
        # order time dataframe by tic and time
        if order:
            self._df = self._df.sort_values(by=[self._tic_column, self._time_column])
        # defining price variation after ordering dataframe
        self._df_price_variation = self._temporal_variation_df()
        # select only stocks in portfolio
        if tics_in_portfolio != "all":
            self._df_price_variation = self._df_price_variation[
                self._df_price_variation[self._tic_column].isin(tics_in_portfolio)
            ]
        # apply normalization
        if normalize:
            self._normalize_dataframe(normalize)
        # transform str to datetime
        if self._time_format is not None:
            self._df[self._time_column] = pd.to_datetime(
                self._df[self._time_column], format=self._time_format
            )
            self._df_price_variation[self._time_column] = pd.to_datetime(
                self._df_price_variation[self._time_column], format=self._time_format
            )
        # transform numeric variables to float32 (compatibility with pytorch)
        self._df[self._features] = self._df[self._features].astype("float32")
        self._df_price_variation[self._features] = self._df_price_variation[
            self._features
        ].astype("float32")

    def _terminal_routine(self):
        """Executes terminal routine (prints and plots). This function also adds a
        "metrics" key to "_info" attribute with the episode's simulation metrics.
        """
        metrics_df = pd.DataFrame(
            {
                "date": self._date_memory,
                "returns": self._portfolio_return_memory,
                "rewards": self._portfolio_reward_memory,
                "portfolio_values": self._asset_memory["final"],
            }
        )
        metrics_df.set_index("date", inplace=True)

        self._info["metrics"] = {
            "value": self._portfolio_value,
            "fapv": self._portfolio_value / self._asset_memory["final"][0],
            "mdd": qs.stats.max_drawdown(metrics_df["portfolio_values"]),
            "sharpe": qs.stats.sharpe(metrics_df["returns"], annualize=False),
        }

        if self._plot_graphs:
            plt.plot(self._asset_memory["final"], "tab:blue")
            plt.title("Portfolio Value Over Time")
            plt.xticks(
                np.round(np.linspace(0, len(self._asset_memory["final"]) - 1, 10))
            )
            plt.xlabel("Simulation Steps")
            plt.ylabel("Portfolio Value")
            plt.savefig(
                self._results_file / "portfolio_value_plot.png", bbox_inches="tight"
            )
            plt.close()

            plt.plot(self._portfolio_reward_memory, "tab:blue")
            plt.title("Reward Over Time")
            plt.xticks(
                np.round(np.linspace(0, len(self._portfolio_reward_memory) - 1, 10))
            )
            plt.xlabel("Simulation Steps")
            plt.ylabel("Reward")
            plt.tight_layout()
            plt.savefig(self._results_file / "reward_plot.png", bbox_inches="tight")
            plt.close()

            actions = np.array(self._actions_memory)
            df_actions = pd.DataFrame(
                actions,
                columns=np.append("Cash", self._tic_list),
            )

            legend_items = list(df_actions)
            legend_items.reverse()

            fig, ax = plt.subplots(figsize=(15, 6))
            ax = df_actions.plot(
                kind="bar",
                ax=ax,
                stacked=True,
                title="Portfolio Distribution Over Time",
                grid=False,
                legend="reverse",
                xticks=np.round(np.linspace(0, len(self._date_memory) - 1, 10)),
                yticks=[],
                xlabel="Simulation Step",
                ylabel="Assets Distribution",
                rot=0,
                colormap="tab20",
                width=1,
                linewidth=0,
            )
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(
                handles[::-1],
                labels[::-1],
                title="Assets",
                loc="center left",
                bbox_to_anchor=(1.0, 0.5),
            )
            fig.savefig(self._results_file / "actions_plot.png", bbox_inches="tight")
            plt.close()

            # this plot cannot be done if index is not datetime
            try:
                qs.plots.snapshot(
                    metrics_df["returns"],
                    show=False,
                    savefig=self._results_file / "portfolio_summary.png",
                )
            except:
                pass

        if self._print_metrics:
            print("=================================")
            print("Initial portfolio value:{}".format(self._asset_memory["final"][0]))
            print(f"Final portfolio value: {self._portfolio_value}")
            print(
                "Final accumulative portfolio value: {}".format(
                    self._portfolio_value / self._asset_memory["final"][0]
                )
            )
            print(
                "Maximum DrawDown: {}".format(
                    qs.stats.max_drawdown(metrics_df["portfolio_values"])
                )
            )
            print(
                "Sharpe ratio: {}".format(
                    qs.stats.sharpe(metrics_df["returns"], annualize=False)
                )
            )
            print("=================================")

    def _softmax_normalization(self, actions):
        """Normalizes the action vector using softmax function.

        Returns:
            Normalized action vector (portfolio vector).
        """
        numerator = np.exp(actions)
        denominator = np.sum(np.exp(actions))
        softmax_output = numerator / denominator
        return softmax_output

    def _normalize_state(self, state):
        """Applies a normalization method to the state matrix.

        Args:
            state: State to apply the normalization method.

        Returns:
            Normalized state.
        """
        norm_state = state.copy()
        feature_size = state.shape[0]
        tic_size = state.shape[1]
        if type(self._state_normalization) == str:
            if self._state_normalization == "by_last_value":
                last_values = state[:, :, -1].reshape(feature_size, tic_size, 1)
                norm_state = state / last_values
            elif self._state_normalization == "by_initial_value":
                initial_values = state[:, :, 0].reshape(feature_size, tic_size, 1)
                norm_state = state / initial_values
            elif self._state_normalization.startswith("by_last_"):
                feature = self._state_normalization[8:]
                feature_index = self._features.index(feature)
                last_values = state[feature_index, :, -1].reshape(1, tic_size, 1)
                norm_state = state / last_values
            elif self._state_normalization.startswith("by_initial_"):
                feature = self._state_normalization[11:]
                feature_index = self._features.index(feature)
                initial_values = state[feature_index, :, 0].reshape(1, tic_size, 1)
                norm_state = state / initial_values
        elif type(self._state_normalization) == callable:
            norm_state = self._state_normalization(state)
        return norm_state

    def _generate_observation(self, state):
        """Generate observation given the observation space. If "return_last_action"
        is set to False, a three-dimensional box is returned. If it's set to True, a
        dictionary is returned. The dictionary follows the standard below::

            {
            "state": Three-dimensional box representing the current state,
            "last_action": One-dimensional box representing the last action
            }
        """
        last_action = self._actions_memory[-1]
        if self._return_last_action:
            return {"state": state, "last_action": last_action}
        else:
            return state

    def _generate_observation_and_info(self, time_index):
        """Generates observation and information given a time index. It also updates
        "_data" and "_price_variations" attributes with information about the current
        simulation step.

        Args:
            time_index: An integer that represents the index of a specific datetime.
                The initial datetime of the dataframe is given by 0.

        Note:
            If the environment was created with "return_last_action" set to
            True, the returned observation will be a Dict. If it's set to False,
            the returned observation will be a Box. You can check the observation
            state through the attribute "observation_space".

        Returns:
            A tuple with the following form: (observation, info).

            observation: The observation of the current time index. It can be a Box
                or a Dict.
            info: A dictionary with some informations about the current simulation
                step. The dict has the following keys::

                {
                "tics": List of ticker symbols,
                "start_time": Start time of current time window,
                "start_time_index": Index of start time of current time window,
                "end_time": End time of current time window,
                "end_time_index": Index of end time of current time window,
                "data": Data related to the current time window,
                "price_variation": Price variation of current time step
                }
        """
        # returns state in form (channels, tics, timesteps)
        end_time = self._sorted_times[time_index]
        start_time = self._sorted_times[time_index - (self._time_window - 1)]

        # define data to be used in this time step
        self._data = self._df[
            (self._df[self._time_column] >= start_time)
            & (self._df[self._time_column] <= end_time)
        ][[self._time_column, self._tic_column] + self._features]

        # define price variation of this time_step
        self._price_variation = self._df_price_variation[
            self._df_price_variation[self._time_column] == end_time
        ][self._valuation_feature].to_numpy()
        self._price_variation = np.insert(self._price_variation, 0, 1)

        # define state to be returned
        state = (
            self._data.set_index([self._time_column, self._tic_column])
            .unstack(-1)
            .to_numpy()
            .reshape(self._time_window, len(self._features), len(self._tic_list))
            .transpose(1, 2, 0)
        )
        info = {
            "tics": self._tic_list,
            "start_time": start_time,
            "start_time_index": time_index - (self._time_window - 1),
            "end_time": end_time,
            "end_time_index": time_index,
            "data": self._data,
            "price_variation": self._price_variation,
        }
        return self._generate_observation(self._normalize_state(state)), info

    def _apply_wvm_fee(self, weights, last_weights):
        """Applies weights vector modifier fee model.

        Args:
            weights: Current portfolio weights (current agent action).
            last_weights: Portfolio weights at the end of last step.

        Returns:
            New portfolio weights after fees were applied and new portfolio value.
        """
        delta_weights = weights - last_weights
        delta_assets = delta_weights[1:]  # disconsider cash
        # calculate fees considering weights modification
        fees = np.sum(np.abs(delta_assets * self._portfolio_value))
        if fees > weights[0] * self._portfolio_value:
            return last_weights, self._portfolio_value
            # maybe add negative reward
        else:
            portfolio = weights * self._portfolio_value
            portfolio[0] -= fees
            new_portfolio_value = np.sum(portfolio)  # new portfolio value
            new_weights = portfolio / new_portfolio_value  # new weights
            return new_weights, new_portfolio_value

    def _apply_trf_fee(self, weights, last_weights):
        """Applies transaction remainder factor fee model.

        Args:
            weights: Current portfolio weights (current agent action).
            last_weights: Portfolio weights at the end of last step.

        Returns:
            Calculated mu factor and new portfolio_value.
        """
        last_mu = 1
        mu = 1 - 2 * self._comission_fee_pct + self._comission_fee_pct**2
        while abs(mu - last_mu) > 1e-10:
            last_mu = mu
            mu = (
                1
                - self._comission_fee_pct * last_weights[0]
                - (2 * self._comission_fee_pct - self._comission_fee_pct**2)
                * np.sum(np.maximum(last_weights[1:] - mu * weights[1:], 0))
            ) / (1 - self._comission_fee_pct * weights[0])
        new_portfolio_value = mu * self._portfolio_value
        return mu, new_portfolio_value

    def _apply_trf_approx_fee(self, weights, last_weights):
        """Applies an approximate version of transaction remainder factor
        fee model. This version is faster and the difference between the
        approximate value and the true value is O(c^2), where c is the
        commission fee.

        Args:
            weights: Current portfolio weights (current agent action).
            last_weights: Portfolio weights at the end of last step.

        Returns:
            Calculated mu factor and new portfolio_value.
        """
        mu = 1 - self._comission_fee_pct * np.sum(
            np.abs(weights[1:] - last_weights[1:])
        )
        new_portfolio_value = mu * self._portfolio_value
        return mu, new_portfolio_value

    def _reset_memory(self):
        """Resets the environment's memory."""
        date_time = self._sorted_times[self._time_index]
        # memorize portfolio value each step
        self._asset_memory = {
            "initial": [self._initial_amount],
            "final": [self._initial_amount],
        }
        # memorize portfolio return and reward each step
        self._portfolio_return_memory = [0]
        self._portfolio_reward_memory = [0]
        # initial action: all money is allocated in cash
        self._actions_memory = [
            np.array([1] + [0] * self.portfolio_size, dtype=np.float32)
        ]
        # memorize portfolio weights at the ending of time step
        self._final_weights = [
            np.array([1] + [0] * self.portfolio_size, dtype=np.float32)
        ]
        # memorize datetimes
        self._date_memory = [date_time]

    def _seed(self, seed=None):
        """Seeds the sources of randomness of this environment to guarantee
        reproducibility.

        Args:
            seed: Seed value to be applied.

        Returns:
            Seed value applied.
        """
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
