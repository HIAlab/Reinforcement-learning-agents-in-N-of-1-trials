from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

import pandas
import numpy
import seaborn as sb
from .basic_types import History

from scipy.stats import entropy


import hvplot.pandas  # noqa

if TYPE_CHECKING:
    from .simulation_data import SimulationData

from sklearn.preprocessing import minmax_scale


class Metric(ABC):
    def __init__(self, outcome_name="outcome", treatment_name="treatment"):
        self.outcome_name = outcome_name
        self.treatment_name = treatment_name

    @abstractmethod
    def score(self, data: SimulationData) -> List[float] | float:
        pass

    def score_simulations(self, simulations: List[SimulationData]):
        df_list = [
            pandas.DataFrame(
                {
                    "t": [
                        observation.t for observation in simulation.history.observations
                    ],
                    "score": self.score(simulation),
                    "simulation": str(simulation),
                    "patient_id": simulation.patient_id,
                    "model": str(simulation.model),
                    "policy": str(simulation.policy),
                }
            )
            for index, simulation in enumerate(simulations)
        ]

        return pandas.concat(df_list)

    @abstractmethod
    def __str__(self) -> str:
        pass


def score_df(histories: list[History], metrics, minmax_normalization=False):
    df_list = []
    scores = {str(metric): metric.score_simulations(histories) for metric in metrics}
    for metric_name, metric_df in scores.items():
        metric_df["metric"] = metric_name
        df_list.append(metric_df)
    df = pandas.concat(df_list)
    if minmax_normalization:
        for metric in metrics:
            df[str(metric)] = minmax_scale(df[str(metric)])
    return df


class SimpleRegret(Metric):
    def score(self, data: SimulationData) -> List[float]:
        return numpy.cumsum(-data.history.to_df()[self.outcome_name])

    def __str__(self) -> str:
        return "Simple Regret"


class Entropy(Metric):
    def score(self, data: SimulationData) -> float:
        value, counts = numpy.unique(
            data.history.to_df()[self.outcome_name], return_counts=True
        )
        return entropy(counts)

    def __str__(self) -> str:
        return f"Entropy ({self.outcome_name})"


class RegretAgainstOtherConfiguration(Metric):
    def __init__(
        self,
        config_to_simulation_data: dict[dict, SimulationData],
        configuration_transform_function,
        name="",
        **kwargs,
    ):
        self.config_to_simulation_data = config_to_simulation_data
        self.configuration_transform_function = configuration_transform_function
        self.name = name
        super().__init__(**kwargs)

    def score(self, data: SimulationData):
        config_to_compare_against = str(
            self.configuration_transform_function(data.configuration)
        )
        assert (
            config_to_compare_against in self.config_to_simulation_data
        ), f"Can not calculate compare metric, desired config not present in dataset\nTried to fetch {config_to_compare_against}"

        counterfactual_df = self.config_to_simulation_data[
            config_to_compare_against
        ].history.to_df()
        data_df = data.history.to_df()

        merge = data_df.merge(
            counterfactual_df,
            how="inner",
            validate="one_to_one",
            on="t",
            suffixes=(None, "_counterfactual"),
        )
        return numpy.cumsum(
            merge[self.outcome_name + "_counterfactual"] - merge[self.outcome_name]
        )

    def __str__(self) -> str:
        return f"RegretAgainstOtherConfiguration({self.name})"


class StandardDeviation(Metric):
    def score(self, data: SimulationData) -> float:
        return numpy.std(data.history.to_df()[self.treatment_name])

    def __str__(self) -> str:
        return f"std({self.treatment_name})"


class ProbabilityInterval(Metric):
    def __init__(
        self,
        policy_filter_names,
        **kwargs,
    ):
        self.policy_filter_names = policy_filter_names
        super().__init__(**kwargs)


class ProbabilityIntervalMax(ProbabilityInterval):
    def score(self, simulation: SimulationData) -> List[float]:
        policy = simulation.policy
        if policy in self.policy_filter_names:
            probability_arrays = [
                d["probabilities"]
                for d in simulation.history.debug_data()
                if "probabilities" in d
            ]
            max_values = pandas.Series([max(array) for array in probability_arrays])
            return [max_values.max()] * len(simulation.history)
        return 0

    def __str__(self) -> str:
        return "ProbabilityIntervalMax"


class ProbabilityIntervalMin(ProbabilityInterval):
    def score(self, simulation: SimulationData) -> List[float]:
        policy = simulation.policy
        if policy in self.policy_filter_names:
            probability_arrays = [
                d["probabilities"]
                for d in simulation.history.debug_data()
                if "probabilities" in d
            ]
            min_values = pandas.Series([min(array) for array in probability_arrays])
            return [min_values.min()] * len(simulation.history)
        return 0

    def __str__(self) -> str:
        return "ProbabilityIntervalMin"
