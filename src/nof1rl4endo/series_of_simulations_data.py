from __future__ import annotations

from dataclasses import dataclass
from typing import List, Callable
from .basic_types import History
from .metrics import score_df
from .metrics import Metric
from .simulation_data import SimulationData

import seaborn
import pandas
import pickle
import numpy


@dataclass
class SeriesOfSimulationsData:
    configuration: dict
    simulations: List[SimulationData]

    def plot_line(self, metric, t_between=None):
        df = score_df(self.simulations, [metric])
        if t_between:
            df = df[df["t"].between(t_between)]
        ax = seaborn.lineplot(data=df, x="t", y="Score", hue="Simulation")
        seaborn.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        return df

    @staticmethod
    def score_data(
        list_of_series: List[SeriesOfSimulationsData],
        metrics,
        renaming: dict[str, Callable] = {},
    ):
        scored_df = score_df(
            [
                simulation
                for series in list_of_series
                for simulation in series.simulations
            ],
            metrics,
            minmax_normalization=False,
        )
        for key, function in renaming.items():
            scored_df[key] = scored_df[key].apply(function)
        return scored_df

    @staticmethod
    def plot_lines(
        list_of_series: List[SeriesOfSimulationsData],
        metrics: List[Metric],
        hue: str = "policy_#_metric_#_model",
        process_df=lambda x: x,
        legend_position=None,
    ):
        scored_df = score_df(
            [
                simulation
                for series in list_of_series
                for simulation in series.simulations
            ],
            metrics,
            minmax_normalization=False,
        )
        scored_df["policy_#_metric_#_model"] = (
            scored_df["policy"]
            + "_#_"
            + scored_df["metric"]
            + "_#_"
            + scored_df["model"]
        )
        ax = seaborn.lineplot(
            data=process_df(scored_df),
            x="t",
            y="score",
            hue=hue,
            # units="patient_id",
            estimator=numpy.median,
            errorbar=lambda x: (numpy.quantile(x, 0.25), numpy.quantile(x, 0.75)),
        )
        ax.set(xlabel="t", ylabel="Regret")
        if not legend_position:
            legend_position = (0, 1 + scored_df[hue].nunique() * 0.1)
        seaborn.move_legend(
            ax, "upper left", title=None, bbox_to_anchor=legend_position
        )
        return ax

    def plot_allocations(self, treatment_name="treatment"):
        data = []
        for patient_id in range(len(self.simulations)):
            patient_history = self.simulations[patient_id].history
            for index in range(len(patient_history)):
                observation = patient_history.observations[index]
                data.append(
                    {
                        "patient_id": patient_id,
                        "index": index,
                        **observation.treatment,
                        "debug_info": str(
                            self.simulations[patient_id].history.debug_information()[
                                index
                            ]
                        ),
                        "context": str(observation.context),
                        "outcome": str(observation.outcome),
                        "counterfactual_outcomes": str(
                            observation.counterfactual_outcomes
                        ),
                        "t": observation.t,
                    }
                )
        df = pandas.DataFrame(data)
        return df.hvplot.heatmap(
            x="t",
            y="patient_id",
            C=treatment_name,
            hover_cols=[
                "debug_info",
                "context",
                "outcome",
                "counterfactual_outcomes",
                treatment_name,
            ],
            cmap="Category10",
            clim=(0, 8),
            grid=True,
        )
