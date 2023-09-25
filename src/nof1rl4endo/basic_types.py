from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict

import matplotlib
import pandas as pd
import seaborn as sb


def flatten(x):
    return [item for row in x for item in row]


def get_member_as_list(list, member_name):
    return [getattr(item, member_name) for item in list]


Context = Dict[str, float]

Treatment = Dict[str, float]

Outcome = Dict[str, float]


@dataclass
class Observation:
    context: Context
    treatment: Treatment
    outcome: Outcome
    t: int
    patient_id: int
    debug_data: List[dict] = field(default_factory=lambda: [])
    debug_information: str = ""
    counterfactual_outcomes: List[Outcome] = field(default_factory=lambda: [])


@dataclass
class History:
    observations: List[Observation]

    @staticmethod
    def fromListOfHistories(histories: List[History]):
        joinedHistory = History(observations=[])
        joinedHistory.observations = flatten(
            [history.observations for history in histories]
        )
        return joinedHistory

    def __len__(self):
        return len(self.observations)

    def add_observation(self, observation):
        self.observations.append(observation)

    def plot(self, ax) -> matplotlib.axis:
        primary_outcomes = [
            observation.outcome.outcome for observation in self.observations
        ]
        treatments = [observation.treatment.i for observation in self.observations]
        df = pd.DataFrame(
            data={"primary_outcomes": primary_outcomes, "treatments": treatments}
        )
        return sb.scatterplot(
            df, x=df.index, y="primary_outcomes", ax=ax, hue="treatments"
        )

    def to_df(self):
        if len(self.observations) == 0:
            return pd.DataFrame(columns=["treatment", "outcome", "patient_id"])

        dict_list = [
            {
                **observation.context,
                **observation.treatment,
                **observation.outcome,
                "patient_id": observation.patient_id,
                "t": observation.t,
            }
            for observation in self.observations
        ]
        df = pd.DataFrame(dict_list).infer_objects()

        # Eliminate duplicate columns
        df = df.loc[:, ~df.columns.duplicated()].copy()
        return df

    def counterfactual_outcomes_df(self, outcome_name="outcome"):
        dict_list = [
            [element[outcome_name] for element in observation.counterfactual_outcomes]
            for observation in self.observations
        ]
        return pd.DataFrame(dict_list)

    def debug_information(self):
        return [observation.debug_information for observation in self.observations]

    def debug_data(self):
        return [observation.debug_data for observation in self.observations]

    def __getitem__(self, index) -> History:
        if isinstance(index, slice):
            return History(
                observations=[
                    self.observations[i]
                    for i in range(*index.indices(len(self.observations)))
                ]
            )
        else:
            return History(observations=[self.observations[index]])
