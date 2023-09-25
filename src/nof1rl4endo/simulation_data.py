from .basic_types import History

from dataclasses import dataclass


@dataclass
class SimulationData:
    history: History
    model: str
    policy: str
    patient_id: int

    def __str__(self):
        return f"SimulationData"

    @property
    def configuration(self):
        return {
            "policy": self.policy,
            "model": self.model,
            "patient_id": self.patient_id,
        }
