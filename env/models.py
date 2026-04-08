from pydantic import BaseModel
from typing import Optional

class Observation(BaseModel):
    ticket_text: str
    urgency_level: int
    customer_tier: int
    waiting_time: float
    churn_risk: float
    sla_deadline: float

class Action(BaseModel):
    assign_priority: int
    response_type: int

class Reward(BaseModel):
    value: float

class StepOutput(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Optional[dict] = {}