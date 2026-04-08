import random
from .models import Observation, Action, Reward, StepOutput

class SupportOpsEnv:

    def __init__(self):
        self.state_data = None
        self.done = False
        self.steps = 0

    def reset(self):
        self.steps = 0
        self.done = False

        self.state_data = Observation(
            ticket_text=random.choice([
                "Payment failed but money deducted",
                "App crashes on checkout",
                "Login issue",
                "Need urgent refund"
            ]),
            urgency_level=random.randint(0, 2),
            customer_tier=random.randint(0, 1),
            waiting_time=random.uniform(1, 24),
            churn_risk=random.uniform(0, 0.5),
            sla_deadline=random.uniform(5, 24)
        )
        return self.state()

    def state(self):
        return self.state_data

    def step(self, action: Action):
        if self.done:
            raise Exception("Episode finished. Call reset().")

        self.steps += 1

        reward_value = self._compute_reward(action)
        self._update_state(action)

        if self.steps >= 4 or self.state_data.sla_deadline <= 0:
            self.done = True

        return StepOutput(
            observation=self.state_data,
            reward=Reward(value=reward_value),
            done=self.done,
            info={"steps": self.steps}
        )

    def _compute_reward(self, action):
        state = self.state_data
        reward = 0

        # Priority match
        if action.assign_priority == state.urgency_level:
            reward += 4
        else:
            reward -= 3

        # Response handling
        if state.urgency_level == 2 and action.response_type == 2:
            reward += 5
        elif action.response_type == 1:
            reward += 3
        else:
            reward += 1

        # SLA penalty
        if state.sla_deadline < 0:
            reward -= 10

        # churn impact
        reward += (1 - state.churn_risk) * 5

        return reward

    def _update_state(self, action):
        self.state_data.waiting_time += 2
        self.state_data.sla_deadline -= 2

        if action.response_type == 0:
            self.state_data.churn_risk += 0.1
        elif action.response_type == 2:
            self.state_data.churn_risk -= 0.2

        self.state_data.churn_risk = max(0, min(1, self.state_data.churn_risk))