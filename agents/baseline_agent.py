from env.models import Action

class BaselineAgent:

    def act(self, obs):
        if obs.urgency_level == 2:
            return Action(assign_priority=2, response_type=2)
        elif obs.urgency_level == 1:
            return Action(assign_priority=1, response_type=1)
        else:
            return Action(assign_priority=0, response_type=0)