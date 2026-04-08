from env.environment import SupportOpsEnv
from env.models import Action

env = SupportOpsEnv()
state = env.reset()

print("Initial:", state)

result = env.step(Action(assign_priority=1, response_type=1))
print("Step:", result)