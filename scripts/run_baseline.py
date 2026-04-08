from agents.baseline_agent import BaselineAgent
from env.tasks import EasyTask, MediumTask, HardTask

agent = BaselineAgent()

tasks = {
    "easy": EasyTask(),
    "medium": MediumTask(),
    "hard": HardTask()
}

for name, task in tasks.items():
    score = task.evaluate(agent)
    print(f"{name} score: {score:.2f}")