from .environment import SupportOpsEnv

class BaseTask:
    def __init__(self):
        self.env = SupportOpsEnv()

    def evaluate(self, agent):
        total_score = 0
        episodes = 5

        for _ in range(episodes):
            obs = self.env.reset()
            done = False

            trajectory = {
                "rewards": [],
                "churn": [],
                "sla_violations": 0
            }

            while not done:
                action = agent.act(obs)
                result = self.env.step(action)

                trajectory["rewards"].append(result.reward.value)
                trajectory["churn"].append(result.observation.churn_risk)

                if result.observation.sla_deadline < 0:
                    trajectory["sla_violations"] += 1

                obs = result.observation
                done = result.done

            score = self._grader(trajectory)
            total_score += score

        return total_score / episodes

    def _grader(self, trajectory):
        avg_reward = sum(trajectory["rewards"]) / len(trajectory["rewards"])
        avg_churn = sum(trajectory["churn"]) / len(trajectory["churn"])

        score = (
            0.5 * (avg_reward / 10) +
            0.3 * (1 - avg_churn) +
            0.2 * (1 - trajectory["sla_violations"])
        )

        return max(0, min(score, 1))


class EasyTask(BaseTask):
    def _grader(self, trajectory):
        avg_reward = sum(trajectory["rewards"]) / len(trajectory["rewards"])
        avg_churn = sum(trajectory["churn"]) / len(trajectory["churn"])

        score = (
            0.4 * (avg_reward / 10) +
            0.4 * (1 - avg_churn) +
            0.2 * (1 - trajectory["sla_violations"])
        )

        return max(0, min(score, 1))

class MediumTask(BaseTask):
    def _grader(self, trajectory):
        avg_reward = sum(trajectory["rewards"]) / len(trajectory["rewards"])
        avg_churn = sum(trajectory["churn"]) / len(trajectory["churn"])

        score = (
            0.5 * (avg_reward / 10) +
            0.3 * (1 - avg_churn) +
            0.2 * (1 - trajectory["sla_violations"])
        )

        return max(0, min(score, 1))

class HardTask(BaseTask):
    def _grader(self, trajectory):
        avg_reward = sum(trajectory["rewards"]) / len(trajectory["rewards"])
        avg_churn = sum(trajectory["churn"]) / len(trajectory["churn"])

        score = (
            0.6 * (avg_reward / 10) +
            0.2 * (1 - avg_churn) +
            0.2 * (1 - trajectory["sla_violations"])
        )

        return max(0, min(score, 1))