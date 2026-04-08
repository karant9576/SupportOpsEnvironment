import os
from openai import OpenAI

# ==============================
# Environment Variables
# ==============================

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# ==============================
# OpenAI Client
# ==============================

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

# ==============================
# Import Environment
# ==============================

from env.environment import SupportOpsEnv
from env.models import Action

# ==============================
# LLM Agent Logic
# ==============================

def get_action_from_llm(observation):
    prompt = f"""
You are a customer support AI.

Observation:
- urgency_level: {observation.urgency_level}
- churn_risk: {observation.churn_risk}
- sla_deadline: {observation.sla_deadline}

Rules:
- If urgency_level is high → assign high priority (2)
- If churn risk is high → escalate (response_type = 2)
- Otherwise choose reasonable values

Return ONLY in format:
priority,response

Example:
2,2
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
        )

        content = response.choices[0].message.content
        if content is None:
            raise ValueError("No content in LLM response")
        
        output = content.strip()
        priority, response_type = map(int, output.split(","))

        return Action(assign_priority=priority, response_type=response_type), output

    except Exception as e:
        return Action(assign_priority=0, response_type=0), str(e)

# ==============================
# Run Environment
# ==============================

def run_episode(task_name="support_ops", env_name="SupportOpsEnv"):
    env = SupportOpsEnv()

    obs = env.reset()
    done = False
    step_count = 0
    rewards = []
    success = False

    print(f"[START] task={task_name} env={env_name} model={MODEL_NAME}")

    try:
        while not done:
            step_count += 1

            action, raw_action = get_action_from_llm(obs)

            result = env.step(action)

            reward = round(result.reward.value, 2)
            rewards.append(f"{reward:.2f}")
            done = result.done

            print(
                f"[STEP] step={step_count} action={raw_action} reward={reward:.2f} done={str(done).lower()} error=null"
            )

            obs = result.observation

        success = True

    except Exception as e:
        print(
            f"[STEP] step={step_count} action=error reward=0.00 done=true error={str(e)}"
        )

    finally:
        print(
            f"[END] success={str(success).lower()} steps={step_count} rewards={','.join(rewards)}"
        )

# ==============================
# LLM Agent Class
# ==============================

class LLMAgent:
    def __init__(self):
        pass

    def act(self, obs):
        action, _ = get_action_from_llm(obs)
        return action

# ==============================
# Evaluate on Tasks
# ==============================

from env.tasks import EasyTask, MediumTask, HardTask

agent = LLMAgent()

tasks = {
    "easy": EasyTask(),
    "medium": MediumTask(),
    "hard": HardTask()
}

for name, task in tasks.items():
    score = task.evaluate(agent)
    print(f"{name} score: {score:.2f}")