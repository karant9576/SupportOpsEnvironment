import gradio as gr
from env.environment import SupportOpsEnv
from env.models import Action

env = SupportOpsEnv()
env.reset()

def step(priority, response):
    result = env.step(Action(
        assign_priority=int(priority),
        response_type=int(response)
    ))

    return (
        result.observation.dict(),
        result.reward.value,
        result.done
    )

def reset():
    return env.reset().dict()

with gr.Blocks() as demo:
    gr.Markdown("# SupportOps Environment")

    priority = gr.Slider(0, 2, step=1, label="Priority")
    response = gr.Slider(0, 2, step=1, label="Response")

    btn = gr.Button("Step")
    reset_btn = gr.Button("Reset")

    state = gr.JSON()
    reward = gr.Number()
    done = gr.Text()

    btn.click(step, [priority, response], [state, reward, done])
    reset_btn.click(reset, outputs=[state])

demo.launch()