from fastapi import FastAPI, Body, HTTPException
from env.environment import SupportOpsEnv
from env.models import Action, StepOutput

app = FastAPI()
env = SupportOpsEnv()

@app.get("/")
def root():
    return {"status": "ok", "message": "SupportOpsEnv server is running."}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/reset")
def reset():
    state = env.reset()
    return state

@app.post("/step")
def step(payload: dict = Body(...)):
    if "action" in payload and isinstance(payload["action"], dict):
        action_data = payload["action"]
    else:
        action_data = payload

    try:
        action = Action(**action_data)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    try:
        result: StepOutput = env.step(action)
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=7860)
