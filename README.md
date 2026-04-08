# SupportOpsEnv

A real-world OpenEnv environment for customer support operations optimization, simulating ticket triage and response management in a support center.

## Overview and Motivation

Customer support teams face challenges with ticket overload, slow response times, and customer churn. This environment provides a realistic simulation where AI agents learn to optimize support operations by prioritizing tickets, selecting appropriate response types, and managing SLA deadlines to minimize churn while maximizing efficiency.

## Action and Observation Spaces

### Observation Space
- `ticket_text`: String description of the customer issue
- `urgency_level`: Integer (0-2) indicating ticket priority
- `customer_tier`: Integer (0-1) representing customer value
- `waiting_time`: Float hours the ticket has been waiting
- `churn_risk`: Float (0-1) probability of customer churn
- `sla_deadline`: Float hours remaining before SLA violation

### Action Space
- `assign_priority`: Integer (0-2) to set ticket priority
- `response_type`: Integer (0-2) for response strategy (0=ignore, 1=standard, 2=escalated)

## Tasks

The environment includes three tasks of increasing difficulty:

- **Easy Task**: Basic priority assignment and response selection with lenient scoring
- **Medium Task**: Balanced evaluation of reward, churn reduction, and SLA compliance
- **Hard Task**: Strict performance requirements emphasizing high rewards and minimal violations

## Setup and Usage

### Installation
```bash
pip install -r requirements.txt
```

### Run Baseline Evaluation
```bash
python -m scripts.run_baseline
```

### Run Interactive Demo
```bash
python -m app.app
```

### Run LLM Evaluation
```bash
export HF_TOKEN=your_huggingface_token
python inference.py
```

## Baseline Performance Scores

- Easy Task: 0.95
- Medium Task: 0.81
- Hard Task: 0.97

## Features

- OpenEnv API compliance with Pydantic models
- Multi-step decision making with continuous rewards
- SLA constraint enforcement
- Churn risk modeling
- Three difficulty levels with programmatic graders
- Baseline agent implementation
- Interactive Gradio UI
- Containerized deployment ready  

🔹 Tech Stack

Python
Pydantic
Gradio