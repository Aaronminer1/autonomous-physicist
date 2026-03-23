# The Autonomous AI Physicist

A completely autonomous, headless AI agent that uses the scientific method to endlessly explore, hypothesize, test, and publish findings about a persistent 3D physical universe.

## Architecture

The system is decoupled into two primary components to allow the 3D physics engine to run synchronously while the AI "thinks" asynchronously:

1. **The Persistent Universe (`world_server.py`)**: A multithreaded MuJoCo physics engine running continuously in the background. It serves a futuristic web Dashboard at `http://localhost:5050/viewer`, streams real-time MJPEG telemetry, and exposes a REST API for the AI to manipulate the world (spawning objects, setting velocities, reading high-speed positional telemetry).
2. **The Curiosity Engine (`researcher.py`)**: The AI "Brain" running a continuous ReAct (Reasoning & Acting) loop via a local Ollama LLM. The AI formulates its own experiments, uses Python math libraries to verify hypotheses, triggers the `world_server` API to execute empirical 3D observations, and finally compiles its findings into formal LaTeX (`.tex`) academic papers.
3. **The Tools Library (`laboratory.py`)**: The native Python API bridge binding the AI's LLM tool schemas to the backend `world_server` endpoints.

## Dependencies & Installation

### 1. Python Environment
Install the core Python dependencies (Flask, MuJoCo, Pillow for MJPEG streaming, and mathematical arrays):
```bash
pip install flask mujoco Pillow requests sympy scipy numpy
```

### 2. Local LLM (Ollama)
The system relies on a local LLM to run the autonomous reasoning loop without hitting API rate limits. Ensure you have Ollama installed and running.
```bash
ollama serve
ollama pull nemotron-3-super:cloud
```
*(Note: You can change the `MODEL` variable at the top of `researcher.py` if you prefer to use `llama3.2:latest` or another model).*

## How to Run

To unleash the autonomous physicist, you need to start the universe server and then boot up the AI brain.

### Step 1: Boot the Persistent Universe
In your first terminal slice, start the MuJoCo server:
```bash
cd /home/aaron/physics
python3 world_server.py
```

### Step 2: Open the Observation Dashboard
Open your web browser and navigate to:
**http://localhost:5050/viewer**

You will see the dual-panel Head-Up Display (Live Telemetry Array on the left, Autonomous Reasoning Matrix on the right).

### Step 3: Spark the Curiosity Engine
In a second terminal slice, boot the AI Physicist:
```bash
cd /home/aaron/physics
python3 -u researcher.py
```

The AI will immediately begin pondering experiments. You can watch its intricate internal reasoning, mathematical proofs, and tool execution stream dynamically into the right panel of the Web Dashboard, while the actual physics simulation plays out visually on the left panel.

## Results & Archives
All formal LaTeX papers authored by the AI (e.g., tests on Galilean free-fall, projectile superposition, thermodynamics) during its multi-hour campaigns are saved to the local directory. Previous 7-hour campaign results have been successfully archived to `results_backup/`.
