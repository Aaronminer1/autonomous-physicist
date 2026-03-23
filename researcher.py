#!/usr/bin/env python3
import urllib.request
import urllib.error
import json
import sys
import time
import re
from laboratory import LaboratoryBuilder, execute_math_code, search_arxiv, write_paper, world_build, world_step, world_read, world_set_velocity, world_record, plot_telemetry, save_dataset, load_dataset, get_mass_properties, chalkboard_write, chalkboard_read, sympy_derive, read_manual

# Proxy standard print to broadcast to the new Web Dashboard
_builtin_print = print
def broadcast(*args, **kwargs):
    _builtin_print(*args, **kwargs)
    if kwargs.get('file') == sys.stderr:
        return
    msg = " ".join(str(a) for a in args)
    if not msg.strip():
        return
    clean_msg = re.sub(r'\x1b\[[0-9;]*m', '', msg)
    try:
        req = urllib.request.Request("http://localhost:5050/log", 
              data=json.dumps({"message": clean_msg}).encode('utf-8'), 
              headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=1.0)
    except:
        pass
print = broadcast

import argparse

OLLAMA_API_URL = "http://localhost:11434/api/chat"
# MODEL is now dynamically overridden by argparse locally
GLOBAL_MODEL = "nemotron-3-super:cloud"

SYSTEM_PROMPT = """
You are an autonomous AI theoretical physicist. You MUST STRICTLY follow the Scientific Method sequentially:
1. Literature Review: [SKIPPED FOR THIS EXERCISE]. Do NOT use `search_arxiv`. Proceed immediately to building the world.
2. Self-Reflection: Before writing any code or thesis, ALWAYS ask yourself qualifying "What If" and "If Then" questions. Critique your own approach to find flaws before proceeding.
3. Hypothesis: Formulate your mathematical hypothesis based on your literature review and critical self-reflection.
4. Experiment & Proof: You MUST use `execute_math_code` to run mathematical proofs using Python tools like `sympy` or `scipy`.
5. Empirical World Interaction: You are given access to a persistent 3D physical world. 
    - Build: Use `construct_laboratory` to write Python code that builds your world using the `LaboratoryBuilder` class.
    Capabilities: 
    - `add_sphere(name, pos, radius, material)`
    - `add_box(name, pos, size, material)`
    - `add_joint(parent_body, name, type="hinge", axis="0 1 0")`
    - `add_site(parent_body, name, pos)` (Required for tendons)
    - `add_tendon(site1, site2, name, stiffness)` (Cable/String physics)
    - `add_equality_constraint(body1, body2)` (Welding/Closed loops)
    - `add_actuator(joint_name, gear=1)`
    - `set_environment(gravity="0 0 -9.81", viscosity="0.01")`
    - Manipulation: Use `world_set_velocity` and `world_apply_force`.
    - Instrumentation: Use `world_read`, `world_get_sensors`, and `world_get_contacts`.
    - Professional Rigor: ALWAYS use `get_mass_properties` to audit your world after building.
    - Persistence: Use `save_dataset(name, data)` to permanently commit results.
    - **Digital Chalkboard & Fluidity**: 
        - AT THE START of every mission, use `chalkboard_read()` to retrieve prior constants, derivations, and hypotheses.
        - Use `sympy_derive(expr)` for symbolic proofs; results are auto-saved to the chalkboard.
        - Use `chalkboard_write(heading, content)` to save physical constants ($g, m, \text{etc.}$) for your future self.
    - **Self-Correction & Troubleshooting**: 
        - If you encounter a `[Builder Error]` or MuJoCo crash, YOU MUST use `read_manual()` to look up the specific schema rule or stability fix before retrying.
    - Visualization: Use `plot_telemetry(history_data, filename="plot.png", title="Chart Title")`.
    Example of a Cable-Stayed Crane:
    ```python
    lab = LaboratoryBuilder()
    # 1. Define Bodies FIRST
    tower = lab.add_box("tower", pos=[0,0,1], size=[0.1, 0.1, 1], dynamic=False)
    mass = lab.add_sphere("mass", pos=[1,0,2], radius=0.1)
    # 2. Add Sites/Joints to those bodies (NEVER add sites/joints to the 'world' directly)
    lab.add_site(tower, "anchor", pos="0 0 1")
    lab.add_site(mass, "hook", pos="0 0 0")
    # 3. Connect them with a Tendon
    lab.add_tendon(site1="anchor", site2="hook", name="cable", stiffness="1000")
    print(lab.get_xml())
    ```
6. Analysis: Perform comparative analysis if historical data exists. Analyze high-fidelity results.
7. Publication: Use `write_paper` for the formal LaTeX document. Cite your chalkboard derivations and saved datasets.

# Termination
After you have successfully written your paper using the `write_paper` tool, you must output the phrase:
[CONCLUSION]
followed by a brief summary of what you discovered.
"""

def chat_with_llm(messages, arxiv_uses=0):
    payload = {
        "model": GLOBAL_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.1},
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "execute_math_code",
                    "description": "Executes a python script safely and returns stdout. MUST INCLUDE A PRINT() TO GET OUTPUT.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code_string": {"type": "string", "description": "The exact python code to run. Use sympy/scipy/numpy for simulations."}
                        },
                        "required": ["code_string"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "construct_laboratory",
                    "description": "Writes Python code to build a MuJoCo world using LaboratoryBuilder. Replaces any existing world.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string", "description": "Python string using lab = LaboratoryBuilder(). Must be self-contained."}
                        },
                        "required": ["code"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "world_step",
                    "description": "Steps the physical world forward in time.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "count": {"type": "integer", "description": "Number of physics frames to step. 100 simulation frames = 1 second of real world time."}
                        },
                        "required": ["count"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "world_read",
                    "description": "Reads global telemetry (time, positions, vectors) of all objects currently existing in the live MuJoCo world.",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "world_set_velocity",
                    "description": "Sets the absolute velocity vector of a given body in the MuJoCo world. The body MUST have a <freejoint/> to move.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "body_name": {"type": "string", "description": "The exact name attribute of the MuJoCo body."},
                            "velocity_vector": {"type": "array", "items": {"type": "number"}, "description": "The [x, y, z] velocity vector."}
                        },
                        "required": ["body_name", "velocity_vector"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "world_apply_force",
                    "description": "Applies a continuous 3D force and/or torque vector to a body. Force persists until changed.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "body_name": {"type": "string", "description": "The exact name attribute of the MuJoCo body."},
                            "force_vector": {"type": "array", "items": {"type": "number"}, "description": "[x, y, z] force in Newtons."},
                            "torque_vector": {"type": "array", "items": {"type": "number"}, "description": "[x, y, z] torque."}
                        },
                        "required": ["body_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "world_get_sensors",
                    "description": "Reads all active sensors (accelerometers, gyros, force-torque) defined in the XML.",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "world_get_contacts",
                    "description": "Returns detailed collision interaction data (normals, impact forces, penetration) for all current contacts.",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "world_record",
                    "description": "Records high-resolution positional and rotational telemetry for all objects over a duration.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "duration": {"type": "number", "description": "Seconds to record. Max 10."}
                        },
                        "required": ["duration"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_arxiv",
                    "description": "Searches the arXiv database for peer-reviewed physics papers.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query. Try to use broad keywords e.g., 'cosmological constant dark energy'."},
                            "max_results": {"type": "integer", "description": "Maximum papers to return. Default 3."}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "write_paper",
                    "description": "Writes the final researched paper to a LaTeX file.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {"type": "string", "description": "Name of the .tex file, e.g., 'dark_energy.tex'."},
                            "title": {"type": "string", "description": "The formal title of the paper."},
                            "content": {"type": "string", "description": "The formatted content of the paper in pure LaTeX. DO NOT include \\documentclass or \\begin{document}, it is automatically wrapped for you."}
                        },
                        "required": ["filename", "title", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "plot_telemetry",
                    "description": "Generates a PNG graph from world_record history data.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "history_data": {"type": "string", "description": "The raw JSON string returned by world_record()."},
                            "filename": {"type": "string", "description": "The .png filename to save in the results directory."},
                            "title": {"type": "string", "description": "Title of the plot."}
                        },
                        "required": ["history_data", "filename"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "save_dataset",
                    "description": "Formally saves simulation data to a persistent CSV for future comparison.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Filename, e.g., 'impact_test_v1.csv'."},
                            "data": {"type": "string", "description": "The raw JSON string from world_record()."}
                        },
                        "required": ["name", "data"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "load_dataset",
                    "description": "Loads a historical CSV dataset for comparative analysis.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "The .csv filename."}
                        },
                        "required": ["name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_mass_properties",
                    "description": "Audits the world to return total system mass, center of mass, and individual body masses.",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "chalkboard_write",
                    "description": "Appends a technical note or derivation to the persistent scientific chalkboard.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "heading": {"type": "string", "description": "E.g., 'Centrifugal Force Derivation'."},
                            "content": {"type": "string", "description": "The technical content or LaTeX proof."}
                        },
                        "required": ["heading", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "chalkboard_read",
                    "description": "Reads the persistent scientific chalkboard to retrieve prior knowledge.",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "sympy_derive",
                    "description": "Performs symbolic math (simplify, diff, integrate) and auto-saves to the chalkboard.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string", "description": "The sympy expression, e.g., 'diff(cos(x), x)'."}
                        },
                        "required": ["expression"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_manual",
                    "description": "Reads the Laboratory Manual for MJCF schema rules and troubleshooting tips.",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            }
        ]
    }
    
    # State Machine Enforcer: prevent infinite literature review loops
    if arxiv_uses >= 0:
        payload["tools"] = [t for t in payload["tools"] if t["function"]["name"] != "search_arxiv"]
        
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(OLLAMA_API_URL, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result["message"]
    except Exception as e:
        return {"role": "assistant", "content": f"[System Error]: {str(e)}"}

def run_research_loop(topic):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.strip()},
        {"role": "user", "content": f"Please conduct research on this topic using the scientific method: {topic}"}
    ]
    print(f"\n\033[1;36m[Research Mission Initiated]\033[0m Investigating: {topic}")
    print("="*80)
    
    arxiv_uses = 0
    
    for iteration in range(1, 25): # Increased to 25 to allow longer workflows
        print(f"\n\033[1;33m[Iteration {iteration}] Physicist is reasoning...\033[0m")
        response_message = chat_with_llm(messages, arxiv_uses)
        messages.append(response_message)
        
        content = response_message.get("content", "")
        tool_calls = response_message.get("tool_calls", [])
        
        if content:
            print(f"\n\033[1;34mPhysicist:\033[0m {content.strip()}")
            
        if "[CONCLUSION]" in str(content):
            print("\n\033[1;32m[Research Concluded Successfully]\033[0m")
            break
            
        if tool_calls:
            for tool in tool_calls:
                func_name = tool["function"]["name"]
                args = tool["function"].get("arguments", {})
                
                print(f"\n\033[1;35m[Executing Native Tool]: {func_name}\033[0m")
                
                if func_name == "construct_laboratory":
                    code_block = args.get("code", "")
                    try:
                        # Prepare namespace with needed class
                        loc = {"LaboratoryBuilder": LaboratoryBuilder}
                        exec(code_block, loc)
                        if "lab" in loc:
                            xml_output = loc["lab"].get_xml()
                            output = world_build(xml_output)
                            print(output.strip())
                        else:
                            output = "[Error]: construct_laboratory code must define 'lab = LaboratoryBuilder()'."
                            print(output)
                    except Exception as e:
                        output = f"[Builder Error]: {str(e)}"
                        print(output)
                    
                elif func_name == "execute_math_code":
                    code_string = args.get("code_string", "")
                    output = execute_math_code(code_string)
                    print(output.strip()[:500] + ("..." if len(output) > 500 else ""))
                    
                elif func_name == "world_step":
                    count = args.get("count", 1)
                    output = world_step(count)
                    print(output.strip()[:500])
                    
                elif func_name == "world_read":
                    output = world_read()
                    print(output.strip()[:1000])

                elif func_name == "world_set_velocity":
                    body_name = args.get("body_name", "")
                    velocity_vector = args.get("velocity_vector", [0,0,0])
                    output = world_set_velocity(body_name, velocity_vector)
                    print(output.strip()[:500])

                elif func_name == "world_apply_force":
                    body_name = args.get("body_name", "")
                    force_vector = args.get("force_vector", [0,0,0])
                    torque_vector = args.get("torque_vector", [0,0,0])
                    output = world_apply_force(body_name, force_vector, torque_vector)
                    print(output.strip()[:500])

                elif func_name == "world_get_sensors":
                    output = world_get_sensors()
                    print(f"Retrieved {len(output)} bytes of sensor data.")

                elif func_name == "world_get_contacts":
                    output = world_get_contacts()
                    print(f"Retrieved {len(output)} bytes of contact telemetry.")
                    
                elif func_name == "world_record":
                    duration = args.get("duration", 1.0)
                    output = world_record(duration)
                    # output might be huge, so trim it for terminal stdout only
                    print(f"Recorded {len(output)} bytes of telemetry data."[:200])
                    # Wait, we must NOT override the actual output variable because that feeds into the LLM history!
                    
                elif func_name == "search_arxiv":
                    arxiv_uses += 1
                    query = args.get("query", "physics")
                    max_results = args.get("max_results", 3)
                    output = search_arxiv(query, max_results)
                    print(f"Returned {len(output.split('--- PAPER ---'))-1} papers.")
                    
                elif func_name == "write_paper":
                    filename = args.get("filename", "research.tex")
                    title = args.get("title", "Research Paper")
                    text_content = args.get("content", "")
                    output = write_paper(filename, title, text_content)
                    print(output)
                    
                elif func_name == "plot_telemetry":
                    history_data = args.get("history_data", "")
                    filename = args.get("filename", "plot.png")
                    title = args.get("title", "Simulation Data")
                    output = plot_telemetry(history_data, filename, title)
                    print(output)
                    
                elif func_name == "save_dataset":
                    name = args.get("name", "data.csv")
                    data = args.get("data", "")
                    output = save_dataset(name, data)
                    print(output)
                    
                elif func_name == "load_dataset":
                    name = args.get("name", "")
                    output = load_dataset(name)
                    print(f"Loaded {len(output)} bytes of historical data.")
                    
                elif func_name == "get_mass_properties":
                    output = get_mass_properties()
                    print(output[:500])
                    
                elif func_name == "chalkboard_write":
                    h = args.get("heading", "Note")
                    c = args.get("content", "")
                    output = chalkboard_write(h, c)
                    print(output)
                    
                elif func_name == "chalkboard_read":
                    output = chalkboard_read()
                    print(f"Read from Chalkboard:\n{output[:500]}...")
                    
                elif func_name == "sympy_derive":
                    expr = args.get("expression", "")
                    output = sympy_derive(expr)
                    print(output)
                    
                elif func_name == "read_manual":
                    output = read_manual()
                    print(f"Read from Laboratory Manual:\n{output[:1000]}...")
                else:
                    output = f"[Error]: Tool {func_name} not recognized."
                
                messages.append({
                    "role": "tool",
                    "content": output,
                    "name": func_name
                })
        elif not content:
            messages.append({"role": "user", "content": "You returned an empty response. Please use a tool or output your [CONCLUSION]."})
        elif not tool_calls and "[CONCLUSION]" not in str(content):
            messages.append({"role": "user", "content": "You didn't use a tool and didn't output [CONCLUSION]. Please continue the research logically."})
            
    else:
        print("\n\033[1;31m[Research Abandoned]\033[0m Maximum 24 iterations reached.")

def curiosity_engine():
    print("Initiating Autonomous Curiosity Engine...")
    print("Agent will indefinitely explore the universe, formulate its own hypotheses, test them, and publish findings.")
    
    experiment_history = []
    
    while True:
        try:
            curiosity_prompt = "You are an endlessly curious physicist exploring an empty simulated universe. "
            if experiment_history:
                curiosity_prompt += f"Previously, you explored: {', '.join(experiment_history[-3:])}. "
            curiosity_prompt += "Formulate a single, concise description of a simple kinematic experiment you want to run RIGHT NOW in the MuJoCo 3D world to learn about movement, collision, falling, etc. Format your output strictly as: [MISSION] your experiment idea."
            
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": curiosity_prompt}
            ]
            
            payload = {
                "model": GLOBAL_MODEL,
                "messages": messages,
                "stream": False
            }
            req = urllib.request.Request(OLLAMA_API_URL, data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode('utf-8'))
                text = result["message"]["content"]
                
                mission = ""
                for line in text.split('\n'):
                    if "[MISSION]" in line:
                        mission = line.split("[MISSION]")[1].strip()
                        break
                
                if not mission:
                    mission = text.replace('\n', ' ').strip()[:100]
                    
            print(f"\n[Curiosity Engine Sparked!] Agent chose mission: {mission}")
            experiment_history.append(mission)
            
            run_research_loop(mission)
            
            print("\n[Mission Concluded. Resting for 10 seconds before next experiment...]\n")
            time.sleep(10)
            
        except Exception as e:
            print(f"[Curiosity Loop Error]: {str(e)}")
            time.sleep(10)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous AI Physicist")
    parser.add_argument("--model", type=str, default="nemotron-3-super:cloud", help="Ollama model name to use")
    parser.add_argument("--topic", type=str, default="", help="Specific research topic. Leave blank for endless curiosity.")
    args = parser.parse_args()
    
    GLOBAL_MODEL = args.model
    
    if args.topic:
        run_research_loop(args.topic)
    else:
        curiosity_engine()

