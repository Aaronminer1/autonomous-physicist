#!/usr/bin/env python3
import urllib.request
import urllib.error
import json
import sys
import time
import re
from laboratory import execute_math_code, search_arxiv, write_paper, world_build, world_step, world_read, world_set_velocity, world_record

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

OLLAMA_API_URL = "http://localhost:11434/api/chat"
MODEL = "nemotron-3-super:cloud"

SYSTEM_PROMPT = """
You are an autonomous AI theoretical physicist. You MUST STRICTLY follow the Scientific Method sequentially:
1. Literature Review: [SKIPPED FOR THIS EXERCISE]. Do NOT use `search_arxiv`. Proceed immediately to building the world.
2. Self-Reflection: Before writing any code or thesis, ALWAYS ask yourself qualifying "What If" and "If Then" questions. Critique your own approach to find flaws before proceeding.
3. Hypothesis: Formulate your mathematical hypothesis based on your literature review and critical self-reflection.
4. Experiment & Proof: You MUST use `execute_math_code` to run mathematical proofs using Python tools like `sympy` or `scipy`.
5. Empirical World Interaction: You are given access to a persistent 3D physical world hosted autonomously. 
   - First, use `world_build` to spawn the necessary physical objects using standard MuJoCo XML string formats. CRITICAL: If you want an object (like a sphere) to be affected by physics and gravity, you MUST include `<freejoint/>` inside its `<body ...>` tag! Otherwise, it will be a frozen statue!
   - Second, use `world_set_velocity` to physically push the bodies and give them initial momentum.
   - Third, use `world_record` to record a continuous array of telemetry (like a high-speed camera). Use this instead of `world_read` if you are expecting fast dynamic kinematics to occur over the next few seconds (e.g., dropping or bouncing).
   Do NOT write your own math scripts to derive kinematic result vectors—the persistent physics engine natively handles true gravity, friction, and collision for you seamlessly!
6. Analysis: Analyze the high-speed laboratory results recursively.
7. Publication: ONLY after you have mathematical results, use `write_paper` to write your final formal LaTeX document summarising your findings.

# Termination
After you have successfully written your paper using the `write_paper` tool, you must output the phrase:
[CONCLUSION]
followed by a brief summary of what you discovered.
"""

def chat_with_llm(messages, arxiv_uses=0):
    payload = {
        "model": MODEL,
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
                    "name": "world_build",
                    "description": "Completely builds the Persistent Physics World using an XML string. This replaces any existing world.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "xml_string": {"type": "string", "description": "The exact full MuJoCo XML (<mujoco>...</mujoco>) modeling the environment and bodies."}
                        },
                        "required": ["xml_string"]
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
                    "name": "world_record",
                    "description": "Records continuous positional telemetry for all objects over a specified duration (in seconds). Use this to watch fast physics events like falling or bouncing in high resolution.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "duration": {"type": "number", "description": "How many seconds of real-time to record. Maximum 10."}
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
                
                if func_name == "execute_math_code":
                    code_string = args.get("code_string", "")
                    output = execute_math_code(code_string)
                    print(output.strip()[:500] + ("..." if len(output) > 500 else ""))
                    
                elif func_name == "world_build":
                    xml_string = args.get("xml_string", "")
                    output = world_build(xml_string)
                    print(output.strip()[:500])
                    
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
                "model": MODEL,
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
    if len(sys.argv) > 1:
        topic = " ".join(sys.argv[1:])
        run_research_loop(topic)
    else:
        curiosity_engine()
