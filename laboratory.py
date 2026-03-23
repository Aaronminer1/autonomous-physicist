import subprocess
import os
import urllib.request
import json
import time
import urllib.parse
import xml.etree.ElementTree as ET

SERVER_URL = "http://localhost:5050"

class LaboratoryBuilder:
    """Helper class to build MuJoCo MJCF XML programmatically."""
    def __init__(self):
        self.xml = ET.Element("mujoco")
        self.xml.set("model", "autonomous_physics_lab")
        ET.SubElement(self.xml, "compiler", angle="degree")
        ET.SubElement(self.xml, "option", gravity="0 0 -9.81", timestep="0.002", integrator="RK4")
        
        visual = ET.SubElement(self.xml, "visual")
        ET.SubElement(visual, "global", offwidth="640", offheight="480")
        
        self.world = ET.SubElement(self.xml, "worldbody")
        ET.SubElement(self.world, "light", pos="0 0 5", dir="0 0 -1", directional="true")
        ET.SubElement(self.world, "geom", name="floor", type="plane", size="0 0 1", rgba=".8 .9 .8 1")
        
        # Internal registries for name-to-element lookup
        self.bodies = {"world": self.world}
        self.joints = {}
        self.sites = {}
        
        # Predefined materials
        self.materials = {
            "steel": {"density": "7850", "friction": "0.3", "rgba": ".5 .5 .5 1"},
            "rubber": {"density": "1100", "friction": "1.0", "rgba": ".2 .2 .2 1", "solref": "0.02 1", "solimp": "0.9 0.95 0.001"},
            "ice": {"density": "917", "friction": "0.01", "rgba": ".9 .9 1 .6"},
            "wood": {"density": "700", "friction": "0.5", "rgba": ".6 .4 .2 1"}
        }

    def _resolve(self, ref, registry, type_name):
        if isinstance(ref, ET.Element):
            return ref
        if ref in registry:
            return registry[ref]
        raise ValueError(f"{type_name} '{ref}' not found. Ensure it was added before referencing.")

    def set_environment(self, gravity="0 0 -9.81", viscosity="0", density="0"):
        """Modifies global physical constants."""
        opt = self.xml.find("option")
        if opt is not None:
            opt.set("gravity", str(gravity))
            opt.set("viscosity", str(viscosity))
            opt.set("density", str(density))

    def add_object(self, obj_type, name, pos, size, material="steel", dynamic=True):
        if isinstance(pos, (list, tuple)): pos = f"{pos[0]} {pos[1]} {pos[2]}"
        if isinstance(size, (list, tuple)): size = " ".join(map(str, size))
        
        body = ET.SubElement(self.world, "body", name=name, pos=str(pos))
        self.bodies[name] = body
        if dynamic:
            ET.SubElement(body, "freejoint")
        
        mat = self.materials.get(material, self.materials["steel"])
        geom = ET.SubElement(body, "geom", 
                             type=obj_type, 
                             size=str(size), 
                             rgba=mat["rgba"], 
                             density=str(mat["density"]),
                             friction=str(mat.get("friction", "0.5")))
        if "solref" in mat: 
            geom.set("solref", str(mat["solref"]))
            geom.set("solimp", str(mat["solimp"]))
        return body

    def add_sphere(self, name, pos, radius, material="steel", dynamic=True):
        return self.add_object("sphere", name, pos, [radius], material, dynamic)

    def add_box(self, name, pos, size, material="steel", dynamic=True):
        return self.add_object("box", name, pos, size, material, dynamic)

    def add_joint(self, parent_body, name, type="hinge", pos="0 0 0", axis="0 1 0"):
        """Adds a joint to a body (useful for chains/pendulums)."""
        parent = self._resolve(parent_body, self.bodies, "Body")
        if parent == self.world:
            raise ValueError("Cannot add a joint directly to the world. Add it to a body.")
        joint = ET.SubElement(parent, "joint", name=name, type=str(type), pos=str(pos), axis=str(axis))
        self.joints[name] = joint
        return joint

    def add_site(self, parent_body, name, pos="0 0 0", size="0.01", rgba="1 0 0 1"):
        """Adds an anchor point (site) to a body, often used for tendons."""
        parent = self._resolve(parent_body, self.bodies, "Body")
        if parent == self.world:
            raise ValueError("Cannot add a site directly to the world. Add it to a body.")
        site = ET.SubElement(parent, "site", name=name, pos=str(pos), size=str(size), rgba=str(rgba))
        self.sites[name] = site
        return site

    def add_tendon(self, site1, site2, name, stiffness="0", damping="0"):
        """Adds a fixed tendon between two sites."""
        tendon_root = self.xml.find("tendon")
        if tendon_root is None:
            tendon_root = ET.SubElement(self.xml, "tendon")
        t = ET.SubElement(tendon_root, "fixed", name=name, stiffness=str(stiffness), damping=str(damping))
        ET.SubElement(t, "site", site=str(site1))
        ET.SubElement(t, "site", site=str(site2))
        return t

    def add_equality_constraint(self, body1, body2, type="connect", name=None):
        """Adds an equality constraint between two bodies (e.g. welding them)."""
        equality = self.xml.find("equality")
        if equality is None:
            equality = ET.SubElement(self.xml, "equality")
        
        # Verify bodies exist
        self._resolve(body1, self.bodies, "Body")
        self._resolve(body2, self.bodies, "Body")
        
        return ET.SubElement(equality, type, name=name or f"const_{body1}_{body2}", body1=body1, body2=body2)

    def add_actuator(self, joint_name, gear="1"):
        """Adds a motor actuator to a specific joint."""
        actuators = self.xml.find("actuator")
        if actuators is None:
            actuators = ET.SubElement(self.xml, "actuator")
        
        # Verify joint exists
        self._resolve(joint_name, self.joints, "Joint")
        
        return ET.SubElement(actuators, "motor", joint=joint_name, gear=gear)

    def get_xml(self):
        return ET.tostring(self.xml, encoding='unicode')

def plot_telemetry(history_data, filename="telemetry_plot.png", title="Simulation Telemetry"):
    """
    Generates a scientific plot from world_record history data.
    history_data: the list of frames returned by world_record()
    """
    import matplotlib.pyplot as plt
    try:
        if isinstance(history_data, str):
            history_data = json.loads(history_data.replace("'", '"'))
        
        history = history_data.get("history", [])
        if not history: return "[Error]: No history data found to plot."

        times = [h["time"] for h in history]
        bodies = {}
        
        for h in history:
            for b in h["bodies"]:
                name = b["name"]
                if name not in bodies: bodies[name] = {"x": [], "y": [], "z": []}
                pos = b["position"]
                bodies[name]["x"].append(pos[0])
                bodies[name]["y"].append(pos[1])
                bodies[name]["z"].append(pos[2])

        plt.figure(figsize=(10, 6))
        for name, data in bodies.items():
            plt.plot(times, data["z"], label=f"{name} (Z-pos)")
        
        plt.xlabel("Time (s)")
        plt.ylabel("Position (m)")
        plt.title(title)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        
        results_dir = os.path.join(os.getcwd(), "results")
        if not os.path.exists(results_dir): os.makedirs(results_dir)
        filepath = os.path.join(results_dir, filename)
        
        plt.savefig(filepath)
        plt.close()
        return f"Successfully saved plot to {filepath}"
    except Exception as e:
        return f"[Plot Error]: {str(e)}"

def save_dataset(name, data):
    """
    Saves a dictionary or list of data to a CSV in results/datasets/.
    Useful for preserving experimental results for future comparison.
    """
    import csv
    try:
        datasets_dir = os.path.join(os.getcwd(), "results", "datasets")
        if not os.path.exists(datasets_dir): os.makedirs(datasets_dir)
        
        if not name.endswith(".csv"): name += ".csv"
        filepath = os.path.join(datasets_dir, name)
        
        if isinstance(data, str): data = json.loads(data.replace("'", '"'))
        # If it's the output of world_record, extract the history
        if isinstance(data, dict) and "history" in data: data = data["history"]
        
        if not data: return "[Error]: No data provided to save."
        
        # Flatten history for CSV
        keys = data[0].keys()
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(data)
        return f"Successfully saved dataset to {filepath}"
    except Exception as e:
        return f"[Save Error]: {str(e)}"

def load_dataset(name):
    """Loads a previously saved CSV dataset from results/datasets/."""
    import csv
    try:
        if not name.endswith(".csv"): name += ".csv"
        filepath = os.path.join(os.getcwd(), "results", "datasets", name)
        if not os.path.exists(filepath): return f"[Error]: Dataset {name} not found."
        
        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            return json.dumps(list(reader))
    except Exception as e:
        return f"[Load Error]: {str(e)}"

def get_mass_properties():
    """Returns the total mass, Center of Mass (COM), and Inertia of the world."""
    import requests
    try:
        r = requests.get("http://localhost:5050/mass", timeout=5)
        return str(r.json())
    except Exception as e:
        return f"[Tool Error]: {str(e)}"

def execute_math_code(code_string):
    """Executes python code safely in a subprocess and returns stdout."""
    filename = "temp_math_sandbox.py"
    with open(filename, "w") as f:
        f.write(code_string)
        
    try:
        result = subprocess.run(
            ["python3", filename],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout
        if result.stderr:
            output += "\n[Error Output from Exception]:\n" + result.stderr
        if not output.strip():
            output = "[Execution complete, but no output. Did you print() the results?]"
        return output
    except subprocess.TimeoutExpired:
        return "[Tool Error]: Execution timed out after 15 seconds."
    except Exception as e:
        return f"[Tool Error]: {str(e)}"
    finally:
        if os.path.exists(filename):
            os.remove(filename)

def chalkboard_write(heading, content):
    """
    Appends a technical note, derivation, or constant to the persistent results/chalkboard.md.
    Use this to store 'Working Memory' for future researcher iterations.
    """
    try:
        results_dir = os.path.join(os.getcwd(), "results")
        if not os.path.exists(results_dir): os.makedirs(results_dir)
        filepath = os.path.join(results_dir, "chalkboard.md")
        
        with open(filepath, "a") as f:
            f.write(f"\n## {heading}\n{content}\n")
        return f"Successfully added '{heading}' to the Chalkboard."
    except Exception as e:
        return f"[Chalkboard Error]: {str(e)}"

def chalkboard_read():
    """Reads the entire persistent chalkboard.md file."""
    try:
        filepath = os.path.join(os.getcwd(), "results", "chalkboard.md")
        if not os.path.exists(filepath): return "The Chalkboard is currently empty."
        with open(filepath, "r") as f:
            return f.read()
    except Exception as e:
        return f"[Chalkboard Error]: {str(e)}"

def read_manual():
    """Reads the laboratory manual for schema rules and troubleshooting."""
    try:
        filepath = os.path.join(os.getcwd(), "results", "lab_manual.md")
        if not os.path.exists(filepath): return "Manual not found. Ensure results/lab_manual.md exists."
        with open(filepath, "r") as f:
            return f.read()
    except Exception as e:
        return f"[Manual Error]: {str(e)}"

def sympy_derive(expression, operation="simplify"):
    """
    Uses SymPy to perform symbolic derivations (simplify, expand, integrate, differentiate).
    Returns LaTeX-formatted output and auto-saves to the Chalkboard.
    Example: sympy_derive("diff(sin(x)*exp(x), x)", "differentiate")
    """
    import sympy
    try:
        # Simple dynamic eval for sympy
        x, y, z, t = sympy.symbols('x y z t')
        f, g = sympy.symbols('f g', cls=sympy.Function)
        
        res = eval(f"sympy.{expression}")
        latex_res = sympy.latex(res)
        
        output = f"Result: {str(res)}\nLaTeX: $${latex_res}$$"
        chalkboard_write(f"Symbolic Derivation: {expression}", output)
        return output
    except Exception as e:
        return f"[Sympy Error]: {str(e)}"

def world_build(xml_string):
    """Sends XML to world_server.py to completely build and replace the physical universe."""
    import requests
    try:
        r = requests.post("http://localhost:5050/build_world", json={"xml": xml_string}, timeout=5)
        return str(r.json())
    except requests.exceptions.ConnectionError:
        return "[Tool Error]: world_server.py is not running. Instruct the user to run it."
    except Exception as e:
        return f"[Tool Error]: {str(e)}"

def world_step(count=1):
    """Steps the MuJoCo simulation forward by `count` timesteps (e.g. count=100 is 1 second). Returns status."""
    import requests
    try:
        r = requests.post("http://localhost:5050/step", json={"count": count}, timeout=30)
        return str(r.json())
    except Exception as e:
        return f"[Tool Error]: {str(e)}"

def world_read():
    """Reads telemetry (positions, velocities) from the active continuous MuJoCo World."""
    import requests
    try:
        r = requests.get("http://localhost:5050/read", timeout=5)
        return str(r.json())
    except Exception as e:
        return f"[Tool Error]: {str(e)}"

def world_set_velocity(body_name, velocity_vector):
    """Sets the absolute velocity ([vx, vy, vz]) of a given body in the MuJoCo world."""
    try:
        req = urllib.request.Request(f"{SERVER_URL}/velocity", 
              data=json.dumps({"body_name": body_name, "velocity": velocity_vector}).encode('utf-8'), 
              headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as r:
            result = json.loads(r.read().decode('utf-8'))
            return result.get("status", result.get("error", "Unknown error"))
    except Exception as e:
        return f"[Error]: {str(e)}"

def world_apply_force(body_name, force_vector=[0,0,0], torque_vector=[0,0,0]):
    """Applies a force and/or torque to a given body in the MuJoCo world."""
    try:
        req = urllib.request.Request(f"{SERVER_URL}/force", 
              data=json.dumps({"body_name": body_name, "force": force_vector, "torque": torque_vector}).encode('utf-8'), 
              headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as r:
            result = json.loads(r.read().decode('utf-8'))
            return result.get("status", result.get("error", "Unknown error"))
    except Exception as e:
        return f"[Error]: {str(e)}"

def world_get_sensors():
    """Reads sensor data from the MuJoCo world."""
    try:
        with urllib.request.urlopen(f"{SERVER_URL}/sensors") as r:
            return r.read().decode('utf-8')
    except Exception as e:
        return f"[Error]: {str(e)}"

def world_get_contacts():
    """Reads contact information from the MuJoCo world."""
    try:
        with urllib.request.urlopen(f"{SERVER_URL}/contacts") as r:
            return r.read().decode('utf-8')
    except Exception as e:
        return f"[Error]: {str(e)}"

def world_record(duration=1.0):
    """Records continuous telemetry from the MuJoCo world for the specified duration (seconds)."""
    import requests
    try:
        r = requests.post("http://localhost:5050/record", json={"duration": float(duration)}, timeout=15)
        return str(r.json())
    except Exception as e:
        return f"[Tool Error]: {str(e)}"

def search_arxiv(query, max_results=3):
    """Searches arXiv for papers matching the query and returns title, authors, and abstract."""
    safe_query = urllib.parse.quote(query.replace(' ', '+'))
    url = f"http://export.arxiv.org/api/query?search_query=all:{safe_query}&start=0&max_results={int(max_results)}"
    
    try:
        response = urllib.request.urlopen(url)
        xml_data = response.read()
        root = ET.fromstring(xml_data)
        
        namespace = {'atom': 'http://www.w3.org/2005/Atom'}
        entries = root.findall('atom:entry', namespace)
        
        if not entries:
            return "No papers found on arXiv for that query."
            
        results = []
        for entry in entries:
            title = entry.find('atom:title', namespace).text.replace('\n', ' ').strip()
            summary = entry.find('atom:summary', namespace).text.replace('\n', ' ').strip()
            authors = [author.find('atom:name', namespace).text for author in entry.findall('atom:author', namespace)]
            authors_str = ", ".join(authors)
            results.append(f"Title: {title}\nAuthors: {authors_str}\nAbstract: {summary}\n")
            
        return "\n--- PAPER ---\n".join(results)
    except Exception as e:
        return f"[ArXiv API Error]: {str(e)}"

def write_paper(filename, title, content):
    """Saves the final academic paper to disk as a LaTeX file."""
    if not filename.endswith(".tex"):
        filename += ".tex"
        
    results_dir = os.path.join(os.getcwd(), "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    filename = os.path.basename(filename)
    filepath = os.path.join(results_dir, filename)
    latex_template = f"\\documentclass{{article}}\n\\usepackage[utf8]{{inputenc}}\n\\usepackage{{amsmath}}\n\\usepackage{{amssymb}}\n\\title{{{title}}}\n\\author{{Autonomous AI Physicist}}\n\\begin{{document}}\n\\maketitle\n\n{content}\n\n\\end{{document}}\n"
    
    try:
        with open(filepath, "w") as f:
            f.write(latex_template)
        return f"Successfully wrote paper to {filepath}!"
    except Exception as e:
        return f"[File Write Error]: {str(e)}"
