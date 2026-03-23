import subprocess
import os
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

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
    import requests
    try:
        r = requests.post("http://localhost:5050/velocity", json={"body_name": body_name, "velocity": velocity_vector}, timeout=5)
        return str(r.json())
    except Exception as e:
        return f"[Tool Error]: {str(e)}"

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
        
    filepath = os.path.join(os.getcwd(), filename)
    latex_template = f"\\documentclass{{article}}\n\\usepackage[utf8]{{inputenc}}\n\\usepackage{{amsmath}}\n\\usepackage{{amssymb}}\n\\title{{{title}}}\n\\author{{Autonomous AI Physicist}}\n\\begin{{document}}\n\\maketitle\n\n{content}\n\n\\end{{document}}\n"
    
    try:
        with open(filepath, "w") as f:
            f.write(latex_template)
        return f"Successfully wrote paper to {filepath}!"
    except Exception as e:
        return f"[File Write Error]: {str(e)}"
