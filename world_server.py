import mujoco
from PIL import Image
import io
import time
import threading
import os
import signal
import atexit
import subprocess
import sys
from flask import Flask, request, jsonify, Response, render_template_string

# Force EGL for GPU accelerated rendering
os.environ["MUJOCO_GL"] = "egl"

app = Flask(__name__)

m = None
d = None
latest_frame = None
lock = threading.Lock()
condition = threading.Condition()
ai_logs = []
active_researcher = None
researcher_status = "STOPPED" # STOPPED, RUNNING, PAUSED

# Initialize latest_frame with a placeholder (Black 1x1 JPEG)
latest_frame = b'\xff\xd8\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xc4\x00\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xda\x00\x08\x01\x01\x00\x00\x01\x05\x02\xbf\xff\xd9'

default_xml = """<mujoco>
  <compiler angle="degree"/>
  <option gravity="0 0 -9.81" timestep="0.002" integrator="RK4"/>
  <visual>
    <global offwidth="640" offheight="480"/>
  </visual>
  <worldbody>
    <light pos="0 0 5" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="0 0 1" rgba=".8 .9 .8 1"/>
  </worldbody>
</mujoco>"""

def physics_loop():
    global m, d
    last_p_time = time.time()
    while True:
        # Get local copies of m and d to reduce lock holding time
        local_m = None
        local_d = None
        with lock:
            local_m = m
            local_d = d
            
        if local_m is not None and local_d is not None:
            now = time.time()
            dt = now - last_p_time
            last_p_time = now
            
            # Catch up physics steps to real time in small batches to avoid blocking APIs
            num_steps = int(dt / local_m.opt.timestep)
            # Max catchup to avoid death spiral
            num_steps = min(num_steps, 100) 
            
            # Step in batches of 10 to allow lock interleaving
            batch_size = 10
            for i in range(0, num_steps, batch_size):
                with lock:
                    for _ in range(min(batch_size, num_steps - i)):
                        mujoco.mj_step(local_m, local_d)
                time.sleep(0.0001) # Shortest yield
                    
        time.sleep(0.001)
                    
        time.sleep(0.001)

def rendering_loop():
    global m, d, latest_frame
    renderer = None
    last_m = None
    last_error_msg = ""
    while True:
        local_m = None
        local_d = None
        with lock:
            local_m = m
            local_d = d
        
        if local_m is not None and local_d is not None:
            try:
                # Initialize renderer outside the lock
                if renderer is None or local_m != last_m:
                    if renderer: renderer.close()
                    renderer = mujoco.Renderer(local_m, 480, 640)
                    last_m = local_m
                    print(f"[Server]: MJRenderer initialized (GL: {os.environ.get('MUJOCO_GL')})")
                
                # Render using a quick look at the data
                with lock:
                    renderer.update_scene(local_d)
                
                pixels = renderer.render()
                img = Image.fromarray(pixels)
                buf = io.BytesIO()
                # Lower quality to 70 for speed and lower bandwidth
                img.save(buf, format='JPEG', quality=70)
                new_frame = buf.getvalue()
                
                with condition:
                    latest_frame = new_frame
                    condition.notify_all()
                last_error_msg = ""
            except Exception as e:
                err_str = str(e)
                if err_str != last_error_msg:
                    print(f"[Renderer Error]: {err_str}", file=sys.stderr)
                    last_error_msg = err_str
                
                # Create a red 'ERROR' frame for visual feedback (outside lock)
                try:
                    from PIL import ImageDraw
                    err_img = Image.new('RGB', (640, 480), color=(50, 0, 0))
                    draw = ImageDraw.Draw(err_img)
                    draw.text((10, 10), f"MUJOCO RENDER ERROR:\n{err_str}", fill=(255, 255, 255))
                    buf = io.BytesIO()
                    err_img.save(buf, format='JPEG')
                    with condition:
                        latest_frame = buf.getvalue()
                        condition.notify_all()
                except:
                    pass
                time.sleep(1.0)
        time.sleep(1/45.0) # Aim for ~45 FPS internally to saturate 30 FPS MJPEG

@app.route('/build_world', methods=['POST'])
def build_world():
    global m, d
    xml = request.json.get("xml", default_xml)
    try:
        new_m = mujoco.MjModel.from_xml_string(xml)
        new_d = mujoco.MjData(new_m)
        with lock:
            m = new_m
            d = new_d
        return jsonify({"status": "World built successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/step', methods=['POST'])
def step():
    return jsonify({"status": "World is running natively in real-time. No manual step needed."})

@app.route('/velocity', methods=['POST'])
def velocity():
    global m, d
    body_name = request.json.get("body_name")
    vel_vec = request.json.get("velocity", [0, 0, 0])
    with lock:
        if m is None:
            return jsonify({"error": "World not built"}), 400
        try:
            body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id == -1:
                return jsonify({"error": f"Body '{body_name}' not found"}), 400
            
            dof_adr = m.body_dofadr[body_id]
            dof_num = m.body_dofnum[body_id]
            if dof_num < 3:
                return jsonify({"error": f"Body '{body_name}' does not have enough degrees of freedom. Did you forget <freejoint/> inside the <body>?"}), 400
                
            d.qvel[dof_adr:dof_adr+3] = vel_vec
            return jsonify({"status": f"Velocity {vel_vec} set for '{body_name}'."})
        except Exception as e:
            return jsonify({"error": str(e)}), 400

@app.route('/read', methods=['GET'])
def read():
    global m, d
    with lock:
        if m is None:
            return jsonify({"error": "World not built"}), 400
        bodies = []
        for i in range(m.nbody):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
            if name != "world":
                pos = d.xpos[i].tolist()
                quat = d.xquat[i].tolist() # [w, x, y, z]
                vel = d.cvel[i].tolist() # [rot_x, rot_y, rot_z, lin_x, lin_y, lin_z]
                bodies.append({
                    "name": name, 
                    "position": pos, 
                    "quaternion": quat,
                    "velocity_angular": vel[:3],
                    "velocity_linear": vel[3:]
                })
        return jsonify({"time": d.time, "bodies": bodies})

@app.route('/force', methods=['POST'])
def apply_force():
    global m, d
    body_name = request.json.get("body_name")
    force = request.json.get("force", [0, 0, 0])
    torque = request.json.get("torque", [0, 0, 0])
    with lock:
        if m is None: return jsonify({"error": "World not built"}), 400
        try:
            body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id == -1: return jsonify({"error": f"Body '{body_name}' not found"}), 400
            d.xfrc_applied[body_id, :3] = force
            d.xfrc_applied[body_id, 3:] = torque
            return jsonify({"status": f"Applied force {force} and torque {torque} to '{body_name}'."})
        except Exception as e:
            return jsonify({"error": str(e)}), 400

@app.route('/sensors', methods=['GET'])
def get_sensors():
    global m, d
    with lock:
        if m is None: return jsonify({"error": "World not built"}), 400
        sensor_data = {}
        for i in range(m.nsensor):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, i)
            adr = m.sensor_adr[i]
            dim = m.sensor_dim[i]
            data = d.sensordata[adr:adr+dim].tolist()
            sensor_data[name or f"sensor_{i}"] = data
        return jsonify({"sensors": sensor_data})

@app.route('/contacts', methods=['GET'])
def get_contacts():
    global m, d
    with lock:
        if m is None: return jsonify({"error": "World not built"}), 400
        contacts = []
        for i in range(d.ncon):
            con = d.contact[i]
            geom1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, con.geom1)
            geom2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, con.geom2)
            contacts.append({
                "geom1": geom1,
                "geom2": geom2,
                "position": con.pos.tolist(),
                "normal": con.frame[:3].tolist(),
                "distance": float(con.dist)
            })
        return jsonify({"contact_count": d.ncon, "contacts": contacts})

@app.route('/record', methods=['POST'])
def record():
    global m, d
    duration = float(request.json.get("duration", 1.0))
    if duration > 10.0:
        duration = 10.0
        
    history = []
    start = time.time()
    
    with lock:
        if m is None:
            return jsonify({"error": "World not built"}), 400
            
    while time.time() - start < duration:
        with lock:
            if m is not None and d is not None:
                positions = []
                for i in range(m.nbody):
                    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
                    if name != "world":
                        pos = d.xpos[i].tolist()
                        positions.append({"name": name, "position": pos})
                history.append({"time": d.time, "bodies": positions})
        time.sleep(0.033)
        
    return jsonify({"history": history, "duration": duration})

@app.route('/mass', methods=['GET'])
def get_mass():
    global m, d
    with lock:
        if m is None: return jsonify({"error": "World not built"}), 400
        
        total_mass = 0
        com = [0, 0, 0]
        body_data = []
        
        for i in range(m.nbody):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
            mass = float(m.body_mass[i])
            ipos = d.xpos[i].tolist() # Using xpos as a proxy for world COM of the body
            
            if name != "world":
                total_mass += mass
                com[0] += mass * ipos[0]
                com[1] += mass * ipos[1]
                com[2] += mass * ipos[2]
            
            body_data.append({"name": name, "mass": mass, "position": ipos})
            
        if total_mass > 0:
            com = [c / total_mass for c in com]
            
        return jsonify({
            "total_mass": total_mass,
            "center_of_mass": com,
            "bodies": body_data
        })

def generate_frames():
    while True:
        with condition:
            if not condition.wait(timeout=1.0):
                # If no frame for 1s, just yield whatever we have
                pass
            frame = latest_frame
        
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            time.sleep(0.1)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/models', methods=['GET'])
def api_models():
    try:
        import urllib.request, json
        req = urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        data = json.loads(req.read())
        models = [m["name"] for m in data.get("models", [])]
        return jsonify(models)
    except Exception as e:
        return jsonify(["nemotron-3-super:cloud", "llama3.2:latest"])

@app.route('/api/start', methods=['POST'])
def api_start():
    global active_researcher, ai_logs
    import subprocess
    import os
    
    data = request.json
    model = data.get("model", "nemotron-3-super:cloud")
    topic = data.get("topic", "").strip()

    if active_researcher is not None:
        try:
            import signal
            os.killpg(os.getpgid(active_researcher.pid), signal.SIGKILL)
            active_researcher.wait(timeout=2)
        except:
            pass

    ai_logs.clear()
    cmd = ["python3", "-u", "researcher.py", "--model", model]
    if topic:
        cmd.extend(["--topic", topic])
        
    import sys
    active_researcher = subprocess.Popen(cmd, preexec_fn=os.setsid, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    researcher_status = "RUNNING"
    
    def monitor_researcher():
        global researcher_status
        for line in iter(active_researcher.stdout.readline, b''):
            msg = line.decode('utf-8').strip()
            if msg:
                ai_logs.append(msg)
                if len(ai_logs) > 500: ai_logs.pop(0)
        researcher_status = "STOPPED"
        
    threading.Thread(target=monitor_researcher, daemon=True).start()
    
    print(f"[Server]: Spawned researcher PID {active_researcher.pid}")
    return jsonify({"status": "started", "state": researcher_status})

@app.route('/api/pause', methods=['POST'])
def api_pause():
    global active_researcher, researcher_status, ai_logs
    if active_researcher is not None and researcher_status == "RUNNING":
        try:
            os.killpg(os.getpgid(active_researcher.pid), signal.SIGSTOP)
            researcher_status = "PAUSED"
            ai_logs.append("[System]: AI Researcher SUSPENDED (Paused).")
            return jsonify({"status": "paused", "state": researcher_status})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"error": "No running researcher to pause"}), 400

@app.route('/api/resume', methods=['POST'])
def api_resume():
    global active_researcher, researcher_status, ai_logs
    if active_researcher is not None and researcher_status == "PAUSED":
        try:
            os.killpg(os.getpgid(active_researcher.pid), signal.SIGCONT)
            researcher_status = "RUNNING"
            ai_logs.append("[System]: AI Researcher RESUMED.")
            return jsonify({"status": "resumed", "state": researcher_status})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"error": "No paused researcher to resume"}), 400
    
@app.route('/api/stop', methods=['POST'])
def api_stop():
    global active_researcher, ai_logs, researcher_status
    if active_researcher is not None:
        try:
            os.killpg(os.getpgid(active_researcher.pid), signal.SIGKILL)
            ai_logs.append("[System]: AI Researcher STOPPED.")
        except Exception as e:
            ai_logs.append(f"[System]: Failed to stop: {e}")
        active_researcher = None
        researcher_status = "STOPPED"
    return jsonify({"status": "stopped", "state": researcher_status})

@app.route('/api/restart', methods=['POST'])
def api_restart():
    # Helper to stop and start immediately
    api_stop()
    return api_start()

@app.route('/api/clear', methods=['POST'])
def api_clear():
    global ai_logs
    ai_logs.clear()
    ai_logs.append("[System]: Terminal Log Cleared.")
    return jsonify({"status": "cleared"})

@app.route('/api/status', methods=['GET'])
@app.route('/status', methods=['GET'])
def api_status():
    global active_researcher, researcher_status
    if active_researcher is not None:
        poll_val = active_researcher.poll()
        if poll_val is not None:
            # Only reset to STOPPED if it actually died, but KEEP the status if it's supposed to be STOPPED anyway
            active_researcher = None
            researcher_status = "STOPPED"
    return jsonify({"state": researcher_status})

@app.route('/viewer')
def viewer():
    html = """<!DOCTYPE html>
<html>
<head>
    <title>Antigravity Research Dashboard</title>
    <style>
        body { background: #0a0a0d; color: #00ffcc; font-family: 'Inter', sans-serif; margin: 0; padding: 20px; display: flex; height: 100vh; overflow: hidden; box-sizing: border-box; }
        .panel { background: rgba(20, 20, 25, 0.8); border: 1px solid #333; border-radius: 12px; backdrop-filter: blur(10px); padding: 20px; box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5); }
        .left-panel { flex: 1; display: flex; flex-direction: column; justify-content: flex-start; align-items: center; margin-right: 20px; }
        .right-panel { flex: 1; display: flex; flex-direction: column; }
        img { max-width: 100%; border-radius: 8px; border: 1px solid #00ffcc; margin-top: auto; margin-bottom: auto;}
        h1 { font-size: 1.5rem; margin-top: 0; text-transform: uppercase; letter-spacing: 2px; text-shadow: 0 0 10px rgba(0,255,204,0.5); margin-bottom: 15px; text-align: center; }
        #terminal { flex: 1; background: #000; font-family: 'Fira Code', 'Courier New', monospace; font-size: 0.9rem; padding: 15px; overflow-y: auto; border-radius: 8px; border: 1px solid #333; color: #eee; white-space: pre-wrap; line-height: 1.5; }
        
        .controls { display: flex; gap: 10px; margin-bottom: 15px; width: 100%; }
        .controls select, .controls input { background: #1a1a24; color: #fff; border: 1px solid #333; padding: 10px; border-radius: 6px; font-family: 'Inter'; outline: none; }
        .controls select:focus, .controls input:focus { border-color: #00ffcc; }
        .controls input { flex: 1; }
        .btn { padding: 10px 20px; border: none; border-radius: 6px; font-weight: bold; cursor: pointer; transition: all 0.2s ease; text-transform: uppercase; letter-spacing: 1px; }
        .btn-start { background: #00ffcc; color: #000; }
        .btn-start:hover { background: #00e6b8; box-shadow: 0 0 15px rgba(0,255,204,0.4); }
        .btn-stop { background: #e74c3c; color: #fff; }
        .btn-stop:hover { background: #c0392b; box-shadow: 0 0 15px rgba(231,76,60,0.4); }
        
        .log-entry { margin-bottom: 8px; line-height: 1.4; }
        .system-log { color: #f39c12; font-weight: bold; }
        .tool-log { color: #e74c3c; font-weight: bold; }
        .ai-log { color: #bdc3c7; }
        .success-log { color: #2ecc71; font-weight: bold; }
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: #111; border-radius: 4px;}
        ::-webkit-scrollbar-thumb { background: #333; border-radius: 4px;}
        ::-webkit-scrollbar-thumb:hover { background: #555; }
    </style>
</head>
<body>
    <div class="panel left-panel">
        <h1>Live Telemetry Array</h1>
        <img src="/video_feed" alt="MJPEG Stream">
    </div>
    <div class="panel right-panel">
        <h1 style="display: flex; justify-content: center; align-items: center; gap: 10px;">
            <div id="statusLed" style="width: 12px; height: 12px; background: #555; border-radius: 50%; box-shadow: 0 0 5px rgba(0,0,0,0.5);"></div>
            Autonomous Reasoning Matrix
        </h1>
        
        <div class="controls">
            <select id="modelSelect" title="Select LLM Engine"></select>
            <input type="text" id="topicInput" placeholder="Enter research subject...">
            <div style="display: flex; gap: 8px;">
                <button id="mainBtn" class="btn btn-start" onclick="toggleResearcher()">Initialize</button>
                <button class="btn btn-stop" onclick="stopResearcher()" style="background: #e74c3c;">Stop</button>
                <button class="btn" onclick="restartResearcher()" style="background: #3498db; color: #fff;">Restart</button>
                <button class="btn" onclick="clearLogs()" style="background: #7f8c8d; color: #fff;">Clear</button>
            </div>
        </div>
        
        <div id="terminal"></div>
    </div>
    <script>
        const terminal = document.getElementById('terminal');
        const mainBtn = document.getElementById('mainBtn');
        const statusLed = document.getElementById('statusLed');
        let lastLogCount = 0;
        let currentState = "STOPPED";
        
        function updateUI(state) {
            currentState = state;
            if (state === "RUNNING") {
                mainBtn.innerHTML = "Pause";
                mainBtn.style.background = "#f39c12"; 
                statusLed.style.background = "#00ffcc";
                statusLed.style.boxShadow = "0 0 10px #00ffcc";
            } else if (state === "PAUSED") {
                mainBtn.innerHTML = "Resume";
                mainBtn.style.background = "#00ffcc";
                statusLed.style.background = "#f39c12";
                statusLed.style.boxShadow = "0 0 10px #f39c12";
            } else {
                mainBtn.innerHTML = "Initialize";
                mainBtn.style.background = "#00ffcc";
                mainBtn.style.color = "#000";
                statusLed.style.background = "#555";
                statusLed.style.boxShadow = "none";
            }
        }

        fetch('/api/models').then(r => r.json()).then(models => {
            const sel = document.getElementById('modelSelect');
            models.forEach(m => {
                let opt = document.createElement('option'); opt.value = m; opt.innerHTML = m; sel.appendChild(opt);
            });
        });

        function toggleResearcher() {
            if (currentState === "STOPPED") {
                updateUI("RUNNING"); // Optimistic UI update
                const model = document.getElementById('modelSelect').value;
                const topic = document.getElementById('topicInput').value;
                fetch('/api/start', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({model, topic})})
                    .then(r => r.json()).then(d => updateUI(d.state));
            } else if (currentState === "RUNNING") {
                fetch('/api/pause', { method: 'POST' }).then(r => r.json()).then(d => updateUI(d.state));
            } else if (currentState === "PAUSED") {
                fetch('/api/resume', { method: 'POST' }).then(r => r.json()).then(d => updateUI(d.state));
            }
        }
        
        function stopResearcher() {
            fetch('/api/stop', { method: 'POST' }).then(r => r.json()).then(d => updateUI(d.state));
        }

        function restartResearcher() {
            updateUI("RUNNING");
            const model = document.getElementById('modelSelect').value;
            const topic = document.getElementById('topicInput').value;
            fetch('/api/restart', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({model, topic})})
                .then(r => r.json()).then(d => updateUI(d.state));
        }

        function clearLogs() {
            fetch('/api/clear', { method: 'POST' });
        }
        
        setInterval(() => {
            fetch('/api/status').then(r => r.json()).then(d => {
                if (d.state !== currentState) updateUI(d.state);
            });

            fetch('/logs').then(r => r.json()).then(logs => {
                if (logs.length !== lastLogCount) {
                    terminal.innerHTML = '';
                    logs.forEach(log => {
                        const div = document.createElement('div');
                        div.className = 'log-entry';
                        if (log.includes('[Research') || log.includes('[Curiosity') || log.includes('[Iteration') || log.includes('Initiating') || log.includes('[System]')) div.className += ' system-log';
                        else if (log.includes('[Executing Native Tool]')) div.className += ' tool-log';
                        else if (log.includes('Physicist:')) { 
                            div.className += ' ai-log'; 
                            log = log.replace('Physicist:', '<strong style="color:#00ffcc">Physicist:</strong>'); 
                        }
                        else if (log.includes('[Research Concluded Successfully]')) div.className += ' success-log';
                        div.innerHTML = log;
                        terminal.appendChild(div);
                    });
                    terminal.scrollTop = terminal.scrollHeight;
                    lastLogCount = logs.length;
                }
            });
        }, 300); 
    </script>
</body>
</html>"""
    return render_template_string(html)

@app.route('/log', methods=['POST'])
def log_message():
    message = request.json.get("message", "")
    ai_logs.append(message)
    if len(ai_logs) > 500:
        ai_logs.pop(0)
    return jsonify({"status": "ok"})

@app.route('/logs', methods=['GET'])
def get_logs():
    return jsonify(ai_logs)

def cleanup():
    global active_researcher
    if active_researcher is not None:
        try:
            os.killpg(os.getpgid(active_researcher.pid), signal.SIGKILL)
        except:
            pass

atexit.register(cleanup)

if __name__ == "__main__":
    # Start separate physics and rendering threads
    p_thread = threading.Thread(target=physics_loop, daemon=True)
    r_thread = threading.Thread(target=rendering_loop, daemon=True)
    p_thread.start()
    r_thread.start()
    
    print("MuJoCo DECOUPLED Engine Running (GPU Accel: EGL)...")
    print("Open http://localhost:5050/viewer in your browser.")
    app.run(host="0.0.0.0", port=5050, threaded=True)
