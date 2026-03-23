import mujoco
from PIL import Image
import io
import time
import threading
from flask import Flask, request, jsonify, Response, render_template_string

app = Flask(__name__)

m = None
d = None
latest_frame = None
lock = threading.Lock()
condition = threading.Condition()
ai_logs = []
active_researcher = None

default_xml = """<mujoco>
  <compiler angle="degree"/>
  <option gravity="0 0 -9.81" timestep="0.01"/>
  <visual>
    <global offwidth="640" offheight="480"/>
  </visual>
  <worldbody>
    <light pos="0 0 5" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="0 0 1" rgba=".8 .9 .8 1"/>
  </worldbody>
</mujoco>"""

def physics_loop():
    global m, d, latest_frame
    
    try:
        local_m = mujoco.MjModel.from_xml_string(default_xml)
        local_d = mujoco.MjData(local_m)
        with lock:
            m = local_m
            d = local_d
        renderer = mujoco.Renderer(local_m, 480, 640)
    except Exception as e:
        print("Renderer init error:", e)
        return

    last_update = time.time()
    while True:
        with lock:
            if m is not None and d is not None:
                if renderer.model != m:
                    renderer.close()
                    renderer = mujoco.Renderer(m, 480, 640)
                    last_update = time.time()
                
                now = time.time()
                dt = now - last_update
                last_update = now
                
                steps = int(dt / m.opt.timestep)
                for _ in range(steps):
                    mujoco.mj_step(m, d)
                
                try:
                    renderer.update_scene(d)
                    pixels = renderer.render()
                    img = Image.fromarray(pixels)
                    buf = io.BytesIO()
                    img.save(buf, format='JPEG')
                    
                    with condition:
                        latest_frame = buf.getvalue()
                        condition.notify_all()
                except Exception as e:
                    pass
        time.sleep(1/30.0)

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
        positions = []
        for i in range(m.nbody):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
            if name != "world":
                pos = d.xpos[i].tolist()
                positions.append({"name": name, "position": pos})
        return jsonify({"time": d.time, "bodies": positions})

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

def generate_frames():
    while True:
        with condition:
            condition.wait()
            frame = latest_frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

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
    
    data = request.json
    model = data.get("model", "nemotron-3-super:cloud")
    topic = data.get("topic", "").strip()

    if active_researcher is not None:
        try:
            active_researcher.terminate()
            active_researcher.wait(timeout=2)
        except:
            pass

    ai_logs.clear()
    cmd = ["python3", "-u", "researcher.py", "--model", model]
    if topic:
        cmd.extend(["--topic", topic])
        
    active_researcher = subprocess.Popen(cmd)
    return jsonify({"status": "started"})
    
@app.route('/api/stop', methods=['POST'])
def api_stop():
    global active_researcher
    if active_researcher is not None:
        try:
            active_researcher.terminate()
        except:
            pass
        active_researcher = None
    return jsonify({"status": "stopped"})

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
        <h1>Autonomous Reasoning Matrix</h1>
        
        <div class="controls">
            <select id="modelSelect"></select>
            <input type="text" id="topicInput" placeholder="Enter research subject (leave blank for Auto-Curiosity)...">
            <button class="btn btn-start" onclick="startResearcher()">Initialize</button>
            <button class="btn btn-stop" onclick="stopResearcher()">Halt</button>
        </div>
        
        <div id="terminal"></div>
    </div>
    <script>
        const terminal = document.getElementById('terminal');
        let lastLogCount = 0;
        
        fetch('/api/models').then(r => r.json()).then(models => {
            const sel = document.getElementById('modelSelect');
            models.forEach(m => {
                let opt = document.createElement('option'); opt.value = m; opt.innerHTML = m; sel.appendChild(opt);
            });
        });
        
        function startResearcher() {
            const model = document.getElementById('modelSelect').value;
            const topic = document.getElementById('topicInput').value;
            fetch('/api/start', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({model, topic})});
        }
        
        function stopResearcher() {
            fetch('/api/stop', { method: 'POST' });
        }
        
        setInterval(() => {
            fetch('/logs').then(r => r.json()).then(logs => {
                if (logs.length !== lastLogCount) {
                    terminal.innerHTML = '';
                    logs.forEach(log => {
                        const div = document.createElement('div');
                        div.className = 'log-entry';
                        if (log.includes('[Research') || log.includes('[Curiosity') || log.includes('[Iteration') || log.includes('Initiating')) div.className += ' system-log';
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

if __name__ == "__main__":
    t = threading.Thread(target=physics_loop, daemon=True)
    t.start()
    print("MuJoCo Real-Time Thread-Safe Engine Running on Port 5050...")
    print("Open http://localhost:5050/viewer in your browser to watch the physics LIVE!")
    app.run(host="0.0.0.0", port=5050, threaded=True)
