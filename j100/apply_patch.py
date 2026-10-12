import re

with open('j100/simulate_j100_los.py', 'r') as f:
    code = f.read()

# 1. Imports
code = code.replace("import numpy as np", "import numpy as np\nimport json\nimport requests")

# 2. LLM Planner
llm_code = """
LLM_PROMPT = \"\"\"You are a robotic commander for the J100 skid-steer mobile robot.
Convert the user's natural language command into the following JSON format for the robot's next target waypoint in meters.
Format:
{
  "action": "move_to_waypoint",
  "x": float,
  "y": float
}
Do not output anything except the JSON.
User: {prompt}
JSON:\"\"\"

def call_ollama_planner(prompt):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "deepseek-r1:8b",
        "prompt": LLM_PROMPT.format(prompt=prompt),
        "stream": False,
        "format": "json"
    }
    try:
        response = requests.post(url, json=payload).json()
        return json.loads(response['response'])
    except Exception as e:
        print(f"Ollama generation failed: {e}")
        return None
"""
if "LLM_PROMPT" not in code:
    code = code.replace("def plot_los_analysis(", llm_code + "\n\ndef plot_los_analysis(")

# 3. Arguments
code = code.replace(
    'choices=["static", "moving", "waypoints"],\n        help="Target scenario: \'static\', \'moving\', or \'waypoints\'",',
    'choices=["static", "moving", "waypoints", "llm"],\n        help="Target scenario: \'static\', \'moving\', \'waypoints\', or \'llm\'",'
)

# 4. Target Dist
code = code.replace(
    'elif args.scenario == "waypoints":',
    'elif args.scenario in ["waypoints", "llm"]:'
)

# 5. Waypoints setup
waypoints_setup = """    # Target Waypoints for 'waypoints' scenario
    waypoint_list = [
        np.array([5.0, 2.5, 0.3]),
        np.array([8.0, -3.0, 0.3]),
        np.array([2.0, -4.5, 0.3]),
        np.array([0.0, 0.0, 0.3]),
    ]
    current_wp_idx = 0"""

llm_waypoints_setup = """    # Target Waypoints for 'waypoints' or 'llm' scenario
    waypoint_list = [
        np.array([5.0, 2.5, 0.3]),
        np.array([8.0, -3.0, 0.3]),
        np.array([2.0, -4.5, 0.3]),
        np.array([0.0, 0.0, 0.3]),
    ]
    current_wp_idx = 0

    if args.scenario == "llm":
        print("\nStarting J100 LLM Waypoint Planner (DeepSeek R1)...")
        user_prompts = [
            "Move the robot to x=5.0, y=2.5",
            "Now go left to x=8, y=-3",
            "Navigate down to x=2, y=-4.5",
            "Return back to origin at x=0, y=0"
        ]
        
        print(f"\nUser: '{user_prompts[0]}'")
        print("Thinking... (deepseek-r1:8b)")
        resp = call_ollama_planner(user_prompts[0])
        waypoint_list = []
        if resp and resp.get("action") == "move_to_waypoint":
            waypoint_list.append(np.array([float(resp["x"]), float(resp["y"]), 0.3]))
            print(f"LLM produced waypoint: {waypoint_list[0][:2]}")
        else:
            waypoint_list.append(np.array([5.0, 2.5, 0.3])) # fallback
"""
code = code.replace(waypoints_setup, llm_waypoints_setup)

# 6. Runtime logic
runtime_logic = """                elif args.scenario == "waypoints":
                    target_wp = waypoint_list[current_wp_idx]
                    beacon_pos = target_wp
                    dist_to_wp = np.linalg.norm(data.xpos[body_id][:2] - target_wp[:2])
                    if dist_to_wp < 1.5:
                        current_wp_idx = (current_wp_idx + 1) % len(waypoint_list)
                        next_wp = waypoint_list[current_wp_idx]"""

llm_runtime_logic = """                elif args.scenario in ["waypoints", "llm"]:
                    target_wp = waypoint_list[current_wp_idx]
                    beacon_pos = target_wp
                    dist_to_wp = np.linalg.norm(data.xpos[body_id][:2] - target_wp[:2])
                    if dist_to_wp < 1.5:
                        if args.scenario == "llm":
                            next_prompt_idx = (current_wp_idx + 1) % len(user_prompts)
                            print(f"\nUser: '{user_prompts[next_prompt_idx]}'")
                            print("Thinking... (deepseek-r1:8b)")
                            resp = call_ollama_planner(user_prompts[next_prompt_idx])
                            if resp and resp.get("action") == "move_to_waypoint":
                                new_wp = np.array([float(resp["x"]), float(resp["y"]), 0.3])
                                waypoint_list.append(new_wp)
                            else:
                                new_wp = target_wp # keep current if fails
                                waypoint_list.append(new_wp)
                            print(f"LLM produced waypoint: {new_wp[:2]}")
                            current_wp_idx += 1
                            next_wp = new_wp
                        else:
                            current_wp_idx = (current_wp_idx + 1) % len(waypoint_list)
                            next_wp = waypoint_list[current_wp_idx]"""

code = code.replace(runtime_logic, llm_runtime_logic)

with open('j100/simulate_j100_los.py', 'w') as f:
    f.write(code)

print("Patch applied.")
