with open('j100/simulate_j100_los.py', 'r') as f:
    code = f.read()

runtime_logic = """                elif args.scenario in ["waypoints", "llm"]:
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

print("Patch 2 applied.")
