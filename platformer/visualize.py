"""Create a self-contained, interactive HTML replay of a policy episode."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from platformer.env import PlatformerEnv
from platformer.planning import MCTSPlanner


ACTION_NAMES = (
    "no-op",
    "left",
    "right",
    "jump",
    "left + jump",
    "right + jump",
    "glide",
    "right + glide",
)


def create_episode_visualization(
    env: PlatformerEnv,
    policy: Any,
    *,
    seed: int,
    output_path: str | Path,
    algorithm: str = "agent",
) -> dict[str, Any]:
    """Run one seeded episode and save it as a playable HTML file."""
    observation, _ = env.reset(seed=seed)
    level = {
        "platforms": [
            {
                "left": platform.left,
                "right": platform.right,
                "y": platform.y,
                "moving": platform.moving,
                "amplitude": platform.amplitude,
                "phase": platform.phase,
                "speed": platform.speed,
            }
            for platform in env.platforms
        ],
        "hazards": [{"x": item.x, "y": item.y} for item in env.hazards],
        "enemies": [
            {
                "x": item.origin_x,
                "y": item.y,
                "amplitude": item.amplitude,
                "phase": item.phase,
                "speed": item.speed,
            }
            for item in env.enemies
        ],
        "gems": [{"x": item.x, "y": item.y} for item in env.gems],
        "goal": {"x": env.goal_x, "y": env.goal_y},
    }
    frames: list[dict[str, Any]] = [
        {
            "x": env.x,
            "y": env.y,
            "time": env._level_time,
            "vx": env.vx,
            "vy": env.vy,
            "gems": [item.collected for item in env.gems],
            "action": "start",
            "reward": 0.0,
            "completed": False,
            "died": False,
        }
    ]
    total_return = 0.0
    while True:
        action = policy.act(env) if isinstance(policy, MCTSPlanner) else policy.act(
            observation, deterministic=True
        )
        observation, reward, terminated, truncated, info = env.step(action)
        total_return += reward
        frames.append(
            {
                "x": env.x,
                "y": env.y,
                "time": env._level_time,
                "vx": env.vx,
                "vy": env.vy,
                "gems": [item.collected for item in env.gems],
                "action": ACTION_NAMES[action],
                "reward": reward,
                "completed": bool(info["completed"]),
                "died": bool(info["died"]),
            }
        )
        if terminated or truncated:
            break
    result = {
        "algorithm": algorithm,
        "seed": seed,
        "difficulty": env.difficulty,
        "steps": len(frames) - 1,
        "return": total_return,
        "completed": bool(info["completed"]),
        "died": bool(info["died"]),
        "gems_collected": int(info["gems_collected"]),
        "gems_available": int(info["gems_available"]),
        "truncated": bool(truncated),
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_render_html(level, frames, result), encoding="utf-8")
    return result


def _render_html(
    level: dict[str, Any], frames: list[dict[str, Any]], result: dict[str, Any]
) -> str:
    level_json = json.dumps(level, separators=(",", ":")).replace("<", "\\u003c")
    frames_json = json.dumps(frames, separators=(",", ":")).replace("<", "\\u003c")
    result_json = json.dumps(result, separators=(",", ":")).replace("<", "\\u003c")
    title = html.escape(f"Platformer replay — {result['algorithm']}")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{ color-scheme: dark; font-family: system-ui, sans-serif; background: #101725; color: #edf3fa; }}
  body {{ max-width: 1100px; margin: 24px auto; padding: 0 18px; }}
  h1 {{ font-size: 1.35rem; margin: 0 0 12px; }}
  .stage {{ overflow: hidden; border: 1px solid #33425a; border-radius: 12px; background: #18243a; }}
  svg {{ display: block; width: 100%; height: auto; aspect-ratio: 16 / 9; }}
  .controls {{ display: grid; grid-template-columns: auto auto auto 1fr auto; align-items: center; gap: 12px; margin: 14px 0; }}
  button {{ border: 1px solid #536783; border-radius: 7px; color: inherit; background: #263751; padding: 7px 12px; cursor: pointer; }}
  button:hover {{ background: #354c6d; }}
  input[type=range] {{ width: 100%; accent-color: #65d6b1; }}
  select {{ border: 1px solid #536783; border-radius: 7px; color: inherit; background: #263751; padding: 6px; }}
  .status {{ display: flex; flex-wrap: wrap; gap: 18px; color: #c4d2e3; font-size: .92rem; }}
  .status strong {{ color: #fff; }}
  .legend {{ color: #a9bad0; font-size: .85rem; margin-top: 12px; }}
  @media(max-width:600px) {{ .controls {{ grid-template-columns: auto auto auto 1fr; }} .controls label {{ grid-column: 1 / -1; }} }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="stage"><svg id="scene" viewBox="0 0 960 540" role="img" aria-label="Animated platformer gameplay"></svg></div>
<div class="controls">
  <button id="play" type="button">▶ Play</button>
  <button id="back" type="button" aria-label="Previous frame">◀</button>
  <button id="forward" type="button" aria-label="Next frame">▶|</button>
  <label><input id="scrubber" type="range" min="0" max="{len(frames) - 1}" value="0"></label>
  <label>Speed <select id="speed"><option value="1">1×</option><option value="2">2×</option><option value="0.5">0.5×</option></select></label>
</div>
<div class="status" id="status"></div>
<p class="legend">🟩 platforms · 💎 gems · 🔺 hazards · 🟠 enemies · 🏁 goal · 🟦 player</p>
<script>
const level = {level_json};
const frames = {frames_json};
const result = {result_json};
const svg = document.getElementById("scene");
const scrubber = document.getElementById("scrubber");
const status = document.getElementById("status");
const playButton = document.getElementById("play");
let current = 0;
let timer = null;
const NS = "http://www.w3.org/2000/svg";
function element(name, attrs, parent = svg) {{
  const node = document.createElementNS(NS, name);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  parent.appendChild(node);
  return node;
}}
function text(value, x, y, attrs = {{}}) {{
  const node = element("text", {{x, y, ...attrs}});
  node.textContent = value;
  return node;
}}
function render(index) {{
  current = Math.max(0, Math.min(index, frames.length - 1));
  scrubber.value = current;
  const frame = frames[current];
  const scale = 38;
  const cameraX = Math.max(-2, Math.min(frame.x - 7, level.goal.x - 18));
  const cameraY = frame.y + 2.3;
  const px = x => (x - cameraX) * scale;
  const py = y => 292 - (y - cameraY) * scale;
  svg.replaceChildren();
  const sky = element("defs", {{}});
  const gradient = element("linearGradient", {{id:"sky", x1:"0", y1:"0", x2:"0", y2:"1"}}, sky);
  element("stop", {{offset:"0%", "stop-color":"#233a5b"}}, gradient);
  element("stop", {{offset:"100%", "stop-color":"#15243a"}}, gradient);
  element("rect", {{x:0,y:0,width:960,height:540,fill:"url(#sky)"}});
  for (const p of level.platforms) {{
    const height = p.y + (p.moving ? p.amplitude * Math.sin(p.phase + frame.time * p.speed) : 0);
    const x = px(p.left), y = py(height);
    if (x + (p.right-p.left)*scale < 0 || x > 960) continue;
    element("rect", {{x, y, width:(p.right-p.left)*scale, height:18, rx:4, fill:p.moving?"#53a7a3":"#5fc79c"}});
    element("rect", {{x, y:y+13, width:(p.right-p.left)*scale, height:5, fill:"#397b72"}});
  }}
  for (const gem of level.gems) {{
    if (frame.gems[level.gems.indexOf(gem)]) continue;
    const x=px(gem.x), y=py(gem.y);
    if (x < -20 || x > 980) continue;
    element("polygon", {{points:`${{x}},${{y-10}} ${{x+8}},${{y}} ${{x}},${{y+10}} ${{x-8}},${{y}}`, fill:"#66d9ff", stroke:"#d7f7ff", "stroke-width":2}});
  }}
  for (const hazard of level.hazards) {{
    const x=px(hazard.x), y=py(hazard.y);
    if (x < -30 || x > 990) continue;
    element("polygon", {{points:`${{x-13}},${{y+12}} ${{x}},${{y-15}} ${{x+13}},${{y+12}}`, fill:"#f25f5c", stroke:"#ffd1c9", "stroke-width":2}});
  }}
  for (const enemy of level.enemies) {{
    const x=px(enemy.x+enemy.amplitude*Math.sin(enemy.phase+frame.time*enemy.speed)), y=py(enemy.y);
    if (x < -30 || x > 990) continue;
    element("circle", {{cx:x,cy:y,r:13,fill:"#f69b52",stroke:"#ffe1b9","stroke-width":2}});
    element("circle", {{cx:x-4,cy:y-2,r:2,fill:"#28364c"}});
    element("circle", {{cx:x+4,cy:y-2,r:2,fill:"#28364c"}});
  }}
  const gx=px(level.goal.x), gy=py(level.goal.y);
  element("line", {{x1:gx,y1:gy-42,x2:gx,y2:gy+4,stroke:"#f5f2d0","stroke-width":4}});
  element("polygon", {{points:`${{gx+2}},${{gy-41}} ${{gx+29}},${{gy-32}} ${{gx+2}},${{gy-22}}`,fill:"#f2c14e"}});
  const playerX=px(frame.x), playerY=py(frame.y);
  element("rect", {{x:playerX-11,y:playerY-27,width:22,height:32,rx:6,fill:"#73a7ff",stroke:"#e0eaff","stroke-width":2}});
  element("circle", {{cx:playerX+4,cy:playerY-18,r:2.5,fill:"#13233c"}});
  text(`seed ${{result.seed}} · frame ${{current}} / ${{frames.length-1}}`, 18, 30, {{fill:"#e8f1ff","font-size":16,"font-weight":600}});
  const terminal = frame.completed ? " · LEVEL COMPLETE!" : frame.died ? " · PLAYER DOWN" : "";
  status.innerHTML = `<span>Step <strong>${{current}} / ${{frames.length-1}}</strong></span><span>Action <strong>${{frame.action}}</strong></span><span>Position <strong>(${{frame.x.toFixed(1)}}, ${{frame.y.toFixed(1)}})</strong></span><span>Velocity <strong>(${{frame.vx.toFixed(1)}}, ${{frame.vy.toFixed(1)}})</strong></span><span>Gems <strong>${{frame.gems.filter(Boolean).length}} / ${{level.gems.length}}</strong></span><span>Reward <strong>${{frame.reward.toFixed(2)}}</strong>${{terminal}}</span>`;
}}
function stop() {{ if (timer !== null) clearInterval(timer); timer = null; playButton.textContent = "▶ Play"; }}
function play() {{
  if (timer !== null) {{ stop(); return; }}
  if (current >= frames.length-1) render(0);
  playButton.textContent = "⏸ Pause";
  const interval = Math.max(30, 100 / Number(document.getElementById("speed").value));
  timer = setInterval(() => {{
    if (current >= frames.length-1) {{ stop(); return; }}
    render(current+1);
  }}, interval);
}}
scrubber.addEventListener("input", () => {{ stop(); render(Number(scrubber.value)); }});
document.getElementById("back").addEventListener("click", () => {{ stop(); render(current-1); }});
document.getElementById("forward").addEventListener("click", () => {{ stop(); render(current+1); }});
playButton.addEventListener("click", play);
document.getElementById("speed").addEventListener("change", () => {{ if (timer !== null) {{ stop(); play(); }} }});
document.addEventListener("keydown", event => {{
  if (event.code === "Space") {{ event.preventDefault(); play(); }}
  if (event.code === "ArrowRight") {{ stop(); render(current+1); }}
  if (event.code === "ArrowLeft") {{ stop(); render(current-1); }}
}});
render(0);
</script>
</body>
</html>
"""
