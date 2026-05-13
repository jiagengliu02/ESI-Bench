from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch as th
import yaml


os.environ.setdefault("OG_DISABLE_EMITTER_APIS", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
ACTIVE_EXPLORE_DIR = REPO_ROOT / "src" / "active_explore"
sys.path.insert(0, str(ACTIVE_EXPLORE_DIR))

import omnigibson as og  # noqa: E402
from omnigibson.macros import gm  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402


gm.ENABLE_FLATCACHE = False
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_TRANSITION_RULES = False


MOVE_STEP = 0.02
TURN_DEG = 0.6
DOORLIKE_KEYWORDS = ("door", "doorway")
DOORLIKE_EXCLUDED_CATEGORIES = ("door", "sliding_door", "doorway")
COGNITIVEMAP_CAMERA_PITCH_DEG = -10.0
COUNTING_SCAN_RADIUS_M = 2.0
COUNTING_SCAN_CAMERA_HEIGHT_M = 1.35
COUNTING_SCAN_LOOK_HEIGHT_M = 0.75
UNOBSERVED_PHASE_START_DISTANCE_M = 2.4
UNOBSERVED_PHASE_END_DISTANCE_M = 0.75
UNOBSERVED_PHASE_CAMERA_HEIGHT_M = 0.72
UNOBSERVED_PHASE_LOOK_HEIGHT_M = 0.12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load an OmniGibson scene and record viewer-camera video.")
    parser.add_argument("--scene", help="Scene model name. Optional when --metadata has a scene field.")
    parser.add_argument("--room", default=None, help="Room instance to load. Ignored with --full-scene.")
    parser.add_argument("--metadata", type=Path, default=None, help="Optional question JSON used for scene/camera/task setup.")
    parser.add_argument("--task", default=None, help="Optional active_explore task module, e.g. unobserved_changes.")
    parser.add_argument("--question-index", type=int, default=0, help="Index when --metadata contains json_paths.")
    parser.add_argument("--json-root", type=Path, default=None, help="Optional root for resolving metadata json_paths.")
    parser.add_argument("--output", type=Path, default=Path("outputs/videos/scene.mp4"))
    parser.add_argument("--frames", type=int, default=None, help="Number of frames to record. Defaults to unlimited for --interactive, 300 otherwise.")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=None, help="Optional output width.")
    parser.add_argument("--height", type=int, default=None, help="Optional output height.")
    parser.add_argument("--robot", default="R1")
    parser.add_argument("--full-scene", action="store_true", help="Load full scene. This is the default unless --room-only is passed.")
    parser.add_argument("--room-only", action="store_true", help="Load only --room / metadata room instead of the default full scene.")
    parser.add_argument("--hide-ceilings", action="store_true", help="Do not load ceilings.")
    parser.add_argument("--hide-walls", action="store_true", help="Do not load walls.")
    parser.add_argument("--keep-doors-closed", action="store_true", help="Keep articulated doors in their dataset/default state.")
    parser.add_argument("--settle-steps", type=int, default=30)
    parser.add_argument("--fov-deg", type=float, default=None)
    parser.add_argument("--camera-position", nargs=3, type=float, metavar=("X", "Y", "Z"))
    parser.add_argument("--camera-quat", nargs=4, type=float, metavar=("X", "Y", "Z", "W"))
    parser.add_argument(
        "--motion",
        choices=[
            "none",
            "orbit",
            "target-orbit",
            "pan-left",
            "pan-right",
            "forward",
            "approach-box",
            "task-demo",
            "cognitivemap-path",
            "deformable-unveil",
            "unobserved-phases",
            "counting-scan",
        ],
        default="target-orbit",
    )
    parser.add_argument("--orbit-radius", type=float, default=1.0)
    parser.add_argument("--orbit-deg", type=float, default=180.0)
    parser.add_argument("--target-position", nargs=3, type=float, metavar=("X", "Y", "Z"), help="Override target point for --motion target-orbit.")
    parser.add_argument("--target-radius", type=float, default=None, help="Orbit radius for --motion target-orbit. Defaults to the initial camera distance, clamped.")
    parser.add_argument("--target-height", type=float, default=None, help="Camera height above target for --motion target-orbit. Defaults to the initial relative height.")
    parser.add_argument("--target-look-height", type=float, default=0.15, help="Look-at point height above target for --motion target-orbit.")
    parser.add_argument("--approach-distance", type=float, default=0.75, help="Final camera distance from target for --motion approach-box.")
    parser.add_argument("--approach-height", type=float, default=0.55, help="Final camera height above target for --motion approach-box.")
    parser.add_argument("--path-height", type=float, default=1.35, help="Camera height for --motion cognitivemap-path.")
    parser.add_argument("--path-lookahead", type=float, default=1.4, help="Meters of lookahead for --motion cognitivemap-path.")
    parser.add_argument("--phase-hold-frames", type=int, default=24, help="Pause frames near each phase target for --motion unobserved-phases.")
    parser.add_argument("--unobserved-phase", choices=["both", "phase1", "phase2"], default="both", help="Which phase to record for --motion unobserved-phases.")
    parser.add_argument("--interactive", action="store_true", help="Control the camera live with WASD + mouse and record that view.")
    parser.add_argument("--move-speed", type=float, default=0.08, help="Meters per keypress/frame in --interactive mode.")
    parser.add_argument("--mouse-sensitivity", type=float, default=0.12, help="Degrees per mouse pixel in --interactive mode.")
    parser.add_argument("--no-task-setup", action="store_true", help="Skip task postprocess_env object setup.")
    args = parser.parse_args()
    if args.frames is None:
        args.frames = 0 if args.interactive else 300
    return args


def load_json(path: Path) -> Any:
    with path.expanduser().open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_json_path(raw_path: str, metadata_path: Path, json_root: Path | None) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute() and path.exists():
        return path.resolve()
    for root in [metadata_path.parent, json_root, Path.cwd()]:
        if root is None:
            continue
        candidate = (root / path).resolve()
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve question JSON path: {raw_path}")


def select_question_json(metadata_path: Path, question_index: int, json_root: Path | None) -> Path:
    metadata_path = metadata_path.expanduser().resolve()
    metadata = load_json(metadata_path)
    if isinstance(metadata, dict) and isinstance(metadata.get("json_paths"), list):
        paths = metadata["json_paths"]
    elif isinstance(metadata, list):
        paths = metadata
    else:
        return metadata_path
    if question_index < 0 or question_index >= len(paths):
        raise IndexError(f"question_index={question_index} out of range for {len(paths)} paths")
    return resolve_json_path(str(paths[question_index]), metadata_path, json_root)


def import_task(task_name: str | None):
    if not task_name:
        return None
    import importlib

    return importlib.import_module(f"tasks.{task_name}")


def scene_room_from_payload(payload: dict[str, Any], task_module) -> tuple[str | None, str | None]:
    if task_module is not None and hasattr(task_module, "scene_room"):
        return task_module.scene_room(payload)
    return payload.get("scene"), payload.get("room")


def build_env_config(scene: str, room: str | None, robot: str, objects: list[dict[str, Any]], full_scene: bool, args: argparse.Namespace) -> dict[str, Any]:
    cfg_file = Path(og.example_config_path) / f"{robot.lower()}_primitives.yaml"
    if cfg_file.exists():
        config = yaml.safe_load(cfg_file.read_text())
    else:
        config = {"scene": {"type": "InteractiveTraversableScene"}, "robots": [], "objects": []}
    config.setdefault("scene", {})
    config["scene"]["scene_model"] = scene
    excluded = ["carpet", *DOORLIKE_EXCLUDED_CATEGORIES]
    if args.hide_ceilings:
        excluded.append("ceilings")
    if args.hide_walls:
        excluded.append("walls")
    config["scene"]["not_load_object_categories"] = excluded
    if room and not full_scene:
        config["scene"]["load_room_instances"] = [room]
    config["robots"] = []
    config["objects"] = [obj for obj in objects if not is_doorlike_record(obj)]
    return config


def is_doorlike_text(value: Any) -> bool:
    text = str(value or "").lower()
    tokens = [token for token in re.split(r"[^a-z0-9]+", text) if token]
    return any(token in DOORLIKE_KEYWORDS for token in tokens)


def is_doorlike_record(record: Any) -> bool:
    if isinstance(record, dict):
        values = [
            record.get("name"),
            record.get("category"),
            record.get("model"),
            record.get("prim_path"),
            record.get("relative_prim_path"),
        ]
        return any(is_doorlike_text(value) for value in values)
    return False


def is_doorlike_scene_object(obj: object) -> bool:
    values = [
        getattr(obj, "name", None),
        getattr(obj, "category", None),
        getattr(obj, "model", None),
        getattr(obj, "prim_path", None),
        getattr(obj, "relative_prim_path", None),
    ]
    return any(is_doorlike_text(value) for value in values)


def scene_objects(scene) -> list[object]:
    objects = getattr(scene, "objects", [])
    if isinstance(objects, dict):
        return list(objects.values())
    return list(objects)


def remove_scene_doorlike_objects(env) -> dict[str, Any]:
    targets = [obj for obj in scene_objects(env.scene) if is_doorlike_scene_object(obj)]
    removed = []
    failed = []
    for obj in targets:
        name = str(getattr(obj, "name", "unknown"))
        try:
            env.scene.remove_object(obj)
            removed.append(name)
        except Exception as exc:
            failed.append({"name": name, "error": f"{exc.__class__.__name__}: {exc}"})
    if targets:
        for _ in range(3):
            try:
                og.sim.step()
            except Exception:
                break
    return {"target_total": len(targets), "removed": removed, "failed": failed}


def open_scene_doors(env) -> int:
    try:
        import omnigibson.object_states as object_states
    except Exception:
        return 0
    opened = 0
    for obj in scene_objects(env.scene):
        category = str(getattr(obj, "category", "")).lower()
        name = str(getattr(obj, "name", "")).lower()
        if not is_doorlike_text(category) and not is_doorlike_text(name):
            continue
        try:
            states = getattr(obj, "states", {})
            if object_states.Open in states:
                states[object_states.Open].set_value(True)
                opened += 1
        except Exception:
            pass
    for _ in range(5):
        try:
            og.sim.step()
        except Exception:
            break
    return opened


def set_viewer_camera_fov(fov_deg: float | None) -> None:
    if fov_deg is None:
        return
    cam = getattr(og.sim, "viewer_camera", None) or getattr(og.sim, "_viewer_camera", None)
    if cam is None:
        return
    aperture_mm = float(cam.horizontal_aperture)
    cam.focal_length = aperture_mm / (2.0 * math.tan(math.radians(float(fov_deg)) * 0.5))


def initial_camera(payload: dict[str, Any] | None, task_module, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    if args.camera_position and args.camera_quat:
        return np.array(args.camera_position, dtype=float), np.array(args.camera_quat, dtype=float)
    if payload is not None and task_module is not None and hasattr(task_module, "initial_camera"):
        pos, quat, _ = task_module.initial_camera(payload)
        return np.array(pos, dtype=float), np.array(quat, dtype=float)
    return np.array([0.0, 0.0, 1.5], dtype=float), np.array([0.0, 0.0, 0.0, 1.0], dtype=float)


def look_at_quat(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = target - eye
    norm = np.linalg.norm(forward)
    if norm < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    forward = forward / norm
    world_up = np.array([0.0, 0.0, 1.0], dtype=float)
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-9:
        right = np.array([1.0, 0.0, 0.0], dtype=float)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    rot = np.array([right, up, -forward]).T
    return Rotation.from_matrix(rot).as_quat()


def ease_in_out(value: float) -> float:
    return 0.5 - 0.5 * math.cos(math.pi * max(0.0, min(1.0, value)))


def as_vec3(value: Any, z_default: float | None = None) -> np.ndarray | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        z = float(value[2]) if len(value) >= 3 else z_default
        if z is None:
            return None
        return np.array([float(value[0]), float(value[1]), float(z)], dtype=float)
    except (TypeError, ValueError):
        return None


def bbox_center(bbox: Any) -> np.ndarray | None:
    if isinstance(bbox, dict) and "min" in bbox and "max" in bbox:
        lo = as_vec3(bbox.get("min"))
        hi = as_vec3(bbox.get("max"))
    elif isinstance(bbox, (list, tuple)) and len(bbox) == 2:
        lo = as_vec3(bbox[0])
        hi = as_vec3(bbox[1])
    else:
        return None
    if lo is None or hi is None:
        return None
    return (lo + hi) * 0.5


def mean_vec3(values: Any, z_default: float | None = None) -> np.ndarray | None:
    if not isinstance(values, list):
        return None
    points = [point for point in (as_vec3(value, z_default=z_default) for value in values) if point is not None]
    if not points:
        return None
    return np.mean(np.stack(points, axis=0), axis=0)


def nested_get(data: dict[str, Any], *keys: str) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def first_camera_look_target(render: dict[str, Any]) -> np.ndarray | None:
    pose = render.get("camera_pose") if isinstance(render, dict) else None
    target = as_vec3((pose or {}).get("look_target") or (pose or {}).get("target")) if isinstance(pose, dict) else None
    if target is not None:
        return target

    camera_poses = render.get("camera_poses") if isinstance(render, dict) else None
    if isinstance(camera_poses, dict):
        for pose in camera_poses.values():
            if isinstance(pose, dict):
                target = as_vec3(pose.get("look_target") or pose.get("target"))
                if target is not None:
                    return target
    if isinstance(camera_poses, list):
        for pose in camera_poses:
            if isinstance(pose, dict):
                target = as_vec3(pose.get("look_target") or pose.get("target"))
                if target is not None:
                    return target
    return None


def target_candidates_from_payload(payload: dict[str, Any] | None) -> list[tuple[str, np.ndarray]]:
    if not isinstance(payload, dict):
        return []

    qd = payload.get("question_data") or {}
    candidates: list[tuple[str, np.ndarray]] = []

    for key in ("target_xyz", "target_position", "goal_position"):
        point = as_vec3(payload.get(key) or qd.get(key))
        if point is not None:
            candidates.append((key, point))

    boxes = qd.get("boxes") or []
    if isinstance(boxes, list) and boxes:
        points = []
        for box in boxes:
            if not isinstance(box, dict):
                continue
            container = box.get("container") or {}
            point = bbox_center(container.get("bbox"))
            if point is None:
                point = as_vec3(nested_get(container, "placement", "position"))
            if point is not None:
                points.append(point)
        if points:
            candidates.append(("box center", np.mean(np.stack(points, axis=0), axis=0)))

    small_item = payload.get("small_item") or {}
    point = bbox_center(small_item.get("bbox_after_cover"))
    if point is None:
        point = as_vec3(nested_get(small_item, "pose_after_cover", "position"))
    if point is None:
        point = as_vec3((payload.get("camera_setup") or {}).get("target_xyz"))
    if point is None:
        point = as_vec3(nested_get(payload, "render", "main_view", "target"))
    if point is not None:
        candidates.append(("deformable item", point))

    count_points = mean_vec3(qd.get("object_positions"), z_default=0.08)
    if count_points is None:
        count_points = mean_vec3(qd.get("ball_positions"), z_default=0.08)
    if count_points is not None:
        candidates.append(("count target cluster", count_points))

    placements = qd.get("placement") or []
    if isinstance(placements, list) and placements:
        points = [point for point in (as_vec3(item.get("real_position")) for item in placements if isinstance(item, dict)) if point is not None]
        if points:
            candidates.append(("placed target objects", np.mean(np.stack(points, axis=0), axis=0)))

    mirror_point = as_vec3(nested_get(qd, "scene_setup", "mirror", "position"))
    if mirror_point is None:
        mirror_point = as_vec3(nested_get(qd, "render", "mirror_pose", "position"))
    if mirror_point is not None:
        candidates.append(("mirror", mirror_point))

    goal_xy = qd.get("goal_xy")
    if isinstance(goal_xy, (list, tuple)) and len(goal_xy) >= 2:
        candidates.append(("navigation goal", np.array([float(goal_xy[0]), float(goal_xy[1]), float(qd.get("floor_z", 0.0)) + 1.0], dtype=float)))

    path_world = qd.get("path_world")
    path_mid = mean_vec3(path_world, z_default=float(qd.get("floor_z", 0.0)) + 1.0)
    if path_mid is not None:
        candidates.append(("navigation path midpoint", path_mid))

    for render in [payload.get("render"), qd.get("render")]:
        if isinstance(render, dict):
            target = first_camera_look_target(render)
            if target is not None:
                candidates.append(("render look target", target))
            multi = render.get("multi_image_input")
            views = (multi or {}).get("views") if isinstance(multi, dict) else None
            view_targets = [point for point in (as_vec3(view.get("target")) for view in views or [] if isinstance(view, dict)) if point is not None]
            if view_targets:
                candidates.append(("multi-view target", np.mean(np.stack(view_targets, axis=0), axis=0)))

    return candidates


def target_from_payload(payload: dict[str, Any] | None, args: argparse.Namespace) -> tuple[np.ndarray | None, str]:
    if args.target_position is not None:
        return np.array(args.target_position, dtype=float), "cli target"
    candidates = target_candidates_from_payload(payload)
    if candidates:
        return candidates[0][1], candidates[0][0]
    return None, ""


def task_demo_motion(task_name: str | None, payload: dict[str, Any] | None) -> str:
    if task_name == "cognitivemap":
        return "cognitivemap-path"
    if task_name == "deformable":
        return "deformable-unveil"
    if task_name == "unobserved_changes":
        return "unobserved-phases"
    if task_name == "counting":
        return "counting-scan"
    return "target-orbit"


def box_target_from_payload(payload: dict[str, Any] | None) -> np.ndarray | None:
    if not isinstance(payload, dict):
        return None
    boxes = ((payload.get("question_data") or {}).get("boxes") or [])
    if not boxes:
        return None
    box = boxes[0]
    bbox = ((box.get("container") or {}).get("bbox"))
    if isinstance(bbox, list) and len(bbox) == 2:
        lo = np.array(bbox[0], dtype=float)
        hi = np.array(bbox[1], dtype=float)
        center = (lo + hi) * 0.5
        center[2] = max(float(center[2]), float(lo[2]) + 0.12)
        return center
    placement = ((box.get("container") or {}).get("placement") or {}).get("position")
    if isinstance(placement, list) and len(placement) >= 3:
        return np.array([float(placement[0]), float(placement[1]), float(placement[2]) + 0.12], dtype=float)
    return None


def object_pose(scene, name: str | None) -> tuple[np.ndarray, np.ndarray] | None:
    if not name:
        return None
    try:
        obj = scene.object_registry("name", name)
    except Exception:
        obj = None
    if obj is None:
        return None
    try:
        pos, quat = obj.get_position_orientation()
        if hasattr(pos, "detach"):
            pos = pos.detach().cpu().numpy()
        if hasattr(quat, "detach"):
            quat = quat.detach().cpu().numpy()
        return np.array(pos, dtype=float), np.array(quat, dtype=float)
    except Exception:
        return None


def set_object_pose(scene, name: str | None, pos: np.ndarray, quat: np.ndarray | None = None) -> None:
    pose = object_pose(scene, name)
    if pose is None:
        return
    _, current_quat = pose
    try:
        obj = scene.object_registry("name", name)
        obj.set_position_orientation(
            position=th.tensor(pos, dtype=th.float32),
            orientation=th.tensor(quat if quat is not None else current_quat, dtype=th.float32),
        )
        try:
            obj.keep_still()
        except Exception:
            pass
    except Exception:
        pass


def runtime_name_by_prefix(task_state: dict[str, Any], prefix: str) -> str | None:
    for name in task_state.get("dynamic_object_names") or []:
        if str(name).startswith(prefix):
            return str(name)
    return None


def item_target_from_runtime(env, payload: dict[str, Any] | None, task_state: dict[str, Any]) -> np.ndarray | None:
    json_target = None
    for source, point in target_candidates_from_payload(payload):
        if source == "deformable item":
            json_target = point
            break
    runtime_item = nested_get(task_state, "item_after_settle", "bbox", "center")
    point = as_vec3(runtime_item)
    if point is not None and (json_target is None or np.linalg.norm(point[:2] - json_target[:2]) < 1.0):
        return point
    runtime_pose = nested_get(task_state, "item_after_settle", "pose", "position")
    point = as_vec3(runtime_pose)
    if point is not None and (json_target is None or np.linalg.norm(point[:2] - json_target[:2]) < 1.0):
        return point
    if json_target is not None:
        return json_target
    item_name = runtime_name_by_prefix(task_state, "cover_small_item_render_item_")
    pose = object_pose(env.scene, item_name)
    if pose is not None and np.linalg.norm(pose[0][:2]) > 1e-6:
        return pose[0]
    return None


def content_phase_pose(payload: dict[str, Any] | None, phase_index: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not isinstance(payload, dict):
        return None, None
    image_key = "image1" if phase_index == 1 else "image2"
    render = ((payload.get("question_data") or {}).get("render") or {})
    gt_view = render.get("gt_view") or {}
    entry = None
    if isinstance(gt_view.get(image_key), dict):
        entry = gt_view.get(image_key)
    elif isinstance(gt_view.get("image"), list):
        for item in gt_view["image"]:
            if isinstance(item, dict) and item.get("_key") == image_key:
                entry = item
                break
    images = (entry or {}).get("images") or []
    for image in images:
        pose = (image or {}).get("camera_pose")
        if isinstance(pose, dict) and pose.get("position") and pose.get("quaternion_xyzw"):
            return np.array(pose["position"], dtype=float), np.array(pose["quaternion_xyzw"], dtype=float)
    return None, None


def switch_unobserved_phase(task_module, env, payload: dict[str, Any] | None, task_state: dict[str, Any], phase_index: int) -> None:
    if payload is None or task_module is None:
        return
    phase_key = "phase1_content" if phase_index == 1 else "phase2_content"
    states = task_state.get("unobserved_change_states")
    try:
        if not states and hasattr(task_module, "build_states_from_payload"):
            states = task_module.build_states_from_payload(payload)
            task_state["unobserved_change_states"] = states
        if states and hasattr(task_module, "_setup_phase_contents"):
            names = task_module._setup_phase_contents(env.scene, states, phase_key)
            task_state["dynamic_object_names"] = list(task_state.get("dynamic_object_names") or []) + list(names)
            log(f"unobserved phase switched to {phase_key}")
    except Exception as exc:
        log(f"warning: could not switch unobserved phase to {phase_key}: {exc}")


def path_points_from_payload(payload: dict[str, Any] | None) -> np.ndarray | None:
    if not isinstance(payload, dict):
        return None
    qd = payload.get("question_data") or {}
    candidates = [
        qd.get("path_world"),
        nested_get(qd, "verification", "successful_path", "path_world_xy"),
    ]
    attempts = nested_get(qd, "verification", "shortest_path_attempts")
    if isinstance(attempts, list):
        for attempt in attempts:
            if isinstance(attempt, dict) and attempt.get("path_world_xy"):
                candidates.append(attempt.get("path_world_xy"))
    path_views = qd.get("path_views") or {}
    view_points = []
    for view in path_views.get("views") or []:
        pose = (view or {}).get("camera_pose") or {}
        point = as_vec3(pose.get("position"))
        if point is not None:
            view_points.append(point)
    if view_points:
        return np.stack(view_points, axis=0)
    for raw in candidates:
        if isinstance(raw, list) and len(raw) >= 2:
            points = [point for point in (as_vec3(value, z_default=0.0) for value in raw) if point is not None]
            if len(points) >= 2:
                return np.stack(points, axis=0)
    source = as_vec3(nested_get(qd, "initial_view", "render_xy"), z_default=0.0)
    goal = as_vec3(qd.get("goal_xy"), z_default=0.0)
    if source is not None and goal is not None:
        return np.stack([source, goal], axis=0)
    return None


def resample_polyline(points: np.ndarray, samples: int) -> np.ndarray:
    if len(points) == 0:
        return points
    if len(points) == 1 or samples <= 1:
        return points[:1].copy()
    deltas = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(deltas)])
    total = float(cumulative[-1])
    if total < 1e-9:
        return np.repeat(points[:1], samples, axis=0)
    targets = np.linspace(0.0, total, samples)
    output = []
    for distance in targets:
        index = int(np.searchsorted(cumulative, distance, side="right") - 1)
        index = min(max(index, 0), len(points) - 2)
        span = max(float(cumulative[index + 1] - cumulative[index]), 1e-9)
        local = (distance - cumulative[index]) / span
        output.append(points[index] * (1.0 - local) + points[index + 1] * local)
    return np.stack(output, axis=0)


def approach_box_pose(
    pos: np.ndarray,
    quat: np.ndarray,
    target: np.ndarray,
    frame: int,
    total_frames: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    progress = ease_in_out(frame / float(max(total_frames - 1, 1)))
    start_pos = pos.copy()
    start_dir = start_pos[:2] - target[:2]
    if np.linalg.norm(start_dir) < 1e-9:
        start_dir = np.array([1.0, 0.0], dtype=float)
    start_dir = start_dir / np.linalg.norm(start_dir)
    end_pos = np.array(
        [
            target[0] + start_dir[0] * float(args.approach_distance),
            target[1] + start_dir[1] * float(args.approach_distance),
            target[2] + float(args.approach_height),
        ],
        dtype=float,
    )
    frame_pos = start_pos * (1.0 - progress) + end_pos * progress
    look_target = target + np.array([0.0, 0.0, 0.05], dtype=float)
    return frame_pos, look_at_quat(frame_pos, look_target)


def approach_target_pose(
    start_pos: np.ndarray,
    start_quat: np.ndarray,
    target: np.ndarray,
    frame: int,
    total_frames: int,
    args: argparse.Namespace,
    start_distance: float = 3.0,
    end_distance: float | None = None,
    start_height: float = 1.45,
    end_height: float | None = None,
    angle: float | None = None,
    look_height: float = 0.12,
) -> tuple[np.ndarray, np.ndarray]:
    progress = ease_in_out(frame / float(max(total_frames - 1, 1)))
    end_distance = float(args.approach_distance) if end_distance is None else float(end_distance)
    end_height = float(args.approach_height) if end_height is None else float(end_height)
    if angle is None:
        offset = start_pos[:2] - target[:2]
        if np.linalg.norm(offset) < 1e-6:
            angle = 0.0
        else:
            angle = math.atan2(float(offset[1]), float(offset[0]))
    direction = np.array([math.cos(angle), math.sin(angle)], dtype=float)
    distance = float(start_distance) * (1.0 - progress) + end_distance * progress
    height = float(start_height) * (1.0 - progress) + end_height * progress
    frame_pos = np.array([target[0] + direction[0] * distance, target[1] + direction[1] * distance, target[2] + height], dtype=float)
    return frame_pos, look_at_quat(frame_pos, target + np.array([0.0, 0.0, float(look_height)], dtype=float))


def target_orbit_pose(
    pos: np.ndarray,
    quat: np.ndarray,
    target: np.ndarray,
    frame: int,
    total_frames: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    progress = frame / float(max(total_frames - 1, 1))
    offset_xy = pos[:2] - target[:2]
    distance = float(np.linalg.norm(offset_xy))
    if distance < 1e-6:
        offset_xy = np.array([1.0, 0.0], dtype=float)
        distance = 1.0
    radius = float(args.target_radius) if args.target_radius is not None else min(max(distance, 0.55), 3.0)
    start_angle = math.atan2(float(offset_xy[1]), float(offset_xy[0]))
    angle = start_angle + math.radians(float(args.orbit_deg)) * progress
    rel_height = float(args.target_height) if args.target_height is not None else max(float(pos[2] - target[2]), 0.35)
    frame_pos = np.array(
        [
            target[0] + math.cos(angle) * radius,
            target[1] + math.sin(angle) * radius,
            target[2] + rel_height,
        ],
        dtype=float,
    )
    look_target = target + np.array([0.0, 0.0, float(args.target_look_height)], dtype=float)
    return frame_pos, look_at_quat(frame_pos, look_target)


def cognitivemap_path_pose(
    pos: np.ndarray,
    quat: np.ndarray,
    payload: dict[str, Any] | None,
    frame: int,
    total_frames: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    path = path_points_from_payload(payload)
    if path is None or len(path) < 2:
        return pos, quat
    sampled = resample_polyline(path, max(total_frames + int(args.path_lookahead * 8), 2))
    index = min(frame, len(sampled) - 1)
    look_index = min(index + max(int(args.path_lookahead * 8), 1), len(sampled) - 1)
    frame_pos = sampled[index].copy()
    frame_pos[2] = float(args.path_height)
    heading = sampled[look_index, :2] - sampled[index, :2]
    if np.linalg.norm(heading) < 1e-6 and index > 0:
        heading = sampled[index, :2] - sampled[index - 1, :2]
    if np.linalg.norm(heading) < 1e-6:
        heading = Rotation.from_quat(quat).apply(np.array([0.0, 0.0, -1.0], dtype=float))[:2]
    if np.linalg.norm(heading) < 1e-6:
        heading = np.array([1.0, 0.0], dtype=float)
    heading = heading / np.linalg.norm(heading)
    look_distance = max(float(args.path_lookahead), 0.5)
    pitch_drop = look_distance * math.tan(math.radians(abs(COGNITIVEMAP_CAMERA_PITCH_DEG)))
    look_target = frame_pos + np.array([heading[0] * look_distance, heading[1] * look_distance, -pitch_drop], dtype=float)
    return frame_pos, look_at_quat(frame_pos, look_target)


def counting_scan_pose(
    pos: np.ndarray,
    quat: np.ndarray,
    target: np.ndarray | None,
    frame: int,
    total_frames: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    if target is None:
        return target_orbit_pose(pos, quat, pos + np.array([0.0, 1.0, 0.0]), frame, total_frames, args)
    third = max(total_frames // 3, 1)
    if frame < third:
        start_distance = max(float(np.linalg.norm(pos[:2] - target[:2])), COUNTING_SCAN_RADIUS_M)
        start_height = max(float(pos[2] - target[2]), COUNTING_SCAN_CAMERA_HEIGHT_M)
        return approach_target_pose(
            pos,
            quat,
            target,
            frame,
            third,
            args,
            start_distance=start_distance,
            end_distance=COUNTING_SCAN_RADIUS_M,
            start_height=start_height,
            end_height=COUNTING_SCAN_CAMERA_HEIGHT_M,
            look_height=COUNTING_SCAN_LOOK_HEIGHT_M,
        )
    local_frame = frame - third
    orbit_frames = max(total_frames - third, 1)
    local_args = argparse.Namespace(**vars(args))
    local_args.orbit_deg = min(max(float(args.orbit_deg), 160.0), 220.0)
    local_args.target_radius = args.target_radius if args.target_radius is not None else COUNTING_SCAN_RADIUS_M
    local_args.target_height = args.target_height if args.target_height is not None else COUNTING_SCAN_CAMERA_HEIGHT_M
    local_args.target_look_height = max(float(args.target_look_height), COUNTING_SCAN_LOOK_HEIGHT_M)
    start_distance = max(float(np.linalg.norm(pos[:2] - target[:2])), COUNTING_SCAN_RADIUS_M)
    start_height = max(float(pos[2] - target[2]), COUNTING_SCAN_CAMERA_HEIGHT_M)
    approach_pos, approach_quat = approach_target_pose(
        pos,
        quat,
        target,
        third - 1,
        third,
        args,
        start_distance=start_distance,
        end_distance=COUNTING_SCAN_RADIUS_M,
        start_height=start_height,
        end_height=COUNTING_SCAN_CAMERA_HEIGHT_M,
        look_height=COUNTING_SCAN_LOOK_HEIGHT_M,
    )
    return target_orbit_pose(approach_pos, approach_quat, target, local_frame, orbit_frames, local_args)


def annotate_frame(rgb: np.ndarray, args: argparse.Namespace, demo_state: dict[str, Any], frame: int, total_frames: int) -> np.ndarray:
    return rgb


def apply_motion(
    pos: np.ndarray,
    quat: np.ndarray,
    frame: int,
    total_frames: int,
    args: argparse.Namespace,
    payload: dict[str, Any] | None = None,
    target: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    pos = pos.copy()
    quat = quat.copy()
    if args.motion == "none" or total_frames <= 1:
        return pos, quat
    progress = frame / float(max(total_frames - 1, 1))
    if args.motion in {"pan-left", "pan-right"}:
        sign = 1.0 if args.motion == "pan-left" else -1.0
        yaw = sign * math.radians(args.orbit_deg) * progress
        quat = (Rotation.from_rotvec([0.0, 0.0, yaw]) * Rotation.from_quat(quat)).as_quat()
    elif args.motion == "forward":
        right = Rotation.from_quat(quat).apply(np.array([1.0, 0.0, 0.0]))
        forward = np.cross(right, np.array([0.0, 0.0, 1.0]))
        forward[2] = 0.0
        norm = np.linalg.norm(forward)
        if norm > 1e-9:
            pos += forward / norm * MOVE_STEP * frame
    elif args.motion == "orbit":
        angle = math.radians(args.orbit_deg) * (progress - 0.5)
        offset = np.array([math.sin(angle) * args.orbit_radius, 0.0, 0.0])
        pos += Rotation.from_quat(quat).apply(offset)
        quat = (Rotation.from_rotvec([0.0, 0.0, angle]) * Rotation.from_quat(quat)).as_quat()
    elif args.motion == "target-orbit":
        if target is not None:
            return target_orbit_pose(pos, quat, target, frame, total_frames, args)
    elif args.motion == "approach-box":
        target = box_target_from_payload(payload)
        if target is not None:
            return approach_box_pose(pos, quat, target, frame, total_frames, args)
    elif args.motion == "cognitivemap-path":
        return cognitivemap_path_pose(pos, quat, payload, frame, total_frames, args)
    elif args.motion == "counting-scan":
        return counting_scan_pose(pos, quat, target, frame, total_frames, args)
    return pos, quat


def prepare_task_demo(
    env,
    task_module,
    payload: dict[str, Any] | None,
    task_state: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    state: dict[str, Any] = {}
    if args.motion == "deformable-unveil":
        item_target = item_target_from_runtime(env, payload, task_state)
        cloth_name = runtime_name_by_prefix(task_state, "cover_small_item_render_cloth_")
        cloth_pose = object_pose(env.scene, cloth_name)
        cloth_center = as_vec3(nested_get(task_state, "cloth_after_settle", "bbox", "center"))
        if item_target is not None:
            state["target"] = item_target
        if cloth_center is not None and item_target is not None and np.linalg.norm(cloth_center[:2] - item_target[:2]) < 1.0:
            state["target"] = (np.array(state.get("target", cloth_center), dtype=float) + cloth_center) * 0.5
        if cloth_name and cloth_pose is not None:
            state["cloth_name"] = cloth_name
            state["cloth_start_pos"] = cloth_pose[0]
            state["cloth_quat"] = cloth_pose[1]
            lift = np.array([0.55, -0.35, 0.9], dtype=float)
            state["cloth_end_pos"] = cloth_pose[0] + lift
        log(f"deformable-unveil target={state.get('target')} cloth={cloth_name}")
    elif args.motion == "unobserved-phases":
        first_phase = 2 if args.unobserved_phase == "phase2" else 1
        switch_unobserved_phase(task_module, env, payload, task_state, first_phase)
        target = box_target_from_payload(payload)
        if target is not None:
            state["target"] = target
        phase1_pos, phase1_quat = content_phase_pose(payload, 1)
        phase2_pos, phase2_quat = content_phase_pose(payload, 2)
        state["phase1_pose"] = (phase1_pos, phase1_quat)
        state["phase2_pose"] = (phase2_pos, phase2_quat)
        state["phase"] = first_phase
        log(f"unobserved-phases target={target}")
    elif args.motion == "cognitivemap-path":
        path = path_points_from_payload(payload)
        if path is not None:
            state["path_points"] = path
            log(f"cognitivemap-path points={len(path)} start={path[0].round(3).tolist()} end={path[-1].round(3).tolist()}")
    elif args.motion == "counting-scan":
        target, source = target_from_payload(payload, args)
        if target is not None:
            state["target"] = target
        log(f"counting-scan target_source={source} target={target.round(4).tolist() if target is not None else None}")
    return state


def deformable_unveil_frame(
    env,
    pos: np.ndarray,
    quat: np.ndarray,
    frame: int,
    total_frames: int,
    args: argparse.Namespace,
    state: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    target = state.get("target")
    if target is None:
        return pos, quat
    target = np.array(target, dtype=float)
    approach_frames = max(int(total_frames * 0.42), 1)
    unveil_frames = max(int(total_frames * 0.34), 1)
    if frame < approach_frames:
        return approach_target_pose(
            pos,
            quat,
            target,
            frame,
            approach_frames,
            args,
            start_distance=2.4,
            end_distance=1.05,
            start_height=1.55,
            end_height=1.05,
            angle=math.radians(-90.0),
            look_height=0.35,
        )
    if "cloth_name" in state and "cloth_start_pos" in state and "cloth_end_pos" in state:
        local = min(max((frame - approach_frames) / float(unveil_frames), 0.0), 1.0)
        eased = ease_in_out(local)
        cloth_pos = np.array(state["cloth_start_pos"], dtype=float) * (1.0 - eased) + np.array(state["cloth_end_pos"], dtype=float) * eased
        set_object_pose(env.scene, state.get("cloth_name"), cloth_pos, state.get("cloth_quat"))
    end_pos, _ = approach_target_pose(
        pos,
        quat,
        target,
        approach_frames - 1,
        approach_frames,
        args,
        start_distance=2.4,
        end_distance=1.05,
        start_height=1.55,
        end_height=1.05,
        angle=math.radians(-90.0),
        look_height=0.35,
    )
    return end_pos, look_at_quat(end_pos, target + np.array([0.0, 0.0, 0.35], dtype=float))


def unobserved_phases_frame(
    env,
    task_module,
    payload: dict[str, Any] | None,
    task_state: dict[str, Any],
    pos: np.ndarray,
    quat: np.ndarray,
    frame: int,
    total_frames: int,
    args: argparse.Namespace,
    state: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    target = state.get("target")
    if target is None:
        return pos, quat
    target = np.array(target, dtype=float)
    if args.unobserved_phase == "phase1":
        phase_index = 1
        phase_frame = frame
        phase_total = total_frames
    elif args.unobserved_phase == "phase2":
        phase_index = 2
        phase_frame = frame
        phase_total = total_frames
    else:
        half = max(total_frames // 2, 1)
        phase_index = 1 if frame < half else 2
        phase_frame = frame if phase_index == 1 else frame - half
        phase_total = half if phase_index == 1 else max(total_frames - half, 1)
    if state.get("phase") != phase_index:
        switch_unobserved_phase(task_module, env, payload, task_state, phase_index)
        state["phase"] = phase_index

    phase_pose = state.get(f"phase{phase_index}_pose")
    phase_pos, phase_quat = phase_pose if isinstance(phase_pose, tuple) else (None, None)
    start_pos = phase_pos if phase_pos is not None else pos
    hold = max(int(args.phase_hold_frames), 0)
    active_total = max(phase_total - hold, 1)
    active_frame = min(phase_frame, active_total - 1)
    offset = start_pos[:2] - target[:2]
    angle = math.atan2(float(offset[1]), float(offset[0])) if np.linalg.norm(offset) > 1e-6 else math.radians(210.0)
    camera_height = max(float(args.approach_height), UNOBSERVED_PHASE_CAMERA_HEIGHT_M)
    return approach_target_pose(
        start_pos,
        quat,
        target,
        active_frame,
        active_total,
        args,
        start_distance=max(float(np.linalg.norm(offset)), UNOBSERVED_PHASE_START_DISTANCE_M),
        end_distance=max(float(args.approach_distance), UNOBSERVED_PHASE_END_DISTANCE_M),
        start_height=camera_height,
        end_height=camera_height,
        angle=angle,
        look_height=UNOBSERVED_PHASE_LOOK_HEIGHT_M,
    )


def frame_pose_and_actions(
    env,
    task_module,
    payload: dict[str, Any] | None,
    task_state: dict[str, Any],
    demo_state: dict[str, Any],
    pos: np.ndarray,
    quat: np.ndarray,
    frame: int,
    total_frames: int,
    args: argparse.Namespace,
    target: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if args.motion == "deformable-unveil":
        return deformable_unveil_frame(env, pos, quat, frame, total_frames, args, demo_state)
    if args.motion == "unobserved-phases":
        return unobserved_phases_frame(env, task_module, payload, task_state, pos, quat, frame, total_frames, args, demo_state)
    demo_target = demo_state.get("target")
    if demo_target is not None and args.motion in {"counting-scan", "target-orbit"}:
        target = np.array(demo_target, dtype=float)
    return apply_motion(pos, quat, frame, total_frames, args, payload, target=target)


def rgb_obs_to_uint8(frame: Any) -> np.ndarray:
    if hasattr(frame, "detach"):
        frame = frame.detach()
    if hasattr(frame, "cpu"):
        frame = frame.cpu()
    image = np.array(frame)
    if not (image.ndim == 3 and image.shape[2] in (3, 4)):
        raise ValueError(f"Viewer camera returned invalid rgb shape {getattr(image, 'shape', None)}")
    image = image[:, :, :3]
    if np.issubdtype(image.dtype, np.floating):
        if image.size and float(np.nanmax(image)) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    return image


def capture_rgb() -> np.ndarray:
    for _ in range(2):
        og.sim.render()
    return rgb_obs_to_uint8(og.sim._viewer_camera.get_obs()[0]["rgb"])


def make_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (int(width), int(height)))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    return writer


def log(message: str) -> None:
    print(f"[record_scene_video] {message}", flush=True)


def call_with_supported_args(func, *args, **kwargs):
    params = inspect.signature(func).parameters
    filtered_kwargs = {key: value for key, value in kwargs.items() if key in params}
    return func(*args, **filtered_kwargs)


def camera_axes(quat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rot = Rotation.from_quat(quat)
    right = rot.apply(np.array([1.0, 0.0, 0.0], dtype=float))
    up = np.array([0.0, 0.0, 1.0], dtype=float)
    forward = np.cross(right, up)
    for vec in (right, forward):
        vec[2] = 0.0
    right_norm = np.linalg.norm(right)
    forward_norm = np.linalg.norm(forward)
    right = right / right_norm if right_norm > 1e-9 else np.array([1.0, 0.0, 0.0], dtype=float)
    forward = forward / forward_norm if forward_norm > 1e-9 else np.array([0.0, 1.0, 0.0], dtype=float)
    return right, forward, up


def rotate_camera(quat: np.ndarray, yaw_deg: float = 0.0, pitch_deg: float = 0.0) -> np.ndarray:
    rot = Rotation.from_quat(quat)
    if abs(yaw_deg) > 1e-9:
        rot = Rotation.from_rotvec([0.0, 0.0, math.radians(yaw_deg)]) * rot
    if abs(pitch_deg) > 1e-9:
        right = rot.apply(np.array([1.0, 0.0, 0.0], dtype=float))
        rot = Rotation.from_rotvec(right * math.radians(pitch_deg)) * rot
    return rot.as_quat()


def record_interactive(
    args: argparse.Namespace,
    pos: np.ndarray,
    quat: np.ndarray,
    out_width: int,
    out_height: int,
) -> int:
    window_name = "ESI-Bench interactive recorder"
    writer = make_writer(args.output, args.fps, out_width, out_height)
    mouse = {"dragging": False, "x": 0, "y": 0, "yaw": 0.0, "pitch": 0.0}
    recording = True
    frame_count = 0
    frame_delay = 1.0 / max(float(args.fps), 1.0)

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse["dragging"] = True
            mouse["x"] = x
            mouse["y"] = y
        elif event == cv2.EVENT_LBUTTONUP:
            mouse["dragging"] = False
        elif event == cv2.EVENT_MOUSEMOVE and mouse["dragging"]:
            dx = x - mouse["x"]
            dy = y - mouse["y"]
            mouse["x"] = x
            mouse["y"] = y
            mouse["yaw"] += -dx * float(args.mouse_sensitivity)
            mouse["pitch"] += -dy * float(args.mouse_sensitivity)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, out_width, out_height)
    cv2.setMouseCallback(window_name, on_mouse)
    print(
        "Interactive controls: WASD move, Q/E move down/up, left-drag mouse to look, "
        "Space pause/resume recording, Esc or X exits.",
        flush=True,
    )
    print(f"Interactive recording limit: {'unlimited' if args.frames <= 0 else str(args.frames) + ' frames'}", flush=True)

    try:
        while args.frames <= 0 or frame_count < args.frames:
            t0 = time.time()

            if mouse["yaw"] or mouse["pitch"]:
                quat = rotate_camera(quat, yaw_deg=mouse["yaw"], pitch_deg=mouse["pitch"])
                mouse["yaw"] = 0.0
                mouse["pitch"] = 0.0

            og.sim._viewer_camera.set_position_orientation(
                position=th.tensor(pos, dtype=th.float32),
                orientation=th.tensor(quat, dtype=th.float32),
            )
            og.sim.step()
            rgb = capture_rgb()
            if (rgb.shape[1], rgb.shape[0]) != (out_width, out_height):
                rgb = cv2.resize(rgb, (out_width, out_height), interpolation=cv2.INTER_AREA)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imshow(window_name, bgr)
            if recording:
                writer.write(bgr)
                frame_count += 1

            key = cv2.waitKey(1) & 0xFF
            right, forward, up = camera_axes(quat)
            if key in (27, ord("x")):
                break
            if key == ord(" "):
                recording = not recording
                print(f"recording={'on' if recording else 'paused'}", flush=True)
            elif key in (ord("w"), ord("W")):
                pos += forward * float(args.move_speed)
            elif key in (ord("s"), ord("S")):
                pos -= forward * float(args.move_speed)
            elif key in (ord("a"), ord("A")):
                pos -= right * float(args.move_speed)
            elif key in (ord("d"), ord("D")):
                pos += right * float(args.move_speed)
            elif key == ord("e") or key == ord("E"):
                pos += up * float(args.move_speed)
            elif key == ord("q") or key == ord("Q"):
                pos -= up * float(args.move_speed)

            elapsed = time.time() - t0
            if elapsed < frame_delay:
                time.sleep(frame_delay - elapsed)
    finally:
        writer.release()
        cv2.destroyWindow(window_name)

    print(json.dumps({"output": str(args.output.resolve()), "frames": frame_count, "fps": args.fps}, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    task_module = import_task(args.task)
    payload = None
    question_json = None
    task_state: dict[str, Any] = {}
    if args.metadata is not None:
        question_json = select_question_json(args.metadata, args.question_index, args.json_root)
        payload = load_json(question_json)
    if args.motion == "task-demo":
        args.motion = task_demo_motion(args.task, payload)
    log(f"args interactive={args.interactive} frames={args.frames} motion={args.motion} output={args.output}")

    scene_from_payload, room_from_payload = scene_room_from_payload(payload or {}, task_module)
    scene = args.scene or scene_from_payload
    room = args.room or room_from_payload
    if not scene:
        raise ValueError("Pass --scene or --metadata containing a scene field.")

    objects = task_module.build_env_objects(payload) if payload is not None and task_module is not None else []
    full_scene = bool(args.full_scene or not args.room_only)
    config = build_env_config(scene, room, args.robot, objects, full_scene=full_scene, args=args)

    env = None
    try:
        log(f"loading scene={scene} room={room} full_scene={full_scene}")
        env = og.Environment(configs=config)
        log("environment loaded")
        doorlike_summary = remove_scene_doorlike_objects(env)
        log(
            "removed doorlike objects="
            f"{len(doorlike_summary['removed'])}/{doorlike_summary['target_total']} "
            f"failed={len(doorlike_summary['failed'])}"
        )
        if not args.keep_doors_closed:
            opened = open_scene_doors(env)
            log(f"opened doors={opened}")
        for _ in range(max(args.settle_steps, 0)):
            og.sim.step()
        log(f"settled {max(args.settle_steps, 0)} steps")

        pos, quat = initial_camera(payload, task_module, args)
        camera_info = {"camera_pose": {"position": pos.tolist(), "quaternion_xyzw": quat.tolist()}}
        if payload is not None and task_module is not None and not args.no_task_setup and hasattr(task_module, "postprocess_env"):
            log(f"running task setup for task={args.task}")
            task_state = {
                "source_json": str(question_json) if question_json is not None else "",
                "step_image_root": str((REPO_ROOT / "outputs" / "steps").resolve()),
            }
            setup_result = call_with_supported_args(task_module.postprocess_env, env, payload, camera_info, task_state=task_state)
            if isinstance(setup_result, dict):
                task_state.update(setup_result)
            doorlike_summary = remove_scene_doorlike_objects(env)
            if doorlike_summary["target_total"]:
                log(
                    "removed task doorlike objects="
                    f"{len(doorlike_summary['removed'])}/{doorlike_summary['target_total']} "
                    f"failed={len(doorlike_summary['failed'])}"
                )
            pos, quat = initial_camera(payload, task_module, args)
            log("task setup complete")

        set_viewer_camera_fov(args.fov_deg)
        target, target_source = target_from_payload(payload, args)
        demo_state = prepare_task_demo(env, task_module, payload, task_state, args)
        if args.motion in {"target-orbit", "counting-scan"}:
            if target is None:
                log(f"{args.motion} could not infer a target; falling back to camera-relative orbit")
            else:
                log(f"{args.motion} target_source={target_source} target={target.round(4).tolist()}")
        first_pos, first_quat = frame_pose_and_actions(
            env,
            task_module,
            payload,
            task_state,
            demo_state,
            pos,
            quat,
            0,
            args.frames,
            args,
            target,
        )
        log("setting initial viewer camera")
        og.sim._viewer_camera.set_position_orientation(
            position=th.tensor(first_pos, dtype=th.float32),
            orientation=th.tensor(first_quat, dtype=th.float32),
        )
        log("capturing first frame")
        first_rgb = capture_rgb()
        log(f"first frame captured shape={first_rgb.shape}")
        height, width = first_rgb.shape[:2]
        out_width = args.width or width
        out_height = args.height or height
        if args.interactive:
            log("entering interactive recorder")
            return record_interactive(args, first_pos, first_quat, out_width, out_height)
        writer = make_writer(args.output, args.fps, out_width, out_height)
        try:
            for frame in range(args.frames):
                frame_pos, frame_quat = frame_pose_and_actions(
                    env,
                    task_module,
                    payload,
                    task_state,
                    demo_state,
                    pos,
                    quat,
                    frame,
                    args.frames,
                    args,
                    target,
                )
                og.sim._viewer_camera.set_position_orientation(
                    position=th.tensor(frame_pos, dtype=th.float32),
                    orientation=th.tensor(frame_quat, dtype=th.float32),
                )
                og.sim.step()
                rgb = capture_rgb()
                if (rgb.shape[1], rgb.shape[0]) != (out_width, out_height):
                    rgb = cv2.resize(rgb, (out_width, out_height), interpolation=cv2.INTER_AREA)
                rgb = annotate_frame(rgb, args, demo_state, frame, args.frames)
                writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
        print(json.dumps({"output": str(args.output.resolve()), "frames": args.frames, "fps": args.fps}, indent=2))
        return 0
    finally:
        if env is not None:
            try:
                og.shutdown()
            except BaseException:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
