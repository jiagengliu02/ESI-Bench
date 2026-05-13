from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import omnigibson as og
import omnigibson.object_states as object_states
import torch as th
from omnigibson.objects.dataset_object import DatasetObject

from utils import normalize_text, resolve_path


TASK_NAME = "unobserved_changes"
FULL_SCENE = True
DEFAULT_MODEL = "gemini-2.5-flash"
ASSETS_ROOT = Path("/home/yininghong/BEHAVIOR-1K/datasets/behavior-1k-assets/objects")

QUALITATIVE_CONTENT_OVERRIDES = {
    # The original bottle_of_sage / lace meshes for this qualitative render can poke
    # through the carton bottom. Use compact, stable contents for the two phases.
    ("change_detection/q_000", "grocery_store_cafe", "dining_room_0", 0, "phase1_content"): {
        "category": "can_of_icetea",
        "display_name": "can of iced tea",
        "representative_model": "ifrjsc",
        "bbox_size_m": [0.07569613647460938, 0.07569613647460938, 0.14924559783935548],
        "sampling_source": "qualitative_render_override",
    },
    ("change_detection/q_000", "grocery_store_cafe", "dining_room_0", 0, "phase2_content"): {
        "category": "candle_holder",
        "display_name": "candle holder",
        "representative_model": "szulaa",
        "bbox_size_m": [0.13289098691940307, 0.1329570770263672, 0.1368186492919922],
        "sampling_source": "qualitative_render_override",
    },
}

VALID_ACTIONS = {
    "move_forward",
    "move_backward",
    "move_left",
    "move_right",
    "move_up",
    "move_down",
    "turn_left",
    "turn_right",
    "turn_up",
    "turn_down",
    "stop",
}

ACTION_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string"},
        "reasoning": {"type": "string"},
        "answer": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["action", "reasoning", "answer", "confidence"],
}

FINAL_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "number"},
        "reasoning": {"type": "string"},
    },
    "required": ["answer", "confidence", "reasoning"],
}


def normalize_choice(value: object) -> str:
    text = normalize_text(value).lower()
    text = re.sub(r"^[a-z]\s*[\).:\-]\s*", "", text)
    text = text.strip("\"' ")
    return re.sub(r"\s+", " ", text)


def canonicalize_prediction(predicted_answer: object, options: list[str]) -> str:
    raw = normalize_choice(predicted_answer)
    if not raw:
        return ""
    option_map = {normalize_choice(option): option for option in options}
    if raw in option_map:
        return option_map[raw]
    for normalized_option, option in option_map.items():
        if raw == f"answer: {normalized_option}" or raw.endswith(f": {normalized_option}"):
            return option
    for normalized_option, option in option_map.items():
        if normalized_option and normalized_option in raw:
            return option
    return normalize_text(predicted_answer)


def compute_accuracy(predicted_answer: object, ground_truth: object) -> float:
    return 1.0 if normalize_choice(predicted_answer) == normalize_choice(ground_truth) else 0.0


def scene_room(payload: dict[str, Any]) -> tuple[str, str]:
    return normalize_text(payload.get("scene")), normalize_text(payload.get("room")) or "full_scene"


def question_id(payload: dict[str, Any], source_path: Path) -> str:
    raw = normalize_text(payload.get("question_id")) or source_path.stem
    return raw.replace("\\", "/").split("/")[-1]


def build_env_objects(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return []


def _model_exists(category: str, model: str) -> bool:
    category = normalize_text(category)
    model = normalize_text(model)
    if not category or not model:
        return False
    return (ASSETS_ROOT / category / model / "usd" / f"{model}.usdz.encrypted").exists()


def _content_override_for_payload(
    payload: dict[str, Any],
    box_index: int,
    phase_key: str,
) -> dict[str, Any] | None:
    key = (
        normalize_text(payload.get("question_id")),
        normalize_text(payload.get("scene")),
        normalize_text(payload.get("room")),
        int(box_index),
        phase_key,
    )
    override = QUALITATIVE_CONTENT_OVERRIDES.get(key)
    if override is None:
        return None
    if not _model_exists(override["category"], override["representative_model"]):
        return None
    return dict(override)


def _step_sim(steps: int) -> None:
    for _ in range(max(int(steps), 0)):
        og.sim.step()


def _set_viewer_camera_fov(fov_deg: float) -> None:
    cam = getattr(og.sim, "viewer_camera", None) or getattr(og.sim, "_viewer_camera", None)
    if cam is None:
        return
    try:
        aperture_mm = float(cam.horizontal_aperture)
        cam.focal_length = aperture_mm / (2.0 * math.tan(math.radians(float(fov_deg)) * 0.5))
    except Exception:
        pass


def _capture_view(path: Path, pose: dict[str, Any], fov_deg: float | None = None) -> Path:
    if fov_deg is not None:
        _set_viewer_camera_fov(float(fov_deg))
    og.sim._viewer_camera.set_position_orientation(
        position=th.tensor(pose["position"], dtype=th.float32),
        orientation=th.tensor(pose["quaternion_xyzw"], dtype=th.float32),
    )
    for _ in range(10):
        og.sim.render()
    obs = og.sim._viewer_camera.get_obs()[0]
    rgb = obs["rgb"].detach().cpu().numpy()[:, :, :3].astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return path


def _remove_scene_object(scene, name: str) -> None:
    obj = scene.object_registry("name", name) if name else None
    if obj is None:
        return
    try:
        scene.remove_object(obj)
    except Exception:
        pass


def _add_dataset_object(
    scene,
    *,
    name: str,
    category: str,
    model: str,
    position: list[float],
    orientation: list[float] | None = None,
    visual_only: bool = True,
    keep_still_after_place: bool = True,
) -> Any | None:
    if not name or not category or not model or position is None:
        return None
    existing = scene.object_registry("name", name)
    if existing is not None:
        _remove_scene_object(scene, name)
        _step_sim(2)
    object_kwargs = {"visual_only": bool(visual_only)}
    if not visual_only:
        object_kwargs.update({"fixed_base": False, "kinematic_only": False})
    obj = DatasetObject(name=name, category=category, model=model, **object_kwargs)
    scene.add_object(obj)
    _step_sim(5)
    obj.set_position_orientation(
        position=th.tensor([float(v) for v in position], dtype=th.float32),
        orientation=th.tensor([float(v) for v in (orientation or [0.0, 0.0, 0.0, 1.0])], dtype=th.float32),
    )
    try:
        obj.visual_only = bool(visual_only)
    except Exception:
        pass
    if keep_still_after_place:
        try:
            obj.keep_still()
        except Exception:
            pass
    return obj


def _set_container_open(obj, value: bool = True) -> None:
    try:
        states = getattr(obj, "states", {})
        if object_states.Open in states:
            states[object_states.Open].set_value(bool(value))
    except Exception:
        pass


def _container_pose(box_state: dict[str, Any]) -> tuple[list[float] | None, list[float]]:
    placement = box_state.get("container_placement") or {}
    position = placement.get("position")
    orientation = (
        placement.get("orientation")
        or placement.get("quaternion_xyzw")
        or box_state.get("container_orientation")
        or [0.0, 0.0, 0.0, 1.0]
    )
    return position, orientation


def _content_position(box_state: dict[str, Any], content: dict[str, Any] | None = None) -> list[float] | None:
    bbox = box_state.get("container_bbox")
    if isinstance(bbox, list) and len(bbox) == 2:
        lo = np.array(bbox[0], dtype=float)
        hi = np.array(bbox[1], dtype=float)
        center = (lo + hi) * 0.5
        content_height = 0.04
        if isinstance(content, dict) and isinstance(content.get("bbox_size_m"), list) and len(content["bbox_size_m"]) >= 3:
            content_height = max(float(content["bbox_size_m"][2]), content_height)
        container_height = max(float(hi[2] - lo[2]), 0.01)
        center[2] = float(lo[2] + min(container_height * 0.72, max(content_height * 0.5 + 0.04, 0.075)))
        return center.tolist()
    container_position, _ = _container_pose(box_state)
    if container_position:
        return [float(container_position[0]), float(container_position[1]), float(container_position[2]) + 0.08]
    return None


def _content_name(phase_key: str, box_index: int, content: dict[str, Any]) -> str:
    category = normalize_text(content.get("category")) or "object"
    return f"render_unobserved_content_{phase_key}_{box_index}_{category}"


def _setup_containers(scene, states: list[dict[str, Any]]) -> list[str]:
    names = []
    objects: list[tuple[dict[str, Any], Any]] = []
    for state in states:
        position, orientation = _container_pose(state)
        obj = _add_dataset_object(
            scene,
            name=state.get("container_name") or f"render_unobserved_box_{state['box_index']}",
            category=state.get("container_category"),
            model=state.get("container_model"),
            position=position,
            orientation=orientation,
            visual_only=False,
            keep_still_after_place=False,
        )
        if obj is not None:
            names.append(obj.name)
            objects.append((state, obj))
    _step_sim(90)
    for state, obj in objects:
        try:
            lo, hi = obj.aabb
            state["container_bbox"] = [np.array(lo, dtype=float).tolist(), np.array(hi, dtype=float).tolist()]
        except Exception:
            pass
        _set_container_open(obj, True)
        _step_sim(5)
        try:
            obj.keep_still()
        except Exception:
            pass
        try:
            obj.visual_only = True
        except Exception:
            pass
    _step_sim(10)
    return names


def _clear_phase_contents(scene, states: list[dict[str, Any]], phase_keys: tuple[str, ...] = ("phase1_content", "phase2_content")) -> None:
    for state in states:
        for phase_key in phase_keys:
            content = state.get(phase_key)
            if isinstance(content, dict):
                _remove_scene_object(scene, _content_name(phase_key, int(state["box_index"]), content))
    _step_sim(5)


def _setup_phase_contents(scene, states: list[dict[str, Any]], phase_key: str) -> list[str]:
    _clear_phase_contents(scene, states)
    names = []
    for state in states:
        content = state.get(phase_key)
        if not isinstance(content, dict):
            continue
        position = _content_position(state, content)
        obj = _add_dataset_object(
            scene,
            name=_content_name(phase_key, int(state["box_index"]), content),
            category=content.get("category"),
            model=content.get("representative_model"),
            position=position,
            visual_only=True,
        )
        if obj is not None:
            names.append(obj.name)
    _step_sim(10)
    return names


def _question_data(payload: dict[str, Any]) -> dict[str, Any]:
    return payload.get("question_data") or {}


def _keyed_list_to_map(items: object) -> dict[str, Any]:
    output = {}
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict) and normalize_text(item.get("_key")):
                output[normalize_text(item.get("_key"))] = item
    return output


def _phase_description_map(payload: dict[str, Any]) -> dict[str, str]:
    phase_description = _question_data(payload).get("phase_description") or {}
    direct = {
        "phase_1": normalize_text(phase_description.get("phase_1")),
        "phase_2": normalize_text(phase_description.get("phase_2")),
    }
    phase_items = _keyed_list_to_map(phase_description.get("phase"))
    for key in ("phase_1", "phase_2"):
        if not direct[key] and isinstance(phase_items.get(key), dict):
            direct[key] = normalize_text(phase_items[key].get("value"))
    return direct


def _phase_content_map(box: dict[str, Any]) -> dict[str, dict[str, Any] | None]:
    direct = {
        "phase1_content": box.get("phase1_content"),
        "phase2_content": box.get("phase2_content"),
    }
    phase_items = _keyed_list_to_map(box.get("phase_content"))
    for key in ("phase1_content", "phase2_content"):
        if direct[key] is None and isinstance(phase_items.get(key), dict):
            item = dict(phase_items[key])
            if item.get("category") is None or item.get("model") is None:
                direct[key] = None
            else:
                direct[key] = item
    return direct


def _content_payload_to_runtime(content: dict[str, Any] | None) -> dict[str, Any] | None:
    if content is None:
        return None
    return {
        "category": content.get("category"),
        "display_name": content.get("display_name"),
        "representative_model": content.get("model"),
        "bbox_size_m": content.get("bbox_size_m"),
        "sampling_source": content.get("sampling_source"),
    }


def build_states_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    states = []
    for box in _question_data(payload).get("boxes") or []:
        container = box.get("container") or {}
        phase_content = _phase_content_map(box)
        box_index = int(box.get("box_index", len(states)))
        phase1_content = _content_override_for_payload(payload, box_index, "phase1_content") or _content_payload_to_runtime(
            phase_content.get("phase1_content")
        )
        phase2_content = _content_override_for_payload(payload, box_index, "phase2_content") or _content_payload_to_runtime(
            phase_content.get("phase2_content")
        )
        states.append(
            {
                "box_index": box_index,
                "position_label": normalize_text(box.get("position_label")) or f"box {len(states)}",
                "change_type": normalize_text(box.get("change_type")) or "no_change",
                "phase1_content": phase1_content,
                "phase2_content": phase2_content,
                "container_name": normalize_text(container.get("name")),
                "container_category": normalize_text(container.get("category")),
                "container_model": normalize_text(container.get("model")),
                "container_placement": container.get("placement"),
                "container_bbox": container.get("bbox"),
            }
        )
    if not states:
        raise ValueError("Question JSON does not contain any boxes.")
    return states


def _gt_view_phase_entries(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    render = _question_data(payload).get("render") or {}
    gt_view = render.get("gt_view") or {}
    if isinstance(gt_view.get("image1"), dict) or isinstance(gt_view.get("image2"), dict):
        return {
            key: value
            for key in ("image1", "image2")
            if isinstance((value := gt_view.get(key)), dict)
        }
    return _keyed_list_to_map(gt_view.get("image"))


def _first_gt_image(payload: dict[str, Any], phase_key: str) -> dict[str, Any] | None:
    entry = _gt_view_phase_entries(payload).get(phase_key)
    images = (entry or {}).get("images") or []
    for image in images:
        if isinstance(image, dict) and image.get("image_path"):
            return image
    return None


def _reference_image_paths(payload: dict[str, Any], source_json: Path, config=None) -> tuple[list[Path], dict[str, Any] | None, str | None]:
    data_root = getattr(config, "json_root", None)
    image1 = _first_gt_image(payload, "image1")
    image2 = _first_gt_image(payload, "image2")
    if image1 is None or image2 is None:
        return [], None, "missing_phase_reference_images"
    path1 = resolve_path(image1.get("image_path"), source_json, data_root=data_root)
    path2 = resolve_path(image2.get("image_path"), source_json, data_root=data_root)
    pose = image2.get("camera_pose")
    if not pose or not pose.get("position") or not pose.get("quaternion_xyzw"):
        return [], None, "missing_phase2_camera_pose"
    paths = []
    if path1 is not None:
        paths.append(path1)
    if path2 is not None:
        paths.append(path2)
    return paths, pose, None


def preprocess(payload: dict[str, Any], source_json: Path, config=None) -> dict[str, Any]:
    reference_paths, pose, skip_reason = _reference_image_paths(payload, source_json, config=config)
    if skip_reason:
        return {"skip_reason": skip_reason}
    return {
        "source_json": str(source_json),
        "reference_image_paths": [str(path) for path in reference_paths],
        "initial_camera_pose": pose,
        "step_image_root": str(getattr(config, "step_image_root", "")) if config is not None else "",
    }


def reference_image_paths(payload: dict[str, Any], task_state: dict[str, Any] | None = None) -> list[Path]:
    return [Path(path) for path in (task_state or {}).get("reference_image_paths", [])]


def initial_camera(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    # Pipeline calls this before preprocess, so read directly from the keyed-list
    # image2 entry here as well.
    image2 = _first_gt_image(payload, "image2")
    pose = (image2 or {}).get("camera_pose")
    if not pose or not pose.get("position") or not pose.get("quaternion_xyzw"):
        raise ValueError("Missing phase-2 camera pose in unobserved_changes JSON")
    return (
        np.array(pose["position"], dtype=float),
        np.array(pose["quaternion_xyzw"], dtype=float),
        {"camera_pose": pose},
    )


def postprocess_env(
    env,
    payload: dict[str, Any],
    camera_info: dict[str, Any] | None = None,
    task_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    states = build_states_from_payload(payload)
    scene = env.scene
    container_names = _setup_containers(scene, states)

    source_json = Path((task_state or {}).get("source_json") or "question.json")
    qid = question_id(payload, source_json)
    reference_dir = source_json.parent / ".active_references" / qid
    if task_state is not None:
        configured_root = task_state.get("step_image_root")
        if configured_root:
            scene_name, room_name = scene_room(payload)
            reference_dir = Path(configured_root) / TASK_NAME / scene_name / room_name / qid / "references"

    generated_reference_paths: list[str] = []
    render = _question_data(payload).get("render") or {}
    for phase_key, content_key, file_name in (
        ("image1", "phase1_content", "phase1_reference.png"),
        ("image2", "phase2_content", "phase2_reference.png"),
    ):
        image = _first_gt_image(payload, phase_key)
        pose = (image or {}).get("camera_pose")
        if pose and pose.get("position") and pose.get("quaternion_xyzw"):
            _setup_phase_contents(scene, states, content_key)
            output_path = reference_dir / file_name
            _capture_view(output_path, pose, fov_deg=(image or {}).get("fov_deg"))
            generated_reference_paths.append(str(output_path))

    phase2_names = _setup_phase_contents(scene, states, "phase2_content")
    _set_viewer_camera_fov(float(render.get("fov_deg") or 90.0))
    if task_state is not None:
        task_state["unobserved_change_states"] = states
        task_state["phase_descriptions"] = _phase_description_map(payload)
        if generated_reference_paths:
            task_state["reference_image_paths"] = generated_reference_paths
        task_state["dynamic_object_names"] = container_names + phase2_names
    return {
        "box_count": len(states),
        "reference_driven": bool(generated_reference_paths),
        "generated_reference_images": generated_reference_paths,
        "container_names": container_names,
        "phase2_content_names": phase2_names,
    }


def get_context(payload: dict[str, Any]) -> dict[str, Any]:
    question_data = _question_data(payload)
    phase_description = _phase_description_map(payload)
    boxes = []
    for box in question_data.get("boxes") or []:
        container = box.get("container") or {}
        boxes.append(
            {
                "label": normalize_text(box.get("position_label")) or f"box {int(box.get('box_index', 0))}",
                "container_category": normalize_text(container.get("category")),
                "change_type": normalize_text(box.get("change_type")),
            }
        )
    return {
        "task_type": normalize_text(payload.get("task_type") or question_data.get("task_type")),
        "question": normalize_text(question_data.get("question")),
        "options": [normalize_text(opt) for opt in question_data.get("options", []) if normalize_text(opt)],
        "phase_1": phase_description["phase_1"],
        "phase_2": phase_description["phase_2"],
        "boxes": boxes,
        "ground_truth": normalize_text(question_data.get("answer")),
    }


def build_system_prompt(
    payload: dict[str, Any],
    threshold: float,
    min_steps: int,
    camera_info: dict[str, Any] | None = None,
    task_state: dict[str, Any] | None = None,
) -> str:
    ctx = get_context(payload)
    lines = [
        "You are an embodied visual reasoning agent for unobserved scene changes.",
        f"Task type: {ctx['task_type']}",
        f"Question: {ctx['question']}",
    ]
    if ctx["phase_1"]:
        lines.append(f"Phase 1 description: {ctx['phase_1']}")
    if ctx["phase_2"]:
        lines.append(f"Phase 2 description: {ctx['phase_2']}")
    if ctx["options"]:
        lines.append("Options: " + ", ".join(ctx["options"]))
    if ctx["boxes"]:
        box_line = ", ".join(f"{item['label']} ({item['container_category'] or 'box'})" for item in ctx["boxes"])
        lines.append(f"Box identities in the scene: {box_line}")
    lines.extend([
        "",
        "Important scene setup:",
        "  - The first reference image is the original Phase 1 image.",
        "  - The second reference image is the original Phase 2 image.",
        "  - The reference images are authoritative for the before/after change.",
        "  - The simulator view is extra spatial context and may not contain generated hidden contents.",
        "",
        "You will receive those two reference images plus recent exploration views, with the CURRENT view always last.",
        "Use the reference images to understand the before/after change, and use exploration views only when they add useful spatial context.",
        "",
        "Output EXACTLY one JSON object and nothing else:",
        "{",
        '  "action": "<move_forward|move_backward|move_left|move_right|move_up|move_down|turn_left|turn_right|turn_up|turn_down|stop>",',
        '  "reasoning": "<brief explanation>",',
        '  "answer": "<one option exactly as written or not sure>",',
        '  "confidence": <float 0.0-1.0>',
        "}",
        "",
        "Rules:",
        "  - Output only valid JSON.",
        "  - The answer should match one listed option exactly when confident.",
        "  - If the current view is insufficient, choose a movement action instead of stopping.",
        "  - Do not assume simulator-only details override the two reference images.",
        "  - Use turn actions when rotation is more helpful than translation.",
        f"  - Before step {min_steps}, confidence should usually remain <= 0.5 unless the answer is extremely obvious.",
        f"  - Do not stop early unless confidence is at least {threshold:.2f} or there is no useful exploration left.",
        "  - If uncertain, answer 'not sure' and continue exploring instead of guessing too early.",
    ])
    return "\n".join(lines)


def build_force_choice_prompt(
    payload: dict[str, Any],
    camera_info: dict[str, Any] | None = None,
    task_state: dict[str, Any] | None = None,
) -> str:
    ctx = get_context(payload)
    lines = ["Exploration budget is exhausted.", f"Question: {ctx['question']}"]
    if ctx["options"]:
        lines.append("Options: " + ", ".join(ctx["options"]))
    lines.extend([
        "Choose the single best option using the Phase 1 image, the Phase 2 image, and the exploration evidence.",
        "Do not answer 'not sure'.",
        "Output EXACTLY one JSON object and nothing else:",
        '{"answer": "<one listed option exactly as written>", "confidence": <float 0.0-1.0>, "reasoning": "<brief explanation>"}',
    ])
    return "\n".join(lines)


def parse_model_output(parsed: dict[str, Any]) -> dict[str, Any]:
    ctx_options = parsed.get("_options") if isinstance(parsed.get("_options"), list) else []
    action = normalize_text(parsed.get("action")).lower() or "move_forward"
    if action not in VALID_ACTIONS:
        action = "move_forward"
    answer = canonicalize_prediction(parsed.get("answer"), ctx_options) if ctx_options else normalize_text(parsed.get("answer"))
    if normalize_choice(answer) in {"", "not sure", "unsure", "unknown"}:
        answer = "not sure"
    try:
        confidence = float(parsed.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    return {
        **parsed,
        "action": action,
        "answer": answer,
        "confidence": max(0.0, min(1.0, confidence)),
        "reasoning": normalize_text(parsed.get("reasoning")) or "no reasoning provided",
    }


def should_stop(parsed: dict[str, Any], history: list[dict[str, Any]], step: int, max_steps: int, min_steps: int, threshold: float) -> tuple[bool, str]:
    if float(parsed.get("confidence", 0.0)) >= threshold and step >= min_steps:
        return True, "confidence_threshold"
    if parsed.get("action") == "stop":
        return True, "model_stop"
    if step == max_steps:
        return True, "max_steps"
    return False, ""


def resolve_final_answer(history: list[dict[str, Any]]) -> tuple[str, int]:
    for item in reversed(history):
        answer = normalize_text(item.get("answer"))
        if answer and answer.lower() != "not sure":
            return answer, int(item["step"])
    if history:
        return normalize_text(history[-1].get("answer")) or "not sure", int(history[-1]["step"])
    return "not sure", -1


def needs_force_final_choice(answer: str, stop_reason: str) -> bool:
    return normalize_choice(answer) in {"", "not sure", "unsure", "unknown"}


def score(
    payload: dict[str, Any],
    final_answer: dict[str, Any],
    camera_info: dict[str, Any] | None = None,
    task_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ctx = get_context(payload)
    predicted_answer = canonicalize_prediction((final_answer or {}).get("answer"), ctx["options"])
    accuracy = compute_accuracy(predicted_answer, ctx["ground_truth"]) if predicted_answer else 0.0
    return {
        "task_type": ctx["task_type"],
        "question": ctx["question"],
        "options": ctx["options"],
        "ground_truth": ctx["ground_truth"],
        "predicted_answer": predicted_answer,
        "accuracy": accuracy,
        "correct": bool(accuracy),
        "reference_images": (task_state or {}).get("reference_image_paths", []),
    }
