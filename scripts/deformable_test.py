from __future__ import annotations

import argparse
import math
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("OG_DISABLE_EMITTER_APIS", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_DEFORMABLE_DIR = REPO_ROOT / "src" / "dataset_generation" / "task_deformable"
TASK_MIRROR_DIR = REPO_ROOT / "src" / "dataset_generation" / "task_mirror"
for path in (TASK_DEFORMABLE_DIR, TASK_MIRROR_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

try:
    import cv2  # noqa: E402
    import numpy as np  # noqa: E402
    import omnigibson as og  # noqa: E402
    import torch as th  # noqa: E402
    from omnigibson.objects.dataset_object import DatasetObject  # noqa: E402
    from omnigibson.utils.constants import PrimType  # noqa: E402
    import batch_deformable as batch  # noqa: E402
    import batch_mirror_distance as scene_utils  # noqa: E402
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local OG env
    if exc.name == "omnigibson":
        raise SystemExit(
            "omnigibson is not available in the current Python environment. "
            "Please run this script inside the same OmniGibson-enabled environment used by the other scene scripts."
        ) from exc
    raise


DEFAULT_SMALL_ITEM_CATEGORY = "can_of_tomato_paste"
DEFAULT_SMALL_ITEM_MODEL = "sqqdzb"
DEFAULT_CLOTH_CATEGORY = "dishtowel"
DEFAULT_CLOTH_MODEL = "ltydgg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load one room, place a small item and a cloth using deformable batch logic, then record a smooth orbit video."
    )
    parser.add_argument("--scene", required=True, help="Scene model name.")
    parser.add_argument("--room", required=True, help="Room instance name to load.")
    parser.add_argument("--floor", default=None, help="Optional floor object name inside the room.")
    parser.add_argument("--robot", default="R1")
    parser.add_argument("--run-idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None, help="Override seed. Defaults to the batch stable seed.")
    parser.add_argument("--small-item-category", type=str, default=DEFAULT_SMALL_ITEM_CATEGORY)
    parser.add_argument("--small-item-model", type=str, default=DEFAULT_SMALL_ITEM_MODEL)
    parser.add_argument("--cloth-category", type=str, default=DEFAULT_CLOTH_CATEGORY)
    parser.add_argument("--cloth-model", type=str, default=DEFAULT_CLOTH_MODEL)
    parser.add_argument("--disable-trav-map-check", action="store_true")
    parser.add_argument("--fast-mode", dest="fast_mode", action="store_true", default=True)
    parser.add_argument("--normal-mode", dest="fast_mode", action="store_false")
    parser.add_argument("--scene-warmup-steps", type=int, default=None)
    parser.add_argument("--item-add-steps", type=int, default=None)
    parser.add_argument("--item-settle-steps", type=int, default=None)
    parser.add_argument("--post-item-freeze-steps", type=int, default=None)
    parser.add_argument("--cloth-add-steps", type=int, default=None)
    parser.add_argument("--cloth-settle-steps", type=int, default=None)
    parser.add_argument("--capture-render-steps", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/qualitative_task_demo/videos/deformable_test_orbit.mp4"),
        help="Output mp4 path.",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Output video FPS.")
    parser.add_argument("--frames", type=int, default=None, help="Override total video frames.")
    parser.add_argument("--video-seconds", type=float, default=4.0, help="Video length when --frames is not provided.")
    parser.add_argument("--width", type=int, default=batch.DEFAULT_CAPTURE_WIDTH, help="Output video width.")
    parser.add_argument("--height", type=int, default=batch.DEFAULT_CAPTURE_HEIGHT, help="Output video height.")
    parser.add_argument("--keep-alive", action="store_true", help="Keep simulator alive after recording finishes.")
    return parser.parse_args()


def _pick_small_item(
    rng: random.Random,
    catalog: dict[str, list[str]],
    category: str | None,
    model: str | None,
) -> dict[str, str]:
    if category is None and model is not None:
        raise ValueError("--small-item-model requires --small-item-category.")
    if category is None:
        sampled = batch._sample_target_and_distractors(rng, catalog)
        return dict(sampled["target"])
    if category not in catalog:
        raise ValueError(f"Unknown small-item category: {category}")
    if model is None:
        model = rng.choice(catalog[category])
    elif model not in catalog[category]:
        raise ValueError(f"Model '{model}' is not available under small-item category '{category}'.")
    return {"category": str(category), "model": str(model)}


def _pick_cloth(
    rng: random.Random,
    catalog: list[dict],
    category: str | None,
    model: str | None,
) -> dict:
    if category is None and model is not None:
        raise ValueError("--cloth-model requires --cloth-category.")
    if category is None:
        return batch._sample_cloth_asset(rng, catalog)

    matches = [entry for entry in catalog if str(entry.get("category")) == str(category)]
    if not matches:
        raise ValueError(f"Unknown cloth category: {category}")
    if model is None:
        return dict(rng.choice(matches))

    for entry in matches:
        if str(entry.get("model")) == str(model):
            return dict(entry)
    raise ValueError(f"Model '{model}' is not available under cloth category '{category}'.")


def _set_viewer_camera_pose(position, quat) -> None:
    cam = batch._create_capture_camera(width=batch.DEFAULT_CAPTURE_WIDTH, height=batch.DEFAULT_CAPTURE_HEIGHT)
    batch._set_capture_camera_pose(cam, position, quat)


def _render_only(frames: int) -> None:
    for _ in range(max(int(frames), 0)):
        try:
            og.sim.render()
        except Exception:
            break


def _rgb_obs_to_uint8(frame) -> np.ndarray:
    if hasattr(frame, "detach"):
        frame = frame.detach()
    if hasattr(frame, "cpu"):
        frame = frame.cpu()
    image = np.array(frame)
    if not (image.ndim == 3 and image.shape[2] in (3, 4)):
        raise ValueError(f"Viewer camera returned invalid rgb shape {getattr(image, 'shape', None)}")
    image = image[..., :3]
    if np.issubdtype(image.dtype, np.floating):
        if image.size and float(np.nanmax(image)) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    return image


def _set_capture_camera_pose_static(cam, position, quat, render_frames: int) -> None:
    cam.set_position_orientation(
        position=th.tensor([float(v) for v in position], dtype=th.float32),
        orientation=th.tensor([float(v) for v in quat], dtype=th.float32),
    )
    try:
        cam.clipping_range = th.tensor([0.01, 100.0], dtype=th.float32)
    except Exception:
        pass
    _render_only(render_frames)


def _capture_rgb(cam, render_frames: int) -> np.ndarray:
    _render_only(render_frames)
    obs, info = cam.get_obs()
    if not isinstance(obs, dict):
        raise RuntimeError(f"get_obs() returned non-dict obs: {type(obs)}")
    if "rgb" not in obs:
        raise RuntimeError(f"RGB not found in obs. keys={list(obs.keys())}, info={info}")
    return _rgb_obs_to_uint8(obs["rgb"])


def _make_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    return writer


def _ease_in_out(progress: float) -> float:
    progress = min(max(float(progress), 0.0), 1.0)
    return progress * progress * progress * (progress * (progress * 6.0 - 15.0) + 10.0)


def _orbit_camera_pose(start_pos, target_xyz, frame_idx: int, total_frames: int):
    start_pos_np = np.array([float(v) for v in start_pos], dtype=float)
    target_np = np.array([float(v) for v in target_xyz], dtype=float)
    progress = 1.0 if total_frames <= 1 else frame_idx / float(total_frames - 1)
    eased = _ease_in_out(progress)

    relative_xy = start_pos_np[:2] - target_np[:2]
    radius = float(np.linalg.norm(relative_xy))
    if radius < 1e-6:
        relative_xy = np.array([1.0, 0.0], dtype=float)
        radius = 1.0

    start_angle = math.atan2(float(relative_xy[1]), float(relative_xy[0]))
    angle = start_angle + math.radians(90.0) * eased
    end_z = float(target_np[2])
    z = float(start_pos_np[2]) * (1.0 - eased) + end_z * eased

    position = np.array(
        [
            float(target_np[0]) + radius * math.cos(angle),
            float(target_np[1]) + radius * math.sin(angle),
            z,
        ],
        dtype=float,
    )
    quaternion = np.array(batch._look_at_quaternion(position.tolist(), target_np.tolist()), dtype=float)
    return position, quaternion


def _record_orbit_video(args: argparse.Namespace, capture_camera, start_pos, target_xyz) -> dict[str, object]:
    total_frames = int(args.frames) if args.frames is not None else max(int(round(float(args.video_seconds) * float(args.fps))), 2)
    render_frames = max(int(args.capture_render_steps), 1)
    writer = _make_writer(args.output, args.fps, args.width, args.height)
    end_position = None

    try:
        for frame_idx in range(total_frames):
            frame_pos, frame_quat = _orbit_camera_pose(start_pos, target_xyz, frame_idx, total_frames)
            _set_capture_camera_pose_static(capture_camera, frame_pos, frame_quat, render_frames=render_frames)
            rgb = _capture_rgb(capture_camera, render_frames=render_frames)
            if rgb.shape[1] != int(args.width) or rgb.shape[0] != int(args.height):
                rgb = cv2.resize(rgb, (int(args.width), int(args.height)), interpolation=cv2.INTER_AREA)
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            end_position = frame_pos
    finally:
        writer.release()

    return {
        "output": str(args.output),
        "fps": float(args.fps),
        "frames": total_frames,
        "start_position": [float(v) for v in start_pos],
        "end_position": None if end_position is None else [float(v) for v in end_position],
        "target_xyz": [float(v) for v in target_xyz],
    }


def _load_room_context(scene, scene_name: str, room_name: str, floor_name: str | None, disable_trav_map_check: bool):
    room_objects = scene_utils._collect_room_objects(scene, room_name)
    floor_record = scene_utils._select_floor(
        room_objects,
        floor_name,
        agent_pos=(0.0, 0.0, 0.0),
        room_name=room_name,
    )
    room_bbox_world_xy, resolved_room_instance = scene_utils._resolve_room_bbox_world_xy(scene, room_name, floor_record)
    blockers = [record for record in room_objects if scene_utils._is_floor_blocker(record, floor_record.bbox_max[2])]

    trav_map = None
    trav_map_img = None
    if not disable_trav_map_check:
        try:
            trav_map, trav_map_img = scene_utils._trav_map_floor_image(scene, floor_idx=0, scene_name=scene_name)
        except Exception as exc:
            print(
                f"[deformable_test] Failed to load traversability map, fallback to bbox/grid placement: "
                f"{exc.__class__.__name__}: {exc}",
                flush=True,
            )

    if room_bbox_world_xy is not None:
        preferred_center_xy = scene_utils._room_bbox_center_xy(room_bbox_world_xy)
    else:
        preferred_center_xy = th.tensor(
            [float(floor_record.center[0]), float(floor_record.center[1])],
            dtype=th.float32,
        )

    return floor_record, room_bbox_world_xy, resolved_room_instance, room_objects, blockers, trav_map, trav_map_img, preferred_center_xy


def _spawn_item(scene, floor_record, item_xy, item_spec: dict[str, str], seed: int, args: argparse.Namespace):
    item_name = f"{batch.RENDER_OBJECT_PREFIX}item_{seed:010d}"
    item_obj = DatasetObject(name=item_name, category=item_spec["category"], model=item_spec["model"])
    scene.add_object(item_obj)
    scene_utils._step_sim(args.item_add_steps)
    item_obj.set_position_orientation(
        position=th.tensor(
            [
                float(item_xy[0]),
                float(item_xy[1]),
                float(floor_record.bbox_max[2]) + batch.DEFAULT_ITEM_DROP_HEIGHT_M,
            ],
            dtype=th.float32,
        ),
        orientation=th.tensor([0.0, 0.0, 0.0, 1.0], dtype=th.float32),
    )
    scene_utils._step_sim(args.item_settle_steps)
    batch._set_velocity_zero(item_obj)
    scene_utils._step_sim(args.post_item_freeze_steps)
    return item_name, item_obj


def _spawn_cloth(scene, item_obj, cloth_spec: dict, seed: int, args: argparse.Namespace):
    cloth_name = f"{batch.RENDER_OBJECT_PREFIX}cloth_{seed:010d}"
    cloth_obj = DatasetObject(
        name=cloth_name,
        category=cloth_spec["category"],
        model=cloth_spec["model"],
        prim_type=PrimType.CLOTH,
        abilities={"cloth": {}},
        load_config={"default_configuration": "settled"},
    )
    scene.add_object(cloth_obj)
    scene_utils._step_sim(args.cloth_add_steps)
    cloth_configuration_used = batch._reset_cloth_to_best_configuration(cloth_obj)
    try:
        cloth_obj.root_link.mass = float(batch.DEFAULT_CLOTH_MASS_KG)
    except Exception:
        pass

    item_bbox_min, item_bbox_max = scene_utils._read_current_aabb(item_obj)
    item_center = [
        float((item_bbox_min[0] + item_bbox_max[0]) * 0.5),
        float((item_bbox_min[1] + item_bbox_max[1]) * 0.5),
        float((item_bbox_min[2] + item_bbox_max[2]) * 0.5),
    ]
    item_top_z = float(item_bbox_max[2])
    cloth_drop_pos = [
        float(item_center[0]),
        float(item_center[1]),
        float(item_top_z) + batch.DEFAULT_CLOTH_CLEARANCE_ABOVE_ITEM_M,
    ]
    cloth_obj.set_position_orientation(
        position=th.tensor(cloth_drop_pos, dtype=th.float32),
        orientation=th.tensor([0.0, 0.0, 0.0, 1.0], dtype=th.float32),
    )
    try:
        cloth_obj.root_link.set_linear_velocity(
            th.tensor([0.0, 0.0, -float(batch.DEFAULT_CLOTH_DOWNWARD_SPEED_MPS)], dtype=th.float32)
        )
    except Exception:
        batch._set_velocity_zero(cloth_obj)
    scene_utils._step_sim(args.cloth_settle_steps)
    return cloth_name, cloth_obj, cloth_configuration_used, cloth_drop_pos


def _compute_main_camera(item_obj, cloth_obj):
    item_bbox_min, item_bbox_max = scene_utils._read_current_aabb(item_obj)
    item_center_after_cover, _ = batch._bbox_center_and_extents(item_bbox_min, item_bbox_max)

    cloth_bbox_min, cloth_bbox_max = scene_utils._read_current_aabb(cloth_obj)
    cloth_center_after_settle, cloth_extents_after_settle = batch._bbox_center_and_extents(cloth_bbox_min, cloth_bbox_max)
    cover_target = [
        float(cloth_center_after_settle[0]),
        float(cloth_center_after_settle[1]),
        float(max(item_center_after_cover[2], cloth_center_after_settle[2])),
    ]
    cover_footprint_xy_m = float(max(cloth_extents_after_settle[0], cloth_extents_after_settle[1]))
    main_distance_m, main_height_offset_m = batch._main_view_camera_params(cover_footprint_xy_m)
    main_eye = batch._camera_eye_for_azimuth(
        cover_target,
        -90.0,
        distance_m=main_distance_m,
        height_offset_m=main_height_offset_m,
    )
    main_quat = batch._look_at_quaternion(main_eye, cover_target)
    return {
        "target_xyz": [float(v) for v in cover_target],
        "position": [float(v) for v in main_eye],
        "quaternion_xyzw": [float(v) for v in main_quat],
        "distance_to_item_m": float(main_distance_m),
        "height_offset_m": float(main_height_offset_m),
        "cloth_footprint_xy_m": float(cover_footprint_xy_m),
    }


def main() -> None:
    args = batch._resolve_runtime_steps(parse_args())
    seed = int(args.seed) if args.seed is not None else int(batch._stable_seed(args.scene, args.room, args.run_idx))
    rng = random.Random(seed)

    item_spec = _pick_small_item(
        rng,
        {str(args.small_item_category): [str(args.small_item_model)]},
        args.small_item_category,
        args.small_item_model,
    )
    cloth_spec = _pick_cloth(
        rng,
        [
            {
                "category": str(args.cloth_category),
                "model": str(args.cloth_model),
                "mass_kg": float(batch.DEFAULT_CLOTH_MASS_KG),
                "group": "manual_default",
            }
        ],
        args.cloth_category,
        args.cloth_model,
    )

    env = None
    item_obj = None
    cloth_obj = None
    capture_camera = None
    try:
        config = batch._build_config(args.scene, args.robot, room_name=args.room)
        env = og.Environment(configs=config)
        scene = env.scene

        batch._configure_sim_for_cloth_drop()
        scene_utils._set_viewer_camera_fov(batch.DEFAULT_CAMERA_FOV_DEG)
        scene_utils._step_sim(args.scene_warmup_steps)

        (
            floor_record,
            room_bbox_world_xy,
            resolved_room_instance,
            room_objects,
            blockers,
            trav_map,
            trav_map_img,
            preferred_center_xy,
        ) = _load_room_context(scene, args.scene, args.room, args.floor, args.disable_trav_map_check)

        item_xy = batch._sample_free_position(
            rng=rng,
            floor_record=floor_record,
            blockers=blockers,
            preferred_center_xy=preferred_center_xy,
            room_bbox_world_xy=room_bbox_world_xy,
            trav_map=trav_map,
            trav_map_img=trav_map_img,
            clearance_m=batch.DEFAULT_ITEM_FREE_RADIUS_M,
        )
        _, item_obj = _spawn_item(scene, floor_record, item_xy, item_spec, seed, args)
        _, cloth_obj, cloth_configuration_used, cloth_drop_pos = _spawn_cloth(scene, item_obj, cloth_spec, seed, args)
        batch._set_velocity_zero(item_obj)
        batch._set_velocity_zero(cloth_obj)
        camera_pose = _compute_main_camera(item_obj, cloth_obj)
        _set_viewer_camera_pose(camera_pose["position"], camera_pose["quaternion_xyzw"])
        capture_camera = batch._create_capture_camera(width=args.width, height=args.height)

        video_info = _record_orbit_video(
            args,
            capture_camera=capture_camera,
            start_pos=camera_pose["position"],
            target_xyz=camera_pose["target_xyz"],
        )

        summary = {
            "scene": args.scene,
            "room": args.room,
            "resolved_room": resolved_room_instance,
            "floor_name": floor_record.name,
            "seed": seed,
            "small_item": {
                **item_spec,
                "placement_xy": [round(float(item_xy[0]), 4), round(float(item_xy[1]), 4)],
            },
            "cloth": {
                "category": cloth_spec["category"],
                "model": cloth_spec["model"],
                "group": cloth_spec.get("group"),
                "configuration_used": cloth_configuration_used,
                "drop_position": [round(float(v), 4) for v in cloth_drop_pos],
            },
            "camera_pose": {
                "position": [round(float(v), 4) for v in camera_pose["position"]],
                "quaternion_xyzw": [round(float(v), 6) for v in camera_pose["quaternion_xyzw"]],
                "target_xyz": [round(float(v), 4) for v in camera_pose["target_xyz"]],
                "distance_to_item_m": round(float(camera_pose["distance_to_item_m"]), 4),
                "height_offset_m": round(float(camera_pose["height_offset_m"]), 4),
                "cloth_footprint_xy_m": round(float(camera_pose["cloth_footprint_xy_m"]), 4),
            },
            "runtime_steps": {
                "scene_warmup_steps": int(args.scene_warmup_steps),
                "item_add_steps": int(args.item_add_steps),
                "item_settle_steps": int(args.item_settle_steps),
                "post_item_freeze_steps": int(args.post_item_freeze_steps),
                "cloth_add_steps": int(args.cloth_add_steps),
                "cloth_settle_steps": int(args.cloth_settle_steps),
                "capture_render_steps": int(args.capture_render_steps),
            },
            "room_object_count": len(room_objects),
            "blocker_count": len(blockers),
            "video": video_info,
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
        print(f"[deformable_test] Orbit video saved to {args.output}", flush=True)
        if args.keep_alive:
            print("[deformable_test] Recording finished. Simulator stays alive. Press Ctrl+C to exit.", flush=True)
            while True:
                _render_only(1)
    except KeyboardInterrupt:
        print("[deformable_test] Exit requested by user (Ctrl+C).", flush=True)
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        try:
            og.clear()
        except Exception:
            pass


if __name__ == "__main__":
    main()
