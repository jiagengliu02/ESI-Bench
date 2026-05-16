from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import traceback
from pathlib import Path

os.environ.setdefault("OG_DISABLE_EMITTER_APIS", "1")

import deformable_camera as base  # noqa: E402


DEFAULT_OUTPUT = Path("outputs/qualitative_task_demo/videos/deformable_cloth_fall_on_can.mp4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record a cloth falling from above onto a can in an OmniGibson room."
    )
    parser.add_argument("--scene", required=True, help="Scene model name.")
    parser.add_argument("--room", required=True, help="Room instance name to load.")
    parser.add_argument("--floor", default=None, help="Optional floor object name inside the room.")
    parser.add_argument("--robot", default="R1")
    parser.add_argument("--run-idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None, help="Override seed. Defaults to the batch stable seed.")
    parser.add_argument("--small-item-category", type=str, default=base.DEFAULT_SMALL_ITEM_CATEGORY)
    parser.add_argument("--small-item-model", type=str, default=base.DEFAULT_SMALL_ITEM_MODEL)
    parser.add_argument("--cloth-category", type=str, default=base.DEFAULT_CLOTH_CATEGORY)
    parser.add_argument("--cloth-model", type=str, default=base.DEFAULT_CLOTH_MODEL)
    parser.add_argument("--disable-trav-map-check", action="store_true")
    parser.add_argument("--fast-mode", dest="fast_mode", action="store_true", default=True)
    parser.add_argument("--normal-mode", dest="fast_mode", action="store_false")
    parser.add_argument("--scene-warmup-steps", type=int, default=None)
    parser.add_argument("--item-add-steps", type=int, default=None)
    parser.add_argument("--item-settle-steps", type=int, default=None)
    parser.add_argument("--post-item-freeze-steps", type=int, default=None)
    parser.add_argument("--cloth-add-steps", type=int, default=None)
    parser.add_argument(
        "--cloth-settle-steps",
        type=int,
        default=None,
        help="Number of cloth fall physics steps to record before freezing the result. Defaults to 60.",
    )
    parser.add_argument("--capture-render-steps", type=int, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output mp4 path.")
    parser.add_argument("--fps", type=float, default=30.0, help="Output video FPS.")
    parser.add_argument("--frames", type=int, default=None, help="Override total video frames.")
    parser.add_argument("--video-seconds", type=float, default=5.0, help="Video length when --frames is not provided.")
    parser.add_argument("--width", type=int, default=base.batch.DEFAULT_CAPTURE_WIDTH)
    parser.add_argument("--height", type=int, default=base.batch.DEFAULT_CAPTURE_HEIGHT)
    parser.add_argument(
        "--cloth-clearance",
        type=float,
        default=0.55,
        help="Initial cloth center height above the can top, in meters.",
    )
    parser.add_argument(
        "--initial-downward-speed",
        type=float,
        default=base.batch.DEFAULT_CLOTH_DOWNWARD_SPEED_MPS,
        help="Initial cloth downward speed in m/s at release time.",
    )
    parser.add_argument(
        "--sim-steps-per-frame",
        type=int,
        default=1,
        help="Physics steps advanced for each recorded frame after release.",
    )
    parser.add_argument(
        "--sim-slowdown",
        type=float,
        default=1.0,
        help="Slow down simulated time during the recorded fall. 4.0 advances one quarter as much sim time per recorded frame.",
    )
    parser.add_argument(
        "--preroll-frames",
        type=int,
        default=12,
        help="Still frames recorded before the cloth is released.",
    )
    parser.add_argument("--camera-azimuth-deg", type=float, default=-90.0)
    parser.add_argument("--camera-distance", type=float, default=None)
    parser.add_argument("--camera-height-offset", type=float, default=None)
    parser.add_argument("--keep-alive", action="store_true", help="Keep simulator alive after recording finishes.")
    parser.add_argument(
        "--cleanup-on-exit",
        action="store_true",
        help="Call env.close() and og.clear() before exiting. Disabled by default because OmniGibson/Kit can segfault during teardown.",
    )
    return parser.parse_args()


def _spawn_cloth_ready_above_item(
    scene,
    item_obj,
    cloth_spec: dict,
    seed: int,
    args: argparse.Namespace,
):
    cloth_name = f"{base.batch.RENDER_OBJECT_PREFIX}falling_cloth_{seed:010d}"
    cloth_obj = base.DatasetObject(
        name=cloth_name,
        category=cloth_spec["category"],
        model=cloth_spec["model"],
        prim_type=base.PrimType.CLOTH,
        abilities={"cloth": {}},
        load_config={"default_configuration": "settled"},
    )
    scene.add_object(cloth_obj)
    base.scene_utils._step_sim(args.cloth_add_steps)
    cloth_configuration_used = base.batch._reset_cloth_to_best_configuration(cloth_obj)
    try:
        cloth_obj.root_link.mass = float(base.batch.DEFAULT_CLOTH_MASS_KG)
    except Exception:
        pass

    item_bbox_min, item_bbox_max = base.scene_utils._read_current_aabb(item_obj)
    item_center = [
        float((item_bbox_min[0] + item_bbox_max[0]) * 0.5),
        float((item_bbox_min[1] + item_bbox_max[1]) * 0.5),
        float((item_bbox_min[2] + item_bbox_max[2]) * 0.5),
    ]
    item_top_z = float(item_bbox_max[2])
    cloth_drop_pos = [
        float(item_center[0]),
        float(item_center[1]),
        float(item_top_z) + float(args.cloth_clearance),
    ]
    cloth_obj.set_position_orientation(
        position=base.th.tensor(cloth_drop_pos, dtype=base.th.float32),
        orientation=base.th.tensor([0.0, 0.0, 0.0, 1.0], dtype=base.th.float32),
    )
    base.batch._set_velocity_zero(cloth_obj)
    return cloth_name, cloth_obj, cloth_configuration_used, cloth_drop_pos


def _compute_fall_camera(item_obj, cloth_obj, cloth_drop_pos, args: argparse.Namespace) -> dict[str, object]:
    item_bbox_min, item_bbox_max = base.scene_utils._read_current_aabb(item_obj)
    item_center, item_extents = base.batch._bbox_center_and_extents(item_bbox_min, item_bbox_max)

    cloth_bbox_min, cloth_bbox_max = base.scene_utils._read_current_aabb(cloth_obj)
    _, cloth_extents = base.batch._bbox_center_and_extents(cloth_bbox_min, cloth_bbox_max)
    footprint_xy_m = float(max(cloth_extents[0], cloth_extents[1], item_extents[0], item_extents[1]))

    target_xyz = [
        float(item_center[0]),
        float(item_center[1]),
        float(item_bbox_max[2]) + min(float(args.cloth_clearance) * 0.45, 0.35),
    ]
    default_distance, default_height_offset = base.batch._main_view_camera_params(max(footprint_xy_m, 0.45))
    distance_m = float(args.camera_distance) if args.camera_distance is not None else max(default_distance, 1.25)
    height_offset_m = (
        float(args.camera_height_offset)
        if args.camera_height_offset is not None
        else max(default_height_offset, 0.55)
    )
    eye = base.batch._camera_eye_for_azimuth(
        target_xyz,
        float(args.camera_azimuth_deg),
        distance_m=distance_m,
        height_offset_m=height_offset_m,
    )
    quat = base.batch._look_at_quaternion(eye, target_xyz)
    return {
        "target_xyz": [float(v) for v in target_xyz],
        "position": [float(v) for v in eye],
        "quaternion_xyzw": [float(v) for v in quat],
        "distance_to_item_m": float(distance_m),
        "height_offset_m": float(height_offset_m),
        "cloth_footprint_xy_m": float(footprint_xy_m),
        "cloth_initial_position": [float(v) for v in cloth_drop_pos],
    }


def _release_cloth(cloth_obj, downward_speed_mps: float) -> None:
    try:
        cloth_obj.root_link.set_linear_velocity(
            base.th.tensor([0.0, 0.0, -float(downward_speed_mps)], dtype=base.th.float32)
        )
    except Exception:
        base.batch._set_velocity_zero(cloth_obj)


def _record_fall_video(
    args: argparse.Namespace,
    capture_camera,
    item_obj,
    cloth_obj,
    camera_pose: dict[str, object],
) -> dict[str, object]:
    total_frames = int(args.frames) if args.frames is not None else max(int(round(float(args.video_seconds) * float(args.fps))), 2)
    render_frames = max(int(args.capture_render_steps), 1)
    preroll_frames = max(min(int(args.preroll_frames), total_frames - 1), 0)
    sim_steps_per_frame = max(int(args.sim_steps_per_frame), 1)
    sim_slowdown = max(float(args.sim_slowdown), 1.0)
    recorded_settle_steps = max(int(args.cloth_settle_steps), 0)
    step_credit = 0.0

    base._set_capture_camera_pose_static(
        capture_camera,
        camera_pose["position"],
        camera_pose["quaternion_xyzw"],
        render_frames=render_frames,
    )
    writer = base._make_writer(args.output, args.fps, args.width, args.height)
    released = False
    settled = False
    simulated_steps = 0
    try:
        for frame_idx in range(total_frames):
            if frame_idx >= preroll_frames and not settled:
                if not released:
                    _release_cloth(cloth_obj, float(args.initial_downward_speed))
                    released = True
                step_credit += float(sim_steps_per_frame) / sim_slowdown
                steps_this_frame = min(
                    int(math.floor(step_credit)),
                    max(recorded_settle_steps - simulated_steps, 0),
                )
                step_credit -= float(steps_this_frame)
                for _ in range(steps_this_frame):
                    base.og.sim.step()
                    simulated_steps += 1
                if simulated_steps >= recorded_settle_steps:
                    base.batch._set_velocity_zero(item_obj)
                    base.batch._set_velocity_zero(cloth_obj)
                    settled = True
            rgb = base._capture_rgb(capture_camera, render_frames=render_frames)
            if rgb.shape[1] != int(args.width) or rgb.shape[0] != int(args.height):
                rgb = base.cv2.resize(rgb, (int(args.width), int(args.height)), interpolation=base.cv2.INTER_AREA)
            writer.write(base.cv2.cvtColor(rgb, base.cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    return {
        "output": str(args.output),
        "fps": float(args.fps),
        "frames": int(total_frames),
        "preroll_frames": int(preroll_frames),
        "sim_steps_per_frame": int(sim_steps_per_frame),
        "sim_slowdown": float(sim_slowdown),
        "recorded_settle_steps": int(recorded_settle_steps),
        "simulated_steps": int(simulated_steps),
        "settled_and_frozen": bool(settled),
        "slowdown_method": "frame_step_scheduler",
        "initial_downward_speed_mps": float(args.initial_downward_speed),
    }


def main() -> None:
    raw_args = parse_args()
    requested_cloth_settle_steps = raw_args.cloth_settle_steps
    args = base.batch._resolve_runtime_steps(raw_args)
    args.cloth_settle_steps = 60 if requested_cloth_settle_steps is None else max(0, int(requested_cloth_settle_steps))
    seed = int(args.seed) if args.seed is not None else int(base.batch._stable_seed(args.scene, args.room, args.run_idx))
    rng = random.Random(seed)

    item_spec = base._pick_small_item(
        rng,
        {str(args.small_item_category): [str(args.small_item_model)]},
        args.small_item_category,
        args.small_item_model,
    )
    cloth_spec = base._pick_cloth(
        rng,
        [
            {
                "category": str(args.cloth_category),
                "model": str(args.cloth_model),
                "mass_kg": float(base.batch.DEFAULT_CLOTH_MASS_KG),
                "group": "manual_default",
            }
        ],
        args.cloth_category,
        args.cloth_model,
    )

    env = None
    try:
        config = base.batch._build_config(args.scene, args.robot, room_name=args.room)
        env = base.og.Environment(configs=config)
        scene = env.scene

        base.batch._configure_sim_for_cloth_drop()
        base.scene_utils._set_viewer_camera_fov(base.batch.DEFAULT_CAMERA_FOV_DEG)
        base.scene_utils._step_sim(args.scene_warmup_steps)

        (
            floor_record,
            room_bbox_world_xy,
            resolved_room_instance,
            room_objects,
            blockers,
            trav_map,
            trav_map_img,
            preferred_center_xy,
        ) = base._load_room_context(scene, args.scene, args.room, args.floor, args.disable_trav_map_check)

        item_xy = base.batch._sample_free_position(
            rng=rng,
            floor_record=floor_record,
            blockers=blockers,
            preferred_center_xy=preferred_center_xy,
            room_bbox_world_xy=room_bbox_world_xy,
            trav_map=trav_map,
            trav_map_img=trav_map_img,
            clearance_m=base.batch.DEFAULT_ITEM_FREE_RADIUS_M,
        )
        _, item_obj = base._spawn_item(scene, floor_record, item_xy, item_spec, seed, args)
        base.batch._set_velocity_zero(item_obj)
        _, cloth_obj, cloth_configuration_used, cloth_drop_pos = _spawn_cloth_ready_above_item(
            scene, item_obj, cloth_spec, seed, args
        )

        camera_pose = _compute_fall_camera(item_obj, cloth_obj, cloth_drop_pos, args)
        base._set_viewer_camera_pose(camera_pose["position"], camera_pose["quaternion_xyzw"])
        capture_camera = base.batch._create_capture_camera(width=args.width, height=args.height)
        video_info = _record_fall_video(args, capture_camera, item_obj, cloth_obj, camera_pose)

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
                "clearance_above_can_m": round(float(args.cloth_clearance), 4),
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
        print(f"[deformable_fall] Cloth fall video saved to {args.output}", flush=True)
        if args.keep_alive:
            print("[deformable_fall] Recording finished. Simulator stays alive. Press Ctrl+C to exit.", flush=True)
            while True:
                base._render_only(1)
        if not args.cleanup_on_exit:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
    except KeyboardInterrupt:
        print("[deformable_fall] Exit requested by user (Ctrl+C).", flush=True)
        if not args.cleanup_on_exit:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(130)
    except Exception:
        traceback.print_exc()
        if not args.cleanup_on_exit:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
    finally:
        if args.cleanup_on_exit:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
            try:
                base.og.clear()
            except Exception:
                pass


if __name__ == "__main__":
    main()
