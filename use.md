# Qualitative task-action videos

Run from the repo root:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate behavior
mkdir -p outputs/qualitative_task_demo/videos
```

These commands use `--motion task-demo`, which picks a task-specific trajectory/action:

- `cognitivemap`: fly the camera along the source-room to destination-room path and draw a small trajectory overlay in the video.
- `deformable`: approach the target bottle and lift the cloth away.
- `unobserved_changes`: render phase 1 and phase 2 as two separate videos, each approaching the same open box.
- `counting`: scan around the count target cluster.

All videos are rendered into the isolated directory `outputs/qualitative_task_demo/videos/`.

## Deformable / cover / bottle under cloth

```bash
python scripts/deforable_camera.py
```

## Mirror / mirror_correspondence

```bash
python scripts/record_scene_video.py \
  --task mirror \
  --metadata "json-tmp/mirror/mirror_correspondence/Beechwood_0_garden/living_room_0/q_006.json" \
  --output outputs/qualitative_task_demo/videos/mirror_correspondence_Beechwood_0_garden_living_room_0_q006.mp4 \
  --motion target-orbit \
  --orbit-deg 180 \
  --frames 300 \
  --fps 30
```

## Counting / milk cartons

`json-tmp/counting` does not contain bottle-counting examples. This command uses a milk-carton counting example instead. The script also corrects the JSON's target proxy from `club_sandwich` back to the semantic target category `milk_carton` when loading count targets.

```bash
python scripts/record_scene_video.py \
  --task counting \
  --metadata "json-tmp/counting/observation_divided/grocery_store_cafe/dining_room_0/q_003.json" \
  --output outputs/qualitative_task_demo/videos/counting_milk_carton_grocery_store_cafe_dining_room_0_q003.mp4 \
  --motion task-demo \
  --frames 300 \
  --fps 30
```

## Cognitivemap / Long-Horizon Navigation

```bash
python scripts/record_scene_video.py \
  --task cognitivemap \
  --metadata "json-tmp/cognitivemap/Long-Horizon Navigation/Beechwood_0_garden/full_scene/q_011.json" \
  --output outputs/qualitative_task_demo/videos/cognitivemap_path_Beechwood_0_garden_q011.mp4 \
  --motion task-demo \
  --frames 360 \
  --fov-deg 100
  --fps 30 \
  --full-scene
```

## Cognitivemap / Topological Connectivity

```bash
python scripts/record_scene_video.py \
  --task cognitivemap \
  --metadata "json-tmp/cognitivemap/Topological Connectivity/restaurant_asian/full_scene/q_004.json" \
  --output outputs/qualitative_task_demo/videos/cognitivemap_path_restaurant_asian_q004.mp4 \
  --motion task-demo \
  --frames 420 \
  --fps 30 \
  --fov-deg 100 \
  --full-scene
```

## Unobserved changes / change_detection / phase 1

```bash
python scripts/record_scene_video.py \
  --task unobserved_changes \
  --metadata "json-tmp/unobserved_changes/change_detection/grocery_store_cafe/dining_room_0/q_000.json" \
  --output outputs/qualitative_task_demo/videos/unobserved_changes_phase1_box_grocery_store_cafe_q000.mp4 \
  --motion task-demo \
  --unobserved-phase phase1 \
  --frames 240 \
  --fps 30
```

## Unobserved changes / change_detection / phase 2

```bash
python scripts/record_scene_video.py \
  --task unobserved_changes \
  --metadata "json-tmp/unobserved_changes/change_detection/grocery_store_cafe/dining_room_0/q_000.json" \
  --output outputs/qualitative_task_demo/videos/unobserved_changes_phase2_box_grocery_store_cafe_q000.mp4 \
  --motion task-demo \
  --unobserved-phase phase2 \
  --frames 240 \
  --fps 30
```
