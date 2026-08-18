# ACT Five-Camera RGBD Design

## Goal

Train dual-arm relative-joint ACT on `730_subtask_with_depth` with five visual inputs: one top RGB camera, two wrist RGB cameras, and two wrist depth cameras.

## Data Flow

The dataset returns depth in metres through `--dataset.depth_output_unit=m`. ACT receives five configured image keys in this fixed order: top RGB, left wrist RGB, right wrist RGB, left wrist depth, and right wrist depth. RGB frames retain the existing fixed crop/resize path. Depth frames use the same spatial crop/resize, are clipped to a finite physical interval, and are expanded from one channel to three channels only at the depth encoder boundary.

## Model

`five_camera_rgbd` is an ACT-specific visual mode. A pretrained ResNet18 encodes the three RGB inputs. A separate pretrained ResNet18 encodes both depth inputs, so depth values are not normalized with RGB ImageNet statistics. Each of the five encoded maps is projected and emitted as its own token sequence to the ACT transformer.

## Training Contract

Training uses the existing dual-arm 14D relative-state/action representation, gripper indices `[6,13]`, separately computed full-data q01/q99 statistics, state noise, batch size 32, 500000 steps, checkpoint interval 50000, and no validation. The RGB-only run remains stopped and its output is retained. The RGBD run uses a new output directory.

## Safety Checks

Configuration validation requires exactly the five expected keys. Tests verify that depth affects the two depth token streams, that all five feature maps reach the transformer path, and that a five-camera launcher passes metre depth output and the established relative-joint parameters.
