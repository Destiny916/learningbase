# Indoor Lighting Baseline

本记录冻结室内主光、补光、轮廓光改造前的已验证版本。

## Lighting Contract

- Headlight：ambient `0.285 0.285 0.285`，diffuse `0.627 0.627 0.627`，specular `0 0 0`。
- 四盏对称点光源，位置为 `(-1.8, ±1.8, 4.2)` 和 `(1.8, ±1.8, 4.2)`。
- 四盏灯均关闭阴影，使用相同的 ambient、diffuse 和 attenuation。
- 场景为渐变 skybox 加无碰撞网格地面。

## Full-run Evidence

- Run：`scheme_b_threshold_full`。
- Checkpoint：`checkpoints/Act/popcorn/0450000/pretrained_model`。
- Source frames：1391。
- Control frames：2784。
- Strict verification：passed。
- Eye camera：1280×720，30 FPS，0 dropped frames。
- Mean render：9.857318 ms。
- P95 render：11.811284 ms。
- Max render：19.150168 ms。
- Run score：93.173883。

## Source Hashes

```text
3d7c5833fd2129dd954cda17c2346e422c36192b89fe68f1327b5b84cf4af7ff  w1_act/simulation/model.py
36d7f37adc350ec0863ee65e585171c872f78ea60773316eaae63fedd0596a77  w1_act/simulation/tests/test_model.py
eed89a18f0ee97463602e5d95d316065ab3ce63fe40ef85548b122bf98102463  w1_act/evaluation/verify.py
```

## Artifact Hashes

```text
e2ed1232281c6e335e732f06aa7b62c8f263cf88a0b8b394daf4d15e124dfaff  summary.json
c03566af356db807fa76f92fa7964acb605d32602b3b15a95236f0e65c8bfc73  trajectory.npz
d9060e32c40c507b82567418cbe960aef4ec3dc000436ad43e60393da181bb70  verification.json
e89737bb30dd5578552165f5e669f75637b6fd12fb8b3a0f6443f143670dc6af  recording.rrd
```
