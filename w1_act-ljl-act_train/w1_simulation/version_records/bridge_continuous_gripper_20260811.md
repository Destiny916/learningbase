# Bridge 连续夹爪版本记录

本记录冻结动作质量评估改造前的当前版本。该版本保留2 Hz异步重规划、200 ms推理延迟模型、
`sample_factor=2` 的60 Hz动作消费和30 Hz图像输入。

## 行为合同

- 纯ACT继续使用同步最新动作块替换逻辑。
- Bridge继续使用绝对step对齐和最多三个候选块的时间集成。
- Bridge最终输出不再对左右夹爪执行二值化与迟滞，夹爪保持连续的0～100标量。
- 共享 `policy_bridge_core.py` 和真机 `policy_bridge_gripfilter.py` 未修改。
- 默认使用kinematic控制，因此 `tracking_rmse=0` 只表示仿真位置精确跟随目标。

## 验证基线

```text
w1_act_sim/tests: 123 passed
Ruff: passed
```

## Source Hashes

```text
ef4633d51b234ae4e1edc650b86b11a347759e0ca069215c47692a98a28a8ab5  w1_act_sim/action_processor.py
0d006a3f36f419d9bd5a9fe4e0be2b9fd35b4f37d7b0d8e7cb9e7020ffb34f78  w1_act_sim/bridge_controller.py
2beef1ab0b281ec05bc6ecbf4f8226a645ce73f6986d6cde17a5dedcda3afb74  w1_act_sim/run.py
6ea23f45755bd9789425d58fef4fc7e5bb2adefd0bb53ac5c2bef88589609679  w1_act_sim/launch.py
a465c94bedf2d9ce88c81041f4384c9b10515bbcbdbe6e1e50d866c2b04f56b9  w1_act_sim/run_act_sim.sh
4a366c5550b1ecdddfc96a960d9c21767a70644955845500d0b144804987540a  w1_act_sim/run_act_sim_bridge.sh
9d191ed374bc539581448090dd97fdb579817451e2a817cccac8aa84cce867b7  w1_act_sim/verify.py
b0423777149e2db7eb3634091e9661e72427e575fc93e10cf5cc2e8bc9bbfa13  w1_act_sim/mapping.py
86e3dbea9a280d8a1c1b54f3aefe06bdd75d28867b9ae118a03a4cef6b786d5b  w1_act_sim/configs/hand_mapping_popcorn.json
974ba3ba774c1b2fff70f6092ac91c9c989ac41063e4e85b024a841fafc0e498  w1_act_sim/tests/test_bridge_controller.py
a9ae13dc1887fc8e911c75ea11a58809977a41255718f89ead5d9c5548a03824  w1_act_sim/tests/test_runner_integration.py
```
