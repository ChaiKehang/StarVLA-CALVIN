# StarVLA + CALVIN ABC→D：E0 Baseline 与 Factorized Spatial Intent

本仓库保留两条已经完整跑通的 StarVLA–CALVIN ABC→D 链路：E0 baseline，
以及最终的 Factorized Aligned-9 Spatial Intent（E1）。两者都使用冻结的
Qwen3-VL-4B-Instruct、QwenPI layer-wise flow-matching Action DiT，并从同一个
Bridge–RT-1 step-50k checkpoint 初始化动作路径。E1 额外预测短时域 XYZ、RPY
与 gripper Intent，并通过九个深度对齐的 Query-FiLM router 注入 Action DiT。

仓库只保留复现 E0 和最终 E1 所需的代码、数据定义、配置、checkpoint、评测结果
与课程材料；旧 E1-B/E1-C、11/23/35 sparse-query、entropy gate、恢复补丁和中间
实验均已移除。

## 1. 最终结果

| 项目 | 配置/结果 |
|---|---|
| 基础 VLM | Qwen3-VL-4B-Instruct，hidden size 2560，36 layers |
| 动作模型 | QwenPI_v3 layer-wise flow-matching cross-DiT，hidden size 1024，36 layers |
| 动作 | 7D `[dx, dy, dz, droll, dpitch, dyaw, gripper]`，horizon 8 |
| 初始化 | `StarVLA/Qwen3VL-PI_v3-Bridge-RT-1` 的 step-50k checkpoint |
| 训练范围 | 冻结 `qwen_vl_interface`；训练动作相关模块和 projection/base 参数组 |
| 数据 | sixpigs CALVIN ABC→D LeRobot v2.1，离线转换为 scaled relative action |
| 训练 | 2 GPUs，batch 8/GPU，global batch 16，90k steps，约 1.37 epoch |
| 用时 | 约 38 h 34 min，约 1.54 s/step（2×RTX A6000） |
| Report canonical E0 ACL | **2.713 / 5** |
| Report canonical E0 Success@1–5 | **88.4%、66.8%、50.2%、38.2%、27.8%** |

最终 report 使用同一组 500 条 CALVIN-D 五子任务序列进行比较：

| 模型 | 平均完成链 | Success@1 | Success@2 | Success@3 | Success@4 | Success@5 |
|---|---:|---:|---:|---:|---:|---:|
| E0 baseline | 2.713 | 88.4% | 66.8% | 50.2% | 38.2% | 27.8% |
| E1 aligned-9，60k | **3.240** | **93.0%** | **78.8%** | **61.8%** | **50.6%** | **39.8%** |
| E1 aligned-9，90k | 3.130 | 90.2% | 76.2% | 59.6% | 48.4% | 38.6% |

课程要求的最终比较采用 E0 与 E1 90k；E1 60k 用于说明 checkpoint 敏感性。
E0 的 report canonical 数据保存在历史命名目录
[`baseline_60000steps_1.37epoch_500`](eval_logs/e0_abc_rel/baseline_60000steps_1.37epoch_500/results.json)，
E1 对应
[`aligned9_steps60000`](eval_logs/e1_factorized_intent/aligned9_steps60000_calvin500_intent_on_20260805/results.json)
和
[`aligned9_steps90000`](eval_logs/e1_factorized_intent/aligned9_steps90000_calvin500_intent_on_20260805/results.json)。
这里保留原始目录名，不对历史产物重命名；论文口径以
[`main_zh.tex`](plan/report/ieee_cv_final_project_template/main_zh.tex) 为准。

## 2. 目录约定

推荐布局：

```text
PROJECT_ROOT/
├── README.md
├── configs/starvla/e0_abc_rel_calvin_scaled_90k.yaml
├── third_party/
│   ├── starvla/
│   ├── calvin/
│   │   └── calvin_env/tacto/
│   └── README.md
├── scripts/
│   ├── e0_abc_rel/
│   └── reference/Evo-1_sixpigs/
└── eval_logs/
    ├── e0_abc_rel/
    └── e1_factorized_intent/

DATA_ROOT/
└── calvin/lerobot/
    ├── sixpigs1_calvin2lerobotV21_ABC_D_scnet_rel_calvin_scaled/
    └── sixpigs1_calvin2lerobotV21_ABC_D_scnet_rel_calvin_scaled_intent_factorized_h8/

MODEL_ROOT/
├── Qwen3-VL-4B-Instruct/
├── pretrained/starvla_qwenpi_pretrain_qwen3_4B_bridge-rt_1/
└── checkpoints/calvin/
    ├── e0_abc_rel/
    ├── e1_factorized_aligned9_spatial_intent_s0_50k/
    └── e1_factorized_aligned9_query_s1_10k_s2_80k/
```

### 通用路径

按自己的机器修改一次：

```bash
export PROJECT_ROOT=/path/to/StarVLA-CALVIN
export DATA_ROOT=/path/to/large-disk/kehang-CALVIN
export MODEL_ROOT=/path/to/large-disk/kehang-StarVLA
export CACHE_ROOT=/path/to/large-disk/cache

export STARVLA_DIR="$PROJECT_ROOT/third_party/starvla"
export CALVIN_DIR="$PROJECT_ROOT/third_party/calvin"
export EVO_DIR="$PROJECT_ROOT/scripts/reference/Evo-1_sixpigs"
export LEROBOT_ROOT="$DATA_ROOT/calvin/lerobot"
export CHECKPOINT_ROOT="$MODEL_ROOT/checkpoints/calvin"
```

### DIDIS 服务器 路径

DIDIS 服务器 已验证路径如下。
Git：

```bash
export PROJECT_ROOT=/home/liuchang/kehang/488project
export DATA_ROOT=/home/data/datasets/kehang-CALVIN
export MODEL_ROOT=/home/data/models/kehang-StarVLA
export CACHE_ROOT=/home/data/datasets/kehang-CALVIN/.cache

export STARVLA_DIR="$PROJECT_ROOT/third_party/starvla"
export CALVIN_DIR="$PROJECT_ROOT/third_party/calvin"
export EVO_DIR="$PROJECT_ROOT/scripts/reference/Evo-1_sixpigs"
export LEROBOT_ROOT="$DATA_ROOT/calvin/lerobot"
export CHECKPOINT_ROOT="$MODEL_ROOT/checkpoints/calvin"
```

大文件必须位于个人的 `/home/data/datasets/kehang-CALVIN` 和
`/home/data/models/kehang-StarVLA` 子目录，不能下载到系统盘或共享目录根部。

## 3. 获取代码

StarVLA、CALVIN、`calvin_env` 和 tacto 的固定
源码快照已经直接保存在 `third_party/`，因此 clone 主仓库后即可得到训练和评测
代码。

```bash
git clone https://github.com/ChaiKehang/StarVLA-CALVIN.git "$PROJECT_ROOT"

export STARVLA_DIR="$PROJECT_ROOT/third_party/starvla"
export CALVIN_DIR="$PROJECT_ROOT/third_party/calvin"

test -f "$STARVLA_DIR/LICENSE"
test -f "$CALVIN_DIR/LICENSE"
test -f "$CALVIN_DIR/calvin_env/LICENSE"
test -f "$CALVIN_DIR/calvin_env/tacto/LICENSE"

mkdir -p "$PROJECT_ROOT/scripts/reference"
git clone https://github.com/sixpigs1/Evo-1_sixpigs.git "$EVO_DIR"
git -C "$EVO_DIR" checkout 3730312cbafc3342add9473ce74abd28728353a1
```

核对 vendored 源码和 E0 必要修改：

```bash
rg -n '6dc01d0781a817c007f74927a75bf63d89d521e2|fa03f01f19c65920e18cf37398a9ce859274af76' \
  "$PROJECT_ROOT/third_party/README.md"

rg -n 'calvin_abc_d_sixpigs_rel_scaled' \
  "$STARVLA_DIR/starVLA/dataloader/gr00t_lerobot/mixtures.py"

git -C "$EVO_DIR" rev-parse HEAD
```

mixture 应等价于：

```python
"calvin_abc_d_sixpigs_rel_scaled": [
    ("sixpigs1_calvin2lerobotV21_ABC_D_scnet_rel_calvin_scaled", 1.0, "libero_franka"),
],
```

## 4. 安装环境



### 4.1 StarVLA 训练/策略服务环境

```bash
export CONDA_EXE=${CONDA_EXE:-$(command -v conda)}
export CONDA_REMOTE_CONNECT_TIMEOUT_SECS=${CONDA_REMOTE_CONNECT_TIMEOUT_SECS:-30}
export CONDA_REMOTE_READ_TIMEOUT_SECS=${CONDA_REMOTE_READ_TIMEOUT_SECS:-180}

# DIDIS Lab 使用固定的个人 Conda；其他机器保留上面的 command -v conda。
# export CONDA_EXE=/home/liuchang/miniconda3/bin/conda

"$CONDA_EXE" create -y -n starvla-e0 \
  --override-channels -c conda-forge python=3.10 pip
"$CONDA_EXE" run -n starvla-e0 python -m pip install --upgrade pip
"$CONDA_EXE" run -n starvla-e0 python -m pip install \
  --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.6.0 torchvision==0.21.0
"$CONDA_EXE" run -n starvla-e0 python -m pip install \
  -r "$STARVLA_DIR/requirements.txt"
"$CONDA_EXE" run -n starvla-e0 python -m pip install \
  flash-attn==2.7.4.post1 --no-build-isolation
"$CONDA_EXE" run -n starvla-e0 python -m pip install \
  "huggingface_hub[cli]" -e "$STARVLA_DIR"
```

验证：

```bash
"$CONDA_EXE" run -n starvla-e0 python - <<'PY'
import accelerate, deepspeed, flash_attn, torch, transformers
print("torch", torch.__version__, "CUDA", torch.version.cuda, torch.cuda.is_available())
print("transformers", transformers.__version__)
print("accelerate", accelerate.__version__)
print("deepspeed", deepspeed.__version__)
print("flash_attn", flash_attn.__version__)
PY
```

已验证组合为 Python 3.10、Torch 2.6.0+cu124、Transformers 4.57.0、
Accelerate 1.5.2、DeepSpeed 0.16.9、FlashAttention 2.7.4.post1。驱动显示的
“CUDA Version”可以更高；真正需要核对的是 `torch.version.cuda` 和
`torch.cuda.is_available()`。

### 4.2 CALVIN evaluator 环境

CALVIN 是旧依赖栈，因此与 StarVLA 分成两个环境。下面是本项目已跑通的 legacy
安装顺序：

```bash
"$CONDA_EXE" create -y -n calvin-eval \
  --override-channels -c conda-forge python=3.8 pip
"$CONDA_EXE" run -n calvin-eval python -m pip install --upgrade \
  "pip<25" wheel "setuptools==57.5.0" "cmake==3.18.4.post1"
"$CONDA_EXE" run -n calvin-eval python -m pip install \
  --no-build-isolation "pyhash==0.9.3"
"$CONDA_EXE" run -n calvin-eval python -m pip install \
  --no-build-isolation -e "$CALVIN_DIR/calvin_env/tacto"
"$CONDA_EXE" run -n calvin-eval python -m pip install \
  --no-build-isolation -e "$CALVIN_DIR/calvin_env"
"$CONDA_EXE" run -n calvin-eval python -m pip install \
  --no-build-isolation -e "$CALVIN_DIR/calvin_models"
"$CONDA_EXE" run -n calvin-eval python -m pip install \
  "tyro==1.0.15" "websockets==12.0" "msgpack==1.0.8" "moviepy==1.0.3"
```

验证：

```bash
"$CONDA_EXE" run -n calvin-eval python - <<'PY'
import calvin_env, msgpack, torch, tyro, websockets
print("torch", torch.__version__)
print("websockets", websockets.__version__)
print("CALVIN imports: OK")
PY
```

本项目使用 Python 3.8.20、Torch 1.13.1、setuptools 57.5.0 和 websockets
12.0。若 upstream 的 `install.sh` 因 `cmake==3.18.4` 或 `pyhash` 失败，使用上面
的 `3.18.4.post1 + setuptools<58 + --no-build-isolation` 组合。

## 5. 缓存、模型与数据

### 5.1 缓存位置

```bash
export HF_HOME="$CACHE_ROOT/hf"
export HUGGINGFACE_HUB_CACHE="$CACHE_ROOT/hf/hub"
export TORCH_HOME="$CACHE_ROOT/torch"
export WANDB_CACHE_DIR="$CACHE_ROOT/wandb"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$WANDB_CACHE_DIR" \
  "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$MODEL_ROOT" "$LEROBOT_ROOT"
```

如 Hugging Face 对共享出口 IP 限流，先执行 `hf auth login`。不要把 token 写入
脚本、README、shell history 导出文件或 Git remote URL。

### 5.2 下载基础 VLM 和初始化 checkpoint

```bash
hf download Qwen/Qwen3-VL-4B-Instruct \
  --local-dir "$MODEL_ROOT/Qwen3-VL-4B-Instruct" \
  --max-workers 2

hf download StarVLA/Qwen3VL-PI_v3-Bridge-RT-1 \
  --local-dir "$MODEL_ROOT/pretrained/starvla_qwenpi_pretrain_qwen3_4B_bridge-rt_1" \
  --max-workers 2
```

本 baseline 需要的是：

```bash
export BASE_VLM="$MODEL_ROOT/Qwen3-VL-4B-Instruct"
export PRETRAINED_CHECKPOINT="$MODEL_ROOT/pretrained/starvla_qwenpi_pretrain_qwen3_4B_bridge-rt_1/checkpoints/steps_50000_pytorch_model.pt"

test -f "$BASE_VLM/config.json"
test -f "$PRETRAINED_CHECKPOINT"
```

`StarVLA/Qwen3-VL-4B-Instruct-Action` 不是这次 90k baseline 的必需输入。

### 5.3 下载 sixpigs LeRobot 数据

```bash
export SIXPIGS_REPO=sixpigs1/calvin2lerobotV21_ABC_D_scnet
export SIXPIGS_REV=78faab6c533506cea5c526ea606f2afdffd43dac
export RAW_DIR="$LEROBOT_ROOT/sixpigs1_calvin2lerobotV21_ABC_D_scnet_raw"

hf download "$SIXPIGS_REPO" \
  --repo-type dataset \
  --revision "$SIXPIGS_REV" \
  --local-dir "$RAW_DIR" \
  --max-workers 1
```

重复执行同一命令会断点续传/补齐缺失文件。该 revision 的元数据为 17,870
episodes、1,071,743 frames。下载后至少检查：

```bash
test -f "$RAW_DIR/meta/info.json"
test -f "$RAW_DIR/meta/episodes.jsonl"
test -d "$RAW_DIR/data"
test -d "$RAW_DIR/videos"
du -sh "$RAW_DIR"
```

## 6. absolute action → scaled relative action

原始 sixpigs parquet 的 action 接近 next-state absolute pose。本项目不修改 raw
数据，而是创建派生目录：

```bash
export SCALED_DIR="$LEROBOT_ROOT/sixpigs1_calvin2lerobotV21_ABC_D_scnet_rel_calvin_scaled"
export CONVERTER="$PROJECT_ROOT/scripts/e0_abc_rel/convert_sixpigs_abs_to_rel.py"
```

转换公式：

```text
xyz     = clip((absolute_next_xyz - state_xyz) * 50, -1, 1)
rpy     = clip(wrap_to_pi(absolute_next_rpy - state_rpy) * 20, -1, 1)
gripper = original absolute gripper command
```

每个 episode 删除没有真实 `t→t+1` target 的最后一帧；视频不复制，而从派生目录
软链接到 raw 目录。`observation.state_starvla` 补成 8D，派生目录同时生成
`modality.json`、`stats_gr00t.json` 和 `action_conversion.json`。

先做 10-episode smoke：

```bash
export SMOKE_DIR="${SCALED_DIR}_smoke"
"$CONDA_EXE" run -n starvla-e0 python "$CONVERTER" \
  --src "$RAW_DIR" \
  --dst "$SMOKE_DIR" \
  --max-episodes 10 \
  --calvin-scaled \
  --overwrite-smoke
```

检查 smoke 后做全量转换：

```bash
"$CONDA_EXE" run -n starvla-e0 python "$CONVERTER" \
  --src "$RAW_DIR" \
  --dst "$SCALED_DIR" \
  --calvin-scaled
```

只有明确重建派生数据时才加 `--overwrite`。成功产物应为 17,870 episodes、
1,053,873 frames：

```bash
test -f "$SCALED_DIR/meta/info.json"
test -f "$SCALED_DIR/meta/modality.json"
test -f "$SCALED_DIR/meta/stats_gr00t.json"
test -f "$SCALED_DIR/meta/action_conversion.json"
test -L "$SCALED_DIR/videos"
find -L "$SCALED_DIR/videos" -type l -print
```

训练配置中的 `action_mode: abs` 是有意的：它表示直接读取 parquet 中已经离线处理
好的 action；不能再让 dataloader 动态减一次 state。

## 7. 训练 90k baseline

权威配置是
[`configs/starvla/e0_abc_rel_calvin_scaled_90k.yaml`](configs/starvla/e0_abc_rel_calvin_scaled_90k.yaml)，
通用启动器是
[`scripts/e0_abc_rel/train_e0_90k_generic.sh`](scripts/e0_abc_rel/train_e0_90k_generic.sh)。
启动器只覆盖机器相关路径和 batch，不改变架构、优化器或 90k schedule。

### 7.1 通用机器/其他集群

先由本机调度器分配两张可用 GPU，再运行：

```bash
conda activate starvla-e0

export LEROBOT_ROOT MODEL_ROOT CHECKPOINT_ROOT CACHE_ROOT STARVLA_DIR
export BASE_VLM="$MODEL_ROOT/Qwen3-VL-4B-Instruct"
export PRETRAINED_CHECKPOINT="$MODEL_ROOT/pretrained/starvla_qwenpi_pretrain_qwen3_4B_bridge-rt_1/checkpoints/steps_50000_pytorch_model.pt"
export NUM_PROCESSES=2
export PER_DEVICE_BATCH_SIZE=8
export RUN_ID=e0_abc_rel

# 示例仅适用于没有外部调度器的独占机器；有调度器时保留其 CUDA 映射。
export CUDA_VISIBLE_DEVICES=0,1
export WANDB_MODE=disabled

bash "$PROJECT_ROOT/scripts/e0_abc_rel/train_e0_90k_generic.sh"
```

如果要用 W&B：

```bash
wandb login
export WANDB_MODE=online
export E0_WANDB_PROJECT=starVLA_Calvin_E0
export E0_WANDB_ENTITY=<your-wandb-entity>
```

### 7.2 DIDIS 服务器

DIDIS 上先检查节点与队列；不得在登录 shell 直接训练：

```bash
sinfo -o '%P %N %G %T %C %E'
squeue -u "$USER" -o '%.18i %.12P %.24j %.8u %.2t %.10M %.3D %R'
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

mkdir -p /home/liuchang/kehang/488project/logs/e0_abc_rel
sbatch /home/liuchang/kehang/488project/scripts/e0_abc_rel/train_e0_90k_didis.sbatch
```

DIDIS wrapper 申请 2×A6000、16 CPU、80GB RAM 和 48 h，并把缓存放到
`/home/data`。它保留 Slurm 的 `CUDA_VISIBLE_DEVICES`，若 allocation 包含保留给
图形服务的物理 GPU 0 会直接退出；不要手工覆盖 Slurm 分配。脚本同时有 RAM/GPU
显存 watchdog。

观察任务：

```bash
squeue -j <JOB_ID>
tail -f "$PROJECT_ROOT/logs/e0_abc_rel/e0_abc_90k_<JOB_ID>.out"
watch -n 2 nvidia-smi
```

### 7.3 训练验收

首次运行必须使用 `trainer.is_resume=false`。Bridge–RT-1 step-50k 只提供模型权重，
本次优化器、scheduler 和 local step 从 0 开始。不要在已有同名 checkpoint 的目录
重新启动；通用脚本会对此拒绝。

关键产物：

```bash
export RUN_DIR="$CHECKPOINT_ROOT/e0_abc_rel"
test -f "$RUN_DIR/config.full.yaml"
test -f "$RUN_DIR/config.yaml"
test -f "$RUN_DIR/dataset_statistics.json"
test -f "$RUN_DIR/checkpoints/steps_90000_pytorch_model.pt"
```

已验证 schedule：warmup 5,000、cosine decay、action LR `1e-4`、base/project LR
`2.5e-5`、minimum LR `1e-6`、每 6,000 steps 保存。VLM 冻结，因此
`qwen_vl_interface: 1e-5` 只是保留的配置字段，不产生可训练 optimizer group。

实现细节：当前 QwenPI_v3 从 `trainer.repeated_diffusion_steps` 读取训练重复数。原
90k run 未显式设置时使用默认值 16；本仓库配置把 16 明写出来，避免错误地把
`framework.action_model.repeated_diffusion_steps: 8` 当成实际生效值。

## 8. CALVIN 闭环评测

### 8.1 评测资产

本仓库 report canonical E0 的 ACL 2.713 使用 Evo-1_sixpigs 中的
config-only 环境目录：

```text
$EVO_DIR/CALVIN_evaluation/ABC_D_validation/validation/.hydra/merged_config.yaml
```

当前 evaluator 用它创建 CALVIN simulator；不读取 validation NPZ 轨迹，所以复现
本结果不需要下载 517GB 的完整 `task_ABC_D.zip`。


### 8.2 一体化评测脚本

[`scripts/e0_abc_rel/eval_e0_90k_generic.sh`](scripts/e0_abc_rel/eval_e0_90k_generic.sh)
会在同一个 GPU allocation 内启动 policy server、等待 websocket 端口、运行 CALVIN
evaluator，并在退出时关闭 server。它要求 checkpoint 同目录保留 `config.yaml` 和
`dataset_statistics.json`；`unnorm_key=franka` 从后者读取本训练数据对应的动作反归一化
统计。

通用 10-chain smoke：

```bash
export CKPT90K="$CHECKPOINT_ROOT/e0_abc_rel/checkpoints/steps_90000_pytorch_model.pt"
export STARVLA_PYTHON=/path/to/envs/starvla-e0/bin/python
export CALVIN_PYTHON=/path/to/envs/calvin-eval/bin/python
export NUM_SEQUENCES=10
export DEBUG=true
export PORT=5694
export EVAL_LOG_DIR="$PROJECT_ROOT/eval_logs/e0_abc_rel/smoke_10"

bash "$PROJECT_ROOT/scripts/e0_abc_rel/eval_e0_90k_generic.sh"
```

DIDIS 10-chain smoke：

```bash
mkdir -p /home/liuchang/kehang/488project/logs/e0_abc_rel
sbatch --export=ALL,NUM_SEQUENCES=10,DEBUG=true \
  /home/liuchang/kehang/488project/scripts/e0_abc_rel/eval_e0_90k_didis.sbatch
```

500-chain 统计建议关闭视频，避免生成数千 GIF 和约 13GB 文件：

```bash
sbatch \
  --export=ALL,NUM_SEQUENCES=500,DEBUG=false,EVAL_LOG_DIR=/home/liuchang/kehang/488project/eval_logs/e0_abc_rel/repro_90000steps_500 \
  /home/liuchang/kehang/488project/scripts/e0_abc_rel/eval_e0_90k_didis.sbatch
```

评测脚本拒绝写入非空目录，避免覆盖已有结果。只在定性分析时用 `DEBUG=true`；
Git 只提交 `results.json`。

### 8.3 关于 `replan_steps`

为保留实验命令，脚本仍传 `REPLAN_STEPS=5`。

### 8.4 结果检查

```bash
python - <<'PY'
import json
from pathlib import Path

p = Path("eval_logs/e0_abc_rel/baseline_60000steps_1.37epoch_500/results.json")
r = json.loads(p.read_text())
print(json.dumps(r, indent=2, ensure_ascii=False))
PY
```

先通过 1/10-chain smoke，再固定 checkpoint、sequence JSON、seed、port 和环境版本
执行 500-chain。不要根据训练 loss 单独选择 checkpoint；应比较闭环 ACL 和 Task
1–5 success。



## 9. E1 Factorized Aligned-9 Intent

仓库只保留最终的 factorized aligned-9 实验。Intent head 从冻结 Qwen3-VL 的
第 `3,7,11,15,19,23,27,31,35` 层提取特征，分别预测 XYZ、RPY 与 gripper
分布；九个独立 Query-FiLM router 在 Action DiT 的相同九层注入条件。

E1 使用 `125 + 125 + 5 = 255` 维 soft distribution：XYZ 和 RPY 分别是
`5×5×5` 类，gripper 是 5 类。每个 router 使用
`255 → 512 → SiLU → LayerNorm → ZeroLinear(2048)`，调制经过
`0.1·tanh` 有界化；condition dropout 为 0.2，router peak LR 为 `2e-6`。
完整流程如下。

### 9.1 构建 factorized Intent 数据与 trajectory split

如果保留数据已经存在，只需检查；从 sixpigs scaled-relative 数据重新构建时运行：

```bash
conda activate starvla-e0

/home/liuchang/miniconda3/envs/starvla-e0/bin/python \
  "$PROJECT_ROOT/scripts/e1_abc_intent/build_factorized_intent_dataset.py"

test -f "$LEROBOT_ROOT/sixpigs1_calvin2lerobotV21_ABC_D_scnet_rel_calvin_scaled_intent_factorized_h8/meta/factorized_intent_config.json"
test -f "$LEROBOT_ROOT/sixpigs1_calvin2lerobotV21_ABC_D_scnet_rel_calvin_scaled_intent_factorized_h8/meta/splits/factorized_seed42_train.json"
test -f "$LEROBOT_ROOT/sixpigs1_calvin2lerobotV21_ABC_D_scnet_rel_calvin_scaled_intent_factorized_h8/meta/splits/factorized_seed42_val.json"
```

ABC 按完整 trajectory、任务分层、seed 42 划分：16,080 条训练 trajectory、
1,790 条验证 trajectory，二者无重叠；环境 D 不参与训练或 Intent validation。

### 9.2 S0：Intent-only 50k

S0 从 Bridge–RT-1 50k 初始化，冻结 Qwen 与完整 Action 路径，只执行并训练
factorized Intent branch。两张 A6000、单卡 batch 32、global batch 64、warmup 5k；
每 1k 做固定 validation subset，每 5k 做完整 validation，并保存
`best_intent_pytorch_model.pt`。

```bash
conda activate starvla-e0

CUDA_VISIBLE_DEVICES=2,3 \
NUM_PROCESSES=2 \
WANDB_MODE=online \
bash "$PROJECT_ROOT/scripts/e1_abc_intent/train_e1_factorized_aligned9_spatial_intent_s0_50k.sh"
```

### 9.3 S1 10k + S2 80k：连续 90k

Main run 的 Action 路径仍从原始 Bridge–RT-1 50k 初始化，Intent branch 单独加载
S0 best checkpoint。S1 前 10k 冻结 Intent，训练 Action 与九个 router；S2 解冻
Intent 并联合训练 80k。两个阶段共享同一个 90k cosine scheduler，不重置 LR。
单卡 batch 4、两卡、gradient accumulation 2，global batch 16。

```bash
conda activate starvla-e0

CUDA_VISIBLE_DEVICES=2,3 \
NUM_PROCESSES=2 \
WANDB_MODE=online \
S0_INTENT_CHECKPOINT="$CHECKPOINT_ROOT/e1_factorized_aligned9_spatial_intent_s0_50k/checkpoints/best_intent_pytorch_model.pt" \
bash "$PROJECT_ROOT/scripts/e1_abc_intent/train_e1_factorized_aligned9_query_s1_10k_s2_80k.sh"
```

两个训练启动器都是 fresh-only：不会覆盖已有 run，也不包含 weight-only recovery、
step offset、LR restart 或 W&B continuation 补丁。可先设置 `CHECK_ONLY=true` 做
路径和参数验证，它不会启动训练。

### 9.4 E1 CALVIN-D 闭环评测

评测脚本在同一 allocation 内启动 policy server 和 CALVIN client，固定 500 条序列、
seed 42、`replan_steps=5`，并检查 client/server checkpoint 一致性：

```bash
conda activate starvla-e0

CUDA_VISIBLE_DEVICES=1 CHECKPOINT_STEP=60000 DEBUG=false \
bash "$PROJECT_ROOT/scripts/e1_abc_intent/eval_e1_factorized_aligned9.sh"

CUDA_VISIBLE_DEVICES=1 CHECKPOINT_STEP=90000 DEBUG=false \
bash "$PROJECT_ROOT/scripts/e1_abc_intent/eval_e1_factorized_aligned9.sh"
```

定性 rollout 才使用 `DEBUG=true` 生成 GIF；完整 500-sequence 统计建议设为 false。
S0 validation 结果为：XYZ top-1/top-5/near-1 `40.0/76.4/86.4%`，RPY
`25.1/54.0/71.9%`，gripper top-1 `87.3%`。

更紧凑的端到端操作清单见
[`README_ALIGNED9_PIPELINE.md`](scripts/e1_abc_intent/README_ALIGNED9_PIPELINE.md)，
保留文件与 SHA-256 见
[`CLEANUP_MANIFEST_20260810.md`](CLEANUP_MANIFEST_20260810.md)。

## 10. 主要来源

- [StarVLA upstream](https://github.com/starVLA/starVLA)
- [CALVIN upstream](https://github.com/mees/calvin)
- [sixpigs CALVIN LeRobot dataset](https://huggingface.co/datasets/sixpigs1/calvin2lerobotV21_ABC_D_scnet)
- [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
- [StarVLA Bridge–RT-1 pretrained checkpoint](https://huggingface.co/StarVLA/Qwen3VL-PI_v3-Bridge-RT-1)
- [Evo-1_sixpigs evaluation assets](https://github.com/sixpigs1/Evo-1_sixpigs)
