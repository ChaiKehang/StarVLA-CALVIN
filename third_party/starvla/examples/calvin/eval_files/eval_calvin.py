"""
Calvin Multi-Step Evaluation Script

Based on RoboFlamingo's evaluation protocol:
https://github.com/RoboFlamingo/RoboFlamingo/blob/main/robot_flamingo/eval/eval_utils.py

Evaluates a policy server on Calvin's long-horizon multi-task benchmark.
Measures success rate on chains of 1-5 consecutive tasks.

Usage:
    python examples/calvin/eval_calvin.py \
        --args.host 0.0.0.0 \
        --args.port 8000 \
        --args.dataset_path /path/to/calvin/task_D_D \
        --args.num_sequences 1000
"""

import copy
import dataclasses
import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
import tyro

# # Add Calvin to path
# CALVIN_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "calvin"
# sys.path.insert(0, str(CALVIN_ROOT))
from calvin_agent.evaluation.utils import (
    collect_plan,
    count_success,
    get_env_state_for_initial_condition,
    get_log_dir,
    print_and_save,
)
from moviepy.editor import ImageSequenceClip
from omegaconf import OmegaConf
from termcolor import colored
from tqdm import tqdm

from deployment.model_server.tools import image_tools
from examples.LIBERO.eval_files.model2libero_interface import ModelClient

# from calvin_env.envs.play_table_env import get_env

# Set OpenGL platform for headless rendering
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ["MUJOCO_GL"] = "osmesa"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EP_LEN = 360  # Max steps per task


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "127.0.0.1"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    pretrained_path: str = ""
    unnorm_key: str = ""

    #################################################################################################################
    # Calvin environment-specific parameters
    #################################################################################################################
    dataset_path: str = "/path/to/calvin/task_D_D"  # Path to Calvin dataset
    calvin_config_path: str = "/path/to/calvin/calvin_models/conf"
    eval_sequences_path: str = "/path/to/calvin/eval_sequences.json"
    num_sequences: int = 1000  # Number of evaluation sequences
    num_workers: int = 1  # For future multi-process support
    seed: int = 0
    create_plan_tsne: bool = False

    #################################################################################################################
    # Evaluation settings
    #################################################################################################################
    debug: bool = False  # Save debug videos
    eval_log_dir: str = "tmp/calvin/eval_logs"  # Path to save evaluation logs and videos
    reset: bool = False  # If True, reset robot state between tasks (easier)
    diverse_inst: bool = False  # Use diverse instructions (zero-shot generalization)
    disable_intent_conditioning: bool = False  # Paired E1-B causal ablation
    inference_seed: int = 42  # Reuse diffusion noise for on/off comparisons


class CalvinPolicyClient:
    """Wrapper around websocket client with Calvin-specific preprocessing."""

    def __init__(
        self,
        host: str,
        port: int,
        resize_size: int = 224,
        replan_steps: int = 5,
        pretrained_path: str = "",
        unnorm_key: str = "",
        disable_intent_conditioning: bool = False,
        inference_seed: int = 42,
    ):
        self.client = ModelClient(
            host=host,
            port=port,
            image_size=[resize_size, resize_size],
            unnorm_key=(unnorm_key or None),
        )
        server_checkpoint = self.client._server_metadata.get("ckpt_path")
        if pretrained_path and server_checkpoint:
            expected = Path(pretrained_path).resolve()
            actual = Path(server_checkpoint).resolve()
            if expected != actual:
                raise ValueError(
                    "CALVIN client/server checkpoint mismatch: "
                    f"client requested {expected}, server loaded {actual}"
                )
        logger.info(
            "Policy server Intent configuration: %s",
            self.client._server_metadata.get("intent_conditioning", "not present"),
        )
        self.resize_size = resize_size
        self.replan_steps = replan_steps
        self.step_count = 0
        self.disable_intent_conditioning = disable_intent_conditioning
        self.inference_seed = inference_seed
        self.intent_records = []
        self._current_intent_record_start = 0
        self._current_subtask_context = {}

    def reset(self):
        """Reset action plan buffer."""
        self.step_count = 0

    def begin_subtask(self, sequence_i: int, subtask_i: int, subtask: str) -> None:
        self._current_intent_record_start = len(self.intent_records)
        self._current_subtask_context = {
            "sequence_index": int(sequence_i),
            "subtask_index": int(subtask_i),
            "subtask": subtask,
        }

    def finish_subtask(self, success: bool, executed_steps: int) -> None:
        for record in self.intent_records[self._current_intent_record_start :]:
            record["subtask_success"] = bool(success)
            record["subtask_executed_steps"] = int(executed_steps)

    @staticmethod
    def _axis_bins(displacement_xyz: np.ndarray) -> np.ndarray:
        """Approximate the training label bins from a predicted action chunk."""

        q20 = 0.0038118734955787693
        q60 = 0.020136535167694092
        bins = np.full(3, 2, dtype=np.int64)
        bins[displacement_xyz < -q60] = 0
        bins[(displacement_xyz >= -q60) & (displacement_xyz < -q20)] = 1
        bins[(displacement_xyz >= q20) & (displacement_xyz < q60)] = 3
        bins[displacement_xyz >= q60] = 4
        return bins

    def _record_intent_prediction(self, model_output: dict) -> None:
        if not model_output.get("intent_refreshed", False):
            return
        prediction = model_output.get("intent_predictions")
        if not prediction:
            return

        def first(name):
            value = np.asarray(prediction[name])
            return value[0]

        predicted_class = int(first("predicted_class_id"))
        predicted_bins = np.asarray(first("predicted_axis_bins"), dtype=np.int64)
        # Server actions use CALVIN-scaled relative XYZ. Dividing by 50 and
        # summing the chunk approximates the label builder's meter displacement.
        action_displacement = np.asarray(self.client.raw_actions[:, :3]).sum(axis=0) / 50.0
        action_bins = self._axis_bins(action_displacement)
        action_class = int(25 * action_bins[0] + 5 * action_bins[1] + action_bins[2])
        self.intent_records.append(
            {
                **self._current_subtask_context,
                "environment_step": int(self.step_count),
                "predicted_class_id": predicted_class,
                "predicted_axis_bins": predicted_bins.tolist(),
                "top5_class_ids": np.asarray(first("top5_class_ids"), dtype=np.int64).tolist(),
                "top5_probabilities": np.asarray(first("top5_probabilities"), dtype=float).tolist(),
                "entropy": float(first("entropy")),
                "max_probability": float(first("max_probability")),
                "conditioning_applied": bool(first("conditioning_applied")),
                "action_implied_class_id_approx": action_class,
                "action_implied_axis_bins_approx": action_bins.tolist(),
                "intent_action_exact_class_agreement_approx": bool(
                    predicted_class == action_class
                ),
                "intent_action_axis_agreement_count_approx": int(
                    np.sum(predicted_bins == action_bins)
                ),
            }
        )

    def save_intent_report(self, eval_log_dir: str) -> None:
        if not self.intent_records:
            logger.warning("No Intent predictions were returned by the policy server")
            return
        log_dir = Path(eval_log_dir)
        records_path = log_dir / "intent_predictions.jsonl"
        with records_path.open("w", encoding="utf-8") as handle:
            for record in self.intent_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        classes = np.asarray(
            [record["predicted_class_id"] for record in self.intent_records]
        )
        summary = {
            "num_replans": len(self.intent_records),
            "conditioning_applied": bool(
                self.intent_records[0]["conditioning_applied"]
            ),
            "mean_entropy": float(
                np.mean([record["entropy"] for record in self.intent_records])
            ),
            "mean_max_probability": float(
                np.mean([record["max_probability"] for record in self.intent_records])
            ),
            "unique_predicted_classes": int(len(np.unique(classes))),
            "predicted_class_counts": np.bincount(classes, minlength=125).tolist(),
            "intent_action_exact_class_agreement_approx": float(
                np.mean(
                    [
                        record["intent_action_exact_class_agreement_approx"]
                        for record in self.intent_records
                    ]
                )
            ),
            "intent_action_mean_axis_agreement_count_approx": float(
                np.mean(
                    [
                        record["intent_action_axis_agreement_count_approx"]
                        for record in self.intent_records
                    ]
                )
            ),
            "note": (
                "CALVIN rollouts have no expert future-action Intent labels. "
                "Agreement with the policy action chunk is a consistency metric, not classification accuracy."
            ),
        }
        (log_dir / "intent_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

    def step(self, obs: dict, lang_annotation: str) -> np.ndarray:
        """
        Query policy for action given observation and language instruction.

        Args:
            obs: Calvin observation dict with keys:
                - rgb_obs: dict with 'rgb_static' (200x200x3) and 'rgb_gripper' (84x84x3)
                - robot_obs: (15,) proprioceptive state [ee_pos(3), ee_ori(3), gripper(2), joint_pos(7)]
            lang_annotation: Natural language task description
            get_action: If True, query model for new action chunk

        Returns:
            action: (7,) array [dx, dy, dz, droll, dpitch, dyaw, gripper]
        """
        # Preprocess images
        rgb_static = obs["rgb_obs"]["rgb_static"]  # (200, 200, 3) uint8
        rgb_gripper = obs["rgb_obs"]["rgb_gripper"]  # (84, 84, 3) uint8

        # Resize and pad images
        image = image_tools.convert_to_uint8(image_tools.resize_with_pad(rgb_static, self.resize_size, self.resize_size))
        wrist_image = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(rgb_gripper, self.resize_size, self.resize_size)
        )

        # Prepare input for policy server (aligned with eval_libero)
        example = {
            "image": [image, wrist_image],
            "lang": lang_annotation,
        }

        # Query model
        model_output = self.client.step(
            example=example,
            step=self.step_count,
            disable_intent_conditioning=self.disable_intent_conditioning,
            inference_seed=self.inference_seed,
        )
        self._record_intent_prediction(model_output)
        raw_action = model_output["raw_action"]
        world_vector = np.asarray(raw_action.get("world_vector"), dtype=np.float32).reshape(-1)
        rotation_delta = np.asarray(raw_action.get("rotation_delta"), dtype=np.float32).reshape(-1)
        open_gripper = np.asarray(raw_action.get("open_gripper"), dtype=np.float32).reshape(-1)

        action = np.concatenate([world_vector, rotation_delta, open_gripper], axis=0).astype(np.float32)
        self.step_count += 1
        return action


def make_env(dataset_path: str):
    """Initialize Calvin environment without tactile sensor (to avoid OpenGL issues)."""
    val_folder = Path(dataset_path) / "validation"

    # Load config and disable tactile sensor to avoid pyrender/OpenGL conflicts
    from omegaconf import OmegaConf

    config_path = val_folder / ".hydra" / "merged_config.yaml"
    cfg = OmegaConf.load(config_path)

    # Remove tactile sensor from camera list if it exists
    if hasattr(cfg.env, "cameras") and "tactile" in cfg.env.cameras:
        # Create a new camera dict without tactile
        new_cameras = OmegaConf.create({k: v for k, v in cfg.env.cameras.items() if k != "tactile"})
        cfg.env.cameras = new_cameras

    # Initialize environment with modified config
    import hydra

    env = hydra.utils.instantiate(cfg.env, show_gui=False, use_vr=False, use_scene_info=True)

    return env


def load_lang_task(dataset_path: str) -> dict:
    """Load language annotations and task oracle for Calvin validation set."""
    conf_dir = Path(dataset_path)
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
    return val_annotations, task_oracle


def evaluate_policy_ddp(
    policy,
    env,
    epoch,
    calvin_conf_path,
    eval_sequences_path,
    num_sequences,
    eval_log_dir=None,
    debug=False,
    create_plan_tsne=False,
    reset=False,
    diverse_inst=False,
):
    """
    Run this function to evaluate a model on the CALVIN challenge.

    Args:
        model: Must implement methods of CalvinBaseModel.
        env: (Wrapped) calvin env.
        epoch:
        eval_log_dir: Path where to log evaluation results. If None, logs to /tmp/evaluation/
        debug: If True, show camera view and debug info.
        create_plan_tsne: Collect data for TSNE plots of latent plans (does not work for your custom model)

    Returns:
        Dictionary with results
    """
    conf_dir = Path(calvin_conf_path)
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)

    # val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
    if diverse_inst:
        with open("/mnt/bn/robotics/lxh/robot-flamingo/lang_annotation_cache.json", "r") as f:
            val_annotations = json.load(f)
    else:
        val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    eval_log_dir = get_log_dir(eval_log_dir)
    with open(eval_sequences_path, "r") as f:
        eval_sequences = json.load(f)
    if num_sequences is not None and num_sequences > 0:
        eval_sequences = eval_sequences[:num_sequences]
    # device_num = int(torch.distributed.get_world_size())
    # device_id = torch.distributed.get_rank()
    # assert num_sequences % device_num == 0
    # interval_len = int(num_sequences // device_num)
    # eval_sequences = eval_sequences[device_id*interval_len:min((device_id+1)*interval_len, num_sequences)]
    results = []
    plans = defaultdict(list)
    local_sequence_i = 0
    base_sequence_i = 0  # device_id * interval_len

    if not debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    for initial_state, eval_sequence in eval_sequences:
        result = evaluate_sequence(
            env,
            policy,
            task_oracle,
            initial_state,
            eval_sequence,
            val_annotations,
            plans,
            debug,
            eval_log_dir,
            base_sequence_i + local_sequence_i,
            reset=reset,
            diverse_inst=diverse_inst,
        )
        results.append(result)
        if not debug:
            eval_sequences.set_description(
                " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(count_success(results))]) + "|"
            )
        local_sequence_i += 1

    def merge_multi_list(res):
        tmp = []
        for l in res:
            tmp.extend(l)
        return tmp

    def extract_iter_from_tqdm(tqdm_iter):
        return [_ for _ in tqdm_iter]

    # if create_plan_tsne:
    #     create_tsne(plans, eval_log_dir, epoch)

    eval_sequences = extract_iter_from_tqdm(eval_sequences)

    print_and_save(results, eval_sequences, eval_log_dir, epoch)
    if hasattr(policy, "save_intent_report"):
        policy.save_intent_report(eval_log_dir)

    return results


def evaluate_sequence(
    env,
    policy,
    task_checker,
    initial_state,
    eval_sequence,
    val_annotations,
    plans,
    debug,
    eval_log_dir="",
    sequence_i=-1,
    reset=False,
    diverse_inst=False,
):
    """
    Evaluates a sequence of language instructions.
    """
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    success_counter = 0
    if debug:
        time.sleep(1)
        print()
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")
    for subtask_i, subtask in enumerate(eval_sequence):
        if reset:
            success = rollout(
                env,
                policy,
                task_checker,
                subtask,
                val_annotations,
                plans,
                debug,
                eval_log_dir,
                subtask_i,
                sequence_i,
                robot_obs=robot_obs,
                scene_obs=scene_obs,
                diverse_inst=diverse_inst,
            )
        else:
            success = rollout(
                env,
                policy,
                task_checker,
                subtask,
                val_annotations,
                plans,
                debug,
                eval_log_dir,
                subtask_i,
                sequence_i,
                diverse_inst=diverse_inst,
            )
        if success:
            success_counter += 1
        else:
            return success_counter
    return success_counter


def rollout(
    env,
    policy,
    task_oracle,
    subtask,
    val_annotations,
    plans,
    debug,
    eval_log_dir="",
    subtask_i=-1,
    sequence_i=-1,
    robot_obs=None,
    scene_obs=None,
    diverse_inst=False,
):
    """
    Run the actual rollout on one subtask (which is one natural language instruction).
    """
    if debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)
    if robot_obs is not None and scene_obs is not None:
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    obs = env.get_obs()
    # get lang annotation for subtask
    if diverse_inst:
        lang_annotation = val_annotations[sequence_i][subtask_i]
    else:
        lang_annotation = val_annotations[subtask][0]
    lang_annotation = lang_annotation.split("\n")[0]
    if "\u2019" in lang_annotation:
        lang_annotation.replace("\u2019", "'")
    policy.reset()
    if hasattr(policy, "begin_subtask"):
        policy.begin_subtask(sequence_i, subtask_i, subtask)
    start_info = env.get_info()

    if debug:
        img_queue = []

    for step in range(EP_LEN):

        action = policy.step(obs, lang_annotation)

        # Ensure action is writable (Calvin env modifies it in-place)
        if not action.flags.writeable:
            action = np.array(action, copy=True)
        action[-1] = 1 if action[-1] > 0 else -1

        obs, _, _, current_info = env.step(action)
        if debug:
            img_copy = copy.deepcopy(obs["rgb_obs"]["rgb_static"])
            img_queue.append(img_copy)
        if step == 0:
            # for tsne plot, only if available
            collect_plan(policy, plans, subtask)

        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            if hasattr(policy, "finish_subtask"):
                policy.finish_subtask(True, step + 1)
            if debug:
                print(colored("success", "green"), end=" ")
                img_clip = ImageSequenceClip(img_queue, fps=30)
                img_clip.write_gif(os.path.join(eval_log_dir, f"{sequence_i}-{subtask_i}-{subtask}-succ.gif"), fps=30)
            return True
    if hasattr(policy, "finish_subtask"):
        policy.finish_subtask(False, EP_LEN)
    if debug:
        print(colored("fail", "red"), end=" ")
        img_clip = ImageSequenceClip(img_queue, fps=30)
        img_clip.write_gif(os.path.join(eval_log_dir, f"{sequence_i}-{subtask_i}-{subtask}-fail.gif"), fps=30)
    return False


def main(args: Args):
    # args = tyro.cli(Args)

    policy = CalvinPolicyClient(
        args.host,
        args.port,
        args.resize_size,
        args.replan_steps,
        pretrained_path=args.pretrained_path,
        unnorm_key=args.unnorm_key,
        disable_intent_conditioning=args.disable_intent_conditioning,
        inference_seed=args.inference_seed,
    )
    env = make_env(args.dataset_path)

    evaluate_policy_ddp(
        policy,
        env,
        0,
        args.calvin_config_path,
        args.eval_sequences_path,
        args.num_sequences,
        args.eval_log_dir,
        args.debug,
        args.create_plan_tsne,
        args.reset,
        args.diverse_inst,
    )


if __name__ == "__main__":
    tyro.cli(main)
