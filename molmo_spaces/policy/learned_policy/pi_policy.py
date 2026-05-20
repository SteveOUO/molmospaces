import logging
import os
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.policy.base_policy import InferencePolicy, StatefulPolicy
from molmo_spaces.policy.learned_policy.utils import resize_with_pad

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class PI_PolicyState:
    actions_buffer: list[np.ndarray] | None = None
    current_buffer_index: int = 0
    starting_time: float | None = None


class PI_Policy(InferencePolicy, StatefulPolicy):
    def __init__(
        self,
        exp_config: MlSpacesExpConfig,
    ) -> None:
        super().__init__(exp_config)
        self.remote_config = exp_config.policy_config.remote_config
        self.checkpoint_path = exp_config.policy_config.checkpoint_path
        self.grasping_type = exp_config.policy_config.grasping_type
        self.chunk_size = exp_config.policy_config.chunk_size
        self.grasping_threshold = exp_config.policy_config.grasping_threshold
        self.openpi_action_mode = exp_config.policy_config.openpi_action_mode
        self.openpi_control_dt = float(exp_config.policy_config.openpi_control_dt)
        self.current_joint_position: np.ndarray | None = None
        self.model = None  # don't init model till inference to allow multiprocessing

    def get_state(self):
        return PI_PolicyState(
            actions_buffer=self.actions_buffer,
            current_buffer_index=self.current_buffer_index,
            starting_time=self.starting_time,
        )

    def set_state(self, state: PI_PolicyState):
        self.actions_buffer = state.actions_buffer
        self.current_buffer_index = state.current_buffer_index
        self.starting_time = state.starting_time

    def reset(self):
        self.actions_buffer = None
        self.current_buffer_index = 0
        self.starting_time = None

    def prepare_model(self):
        self.model_name = os.path.basename(self.checkpoint_path)
        if self.remote_config is not None:
            self._prepare_remote_model(self.checkpoint_path)
        else:
            self._prepare_local_model(self.checkpoint_path)
        # self.reset()

    def _prepare_local_model(self, checkpoint_path: str):
        from openpi.policies import policy_config as _policy_config
        from openpi.training import config as _config

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
        self.config = _config.get_config(os.path.basename(checkpoint_path.rstrip("/")))
        self.model = _policy_config.create_trained_policy(self.config, checkpoint_path)

    def _prepare_remote_model(self, checkpoint_path: str):
        from openpi_client import websocket_client_policy

        host = self.remote_config["host"]
        port = self.remote_config["port"]
        self.checkpoint_path = checkpoint_path
        self.model = websocket_client_policy.WebsocketClientPolicy(
            host=host,
            port=port,
        )
        log.info(f"Successfully connected to remote model at {host}:{port}")

    def render(self, obs):
        views = np.concatenate([obs["wrist_camera"], obs["exo_camera_1"]], axis=1)
        cv2.imshow("views", cv2.cvtColor(views, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    def obs_to_model_input(self, obs):
        # self.render(obs)
        if isinstance(obs, list):
            if len(obs) > 1:
                log.warning(
                    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                    "WARNING: obs list has %d elements but only using the first one!\n"
                    "This may indicate a batching issue - expected single observation.\n"
                    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",
                    len(obs),
                )
            obs = obs[0]
        prompt = self.task.get_task_description()

        # For local eval
        if isinstance(obs, list | tuple):
            obs = obs[0]

        self.current_joint_position = np.asarray(obs["qpos"]["arm"][:7], dtype=np.float32).reshape(1, 7)
        grip = np.clip(obs["qpos"]["gripper"][0] / 0.824033, 0, 1)
        exo_camera_key = (
            "droid_shoulder_light_randomization"
            if "droid_shoulder_light_randomization" in obs
            else "exo_camera_1"
        )
        wrist_camera_key = (
            "wrist_camera_zed_mini" if "wrist_camera_zed_mini" in obs else "wrist_camera"
        )
        model_input = {
            "observation/exterior_image_1_left": resize_with_pad(obs[exo_camera_key], 224, 224),
            "observation/wrist_image_left": resize_with_pad(obs[wrist_camera_key], 224, 224),
            "observation/joint_position": self.current_joint_position.reshape(7),
            "observation/gripper_position": np.array(grip).reshape(
                1,
            ),
            "prompt": prompt.lower(),
        }
        return model_input

    def inference_model(self, model_input):
        if self.model is None:
            self.prepare_model()
        if self.starting_time is None:
            self.starting_time = time.time()
        if self.actions_buffer is None or self.current_buffer_index >= self.chunk_size:
            self.actions_buffer = self._convert_openpi_actions(self.model.infer(model_input)["actions"])
            self.current_buffer_index = 0
        model_output = self.actions_buffer[self.current_buffer_index]
        self.current_buffer_index += 1
        return model_output

    def _convert_openpi_actions(self, actions: np.ndarray) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float32)
        if self.openpi_action_mode == "joint_position":
            return actions
        if self.openpi_action_mode != "joint_velocity":
            raise ValueError(f"Unsupported openpi_action_mode={self.openpi_action_mode!r}.")
        if self.current_joint_position is None:
            raise RuntimeError("Missing joint position for OpenPI joint-velocity conversion.")
        converted = actions.copy()
        joint_velocity = converted[:, :7]
        converted[:, :7] = self.current_joint_position + np.cumsum(joint_velocity, axis=0) * self.openpi_control_dt
        return converted

    def model_output_to_action(self, model_output):
        if self.grasping_type == "continuous":
            gripper_pos = model_output[7] * np.array([255.0])
        else:  # binary
            gripper_pos = (
                np.array([255.0]) if model_output[7] > self.grasping_threshold else np.array([0.0])
            )

        arm_output = model_output[:7].reshape(
            7,
        )
        action = {
            "arm": arm_output,
            "gripper": gripper_pos,
        }
        return action

    def get_info(self) -> dict:
        info = super().get_info()
        info["policy_name"] = (
            self.model.get_server_metadata().get("policy_name", "pi")
            if hasattr(self.model, "get_server_metadata")
            else "pi"
        )
        info["policy_checkpoint"] = self.model_name
        info["policy_buffer_length"] = self.chunk_size
        info["policy_grasping_threshold"] = self.grasping_threshold
        info["policy_grasping_type"] = self.grasping_type
        info["prompt"] = self.task.get_task_description()
        log.info(f"Current prompt: {info['prompt']}")

        info["time_spent"] = time.time() - self.starting_time if self.starting_time else None
        info["timestamp"] = time.time()
        return info
