import logging
import os
import time
import functools

import msgpack
import numpy as np
import websockets.exceptions
import websockets.sync.client

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.policy.base_policy import InferencePolicy
from molmo_spaces.policy.learned_policy.utils import resize_with_pad

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

PING_INTERVAL_SECS = 60
PING_TIMEOUT_SECS = 600


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _pack_array(obj):
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")

    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }

    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }

    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])

    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])

    return obj


_Packer = functools.partial(msgpack.Packer, default=_pack_array)
_unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


class SmartWorldWebsocketClient:
    """Thin client for the SmartWorld DROID policy server."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}:{port}"
        self._packer = _Packer()
        self._ws, self._server_metadata = self._wait_for_server()

    def _connect_once(self) -> tuple[websockets.sync.client.ClientConnection, dict]:
        conn = websockets.sync.client.connect(
            self._uri,
            compression=None,
            max_size=None,
            ping_interval=PING_INTERVAL_SECS,
            ping_timeout=PING_TIMEOUT_SECS,
        )
        metadata = _unpackb(conn.recv())
        return conn, metadata

    def _wait_for_server(self) -> tuple[websockets.sync.client.ClientConnection, dict]:
        log.info("Waiting for SmartWorld server at %s...", self._uri)
        while True:
            try:
                conn, metadata = self._connect_once()
                log.info("Connected to SmartWorld server at %s", self._uri)
                return conn, metadata
            except ConnectionRefusedError:
                log.info("SmartWorld server is not ready; retrying...")
                time.sleep(2)

    def _reconnect(self) -> None:
        while True:
            try:
                log.warning("Reconnecting to SmartWorld server at %s...", self._uri)
                self._ws, self._server_metadata = self._connect_once()
                return
            except Exception as exc:
                log.warning("Reconnect failed: %s. Retrying...", exc)
                time.sleep(2)

    def infer(self, request: dict) -> dict:
        data = self._packer.pack(request)
        try:
            self._ws.send(data)
            response = self._ws.recv()
        except websockets.exceptions.ConnectionClosedError:
            self._reconnect()
            self._ws.send(data)
            response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in SmartWorld inference server:\n{response}")
        return _unpackb(response)

    def reset(self) -> dict:
        return self.infer({"reset": True})

    def close(self) -> None:
        if self._ws is not None:
            self._ws.close()
            self._ws = None

    def get_server_metadata(self) -> dict:
        return self._server_metadata


class SmartWorld_Policy(InferencePolicy):
    """MolmoSpaces policy adapter for the SmartWorld DROID websocket server."""

    def __init__(self, exp_config: MlSpacesExpConfig, task=None) -> None:
        super().__init__(exp_config)
        self.task = task
        self.remote_config = exp_config.policy_config.remote_config
        self.checkpoint_path = exp_config.policy_config.checkpoint_path
        self.grasping_type = os.environ.get(
            "SMARTWORLD_GRASPING_TYPE", exp_config.policy_config.grasping_type
        )
        self.grasping_threshold = float(
            os.environ.get(
                "SMARTWORLD_GRASPING_THRESHOLD",
                exp_config.policy_config.grasping_threshold,
            )
        )
        self.chunk_size = int(
            os.environ.get("SMARTWORLD_CHUNK_SIZE", exp_config.policy_config.chunk_size)
        )
        self.duplicate_exo_to_exterior_0 = _parse_bool(
            os.environ.get(
                "SMARTWORLD_DUPLICATE_EXO_TO_EXTERIOR_0",
                exp_config.policy_config.duplicate_exo_to_exterior_0,
            )
        )
        self.model_name = (
            os.path.basename(self.checkpoint_path) if self.checkpoint_path else "smartworld"
        )
        self.model = None
        self.actions_buffer = None
        self.current_buffer_index = 0
        self.control_step = 0
        self.starting_time = None
        self._logged_first_request = False
        self._executed_actions_since_request = []

    def reset(self):
        self.actions_buffer = None
        self.current_buffer_index = 0
        self.control_step = 0
        self.starting_time = None
        self._executed_actions_since_request = []
        if self.model is not None:
            self.model.reset()

    def prepare_model(self):
        host = os.environ.get("SMARTWORLD_SERVER_HOST") or self.remote_config.get(
            "host", "127.0.0.1"
        )
        port = int(
            os.environ.get("SMARTWORLD_SERVER_PORT")
            or self.remote_config.get("port", 8000)
        )
        self.model = SmartWorldWebsocketClient(host=host, port=port)

    def obs_to_model_input(self, obs):
        if isinstance(obs, list):
            if len(obs) > 1:
                log.warning("Received batched obs; using the first element for SmartWorld policy.")
            obs = obs[0]

        prompt = self.task.get_task_description()
        exterior_0_key = (
            "droid_shoulder_light_randomization"
            if "droid_shoulder_light_randomization" in obs
            else "exo_camera_1"
        )
        exterior_1_key = "exo_camera_2" if "exo_camera_2" in obs else None
        wrist_camera_key = (
            "wrist_camera_zed_mini" if "wrist_camera_zed_mini" in obs else "wrist_camera"
        )

        exterior_0 = resize_with_pad(obs[exterior_0_key], 180, 320)
        wrist_image = resize_with_pad(obs[wrist_camera_key], 180, 320)
        if self.duplicate_exo_to_exterior_0:
            exterior_1 = exterior_0
            exterior_1_source = f"{exterior_0_key} (duplicated)"
        elif exterior_1_key is not None:
            exterior_1 = resize_with_pad(obs[exterior_1_key], 180, 320)
            exterior_1_source = exterior_1_key
        else:
            image_keys = sorted(
                key for key, value in obs.items() if hasattr(value, "shape") and value.ndim >= 2
            )
            raise KeyError(
                "SmartWorld requires two distinct exterior cameras. Expected one of "
                "`exo_camera_2` for exterior_1, and "
                f"`{exterior_0_key}` for exterior_0. Available image keys: {image_keys}"
            )

        grip = np.clip(obs["qpos"]["gripper"][0] / 0.824033, 0, 1)
        if self.control_step > 0:
            executed_action = np.concatenate([
                np.asarray(obs["qpos"]["arm"][:7], dtype=np.float32).reshape(7),
                np.asarray([grip], dtype=np.float32),
            ])
            self._executed_actions_since_request.append(executed_action)

        if not self._logged_first_request:
            log.info(
                "SmartWorld obs image shapes: exterior_0=%s[%s] exterior_1=%s[%s] wrist=%s[%s] dtypes=(%s,%s,%s)",
                getattr(exterior_0, "shape", None),
                exterior_0_key,
                getattr(exterior_1, "shape", None),
                exterior_1_source,
                getattr(wrist_image, "shape", None),
                wrist_camera_key,
                getattr(exterior_0, "dtype", None),
                getattr(exterior_1, "dtype", None),
                getattr(wrist_image, "dtype", None),
            )
            self._logged_first_request = True
        model_input = {
            "observation/exterior_image_0_left": exterior_0,
            "observation/exterior_image_1_left": exterior_1,
            "observation/wrist_image_left": wrist_image,
            "observation/joint_position": np.asarray(obs["qpos"]["arm"][:7], dtype=np.float32),
            "observation/gripper_position": np.asarray([grip], dtype=np.float32),
            "prompt": prompt.lower(),
            "control_step": int(self.control_step),
        }
        if len(self._executed_actions_since_request) > 0:
            model_input["history/executed_action_count"] = len(self._executed_actions_since_request)
            model_input["history/executed_actions"] = np.stack(self._executed_actions_since_request, axis=0).astype(np.float32)
        return model_input

    def inference_model(self, model_input):
        if self.model is None:
            self.prepare_model()
        if self.starting_time is None:
            self.starting_time = time.time()

        if self.actions_buffer is None or self.current_buffer_index >= len(self.actions_buffer):
            result = self.model.infer(model_input)
            if "actions" not in result:
                raise KeyError(f"SmartWorld server response missing `actions`; keys={list(result)}")
            actions = np.asarray(result["actions"])
            if actions.ndim != 2 or actions.shape[1] < 8:
                raise ValueError(
                    f"Expected SmartWorld actions with shape [T, >=8], got {actions.shape}"
                )
            self.actions_buffer = actions[: self.chunk_size]
            self.current_buffer_index = 0
            self._executed_actions_since_request = []

        model_output = self.actions_buffer[self.current_buffer_index]
        self.current_buffer_index += 1
        self.control_step += 1
        return model_output

    def model_output_to_action(self, model_output):
        gripper_pos = float(np.clip(model_output[7], 0.0, 1.0))
        if self.grasping_type == "binary":
            gripper = np.array([255.0 if gripper_pos >= self.grasping_threshold else 0.0])
        elif self.grasping_type == "continuous":
            gripper = np.array([gripper_pos * 255.0])
        else:
            raise ValueError(f"Invalid grasping type: {self.grasping_type}")

        return {
            "arm": np.asarray(model_output[:7], dtype=np.float32).reshape(7),
            "gripper": gripper.astype(np.float32),
        }

    def get_info(self) -> dict:
        info = super().get_info()
        info["policy_name"] = "smartworld"
        info["policy_checkpoint"] = self.model_name
        info["policy_buffer_length"] = self.chunk_size
        info["policy_grasping_threshold"] = self.grasping_threshold
        info["policy_grasping_type"] = self.grasping_type
        info["prompt"] = self.task.get_task_description()
        info["time_spent"] = time.time() - self.starting_time if self.starting_time else None
        info["timestamp"] = time.time()
        if self.model is not None:
            info["server_metadata"] = self.model.get_server_metadata()
        return info
