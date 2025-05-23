# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import os
import copy
import torch
import numpy as np
import random
from typing import Tuple
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym.torch_utils import *

from mqe import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR


def is_primitive_type(obj):
    return not hasattr(obj, '__dict__')


def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__") or isinstance(obj, dict):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


def update_class_from_dict(obj, dict_, strict=False):
    """ If strict, attributes that are not in dict_ will be removed from obj """
    attr_names = [n for n in obj.__dict__.keys() if not (n.startswith("__") and n.endswith("__"))]
    for attr_name in attr_names:
        if not attr_name in dict_:
            delattr(obj, attr_name)
    for key, val in dict_.items():
        attr = getattr(obj, key, None)
        if attr is None or is_primitive_type(attr):
            if isinstance(val, dict):
                setattr(obj, key, copy.deepcopy(val))
                update_class_from_dict(getattr(obj, key), val)
            else:
                setattr(obj, key, val)
        else:
            update_class_from_dict(attr, val)
    return


def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_sim_params(args, cfg):
    # code from Isaac Gym Preview 2
    # initialize sim params
    sim_params = gymapi.SimParams()

    # set some values from args
    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params


def get_load_path(root, load_run=-1, checkpoint=-1):
    if load_run == -1:
        try:
            runs = os.listdir(root)
            # TODO sort by date to handle change of month
            runs.sort()
            if 'exported' in runs: runs.remove('exported')
            last_run = os.path.join(root, runs[-1])
        except:
            raise ValueError("No runs in this directory: " + root)
        load_run = last_run
    elif os.path.isabs(load_run):
        print("Loading load_run as absolute path:", load_run)
    else:
        load_run = os.path.join(root, load_run)

    if checkpoint == -1:
        models = [file for file in os.listdir(load_run) if 'model' in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint)

    load_path = os.path.join(load_run, model)
    return load_path


def update_cfg_from_args(env_cfg, cfg_train, args):
    # seed
    if env_cfg is not None:
        # num envs
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs
    if cfg_train is not None:
        if args.seed is not None:
            cfg_train.seed = args.seed
        # alg runner parameters
        if args.max_iterations is not None:
            cfg_train.runner.max_iterations = args.max_iterations
        if args.resume:
            cfg_train.runner.resume = args.resume
        if args.experiment_name is not None:
            cfg_train.runner.experiment_name = args.experiment_name
        if args.run_name is not None:
            cfg_train.runner.run_name = args.run_name
        if args.load_run is not None:
            cfg_train.runner.load_run = args.load_run
        if args.checkpoint is not None:
            cfg_train.runner.checkpoint = args.checkpoint

    return env_cfg, cfg_train


def get_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "anymal_c_flat",
         "help": "Resume training or start testing from a checkpoint. Overrides config file if provided."},
        {"name": "--resume", "action": "store_true", "default": False, "help": "Resume training from a checkpoint"},
        {"name": "--experiment_name", "type": str,
         "help": "Name of the experiment to run or load. Overrides config file if provided."},
        {"name": "--run_name", "type": str, "help": "Name of the run. Overrides config file if provided."},
        {"name": "--load_run", "type": str,
         "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided."},
        {"name": "--checkpoint", "type": int,
         "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided."},

        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--rl_device", "type": str, "default": "cuda:0",
         "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
        {"name": "--num_envs", "type": int,
         "help": "Number of environments to create. Overrides config file if provided."},
        {"name": "--seed", "type": int, "default": 0, "help": "Random seed. Overrides config file if provided."},
        {"name": "--max_iterations", "type": int,
         "help": "Maximum number of training iterations. Overrides config file if provided."},
    ]
    # parse arguments
    args = gymutil.parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)

    # name allignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == 'cuda':
        args.sim_device += f":{args.sim_device_id}"
    return args


def export_policy_as_jit(actor_critic, path):
    if hasattr(actor_critic, 'memory_a'):
        # assumes LSTM: TODO add GRU
        exporter = PolicyExporterLSTM(actor_critic)
        exporter.export(path)
    else:
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_1.pt')
        model = copy.deepcopy(actor_critic.actor).to('cpu')
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path)


class PolicyExporterLSTM(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
        self.memory.cpu()
        self.register_buffer(f'hidden_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))
        self.register_buffer(f'cell_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))

    def forward(self, x):
        out, (h, c) = self.memory(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        return self.actor(out.squeeze(0))

    @torch.jit.export
    def reset_memory(self):
        self.hidden_state[:] = 0.
        self.cell_state[:] = 0.

    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_lstm_1.pt')
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)


def merge_dict(this: dict, other: dict):
    """ Merging two dicts. if a key exists in both dict, the other's value will take priority
    NOTE: This method is implemented in python>=3.9
    """
    output = this.copy()
    output.update(other)
    return output


def make_env(task_class, env_cfg, args=None):
    if args is None:
        args = get_args()
    # check if there is a registered env with that name
    # override cfg from args (if specified)
    env_cfg, _ = update_cfg_from_args(env_cfg, None, args)
    set_seed(args.seed)
    # parse sim params (convert to dict first)
    sim_params = {"sim": class_to_dict(env_cfg.sim)}
    sim_params = parse_sim_params(args, sim_params)
    print(env_cfg)
    env = task_class(cfg=env_cfg,
                     sim_params=sim_params,
                     physics_engine=args.physics_engine,
                     sim_device=args.sim_device,
                     headless=args.headless)
    return env, env_cfg


class Sensor:
    def __init__(self, env):
        self.env = env

    def get_observation(self):
        raise NotImplementedError

    def get_noise_vec(self):
        raise NotImplementedError

    def get_dim(self):
        raise NotImplementedError


class FloatingCameraSensor(Sensor):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

        camera_props = gymapi.CameraProperties()
        camera_props.width = self.env.cfg.env.recording_width_px
        camera_props.height = self.env.cfg.env.recording_height_px
        self.rendering_camera = self.env.gym.create_camera_sensor(self.env.envs[0], camera_props)

    def set_position(self, pos=None, lookat=None):
        if pos is None:
            bx, by, bz = self.env.root_states[0, 0], self.env.root_states[0, 1], self.env.root_states[0, 2]
            lookat = [bx, by, bz]
            pos = [bx, by - 1.0, bz + 1.0]
        self.env.gym.set_camera_location(self.rendering_camera, self.env.envs[0], gymapi.Vec3(*pos),
                                         gymapi.Vec3(*lookat))

    def get_observation(self, env_ids=None):
        self.env.gym.step_graphics(self.env.sim)
        self.env.gym.render_all_camera_sensors(self.env.sim)
        img = self.env.gym.get_camera_image(self.env.sim, self.env.envs[0], self.rendering_camera, gymapi.IMAGE_COLOR)
        w, h = img.shape
        return img.reshape([w, h // 4, 4])


class AttachedCameraSensor(Sensor):
    def __init__(self, env, attached_robot_asset=None):
        super().__init__(env)
        self.env = env
        self.attached_robot_asset = attached_robot_asset

    def initialize(self, camera_label, camera_pose, camera_rpy, env_ids=None):
        if env_ids is None: env_ids = range(self.env.num_envs)

        camera_props = gymapi.CameraProperties()
        camera_props.width = self.env.cfg.perception.image_width
        camera_props.height = self.env.cfg.perception.image_height
        camera_props.horizontal_fov = self.env.cfg.perception.image_horizontal_fov

        self.cams = []

        for env_id in env_ids:
            cam = self.env.gym.create_camera_sensor(self.env.envs[env_id], camera_props)
            # initialize camera position
            # attach the camera to the base
            trans_pos = gymapi.Vec3(camera_pose[0], camera_pose[1], camera_pose[2])
            quat_pitch = quat_from_angle_axis(torch.Tensor([-camera_rpy[1]]), torch.Tensor([0, 1, 0]))[0]
            quat_yaw = quat_from_angle_axis(torch.Tensor([camera_rpy[2]]), torch.Tensor([0, 0, 1]))[0]
            quat = quat_mul(quat_yaw, quat_pitch)
            trans_quat = gymapi.Quat(quat[0], quat[1], quat[2], quat[3])
            transform = gymapi.Transform(trans_pos, trans_quat)
            follow_mode = gymapi.CameraFollowMode.FOLLOW_TRANSFORM
            self.env.gym.attach_camera_to_body(cam, self.env.envs[env_id], 0, transform, follow_mode)

            self.cams.append(cam)

        return self.cams

    def get_observation(self, env_ids=None):

        raise NotImplementedError

    def get_depth_images(self, env_ids=None):
        if env_ids is None: env_ids = range(self.env.num_envs)

        depth_images = []
        for env_id in env_ids:
            img = self.env.gym.get_camera_image(self.env.sim, self.env.envs[env_id], self.cams[env_id],
                                                gymapi.IMAGE_DEPTH)
            w, h = img.shape
            depth_images.append(torch.from_numpy(img.reshape([1, w, h])).to(self.env.device))
        depth_images = torch.cat(depth_images, dim=0)
        return depth_images

    def get_rgb_images(self, env_ids):
        if env_ids is None: env_ids = range(self.env.num_envs)

        rgb_images = []
        for env_id in env_ids:
            img = self.env.gym.get_camera_image(self.env.sim, self.env.envs[env_id], self.cams[env_id],
                                                gymapi.IMAGE_COLOR)
            w, h = img.shape
            rgb_images.append(
                torch.from_numpy(img.reshape([1, w, h // 4, 4]).astype(np.int32)).to(self.env.device))
        rgb_images = torch.cat(rgb_images, dim=0)
        return rgb_images

    def get_segmentation_images(self, env_ids):
        if env_ids is None: env_ids = range(self.env.num_envs)

        segmentation_images = []
        for env_id in env_ids:
            img = self.env.gym.get_camera_image(self.env.sim, self.env.envs[env_id], self.cams[env_id],
                                                gymapi.IMAGE_SEGMENTATION)
            w, h = img.shape
            segmentation_images.append(
                torch.from_numpy(img.reshape([1, w, h]).astype(np.int32)).to(self.env.device))
        segmentation_images = torch.cat(segmentation_images, dim=0)
        return segmentation_images