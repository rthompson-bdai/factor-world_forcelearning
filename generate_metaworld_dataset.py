from dataclasses import dataclass, field
import dataclasses
import pyrallis
from typing import List
import re
import os
import json
import numpy as np
import random
import h5py
from PIL import Image
from copy import copy
import shutil
import sys
sys.path.insert(0,'/workspaces/bdai/projects/foundation_models/src/force_learning')

from ibrl_forcelearning.env.vpl_metaworld_wrapper import VPLMetaWorld
import yaml


"""
Sample run for generating a dataset:
python generate_metaworld_dataset.py \
    --num_episodes 5 \
    --save_gifs 5 \
    --output_path data/metaworld/Assembly_frame_stack_1_96x96_end_on_success \
    --env_cfg.env_name Assembly \
    --env_cfg.frame_stack 1 \
    --env_cfg.rl_image_size 96 \
    --env_cfg.end_on_success true

Sample run for generating a dataset and also saving it in MoDem format:
python generate_metaworld_dataset.py \
    --num_episodes 5 \
    --save_gifs 5 \
    --output_path data/metaworld/Assembly_frame_stack_1_224x224_modem \
    --env_cfg.env_name Assembly \
    --env_cfg.frame_stack 1 \
    --env_cfg.rl_image_size 224 \
    --env_cfg.end_on_success false \
    --add_modem_format true
"""

def run(cfg):
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # Make the directory if it doesn"t exist
    os.makedirs(cfg.output_path, exist_ok=True)
    
    
    env_kwargs = dataclasses.asdict(cfg.env_cfg)

    factor_cfg_path = f'/workspaces/bdai/projects/foundation_models/src/force_learning/factor-world_forcelearning/cfgs/{env_kwargs["env_name"]}.yaml'

    with open(factor_cfg_path) as f:
        factor_args = yaml.safe_load(f)['env']['factors']
    cfg.env_cfg.factor_kwargs.insert(0,"object_pos")

    # Make an `env_cfg.json` file that looks like the ones made by robomimic
    cfg.env_cfg.factor_kwargs = {factor:factor_args[factor] for factor in cfg.env_cfg.factor_kwargs if factor != "None"}
    env_kwargs = dataclasses.asdict(cfg.env_cfg)

    env_cfg_json_dict = dict()
    env_cfg_json_dict["env_name"] = env_kwargs["env_name"]
    env_cfg_json_dict["env_kwargs"] = env_kwargs
    with open(os.path.join(cfg.output_path, "env_cfg.json"), "w") as f:
        json.dump(env_cfg_json_dict, f)

    

    # Make an instance of the environment
    env = VPLMetaWorld(**env_kwargs)


    #PAUSE HERE



    list_ep_dict = []
    list_ep_dict_np = []

    n_steps = []

    for ep in range(cfg.num_episodes):
        os.makedirs(os.path.join(cfg.output_path, f"episode_{ep}"))
        output_file = os.path.join(cfg.output_path, f"episode_{ep}", f"episode_{ep}.h5")
        f = h5py.File(output_file, "w")

        ep_dict = dict()
        print(f"Generating episode {ep + 1} / {cfg.num_episodes}...")
        rl_obs, _ = env.reset()

        for key in rl_obs.keys():
            ep_dict[f"{key}"] = [rl_obs[key].cpu().numpy()]
        ep_dict["action"] = []
        ep_dict["rewards"] = []
        ep_dict["dones"] = []
        ep_dict["infos"] = []

        steps = 0
        for _ in range(cfg.env_cfg.episode_length):
            steps += 1
            heuristic_action = env.get_heuristic_action(clip_action=True)
            rl_obs, reward, terminal, _, _ = env.step(heuristic_action)

            for key in rl_obs.keys():
                print(key)
                if key not in ep_dict.keys():
                    ep_dict[f"{key}"] = [rl_obs[key].cpu().numpy()]
                else:
                    ep_dict[f"{key}"].append(rl_obs[key].cpu().numpy())

            ep_dict["action"].append(np.array(heuristic_action))

            if terminal:
                break
        n_steps.append(steps)

        ep_dict_np = dict()
        f.create_dataset("action", data=np.array(ep_dict["action"]))
        for k in rl_obs.keys():
            if "color" in k:
                images_1 = np.expand_dims(np.array(ep_dict[f"{k}"], \
                                                                     dtype=np.uint8).transpose((0, 2, 3, 1)), 1)
                f.create_dataset(f"{k}",data=np.concatenate([images_1, images_1], axis=1))
            else:
                f.create_dataset(f"{k}", data=np.array(ep_dict[f"{k}"]))

        for key in ep_dict.keys():
            if "color" in key:
                ep_dict_np[key] = np.array(ep_dict[key], dtype=np.uint8)
 
        #f.create_dataset("color", data=np.array(ep_dict["color"]))
        list_ep_dict.append(ep_dict)
        list_ep_dict_np.append(ep_dict_np)

        # if ep < cfg.save_gifs:
        #     print("Saving gif...")
        #     gif_output_file = os.path.join(cfg.output_path, f"episode_{ep}.gif")
        #     camera_name = cfg.env_cfg.rl_camera
        #     images = ep_dict_np["color"].transpose((0, 2, 3, 1))
        #     print(images.shape)
        #     exit(0)
        #     images = [Image.fromarray(img[:, :, -3:]) for img in images]
        #     images[0].save(
        #         gif_output_file, save_all=True, append_images=images[1:], duration=50, loop=0
        #     )
        f.close()
        metadata ={"num_timesteps": n_steps, "num_episodes": len(n_steps), }

        with open(os.path.join(cfg.output_path, 'metadata.json'), 'w') as jf:
            json.dump(metadata, jf)

        shutil.copy(factor_cfg_path, os.path.join(cfg.output_path, f'{env_kwargs["env_name"]}.yaml'))

        print(f"Successfully wrote hdf5 file at {output_file}")

@dataclass
class EnvironmentConfig:
    # Below are all the arguments to VPLMetaWorld
    env_name: str = "Assembly"
    robots: List[str] = field(default_factory=lambda: ["Sawyer"])
    episode_length: int = 100
    action_repeat: int = 2
    frame_stack: int = 2
    obs_stack: int = 1
    reward_shaping: bool = False
    rl_image_size: int = 84  # This is the image size that gets saved in the dataset
    device: str = "cuda"
    camera_names: List[str] = field(default_factory=lambda: ["corner2"])
    rl_camera: str = "corner2"  # This is the camera that gets saved in the dataset
    env_reward_scale: float = 1.0
    end_on_success: bool = False
    use_state: bool = True
    use_force: bool = True
    norm: bool = False
    norm_dataset: str = None
    factor_kwargs: List[str] = field(default_factory=lambda: [])


@dataclass
class MainConfig:
    output_path: str = "data/metaworld/Assembly"
    num_episodes: int = 1
    save_gifs: int = 0
    add_modem_format: bool = False
    env_cfg: EnvironmentConfig = field(default_factory=lambda: EnvironmentConfig())
    seed: int = 0

if __name__ == "__main__":
    import rich.traceback

    rich.traceback.install()
    cfg = pyrallis.parse(config_class=MainConfig)  # type: ignore

    run(cfg)
