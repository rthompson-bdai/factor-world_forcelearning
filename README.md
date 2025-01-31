# Decomposing the Generalization Gap in Imitation Learning for Visual Robotic Manipulation


This is a modification of the factor-world repo packaged and wrapped to resemble the robomimic interface for compatibility with a wider range of experiments. 

Environments consist of a base task and a set of 'factor' wrappers for varying that factor at every new episode. Each factor also has a configuration file with a 'train' and 'test' set of parameter ranges to evaluate performance outside of training context. 

The factor wrappers in this repo that work for all environments are: `['arm_pos', 'camera_pos', 'floor_texture', 'table_pos', 'table_texture', 'light', 'object_size']`

The environments with existing configuration files are: `[bin-picking', 'button-press', 'door-lock', 'door-open', 'drawer-close', 'faucet-open', 'handle-press', 'lever-pull', 'pick-place', 'window-open']`

To install, after cloning the repo:
`cd factor-world_forcelearning`
`pip install -e .`

You may also have to `pip` install metaworld, gym, and mujoco_py as dependencies.

To generate data in environments, run `shell_scripts\datagen_factors.sh`

This produces a folder with the saved environment configuration and a number of recorded episodes, in robomimic format.

To load the same environment, you can pass that configuration to VPLMetaworldWrapper. 

Previous README notes:

This is the official codebase for the [paper](https://sites.google.com/view/generalization-gap):
```
@misc{xie2023decomposing,
      title={Decomposing the Generalization Gap in Imitation Learning for Visual Robotic Manipulation}, 
      author={Annie Xie and Lisa Lee and Ted Xiao and Chelsea Finn},
      year={2023},
      eprint={2307.03659},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}

# Acknowledgements

This repository builds upon the following codebases:
* Metaworld: https://github.com/rlworkgroup/metaworld
* Weakly Supervised Control: https://github.com/google-research/weakly_supervised_control
* Mujoco Scanned Objects: https://github.com/kevinzakka/mujoco_scanned_objects
* DrQ-v2: https://github.com/facebookresearch/drqv2
