import re
import sys
import torch
import numpy as np
import collections
import h5py


# try:
import gym
import metaworld
import metaworld.policies
from envs.env_dict import ALL_V2_ENVIRONMENTS_GOAL_HIDDEN, ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from wrappers import make_env_with_factors

# except Exception as e:
#     sys.path.append('/workspaces/bdai/projects/foundation_models/src/force_learning/factor-world_forcelearning')
#     import gym
#     import metaworld
#     import metaworld.policies
#     from envs.env_dict import ALL_V2_ENVIRONMENTS_GOAL_HIDDEN, ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
#     from wrappers import make_env_with_factors

#     print("warning: failed to import metaworld")
#     print("========================================", e)
#     print("========================================")


GOOD_CAMERAS = {
    "Assembly": ["corner2"],
    "Basketball": ["corner2"],
    "CoffeePush": ["corner2"],
    "BoxClose": ["corner2"],
    "StickPull": ["corner2"],
    "StickPull": ["corner2"],
    "PegInsertSide": ["corner2"],
    "Soccer": ["corner2"],
    "button-press": ["corner2"],
    "pick-place": ["corner2"],
    "bin-picking": ["corner2"],
    "button-press-topdown": ["corner2"], 
    "button-press-topdown-wall": ["corner2"], 
    "door-lock": ["corner2"], 
    "door-open": ["corner2"],
    "door-unlock": ["corner2"],
    "drawer-close": ["corner2"],
    "drawer-open": ["corner2"],
    "faucet-close": ["corner2"],
    "faucet-open": ["corner2"],
    "handle-press": ["corner2"], 
    "handle-pull": ["corner2"],
    "handle-pull-side": ["corner2"],
    "lever-pull": ["corner2"],
    "window-close": ["corner2"], 
    "window-open": ["corner2"],

}
DEFAULT_CAMERA = "corner2"


# All V2 environments have 39 dimensional states. Some of the
# dimensions are unused (always 0), but we keep them all.
STATE_IDXS = {
    "Assembly": list(range(39)),
    "Basketball": list(range(39)),
    "CoffeePush": list(range(39)),
    "BoxClose": list(range(39)),
    "HandInsert": list(range(39)),
    "StickPull": list(range(39)),
    "PegInsertSide": list(range(39)),
    "Soccer": list(range(39)),
    "button-press": list(range(39)),
    "pick-place": list(range(39)),
    "bin-picking": list(range(39)),
    "button-press-topdown": list(range(39)), 
    "button-press-topdown-wall": list(range(39)), 
    "door-lock": list(range(39)), 
    "door-open": list(range(39)),
    "door-unlock": list(range(39)),
    "drawer-close": list(range(39)),
    "drawer-open": list(range(39)),
    "faucet-close": list(range(39)),
    "faucet-open": list(range(39)),
    "handle-press": list(range(39)), 
    "handle-pull": list(range(39)),
    "handle-pull-side": list(range(39)),
    "lever-pull": list(range(39)),
    "window-close": list(range(39)), 
    "window-open": list(range(39)),
}
STATE_SHAPE = {env_name: (len(STATE_IDXS[env_name]),) for env_name in STATE_IDXS.keys()}

# We can find out what state dimensions correspond to by inspecting
# the observation_space definition in the base SawyerXYZEnv,
# (metaworld/envs/mujoco/sawyer_xyz_env.py)
# For more per-environment information, can look at the oracle policy,
# (e.g. metaworld/policies/sawyer_assembly_v2_policy.py).
# For all V2 environments, the first four dimensions are x, y, z, gripper;
# though for some environments, the gripper is not necessary.
PROP_IDXS = {
    "Assembly": list(range(4)),
    "Basketball": list(range(4)),
    "CoffeePush": list(range(4)),
    "BoxClose": list(range(4)),
    "StickPull": list(range(4)),
    "HandInsert": list(range(4)),
    "PegInsertSide": list(range(4)),
    "Soccer": list(range(4)),
    "button-press": list(range(4)),
    "pick-place": list(range(4)),
    "bin-picking": list(range(4)),
    "button-press-topdown": list(range(4)), 
    "button-press-topdown-wall": list(range(4)), 
    "door-lock": list(range(4)), 
    "door-open": list(range(4)),
    "door-unlock": list(range(4)),
    "drawer-close": list(range(4)),
    "drawer-open": list(range(4)),
    "faucet-close": list(range(4)),
    "faucet-open": list(range(4)),
    "handle-press": list(range(4)), 
    "handle-pull": list(range(4)),
    "handle-pull-side": list(range(4)),
    "lever-pull": list(range(4)),
    "window-close": list(range(4)), 
    "window-open": list(range(4)),
}
PROP_SHAPE = {env_name: (len(PROP_IDXS[env_name]),) for env_name in STATE_IDXS.keys()}


class MetaWorldEnv(gym.Env):
    """
    Fully-observable state-only (noimage) MetaWorld environment
    `camera_name`, `width`, and `height` only affect the output of `render`.
    """

    def __init__(
        self,
        env_name,
        camera_name,
        width,
        height,
        env_kwargs = {},
        factor_kwargs = None,
        use_train_xml = True,
    ):
        self.env_name = env_name

        # Convert, e.g., CoffeePush to coffee-push
        env_id = re.sub(r"([a-z])([A-Z])", r"\1-\2", self.env_name).lower()
        env_id = f"{env_id}-v2-goal-observable"

        env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_id]

        if 'camera_pos' in list(factor_kwargs.keys()):
            env_kwargs['camera_name'] = "movable"
        #env_kwargs['random_init'] = True

        self.env = make_env_with_factors(
            env_cls, env_kwargs,
            factor_kwargs,
            use_train_xml,
        )


        # Ensures that every time `reset` is called, the goal position is randomized
        self.env._freeze_rand_vec = False

        # Set the heuristic (scripted) policy
        policy_name = "Sawyer" + env_cls.__name__.replace("GoalObservable", "") + "Policy"
        self.heuristic_policy = vars(metaworld.policies)[policy_name]()

        # Redefine corner2 camera to be zoomed in, as in Seo et al. 2022 and Hansen et al. 2022
        index = self.env.model.camera_name2id("corner2")
        # self.env.model.cam_fovy[index] = 22  # FOV
        # self.env.model.cam_pos[index][0] = 1.5  # X
        # self.env.model.cam_pos[index][1] = -0.4  # Y
        # self.env.model.cam_pos[index][2] = 1.1  # Z
        self.env.model.cam_pos[index] = [0.75, 0.075, 0.7]

        self.camera_name = camera_name

        self.width = width
        self.height = height

    @property
    def action_space(self):
        return self.env.action_space

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, _, _ = self.env.step(np.zeros_like(self.env.action_space.sample()))
        obs = np.take(obs['proprio'], STATE_IDXS[self.env_name])
        return dict(state=obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = dict(state=obs['proprio'])
        return obs, reward, done, info

    def get_heuristic_action(self, clip_action=True):
        state_obs = self.env._get_obs()
        action = self.heuristic_policy.get_action(state_obs['proprio'])
        if clip_action:
            action = action.clip(-1, 1)
        return action

    def render(self, mode="rgb_array", camera_name=None, width=None, height=None):
        assert mode == "rgb_array"
        # Use the defaults we initialized the environment with
        # unless different values are specifically passed to `render`
        camera_name = camera_name or self.camera_name
        width = width or self.width
        height = height or self.height

        img = self.env.render(
            offscreen=True,
            camera_name=camera_name,
            resolution=(width, height),
        )

        return img


class ProprioObsWrapper(gym.Wrapper):
    """
    Takes a MetaWorld environment and adds an observation key
    for the proprioceptive state
    """

    def __init__(self, env, idx_list, use_force, no_prop, norm=False, norm_dataset=None):
        super().__init__(env)
        self.idx_list = idx_list
        self.model = self.env.env.env.model
        self.data = self.env.env.env.data
        self.use_force = use_force
        self.no_prop = no_prop
        self.norm = norm
        
        if self.use_force and self.norm:
            assert norm_dataset is not None
            f = h5py.File(norm_dataset, "r")
            demos = list(f["data"].keys())
            inds = np.argsort([int(elem[5:]) for elem in demos])
            demos = [demos[i] for i in inds]
            force_data  = []
            torque_data = []
            for demo_name in demos:
                demo = f["data/{}/obs".format(demo_name)]
                force_data.append(demo['prop'][:, -6:-3])
                torque_data.append(demo['prop'][:, -3:])

            force_data = np.concatenate(force_data, 0)
            torque_data = np.concatenate(torque_data, 0)
            self.force_mean = np.mean(force_data, axis=0)
            self.torque_mean = np.mean(torque_data, axis=0)
            self.force_std= np.std(force_data, axis=0)
            self.torque_std = np.std(torque_data, axis=0)
        else: 
            self.force_mean = np.array([0,0,0])
            self.torque_mean = np.array([0,0,0])
            self.force_std= np.array([1,1,1])
            self.torque_std = np.array([1,1,1])

        print(self.force_mean, self.torque_mean)


    def _get_force_data(self):
        sensor_idx = np.sum(self.model.sensor_dim[: self.model.sensor_name2id("force_ee")])
        sensor_dim = self.model.sensor_dim[self.model.sensor_name2id("force_ee")]
        force_data = np.array(self.data.sensordata[sensor_idx : sensor_idx + sensor_dim])

        sensor_idx = np.sum(self.model.sensor_dim[: self.model.sensor_name2id("torque_ee")])#torque_ee")])
        sensor_dim = self.model.sensor_dim[self.model.sensor_name2id("torque_ee")]#torque_ee")]
        torque_data = np.array(self.data.sensordata[sensor_idx : sensor_idx + sensor_dim])

        return np.concatenate([(force_data - self.force_mean)/self.force_std, \
                               (torque_data - self.torque_mean)/self.torque_std], axis=0)

    def _modify_observation(self, obs): 
        #obs["prop"] = {}  
        obs["robot0_eef_pos"] = np.take(obs["state"], self.idx_list[:3])
        obs["robot0_gripper_qpos"] = np.take(obs["state"], self.idx_list[3:])
        
        if self.use_force:
            force_obs = self._get_force_data()
            obs['robot0_ee_force'] = np.array([force_obs[:3]])
            obs['robot0_ee_torque'] = np.array([force_obs[3:]])

    def reset(self):
        obs = self.env.reset()
        self._modify_observation(obs)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._modify_observation(obs)
        return obs, reward, done, info


class ImageObsWrapper(gym.Wrapper):
    """
    Takes a MetaWorld environment and adds an observation key
    for image from one or more cameras
    """

    def __init__(self, env, camera_names):
        super().__init__(env)
        self.camera_names = camera_names

    def _modify_observation(self, obs):
        images = []
        for camera_name in self.camera_names:
            image_key = f"{camera_name}_image"
            img = self.env.render(camera_name=camera_name)
            img = img.transpose(2, 0, 1)
            if camera_name == "movable":
                img = np.flip(obs[img], 1)
            images.append(img)
            obs[image_key] = img
        obs['color'] = np.concatenate(images)

    def reset(self):
        obs = self.env.reset()
        self._modify_observation(obs)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._modify_observation(obs)
        return obs, reward, done, info


class ActionRepeatWrapper(gym.Wrapper):
    """
    Executes multiple inner `step` calls for each `step` call
    """

    def __init__(self, env, num_repeats):
        super().__init__(env)
        self.env = env
        self.num_repeats = num_repeats

    def step(self, action):
        obs = None
        reward = 0.0
        done = False
        info = None
        discount = 1.0
        for _ in range(self.num_repeats):
            _obs, _reward, _done, _info = self.env.step(action)
            obs = _obs
            reward += _reward * discount
            done = done or _done
            info = _info
            if done:
                break

        return obs, reward, done, info


class SparseRewardWrapper(gym.Wrapper):
    """
    Overwrite the default environment reward with binary success flag
    given in `info`. Does not overwrite the value of `done`.
    """

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        # NOTE: This is different than MoDem, which uses a -1 / 0 reward
        # instead of 0, 1
        obs, reward, done, info = self.env.step(action)
        info["original_reward"] = reward
        reward = float(info["success"])
        return obs, reward, done, info


class StackObsWrapper(gym.Wrapper):
    """
    Stacks observations from a history of the specified length.
    Keeps track of the history in `self.past_obses` and `self.past_frames`
    and returns stacked versions in `step`

    For non-image keys, concatenates along the first/only dimension
    For image keys, concatenates along the first/channel dimension
    """

    def __init__(self, env, obs_stack=1, frame_stack=1):
        super().__init__(env)
        self.obs_stack = obs_stack
        self.frame_stack = frame_stack
        self.past_obses = collections.defaultdict(lambda: collections.deque(maxlen=self.obs_stack))
        self.past_frames = collections.defaultdict(
            lambda: collections.deque(maxlen=self.frame_stack)
        )

    def _get_stacked_observation(self):
        # Concatenate along the first (only) dimension
        obses = {k: np.concatenate(v, axis=0) for k, v in self.past_obses.items()}

        # Concatenate along the first (channel) dimension
        frames = {k: np.concatenate(v, axis=0) for k, v in self.past_frames.items()}
        obses.update(frames)
        return obses

    def reset(self):
        obs = super().reset()
        self.past_obses.clear()
        self.past_frames.clear()

        # Fill up history with multiple copies of the first observation
        # NOTE: This consistent with what is done in the MoDem implementation
        # but not with L153 of robosuite_wrapper.py
        for key in obs:
            print(key)
            if "image" in key:
                for _ in range(self.frame_stack):
                    self.past_frames[key].append(obs[key].copy())
            else:
                for _ in range(self.obs_stack):
                    self.past_obses[key].append(obs[key].copy())

        return self._get_stacked_observation()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        for key in obs:
            if "image" in key:
                self.past_frames[key].append(obs[key])
            else:
                self.past_obses[key].append(obs[key])

        return self._get_stacked_observation(), reward, done, info


class TimeLimitWrapper(gym.Wrapper):
    def __init__(self, env, max_episode_steps):
        super().__init__(env)
        self._max_episode_steps = max_episode_steps
        assert self._max_episode_steps > 0
        self._elapsed_steps = 0

    def reset(self):
        self._elapsed_steps = 0
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        done = done or self._elapsed_steps >= self._max_episode_steps
        return obs, reward, done, info


class VPLMetaWorld:
    def __init__(
        self,
        env_name,
        robots,
        episode_length,  # This measures the number of outer environment steps
        action_repeat,  # Number of inner steps per outer step
        frame_stack,  # Number of outer steps to stack frames over
        obs_stack,  # Number of outer steps to stack obses over
        *,
        reward_shaping=False,
        rl_image_size=None,
        device="cuda",
        camera_names=[DEFAULT_CAMERA],
        rl_camera=DEFAULT_CAMERA,
        env_reward_scale=1.0,
        end_on_success=True,
        use_state=False,
        use_force = True,
        no_prop = False,
        use_train_xml = True,
        norm=False, 
        norm_dataset=None,
        env_kwargs={}, 
        factor_kwargs={},
    ):
        assert robots == None or robots == [] or robots == "Sawyer" or robots == ["Sawyer"]
        assert reward_shaping == False, "reward_shaping is not a supported argument"

        assert isinstance(camera_names, list)
        self.camera_names = camera_names

        if 'camera_pos' in list(factor_kwargs.keys()):
            self.camera_names = ["movable"]
            rl_camera = "movable"

        # Make a state-only environment
        self.base_env = MetaWorldEnv(
            env_name=env_name,
            camera_name=self.camera_names[0],
            width=rl_image_size,
            height=rl_image_size,
            env_kwargs=env_kwargs,
            factor_kwargs=factor_kwargs,
            use_train_xml=use_train_xml,
        )
        self.unwrapped = self.base_env.env.unwrapped
        # For every outer call to step, make multiple inner calls to step
        self.env = ActionRepeatWrapper(env=self.base_env, num_repeats=action_repeat)
        # Add a key `prop` to the observation with proprioceptive dimensions
        self.env = ProprioObsWrapper(env=self.env, idx_list=PROP_IDXS[env_name], use_force=use_force, no_prop=no_prop, norm=norm, norm_dataset=norm_dataset)
        # Add keys to the observation with each camera rendering
        self.env = ImageObsWrapper(env=self.env, camera_names=self.camera_names)
        # Add observation stacking for specified number of steps
        self.env = StackObsWrapper(env=self.env, obs_stack=obs_stack, frame_stack=frame_stack)
        # Overwrite environment rewards with sparse rewards
        self.env = SparseRewardWrapper(env=self.env)
        # Set max horizon --> if we get to episode_length steps, done is True
        if episode_length is not None:
            self.env = TimeLimitWrapper(env=self.env, max_episode_steps=episode_length)

        self.rl_camera = rl_camera
        self.frame_stack = frame_stack
        # self.image_size = image_size
        self.rl_image_size = rl_image_size
        self.env_reward_scale = env_reward_scale
        self.end_on_success = end_on_success
        self.use_state = use_state
        # self.resize_transform = None
        # if self.rl_image_size != self.image_size:
        #     self.resize_transform = utils.get_rescale_transform(self.rl_image_size)
        self.num_action = self.env.action_space.shape[0]
        self.observation_shape = (3 * self.frame_stack, rl_image_size, rl_image_size)
        self.state_shape = (STATE_SHAPE[env_name][0] * obs_stack,)
        self.prop_shape = (PROP_SHAPE[env_name][0] * obs_stack,)
        self.device = device
        self.reward_model = None

        self.time_step = 0
        self.episode_reward = 0
        self.episode_extra_reward = 0
        self.terminal = True

        self.most_recent_info = None

    @property
    def action_dim(self):
        return self.num_action

    def set_reward_model(self, reward_model):
        self.reward_model = reward_model

    def _extract_images(self, obs):
        # NOTE: A couple differences from `_extract_images` in PixelRobosuite:
        # - Logic for adding proprio and image observations is handled by
        #   MetaWorldProprioWrapper and MetaWorldImageWrapper
        # - Logic for frame stacking is handled in StackWrapper

        state = None
        #if self.use_state:
        state = torch.from_numpy(obs["state"]).to(self.device)

        #prop = torch.from_numpy(obs["prop"]).to(self.device)

        rl_image_obs = None
        all_image_obs = {}
        for camera_name in self.camera_names:
            image_key = f"{camera_name}_image"
            #print(obs.keys())
            image_obs = torch.from_numpy(obs[image_key].copy())

            # keep the high-res version for rendering
            # Include just the most recent image if we're using frame stacking
            all_image_obs[camera_name] = image_obs[-3:, :, :]
            if self.rl_camera == camera_name:
                rl_image_obs = image_obs

        assert rl_image_obs is not None
        rl_image_obs = rl_image_obs.to(self.device)
        # if self.resize_transform is not None:
        #     # set the device here because transform is 5x faster on GPU
        #     rl_image_obs = self.resize_transform(rl_image_obs)

        rl_obs = {"obs": rl_image_obs}
        for k, v in obs.items():
            rl_obs[k] = torch.from_numpy(v).to(self.device)
        #rl_obs["prop"] = prop.to(self.device)

        if self.use_state:
            assert state is not None
        rl_obs["state"] = state.to(self.device)

        return rl_obs, all_image_obs

    def reset(self):
        self.time_step = 0
        self.episode_reward = 0
        self.episode_extra_reward = 0
        self.terminal = False

        obs = self.env.reset()
        rl_obs, image_obs = self._extract_images(obs)

        if self.reward_model is not None:
            self.reward_model.reset()

        self.most_recent_info = None

        return rl_obs, image_obs

        

    def step(self, action):
        
        self.time_step += 1
        obs, reward, terminal, info = self.env.step(action)
        self.most_recent_info = info

        rl_obs, image_obs = self._extract_images(obs)
        self.episode_reward += reward

        if self.end_on_success and (reward == 1):
            terminal = True
        success = reward == 1

        reward = reward * self.env_reward_scale
        if self.reward_model is not None:
            reward_ret = self.reward_model.get_reward(image_obs)
            reward += reward_ret.reward
            self.episode_extra_reward += reward_ret.reward

        self.terminal = terminal
        return rl_obs, reward, terminal, success, image_obs

    def get_heuristic_action(self, clip_action=False):
        return self.env.get_heuristic_action(clip_action=clip_action)


if __name__ == "__main__":
    from torchvision.utils import save_image

    env = VPLMetaWorld(
        env_name="Assembly",
        robots="Sawyer",
        episode_length=100,
        action_repeat=2,
        frame_stack=2,
        obs_stack=1,
        rl_image_size=96,
        device="cpu",
        camera_names=GOOD_CAMERAS["Assembly"],
        use_state=False,
    )
    x = env.reset()[0]["obs"].float() / 255
    print(x.dtype)
    print(x.shape)
    save_image(x[-3:, :, :], "test_env.png")
