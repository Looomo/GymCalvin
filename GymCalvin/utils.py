from .calvin import CalvinEnv
from hydra import compose, initialize
from .gym_env import GymWrapper
from .gym_env import wrap_env
from hydra.core.global_hydra import GlobalHydra


def make(env_name = "calvin"):
    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()
    initialize(config_path='conf')
    cfg = compose(config_name='calvin')
    env = CalvinEnv(**cfg)
    env.max_episode_steps = cfg.max_episode_steps = 360
    env = GymWrapper(
        env=env,
        from_pixels=cfg.pixel_ob,
        from_state=cfg.state_ob,
        height=cfg.screen_size[0],
        width=cfg.screen_size[1],
        channels_first=False,
        frame_skip=cfg.action_repeat,
        return_state=False,
    )
    env = wrap_env(env, cfg)

    return env