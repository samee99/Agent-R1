from agent_r1.tool.envs.nous import NousToolEnv

def _default_env(name):
    if name == "nous":
        return NousToolEnv
    else:
        raise NotImplementedError