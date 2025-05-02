from agent_r1.tool.envs.nous import NousToolEnv
from agent_r1.tool.envs.retool import ReToolEnv

def _default_env(name):
    if name == "nous":
        return NousToolEnv
    elif name == "retool":
        return ReToolEnv
    else:
        raise NotImplementedError(f"Tool environment {name} is not implemented")