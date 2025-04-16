from agent_r1.tool.tools.search_tool import SearchTool

def _default_tool(name):
    if name == "search":
        return SearchTool()
    else:
        raise NotImplementedError