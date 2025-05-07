from agent_r1.tool.tools.search_tool import SearchTool
from agent_r1.tool.tools.python_tool import PythonTool
from agent_r1.tool.tools.wiki_search_tool import WikiSearchTool

def _default_tool(name):
    if name == "search":
        return SearchTool()
    elif name == "wiki_search":
        return WikiSearchTool()
    elif name == "python":
        return PythonTool()
    else:
        raise NotImplementedError(f"Tool {name} not implemented")