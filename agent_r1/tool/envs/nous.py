from agent_r1.tool.base import BaseToolEnv, BaseTool
from typing import List, Dict, Tuple
import re
import json

class NousToolEnv(BaseToolEnv):
    def __init__(self, tools: List[BaseTool], config: Dict):
        self.tools = tools
        self.tool_map = {tool.name: tool for tool in self.tools}
        self.tool_call_start = "<tool_call>"
        self.tool_call_end = "</tool_call>"
        self.tool_response_start = "<tool_response>"
        self.tool_response_end = "</tool_response>"
        self.eos_token = "<|im_end|>"
        self.parallel_tool_calls = config.get("parallel_tool_calls", False)
        self.cut_tool_call = config.get("cut_tool_call", True)

    def step(self, raw_assistant_message: str) -> Tuple[str, str, List[bool], bool]:
        tool_calls = self.extract_tool_calls(raw_assistant_message)
        if len(tool_calls) == 0:
            return raw_assistant_message + self.eos_token, "", [], False
        if not self.parallel_tool_calls:
            tool_calls = [tool_calls[0]]
        tool_responses = []
        tool_successes = []
        for tool_call in tool_calls:
            if tool_call is None:
                tool_responses.append("Error: JSONDecodeError")
                tool_successes.append(False)
            else:
                if "name" not in tool_call:
                    tool_responses.append("Error: No tool name")
                    tool_successes.append(False)
                else:
                    tool_name = tool_call["name"]
                    if tool_name not in self.tool_map:
                        tool_responses.append("Error: ToolNotFoundError")
                        tool_successes.append(False)
                    else:
                        tool = self.tool_map[tool_name]
                        if not tool.validate_args(tool_call["arguments"]):
                            tool_responses.append("Error: Invalid tool arguments")
                            tool_successes.append(False)
                        else:
                            tool_result = tool.execute(tool_call["arguments"])
                            tool_responses.append(tool_result["content"])
                            tool_successes.append(tool_result["success"])
        if self.cut_tool_call:
            assistant_message = raw_assistant_message.split(self.tool_call_end)[0] + self.tool_call_end + self.eos_token
        else:
            assistant_message = raw_assistant_message + self.eos_token
        tool_message = self.format_tool_response(tool_responses)
        return assistant_message, tool_message, tool_successes, True

    def batch_step(self, raw_assistant_messages: List[str]) -> Tuple[List[str] | List[bool]]:
        # TODO: 改成并行执行
        # return super().batch_step(raw_assistant_messages)
        # 使用batch_excute执行
        batch_assistant_messages = [""] * len(raw_assistant_messages)
        batch_tool_responses = [[]] * len(raw_assistant_messages)
        batch_tool_messages = [""] * len(raw_assistant_messages)
        batch_tool_successes = [[]] * len(raw_assistant_messages)
        batch_active = [False] * len(raw_assistant_messages)
        success_tool_calls_arguments = {} # batch 内成功的工具调用。key: tool_name，value: [arguments]
        success_tool_calls_index = {} # batch 内成功的工具调用。key: tool_name，value: [(i,j)]
        for i, raw_assistant_message in enumerate(raw_assistant_messages):
            tool_calls = self.extract_tool_calls(raw_assistant_message)
            if len(tool_calls) == 0:
                batch_assistant_messages[i] = raw_assistant_message + self.eos_token
                batch_tool_messages[i] = ""
                batch_tool_successes[i] = []
                batch_active[i] = False
                batch_tool_responses[i] = []
                continue

            if self.cut_tool_call:
                assistant_message = raw_assistant_message.split(self.tool_call_end)[0] + self.tool_call_end + self.eos_token
            else:
                assistant_message = raw_assistant_message + self.eos_token
            batch_assistant_messages[i] = assistant_message

            if not self.parallel_tool_calls:
                tool_calls = [tool_calls[0]]
            tool_responses = []
            tool_successes = []
            for j, tool_call in enumerate(tool_calls):
                if tool_call is None:
                    tool_responses.append("Error: JSONDecodeError")
                    tool_successes.append(False)
                else:
                    tool_name = tool_call["name"]
                    if tool_name not in self.tool_map:
                        tool_responses.append("Error: ToolNotFoundError")
                        tool_successes.append(False)
                    else:
                        tool = self.tool_map[tool_name]
                        if not tool.validate_args(tool_call["arguments"]):
                            tool_responses.append("Error: Invalid tool arguments")
                            tool_successes.append(False)
                        else:
                            # 默认success_tool_calls[tool_name]
                            if tool_name not in success_tool_calls_arguments:
                                success_tool_calls_arguments[tool_name] = []
                                success_tool_calls_index[tool_name] = []
                            tool_responses.append("Executing...")
                            tool_successes.append(False)
                            success_tool_calls_arguments[tool_name].append(tool_call["arguments"])
                            success_tool_calls_index[tool_name].append((i,j))
            batch_tool_responses[i] = tool_responses
            batch_tool_successes[i] = tool_successes
            batch_active[i] = True
        
        # batch excute
        for tool_name, args_list in success_tool_calls_arguments.items():
            tool = self.tool_map[tool_name]
            batch_results = tool.batch_execute(args_list)
            # print(f"[DEBUG] batch_results: {batch_results}")
            # print(f"[DEBUG] success_tool_calls_index: {success_tool_calls_index[tool_name]}")
            # print(f"[DEBUG] batch_tool_responses: {batch_tool_responses}")
            for batch_result, (i,j) in zip(batch_results, success_tool_calls_index[tool_name]):
                assert batch_tool_responses[i][j] == "Executing..."
                batch_tool_responses[i][j] = batch_result["content"]
                batch_tool_successes[i][j] = batch_result["success"]
        
        batch_tool_messages = [self.format_tool_response(tool_responses) for tool_responses in batch_tool_responses]
        
        return batch_assistant_messages, batch_tool_messages, batch_tool_successes, batch_active

    def stop(self, raw_assistant_message: str) -> bool:
        tool_calls = self.extract_tool_calls(raw_assistant_message)
        if len(tool_calls) == 0:
            return True
        else:
            return False
        
    def extract_tool_calls(self, raw_assistant_message: str) -> List[str]:
        tool_calls = []
        pattern = re.compile(f"{re.escape(self.tool_call_start)}(.*?){re.escape(self.tool_call_end)}", re.DOTALL)
        for tool_call in re.findall(pattern, raw_assistant_message):
            try:
                tool_call = json.loads(tool_call)
                tool_calls.append(tool_call)
            except json.JSONDecodeError:
                tool_calls.append(None)
        
        return tool_calls
        
    def format_tool_response(self, tool_responses: List[str]) -> str:
        tool_message = "\n<|im_start|>user\n"
        for i, tool_response in enumerate(tool_responses):
            tool_message += f"<tool_response>\n{tool_response}\n</tool_response>"
            if i < len(tool_responses) - 1:
                tool_message += "\n"
        tool_message += "<|im_end|>\n<|im_start|>assistant\n<think>\n"
        return tool_message
        