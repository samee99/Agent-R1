# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re


def extract_solution(solution_str):
    if solution_str is None:
        return None
        
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.search(answer_pattern, solution_str, re.DOTALL)
    
    if match:
        # 首先尝试从<answer>标签内提取数字
        answer_content = match.group(1)
        numbers_in_answer = re.findall(r"(\-?[0-9\\.\\,]+)", answer_content)
        
        if numbers_in_answer:
            # 从<answer>标签内找到数字
            invalid_str = ['', '.']
            # 找到最后一个非无效的数字
            for final_answer in reversed(numbers_in_answer):
                if final_answer not in invalid_str:
                    return final_answer
        
        # 如果<answer>标签内没有找到有效数字，则在整个solution_str中查找
        numbers_in_solution = re.findall(r"(\-?[0-9\\.\\,]+)", solution_str)
        final_answer = None
        if numbers_in_solution:
            invalid_str = ['', '.']
            # 找到最后一个非无效的数字
            for final_answer in reversed(numbers_in_solution):
                if final_answer not in invalid_str:
                    break
        return final_answer
    else:
        return None

def compute_score_format(solution_str):
    """The scoring function for format reward.

    Args:
        solution_str: the solution text
    
    """
    if solution_str is None:
        return 0.0
        
    try:
        # Perfect format match for the new structure
        # First <|im_start|>assistant should have <think> and possibly <tool_call>
        # Then <|im_start|>tool with <tool_response> (can repeat with assistant/tool pairs)
        # Final <|im_start|>assistant with the answer and <|im_end|>
        
        # Check for basic structure with <|im_start|>assistant and <|im_end|> tags
        assistant_blocks = re.findall(r'<\|im_start\|>assistant\n?(.*?)<\|im_end\|>', solution_str, re.DOTALL)
        tool_blocks = re.findall(r'<\|im_start\|>user\n?(.*?)<\|im_end\|>', solution_str, re.DOTALL)

        format_reward = 0.0
        
        # If no blocks found, return 0
        if not assistant_blocks:
            return 0.0
        
        # Perfect format requires at least one assistant block and matching tool blocks if tool calls exist
        # Check first assistant block contains <think> tags
        for i, assistant_block in enumerate(assistant_blocks[:-1]):
            if assistant_block.count('<think>') == 1 and assistant_block.count('</think>') == 1 and assistant_block.count('<tool_call>') == 1 and assistant_block.count('</tool_call>') == 1:
                think_match = re.search(r'^<think>(.*?)</think>\n<tool_call>(.*?)</tool_call>$', assistant_block, re.DOTALL)
                soft_think_match = re.search(r'<think>(.*?)</think>(.*?)<tool_call>(.*?)</tool_call>', assistant_block, re.DOTALL)
                if think_match:
                    # format_reward += 0.2 * (0.8 ** i)
                    format_reward += max(0, 0.2 - 0.05 * i)
                elif soft_think_match:
                    # format_reward += 0.1 * (0.8 ** i)
                    format_reward += max(0, 0.1 - 0.05 * i)
        
        # Check the last assistant block contains <answer> tags
        if assistant_blocks:  # 确保有至少一个assistant块
            last_assistant_block = assistant_blocks[-1]
            think_answer_match = re.search(r'^<think>(.*?)</think>\n<answer>(.*?)</answer>$', last_assistant_block, re.DOTALL)
            think_match = re.search(r'<think>(.*?)</think>', last_assistant_block, re.DOTALL)
            answer_match = re.search(r'<answer>(.*?)</answer>', last_assistant_block, re.DOTALL)
            if think_answer_match:
                format_reward += 0.2
            elif think_match and answer_match:
                format_reward += 0.15
            elif think_match and not answer_match:
                format_reward += 0.1
            elif not think_match and answer_match:
                format_reward += 0.05

        if tool_blocks:
            for i, tool_block in enumerate(tool_blocks):
                if "Result:" in tool_block:
                    format_reward += 0.4 - 0.1 * i
        
        return format_reward
    except Exception as e:
        print(f"Error in compute_score_format: {e}")
        return 0.0

def compute_score_answer(solution_str, ground_truth):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
    """
    if solution_str is None:
        return 0.0
        
    try:
        assistant_blocks = re.findall(r'<\|im_start\|>assistant\n?(.*?)<\|im_end\|>', solution_str, re.DOTALL)
        
        if not assistant_blocks:
            # 如果没有找到assistant块，直接在整个solution_str中查找答案
            answer = extract_solution(solution_str=solution_str)
        else:
            # 使用最后一个assistant块
            last_block = assistant_blocks[-1]
            answer = extract_solution(solution_str=last_block)

        if answer is None:
            if ground_truth in solution_str:
                return 0.2
            else:
                return 0.0
        else:
            if answer == ground_truth:
                return 1.0
            else:
                return 0.0
    except Exception as e:
        print(f"Error in compute_score_answer: {e}")
        return 0.0
        
def compute_score_format_answer(solution_str, ground_truth):
    """The scoring function for GSM8k.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
    """
    if solution_str is None or ground_truth is None:
        return 0.0
        
    try:
        format_score = compute_score_format(solution_str)
        answer_score = compute_score_answer(solution_str, ground_truth)
        return format_score + answer_score
    except Exception as e:
        print(f"Error in compute_score_format_answer: {e}")
        return 0.0

def compute_score_em(solution_str, ground_truth):
    """The scoring function for GSM8k.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
    """
    if solution_str is None or ground_truth is None:
        return 0.0
        
    try:
        assistant_blocks = re.findall(r'<\|im_start\|>assistant\n?(.*?)<\|im_end\|>', solution_str, re.DOTALL)
        
        if not assistant_blocks:
            return 0.0
        else:
            # 使用最后一个assistant块
            last_block = assistant_blocks[-1]
            answer = extract_solution(solution_str=last_block)
            
        if answer == ground_truth:
            return 1.0
        else:
            return 0.0
    except Exception as e:
        print(f"Error in compute_score_em: {e}")
        return 0.0

if __name__ == "__main__":
    test_case = """<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"name": "calculator", "description": "\n        Perform basic arithmetic calculations (addition, subtraction, multiplication, division).\n        Expression format:\n        - Use standard mathematical notation: 2 + 3, 10 - 5, 4 * 7, 20 / 5\n        - Parentheses are supported for grouping: (2 + 3) * 4\n        - Nested operations are allowed: 2 + 3 * 4\n        - Division by zero will result in an error\n        - Only numbers and basic arithmetic operators (+, -, *, /) are supported\n        ", "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "Arithmetic expression to evaluate (using operators +, -, *, / and parentheses)"}}, "required": ["expression"]}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
<|im_start|>user
Evan's dog weighs 63 pounds; it weighs 7 times as much as Ivan's dog.  Together, what is the weight of the dogs? You can use the tools provided to you to answer the question. You can use the tool as many times as you want.
You must first conduct reasoning inside <think>...</think>. If you get the final answer, you can output the answer inside <answer>...</answer>.

Output format for tool call:
<think>
...
</think>
<tool_call>
...
</tool_call>

Output format for answer:
<think>
...
</think>
<answer>
...
</answer>
<|im_end|>
<|im_start|>assistant
<think>
First, we need to find out the weight of Ivan's dog. According to the problem, Evan's dog weighs 63 pounds and it weighs 7 times as much as Ivan's dog. So, the weight of Ivan's dog can be calculated as 63 pounds divided by 7. After finding the weight of Ivan's dog, we add the weights of both dogs to find the total weight.
</think>
<think>
Let's calculate the weight of Ivan's dog.
</think>
<tool_call>
{"name": "calculator", "arguments": {"expression": "63 / 7"}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
Result: 9.0
</tool_response>
<|im_end|>
<|im_start|>assistant
Now we know that Ivan's dog weighs 9 pounds.

Next, we calculate the total weight of the dogs by adding Evan's dog weight (63 pounds) and Ivan's dog weight (9 pounds).
</im_start>
<think>
The total weight is 63 + 9.
</think>
<think>
Let's calculate the total weight.
</think>
<tool_call>
{"name": "calculator", "arguments": {"expression": "63 + 9"}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
Result: 72
</tool_response>
<|im_end|>
<|im_start|>assistant
The total weight of the dogs is 72 pounds.

<think>
Therefore, the answer is 72 pounds.
</think>
<answer>
72
</answer>
<|im_end|>"""
    print(compute_score_format_answer(test_case, "72"))
    print(compute_score_em(test_case, "72"))