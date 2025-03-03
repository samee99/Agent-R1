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
import string
import random

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score


def extract_solution(solution_str):
    """Extract the answer from the solution string."""
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.search(answer_pattern, solution_str, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    return None

def compute_score_format(solution_str):
    """The scoring function for format reward.

    Args:
        solution_str: the solution text
    
    """
    # Perfect format match
    perfect_format_pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"
    if re.match(perfect_format_pattern, solution_str, re.DOTALL):
        return 0.2
    
    # Soft match - <think> and <answer> tags exist but not in perfect format
    soft_format_pattern = r"<think>.*?</think>.*?<answer>.*?</answer>"
    if re.search(soft_format_pattern, solution_str, re.DOTALL):
        return 0.1
    
    # Check if <think> tag exists
    think_pattern = r"<think>.*?</think>"
    if re.search(think_pattern, solution_str, re.DOTALL):
        return 0.05
    
    return 0


def compute_score_answer(solution_str, ground_truth):
    """The scoring function for exact match (EM) with format reward.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
    
    Returns:
        float: Total reward score (format reward + answer reward)
    """
    # Extract answer from <answer> tags
    answer = extract_solution(solution_str)

    answer_reward = 0.0
    
    if answer is not None:
        # Check for exact match within <answer>
        if em_check(answer, ground_truth):
            answer_reward = 1.0
        # Check for substring match within <answer>
        elif subem_check(answer, ground_truth):
            answer_reward = 0.5
    
    # If no match found within <answer>, check entire solution for substring match
    if answer_reward == 0.0:
        if subem_check(solution_str, ground_truth):
            answer_reward = 0.1
    
    return answer_reward

def compute_score_format_answer(solution_str, ground_truth):
    """The scoring function for format reward.

    Args:
        solution_str: the solution text
    
    """
    format_reward = compute_score_format(solution_str)
    answer_reward = compute_score_answer(solution_str, ground_truth)
    return format_reward + answer_reward

def compute_score_em(solution_str, ground_truth):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
    
    """
    answer = extract_solution(solution_str)
    return subem_check(answer, ground_truth)
