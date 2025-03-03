"""
Tool generation manager for LLM agents
"""

import torch
import re
import json
import os
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

import random

from .tensor_helper import TensorHelper, TensorConfig
from agent_r1.tool.tool_env import ToolEnv
from verl import DataProto
from verl.utils.tracking import Tracking

@dataclass
class ToolGenerationConfig:
    """Configuration for tool-based generation"""
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_tool_response_length: int  # Renamed from max_obs_length
    num_gpus: int

class ToolGenerationManager:
    """Manager for handling LLM tool-based generation and interaction"""
    
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: ToolGenerationConfig,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation
        
        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_tool_response_length=config.max_tool_response_length,  # Renamed
            max_start_length=config.max_start_length,
        ))

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    @staticmethod
    def _process_tool_call(responses_str) -> Tuple[List[str], List[bool]]:
        """
        Process a list of response strings to extract the first tool call
        while preserving the rest of the string content.
        
        Args:
            responses_str (List[str]): List of response strings potentially containing tool calls
            
        Returns:
            List[str]: Processed responses with only first tool call preserved
        """
        def process_single_response(resp):
            # Look for tool call pattern: <tool_call>tool_name(args)</tool_call>
            tool_pattern = r'<tool_call>(.*?)</tool_call>'
            match = re.search(tool_pattern, resp, re.DOTALL)
            
            if not match:
                return resp, False  # No tool call found
            
            # resp = resp.split('</tool_call>')[0] + '</tool_call>'
            tool_content = match.group(0)
            
            # Replace all subsequent answer tag pairs with their content
            rest_of_string = resp[match.end():]
            cleaned_rest = re.sub(r'<tool_call>(.*?)</tool_call>', r'\1', rest_of_string, flags=re.DOTALL)
            
            return resp[:match.start()] + tool_content + cleaned_rest, True
        
        # Process each response string
        return [process_single_response(resp)[0] for resp in responses_str], [process_single_response(resp)[1] for resp in responses_str]

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """Process responses to extract tool calls."""
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )

        # Extract the first tool call from each response
        responses_str, active_masks = self._process_tool_call(responses_str)
        
        # Tokenize processed responses
        cleaned_token_ids = self._batch_tokenize(responses_str)
        
        return cleaned_token_ids, responses_str, torch.tensor(active_masks, dtype=torch.bool)
    
    def _process_tool_responses(self, tool_responses: List[str]) -> torch.Tensor:
        """Process tool responses to token ids"""
        
        tool_responses_ids = self.tokenizer(
            tool_responses, 
            padding='longest',
            return_tensors='pt'
        )['input_ids']
        
        if tool_responses_ids.shape[1] > self.config.max_tool_response_length:
            print("[WARNING] TOOL RESPONSE TOO LONG, CONSIDER CHANGING YOUR CONFIG")
            tool_responses_ids = tool_responses_ids[:, :self.config.max_tool_response_length]
            
        return tool_responses_ids
    
    def _execute_tool_calls(self, response_strs: List[str], 
                          envs: List[ToolEnv], 
                          active_mask: torch.Tensor) -> Tuple[List[str], List[float], List[bool]]:
        """Execute tool calls and return tool responses, rewards, and done flags."""
        tool_responses = []
        
        for resp, env, active in zip(response_strs, envs, active_mask):
            if not active:
                tool_responses.append("")
                continue
            # Step the environment using the agent's response
            tool_response, _, _, _ = env.step(resp)
            tool_response = ToolEnv.formulate_output(tool_response)
            
            # Collect results
            tool_responses.append(tool_response)
            
        return tool_responses
    
    def _update_rolling_state(self, rollings, cur_responses: torch.Tensor, 
                            tool_responses_ids: torch.Tensor) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            tool_responses_ids
        ])
        
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          tool_responses_ids: torch.Tensor) -> Dict:
        """Update right side state."""
        responses = self.tensor_fn.concatenate_with_padding([
            right_side['responses'],
            cur_responses,
            tool_responses_ids
        ], pad_to_left=False)
        
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {'responses': responses[:, :max_len]}


    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)
            
        padded_active_batch = DataProto.from_dict(padded_batch)
        
        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)
        
        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output
    
    def run_llm_loop(self, gen_batch, envs: List[Any] = None,
                    initial_input_ids: torch.Tensor = None) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""
        
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {'responses': initial_input_ids[:, []]}
        
        batch_size = gen_batch.batch['input_ids'].shape[0]
        
        active_mask = torch.ones(batch_size, dtype=torch.bool)
        turns = torch.zeros(batch_size, dtype=torch.int32)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch

        # print(f"[DEBUG] BATCH SIZE: {batch_size}")

        # Main generation loop
        for step in range(self.config.max_turns):
            # print(f"[DEBUG] ACTIVE MASK SHAPE: {active_mask.shape}")
            if not active_mask.sum():
                break
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info            
            responses_ids, responses_str, new_active_masks = self._postprocess_responses(gen_output.batch['responses'])

            # print(f"[DEBUG] RESPONSES IDS SHAPE: {responses_ids.shape}")
            print(f"[DEBUG] RESPONSES STR EXAMPLE: {responses_str[0]}")

            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            active_mask[active_mask.clone()] = new_active_masks

            turns[active_mask] += 1

            tool_responses = self._execute_tool_calls(responses_str, envs, active_mask)

            # print(f"[DEBUG] TOOL RESPONSES EXAMPLE: {tool_responses[0]}")

            active_num_list.append(active_mask.sum().item())
            tool_responses_ids = self._process_tool_responses(tool_responses)

            # print(f"[DEBUG] TOOL RESPONSES IDS SHAPE: {tool_responses_ids.shape}")
            
            # Update states
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                tool_responses_ids
            )
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                tool_responses_ids
            )
        
        print("ACTIVE_TRAJ_NUM:", active_num_list)
        
        original_right_side['turns'] = turns
        
        # Save trajectory and return final output
        return self._compose_final_output(original_left_side, original_right_side, meta_info)


    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']

        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        
        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)

        return final_output