"""
Tool generation manager for LLM agents
"""

import torch
import re
from typing import List, Dict, Any, Tuple, Union
from dataclasses import dataclass
from PIL import Image
import numpy as np

from .tensor_helper import TensorHelper, TensorConfig
from agent_r1.tool.base import BaseToolEnv, BaseImageToolEnv

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

@dataclass
class ToolGenerationConfig:
    """Configuration for tool-based generation"""
    max_turns: int
    max_prompt_length: int 
    max_response_length: int
    max_response_length_single_turn: int
    max_tool_response_length: int
    use_batch_tool_calls: bool = False
    
class ToolGenerationManager:
    """Manager for handling LLM tool-based generation and interaction"""
    
    def __init__(
        self,
        tokenizer,
        processor,
        actor_rollout_wg,
        config: ToolGenerationConfig,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_tool_response_length=config.max_tool_response_length,
        ))

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    def _example_level_pad(self, data: Union[List[Any], np.ndarray],
                           active_mask: torch.Tensor,
                           pad_value: Any = None) -> Union[List[Any], np.ndarray]:
        """Pad data according to active mask.
        
        Args:
            data: Data to be padded. Can be list of any type (str, list, dict, etc.)
            active_mask: Boolean tensor indicating which positions are active
            pad_value: Value to use for padding. If None, will use:
                - Empty string "" for strings
                - Empty list [] for lists
                - Empty dict {} for dicts
                - None for other types
                
        Returns:
            Padded data with same type as input
        """
        # Get batch size from active mask
        batch_size = active_mask.shape[0]
        
        # Determine pad value if not provided
        if pad_value is None:
            if len(data) > 0:
                first_elem = data[0]
                if isinstance(first_elem, str):
                    pad_value = ""
                elif isinstance(first_elem, list):
                    pad_value = []
                else:
                    raise NotImplementedError(f"[WARNING] Unsupported data type: {type(first_elem)}")
                
        # Create padded output
        padded_data = [pad_value] * batch_size
        
        # Fill active positions
        s = 0
        for i, is_active in enumerate(active_mask):
            if is_active:
                padded_data[i] = data[s]
                s += 1
        
        # Convert to numpy array if input was numpy array
        if isinstance(data, np.ndarray):
            padded_data = np.array(padded_data, dtype=object)
            
        return padded_data
 
    def _update_rolling_state(self, rollings, assistant_messages: List[str], 
                            tool_messages: List[str], tool_images: List[List[Image.Image]]) -> Dict:
        """Update rolling state with new responses and observations.
        rollings : last llm input DataProto
        assistant_messages: assistant message
        tool_messages: tool message
        tool_images: tool image 
        """

        is_multi_modal = "multi_modal_data" in rollings.non_tensor_batch.keys()

        row_dict_list = []
        responses = []
        raw_prompts = []
        
        for i, (tool_message, assistant_message, tool_image) in enumerate(zip(tool_messages, assistant_messages, tool_images)):
            row_dict={}
            if is_multi_modal and '<image>' in tool_message and len(tool_image) > 0:
                assert len(tool_image) == tool_message.count('<image>'), f"[WARNING] TOOL RESPONSE IMAGE NUMBER NOT MATCH, {len(tool_image)} != {tool_message.count('<image>')} for {assistant_message}"
                raw_prompts.append(assistant_message + tool_message.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>'))
                row_dict['multi_modal_data'] = {'image': tool_image}
                image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
                row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}
                image_grid_thw = image_inputs['image_grid_thw']
                if image_grid_thw is not None:
                    merge_length = self.processor.image_processor.merge_size**2
                    index = 0
                    while '<image>' in tool_message:
                        tool_message = tool_message.replace(
                            '<image>',
                            '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                            '<|vision_end|>',
                            1,
                        )
                        index += 1

                    tool_message = tool_message.replace('<|placeholder|>', self.processor.image_token)

            else:
                raw_prompts.append(assistant_message + tool_message)
            responses.append(assistant_message + tool_message)
            row_dict_list.append(row_dict)

        responses_ids = self._batch_tokenize(responses)

        if "responses" not in rollings.batch.keys():
            rollings.batch['responses'] = responses_ids[:, :self.config.max_response_length_single_turn]
        else:
            rollings.batch['responses'] = self.tensor_fn.concatenate_with_padding([
                rollings.batch['responses'],
                responses_ids[:, :self.config.max_response_length_single_turn]
            ], pad_to_left=False)

        rollings.batch['responses'] = rollings.batch['responses'][:, :self.config.max_response_length]

        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            responses_ids
        ])

        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        
        if is_multi_modal:
            multi_modal_data = rollings.non_tensor_batch['multi_modal_data']
            multi_modal_inputs = rollings.non_tensor_batch['multi_modal_inputs']

            new_multi_modal_data = []
            new_multi_modal_inputs = []

            for row_dict, multi_modal_data_, multi_modal_inputs_ in zip(row_dict_list, multi_modal_data, multi_modal_inputs):
                if 'multi_modal_data' in row_dict.keys():
                    new_multi_modal_data.append({"image":multi_modal_data_['image'] + row_dict['multi_modal_data']['image']})
                else:
                    new_multi_modal_data.append({"image":multi_modal_data_['image']})
                if 'multi_modal_inputs' in row_dict.keys():
                    new_multi_modal_inputs.append({key: torch.cat((val,row_dict['multi_modal_inputs'][key]),dim=0) for key, val in multi_modal_inputs_.items()})
                else:
                    new_multi_modal_inputs.append({key: val for key, val in multi_modal_inputs_.items()})

            rollings.non_tensor_batch['multi_modal_data'] = np.array(new_multi_modal_data, dtype=object)
            rollings.non_tensor_batch['multi_modal_inputs'] = np.array(new_multi_modal_inputs, dtype=object)

            from verl.models.transformers.qwen2_vl import get_rope_index
            new_postion_ids = []
            for i in range(len(new_multi_modal_data)):
                new_postion_ids.append(get_rope_index(
                    processor=self.processor,
                    input_ids=new_input_ids[i],
                    image_grid_thw=new_multi_modal_inputs[i]['image_grid_thw'],
                    attention_mask=new_attention_mask[i],
                ))

            new_position_ids = torch.stack(new_postion_ids, dim=0)
        else:
            new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        rollings.batch['input_ids'] = new_input_ids
        rollings.batch['position_ids'] = new_position_ids
        rollings.batch['attention_mask'] = new_attention_mask

        raw_prompt_ids = rollings.non_tensor_batch['raw_prompt_ids'].tolist()
        new_raw_prompt_ids = []

        for raw_prompt_id, raw_prompt in zip(raw_prompt_ids, raw_prompts):
            if len(raw_prompt) > 0:
                new_raw_prompt_id = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
                if len(new_raw_prompt_id) > self.config.max_response_length_single_turn:
                    print(f"[WARNING] RESPONSE TOO LONG ({len(new_raw_prompt_id)}/{self.config.max_response_length_single_turn}), TRUNCATED: {raw_prompt}")
                    new_raw_prompt_id = new_raw_prompt_id[:self.config.max_response_length_single_turn]
                # Create a new list instead of extending the existing one
                new_raw_prompt_ids.append(raw_prompt_id + new_raw_prompt_id)
            else:
                new_raw_prompt_ids.append(raw_prompt_id)

        rollings.non_tensor_batch['raw_prompt_ids'] = np.array(new_raw_prompt_ids, dtype=object)

        return rollings
    
    def run_llm_loop(self, gen_batch, env: Union[BaseToolEnv, BaseImageToolEnv]) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""

        batch_size = gen_batch.batch['input_ids'].shape[0]
        
        active_mask = torch.ones(batch_size, dtype=torch.bool)
        turns = torch.zeros(batch_size, dtype=torch.int32)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch
        prompts = gen_batch.batch['input_ids'][:, -self.config.max_prompt_length:].clone()

        # Main generation loop
        for _ in range(self.config.max_turns):
            if not active_mask.sum():
                break

            # Check if any sequence exceeds max length
            effective_len = rollings.batch['attention_mask'].sum(dim=1)
            length_exceeded = effective_len > self.config.max_prompt_length

            if length_exceeded.sum() > 0:
                print("[WARNING] SEQUENCE LENGTH EXCEEDED MAX PROMPT LENGTH")
                active_mask[length_exceeded] = 0

            raw_prompt_ids = rollings.non_tensor_batch['raw_prompt_ids']
            length_exceeded = [len(prompt_id) > self.config.max_response_length for prompt_id in raw_prompt_ids]
            if any(length_exceeded):
                print("[WARNING] SEQUENCE LENGTH EXCEEDED MAX PROMPT LENGTH")
                for prompt_id, length_exceeded_ in zip(raw_prompt_ids, length_exceeded):
                    if length_exceeded_:
                        print(f"[DEBUG] LENGTH {len(prompt_id)}: {self.tokenizer.decode(prompt_id)}")
                active_mask[length_exceeded] = 0
            
            if not active_mask.sum():
                print("[WARNING] NO ACTIVE SEQUENCES")
                break
            
            if hasattr(rollings, 'non_tensor_batch') and rollings.non_tensor_batch:
                # Create active batch with tensor data
                rollings_active = DataProto.from_dict(
                    tensors={
                        k: v[active_mask] for k, v in rollings.batch.items()
                    },
                    non_tensors={
                        k: v[active_mask.numpy()] for k, v in rollings.non_tensor_batch.items()
                    }
                )
            else:
                rollings_active = DataProto.from_dict(
                    batch={
                        k: v[active_mask] for k, v in rollings.batch.items()
                    },
                )

            rollings_active, pad_size = pad_dataproto_to_divisor(rollings_active, self.actor_rollout_wg.world_size)
            gen_output = self.actor_rollout_wg.generate_sequences(rollings_active)
            gen_output = unpad_dataproto(gen_output, pad_size=pad_size)

            raw_assistant_messages = self.tokenizer.batch_decode(gen_output.batch['responses'], skip_special_tokens=True)
            if isinstance(env, BaseToolEnv):
                if self.config.use_batch_tool_calls:
                    assistant_messages, tool_messages, _, new_active_masks = env.batch_step(raw_assistant_messages)
                else:
                    assistant_messages = []
                    tool_messages = []
                    new_active_masks = []
                    for raw_assistant_message in raw_assistant_messages:
                        assistant_message, tool_message, _, active = env.step(raw_assistant_message)
                        assistant_messages.append(assistant_message)
                        tool_messages.append(tool_message)
                        new_active_masks.append(active)
                tool_images = [[]] * len(raw_assistant_messages)
            elif isinstance(env, BaseImageToolEnv):
                if self.config.use_batch_tool_calls:
                    assistant_messages, tool_messages, tool_images, _, new_active_masks = env.batch_step(raw_assistant_messages)
                else:
                    assistant_messages = []
                    tool_messages = []
                    tool_images = []
                    new_active_masks = []
                    for raw_assistant_message in raw_assistant_messages:
                        assistant_message, tool_message, tool_image, success, stop = env.step(raw_assistant_message)

            print(f"[DEBUG] Assistant message Example: {assistant_messages[0]}")
            print(f"[DEBUG] Tool message Example: {tool_messages[0]}")

            assistant_messages = self._example_level_pad(assistant_messages, active_mask, pad_value="")
            tool_messages = self._example_level_pad(tool_messages, active_mask, pad_value="")
            tool_images = self._example_level_pad(tool_images, active_mask, pad_value=[])

            active_mask[active_mask.clone()] = torch.tensor(new_active_masks, dtype=torch.bool)

            turns[active_mask] += 1

            active_num_list.append(active_mask.sum().item())

            # Update states
            rollings = self._update_rolling_state(
                rollings,
                assistant_messages,
                tool_messages,
                tool_images
            )
 
        print("ACTIVE_TRAJ_NUM:", active_num_list)

        # Compose final output
        final_output = {}
        final_output['turns'] = turns
        final_output['prompts'] = prompts
        final_output['responses'] = rollings.batch['responses']
        final_output['input_ids'] = torch.cat([
            prompts,
            rollings.batch['responses']
        ], dim=1)
        final_output['attention_mask'] = self.tensor_fn.create_attention_mask(final_output['input_ids'])
        if "multi_modal_data" in rollings.non_tensor_batch.keys():
            from verl.models.transformers.qwen2_vl import get_rope_index
            position_ids = []
            for i in range(len(rollings.non_tensor_batch['multi_modal_data'])):
                position_ids.append(get_rope_index(
                    processor=self.processor,
                    input_ids=final_output['input_ids'][i],
                    image_grid_thw=rollings.non_tensor_batch['multi_modal_inputs'][i]['image_grid_thw'],
                    attention_mask=final_output['attention_mask'][i],
                ))

            position_ids = torch.stack(position_ids, dim=0)
            final_output['position_ids'] = position_ids
        else:
            final_output['position_ids'] = self.tensor_fn.create_position_ids(final_output['attention_mask'])
        
        final_output = DataProto.from_dict(final_output)
        final_output.non_tensor_batch = rollings.non_tensor_batch
        
        return final_output