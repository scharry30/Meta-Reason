import torch
from typing import List, Union
from verl import DataProto
from verl.utils.reward_score import gsm8k, math
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from rewards_types import RewardConfig, RewardType
from control_reward import control_reward_fn  
from utils import extract_control_spans

import ray
import hydra
import numpy as np



class CSPORewardManager():
    def __init__(self, tokenizer, reward_config, max_control_spans=4):
        self.tokenizer = tokenizer
        self.reward_config = reward_config
        self.max_control_spans = max_control_spans

    def __call__(self, data: DataProto, return_dict=True):
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        ctrl_mask_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        debug_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        from concurrent.futures import ThreadPoolExecutor

        def process_item(args):
            i, data_item = args
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            full_text = self.tokenizer.decode(torch.cat((valid_prompt_ids, valid_response_ids)))
            control_spans, control_token_indices = extract_control_spans(full_text, self.tokenizer, valid_prompt_length)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            num_tokens = data_item.non_tensor_batch['reward_model']['num_tokens']
            data_source = data_item.non_tensor_batch['data_source']

            compute_score_fn = _select_rm_score_fn(data_source)
            score, ctrl_mask = compute_score_fn(
                full_text,
                ground_truth=ground_truth,
                control_token_indices=control_token_indices,
                reward_config=self.reward_config,
                return_ctrl_mask=True
            )
           
            for idx in control_token_indices:
                reward_tensor[i, idx] = score
                ctrl_mask_tensor[i, idx] = 1.0
            debug_tensor[i, :len(control_token_indices)] = torch.tensor(control_token_indices)
            return i, score, ctrl_mask

        with ThreadPoolExecutor(max_workers=96) as executor:
            args = [(i, data[i]) for i in range(len(data))]
            results = list(executor.map(process_item, args))

        for i, score, ctrl_mask in results:
            pass  

        if return_dict:
            return {
                'reward_tensor': reward_tensor,
                'ctrl_mask_tensor': ctrl_mask_tensor,
                'reward_extra_info': {'debug_control_indices': debug_tensor}
            }
        else:
            return reward_tensor, ctrl_mask_tensor


def main(config):
    if not ray.is_initialized():
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from verl.utils import hf_tokenizer
    from transformers import AutoTokenizer
    from omegaconf import OmegaConf
    from pprint import pprint

    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)
    tokenizer = hf_tokenizer(local_path)

    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup
    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup
    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker)
    }
    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    reward_config = RewardConfig(
        alpha=config.reward_config.alpha,
        sigmoid_reward=config.reward_config.sigmoid_reward,
        linear_reward=config.reward_config.linear_reward,
        multiplier_reward=config.reward_config.multiplier_reward
    )
    reward_fn = CSPORewardManager(tokenizer=tokenizer, reward_config=reward_config)
    val_reward_fn = CSPORewardManager(tokenizer=tokenizer, reward_config=reward_config)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    trainer = RayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn
    )
    trainer.init_workers()
    trainer.fit()

if __name__ == '__main__':
    main()
