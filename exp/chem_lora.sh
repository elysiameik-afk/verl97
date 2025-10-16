#!/bin/bash
set -x

# 激活环境
# eval "$(conda shell.bash hook)"
# conda deactivate
# conda activate verl

# 设置环境变量
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export HYDRA_FULL_ERROR=1

# ===================================================================
# FIX: 定义一个Shell变量来存储响应长度，避免'bad substitution'错误
# ===================================================================
MAX_RESPONSE_LENGTH=512

# FIX: 重新使用反斜杠 `\` 来分割长命令，确保脚本可读性和正确性
# ===================================================================
# LoRA版本配置 - 启用LoRA训练以节省显存和加速训练
# ===================================================================
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.actor.policy_loss.loss_mode=plic_p \
    actor_rollout_ref.actor.policy_loss.plic_p=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.0003 \
    actor_rollout_ref.actor.clip_ratio_high=0.0004 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    data.train_files=/root/autodl-tmp/verl97/verl/data/molecule_generation_train.parquet \
    data.val_files=/root/autodl-tmp/verl97/verl/data/molecule_generation_val.parquet \
    data.train_batch_size=16 \
    data.val_batch_size=8 \
    data.max_prompt_length=512 \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.shuffle=true \
    data.prompt_key=prompt \
    data.truncation='error' \
    data.filter_overlong_prompts=true \
    actor_rollout_ref.model.path=/root/autodl-tmp/LlaSMol-EGFR-Final-exp3 \
    actor_rollout_ref.actor.optim.lr=3e-5 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=true \
    'actor_rollout_ref.model.custom_chat_template=[INST] {{ messages[0]["content"] }} [/INST]' \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.entropy_checkpointing=true \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.loss_agg_mode="seq-mean-token-mean" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=true \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    reward_model.reward_manager=chem_rl \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=false \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=2000 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward_model.reward_kwargs.max_resp_len=${MAX_RESPONSE_LENGTH} \
    trainer.project_name=CHEMRL \
    trainer.experiment_name=egfr-1-lora \
    trainer.default_local_dir=/root/autodl-tmp/verl97/verl/ckpts/CHEMRL/egfr-1-lora\
    trainer.critic_warmup=0 \
    trainer.save_freq=16 \
    trainer.test_freq=1 \
    trainer.total_epochs=4 \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.logger='["console","wandb"]' \
    trainer.val_before_train=false \
    trainer.resume_mode=auto \
    trainer.log_val_generations=2 \
    $@

