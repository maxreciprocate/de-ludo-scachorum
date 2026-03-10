set -xeuo pipefail

# export NCCL_IBEXT_DISABLE=1
# export NCCL_NVLS_ENABLE=1
# export NCCL_IB_HCA=mlx5
# export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1

smol_gpu=1
if [[ $smol_gpu ]]; then
    gpu_memory_utilization=0.5
    train_batch_size=16
    max_reward_proc=1
else
    gpu_memory_utilization=0.9
    train_batch_size=64
    max_reward_proc=2
fi

echo $max_reward_proc

export GPUS_PER_NODE=1
NNODES=${SLURM_JOB_NUM_NODES:-1}
export NNODES

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export RAY_LOGGING_LEVEL=DEBUG
export HYDRA_FULL_ERROR=1

echo "Using $NNODES nodes for training..."

project_name='opus-verl'
MODEL_PATH=reciprocate/puzzle-0b6-mar03

loss_mode=gspo
loss_agg_mode="seq-mean-token-mean"
rollout_engine=vllm
rollout_mode=async

adv_estimator=gae
shuffle_dataset=true

test_freq=100
save_freq=-1
total_epochs=100
total_training_steps=500
val_before_train=false

use_kl_in_reward=false
kl_coef=0.00
use_kl_loss=false
kl_loss_coef=0.0

clip_ratio_low=0.0003 # as recommended by the paper, see Sec. 5.1
clip_ratio_high=0.0004 # as recommended by the paper, see Sec. 5.1
ppo_mini_batch_size=16 # maintain 4 mini-batches as recommended by the paper, see Sec. 5.1
ppo_micro_batch_size_per_gpu=4 # setup depending on your GPU memory
n_resp_per_prompt=1

max_prompt_length=128
max_response_length=128

# dapo reward manager params
# enable_overlong_buffer=false # true
# overlong_buffer_len=$((1024 * 4))
# overlong_penalty_factor=1.0

CKPTS_DIR=checkpoints

temperature=0.9
top_p=1.0
top_k=-1
val_top_p=0.9

sp_size=1
use_dynamic_bsz=true
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 3))
offload=true
gen_tp=1
entropy_checkpointing=true

train_files="['/workspace/data/opus/rl-prompts-v0-mar06.parquet']"
test_files="['/workspace/data/opus/rl-prompts-v0-mar06.parquet']"

exp_name="ppo-epslow-${clip_ratio_low}-epshigh-${clip_ratio_high}-$(basename $MODEL_PATH)"
# actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
# actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    # algorithm.adv_estimator=${adv_estimator} \

python3 -m verl.trainer.main_ppo \
    critic.model.path=$MODEL_PATH \
    critic.optim.lr=1e-6 \
    critic.ppo_micro_batch_size_per_gpu=8 \
    data.train_files="${train_files}" \
    data.val_files="${test_files}" \
    data.shuffle=$shuffle_dataset \
    data.prompt_key=prompt \
    data.truncation='error' \
    data.filter_overlong_prompts=true \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.name=${rollout_engine} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=false \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
    actor_rollout_ref.actor.optim.weight_decay=0.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0.00 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.ref.strategy=fsdp \
    custom_reward_function.path="reward.py" \
    custom_reward_function.name="compute_score_uniq" \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=true \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.max_model_len=512 \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=true \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.entropy_checkpointing=${entropy_checkpointing} \
    reward_model.reward_manager="rate_limited" \
    reward_model.enable=false \
    +reward.max_concurrent=$max_reward_proc \
    reward.num_workers=4 \
    reward_model.enable=false \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${GPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=${val_before_train} \
    trainer.test_freq=${test_freq} \
    trainer.save_freq=${save_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.total_training_steps=${total_training_steps} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    trainer.log_val_generations=2 \
    $@
