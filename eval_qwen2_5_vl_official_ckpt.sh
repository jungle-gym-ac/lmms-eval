. /data/miniconda3/etc/profile.d/conda.sh
conda activate lmms-eval
pip install -e lmms-eval[qwen]

export HF_HUB_ENABLE_HF_TRANSFER="0" #https://lmms-lab.github.io/posts/lmms-eval-0.2/
export WANDB_API_KEY=''
export HF_TOKEN=''
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1

#--------------------------------------------------
# Checkpoint path
#--------------------------------------------------
export HF_HOME="/group/40034/jimjunzhang/.cache/huggingface"
ckpt="/group/40033/public_models/Qwen2.5-VL-3B-Instruct"
# export HF_HOME="/group/40007/jimjunzhang/.cache/huggingface"
# ckpt="/group/40007/jimjunzhang/Qwen2.5-VL-3B-Instruct"

#--------------------------------------------------
# Hyperparameters
#--------------------------------------------------
max_num_frames=192
fps=2

max_tokens=256 # 16384(default setting, native resolution) 1280(for performance boost)
min_tokens=128 # 4(default setting native resolution), 256(for performance boost)
max_pixels=$(($max_tokens * 28 * 28))
min_pixels=$(($min_tokens * 28 * 28))

# max_tokens * max_num_frames / 2 already controls total token number, so we don't need to set total_pixels
# total_pixels=$((24576 * 28 * 28)) # 128000 the default value, 24576 the video eval setting in Qwen2.5-VL paper
# 256 * 192 / 2 = 128 * 384 / 2 = 24576


#-----------------------------------------------------------------
# Evaluate Task
#-----------------------------------------------------------------
eval_tasks="temporal_grounding_charades" # "temporal_grounding_charades" "videomme"

#--------------------------------------------------
# Run evaluation
#--------------------------------------------------
# TODO: change the output path to your own path
output_prefix="/group/40034/yuyingge/qwen2.5-vl-eval/logs"
output_path="$output_prefix/${eval_tasks}/max_tokens-${max_tokens}_min_tokens-${min_tokens}_fps-${fps}_max_num_frames-${max_num_frames}"

bash jim_dev_scripts/eval.sh \
    --ckpt $ckpt \
    --fps $fps \
    --max_num_frames $max_num_frames \
    --min_pixels $min_pixels \
    --max_pixels $max_pixels \
    --eval_tasks $eval_tasks \
    --output_path $output_path

# python lmms-eval/lmms_eval/tasks/charades_sta/eval_tvg.py -f xxxxxxxx.json
