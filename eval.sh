#!/bin/bash

GPUS=`nvidia-smi -L | wc -l` #count all GPUs
master_port=12345
# accelerate config
accelerate_config=$HF_HOME/accelerate/default_config.yaml

eval_tasks_list=(
"temporal_grounding_charades"
)
eval_tasks=$(IFS=,; echo "${eval_tasks_list[*]}")

ckpt="/group/40033/jimjunzhang/Qwen2.5-VL-3B-Instruct"
run_name="qwen2_5_vl"
project_name="qwen"
output_path="./logs" #"$ckpt/logs/"

# Qwen Model Args
use_flash_attention_2="True"
fps=1
min_pixels=$((4 * 28 * 28)) # 4(native resolution), 256(for performance boost, 448*448)
max_pixels=$((16384 * 28 * 28)) # 16384(native resolution) 1280(for performance boost)
total_pixels=$((128000 * 28 * 28)) # 128000 the default value, 24576 the video eval setting in Qwen2.5-VL paper
max_num_frames=768

# Parse keyword arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --eval_tasks) eval_tasks=("$2"); shift ;;
        --ckpt) ckpt="$2"; shift ;;
        --GPUS) GPUS="$2"; shift ;;
        --master_port) master_port="$2"; shift ;;
        --run_name) run_name="$2"; shift ;;
        --project_name) project_name="$2"; shift ;;
        --accelerate_config) accelerate_config="$2"; shift ;;
        --output_path) output_path="$2"; shift ;;
        --use_flash_attention_2) use_flash_attention_2="$2"; shift ;;
        --fps) fps="$2"; shift ;;
        --min_pixels) min_pixels="$2"; shift ;;
        --max_pixels) max_pixels="$2"; shift ;;
        --total_pixels) total_pixels="$2"; shift ;;
        --max_num_frames) max_num_frames="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Check for commas in strings that will be used in --model_args, which will raise error
if [[ "$ckpt" == *","* ]]; then
    echo "Error: pretrained=($ckpt) contains a comma."
    exit 1
fi

echo "Running script with:"
echo "Checkpoint: $ckpt"
echo "Evaluation Tasks: $eval_tasks"
echo "GPUs: $GPUS"
echo "Master Port: $master_port"
echo "Project Name: $project_name"
echo "Run Name: $run_name"
echo "Output Path: $output_path"
echo "Use Flash Attention 2: $use_flash_attention_2"
echo "FPS: $fps"
echo "Min Pixels: $min_pixels"
echo "Max Pixels: $max_pixels"
echo "Total Pixels: $total_pixels"
echo "Max Num Frames: $max_num_frames"

# Check if the acclerate config file does not exist
if [ ! -f "$accelerate_config" ]; then
    echo "Accelerate config file does not exist. Will use \$HF_HOME/accelerate/default_config.yaml if it exists.
    You can modify default_config.yaml or create your own config with 'accelerate config --config_file $accelerate_config' "

    python3 -m accelerate.commands.launch  --num_processes=$GPUS --main_process_port=${master_port} \
        -m lmms_eval \
        --model qwen2_5_vl   \
        --model_args="pretrained=$ckpt,use_flash_attention_2=$use_flash_attention_2,fps=$fps,min_pixels=$min_pixels,max_pixels=$max_pixels,total_pixels=$total_pixels,max_num_frames=$max_num_frames" \
        --tasks=$eval_tasks  \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix qwen \
        --output_path="$output_path" #\
        #--wandb_args="project=$project_name,job_type=eval,name=$run_name"
else
    echo "Will use $accelerate_config as accelerate config."

    python3 -m accelerate.commands.launch --config_file $accelerate_config --num_processes=$GPUS --main_process_port=${master_port} \
        -m lmms_eval \
        --model qwen2_5_vl   \
        --model_args="pretrained=$ckpt,use_flash_attention_2=$use_flash_attention_2,fps=$fps,min_pixels=$min_pixels,max_pixels=$max_pixels,total_pixels=$total_pixels,max_num_frames=$max_num_frames" \
        --tasks=$eval_tasks  \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix qwen \
        --output_path="$output_path" #\
        #--wandb_args="project=$project_name,job_type=eval,name=$run_name"
fi
