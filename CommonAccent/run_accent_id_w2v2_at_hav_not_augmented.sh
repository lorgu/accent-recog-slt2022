#!/bin/bash
set -euo pipefail

cmd=none

# training vars

# model from HF hub, it could be another one, e.g., facebook/wav2vec2-base
wav2vec2_hub="facebook/wav2vec2-large-xlsr-53"
seed="1002"
apply_augmentation="True"
max_batch_len=10

# data folder:
csv_prepared_folder="/home/projects/vokquant/accent-recog-slt2022/CommonAccent/data/at_not_augmented_regions"
output_dir="/home/projects/vokquant/accent-recog-slt2022/CommonAccent/results/W2V2/AT"

# If augmentation is defined:
if [ "$apply_augmentation" == 'True' ]; then
    output_folder="$output_dir/$(basename $wav2vec2_hub)-augmented/$seed"
    rir_folder="data/rir_folder/"
else
    output_folder="$output_dir/$(basename $wav2vec2_hub)/$seed"
    rir_folder=""
fi

# configure a GPU to use if we a defined 'CMD'
if [ ! "$cmd" == 'none' ]; then
  basename=train_$(basename $wav2vec2_hub)_${apply_augmentation}_augmentation
  cmd="$cmd -N ${basename} ${output_folder}/log/train_log"
else
  cmd=''
fi

echo "*** About to start the training ***"
echo "*** output folder: $output_folder ***"

$cmd python3 accent_id/train_w2v2_hav.py accent_id/hparams/train_w2v2_xlsr_at_hav_not_augmented.yaml \
    --seed=$seed \
    --skip_prep="True" \
    --rir_folder="$rir_folder" \
    --csv_prepared_folder=$csv_prepared_folder \
    --apply_augmentation="$apply_augmentation" \
    --max_batch_len="$max_batch_len" \
    --output_folder="$output_folder" \
    --wav2vec2_hub="$wav2vec2_hub" 

echo "Done training of model $wav2vec2_hub in $output_folder"
exit 0
