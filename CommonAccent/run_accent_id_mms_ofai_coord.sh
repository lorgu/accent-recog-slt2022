#!/bin/bash
set -euo pipefail

cmd=none
config_file="/home/projects/vokquant/config_telbot.sh"  #TELEGRAM_BOT_TOKEN="..." TELEGRAM_CHAT_ID="..."\n
if [ -f $config_file ]; then
    source /home/projects/vokquant/config_telbot.sh
    send_telegram_message() {
        local message=$1
        curl -s -X POST "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/sendMessage" \
            -d chat_id="$TELEGRAM_CHAT_ID" \
            -d text="$message" > /dev/null
    }

    #send start notification
    send_telegram_message "Training started"
    # Trap errors and send a notification
    trap 'send_telegram_message "failed"; exit 1' ERR
fi

# model from HF hub, it could be another one, e.g., facebook/wav2vec2-base
wav2vec2_hub="facebook/mms-lid-256"
seed="101"
apply_augmentation="True"
max_batch_len=10

# data folder:
csv_prepared_folder="/home/projects/vokquant/accent-recog-slt2022/CommonAccent/data/at_augmented_ofai_fold_cv_regions/at_augmented_ofai_fold_cv_regions_1"
output_dir="/home/projects/vokquant/accent-recog-slt2022/CommonAccent/results/W2V2/OFAI"

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

$cmd python3 accent_id/train_w2v2_coord_mms.py accent_id/hparams/mms_cv_fold/train_mms_ofai_coord_fold.yaml \
    --seed=$seed \
    --skip_prep="True" \
    --rir_folder="$rir_folder" \
    --csv_prepared_folder=$csv_prepared_folder \
    --apply_augmentation="$apply_augmentation" \
    --max_batch_len="$max_batch_len" \
    --output_folder="$output_folder" \
    --wav2vec2_hub="$wav2vec2_hub" 

if [ -f $config_file ]; then
    #send end notification
    send_telegram_message "Training completed"
fi
echo "Done training of model $wav2vec2_hub in $output_folder"
exit 0
