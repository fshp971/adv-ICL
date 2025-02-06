while [[ $# -gt 0 ]]; do
    if [[ $1 == "--src-path" ]]; then
        src_path=$2
        shift 2
    elif [[ $1 == "--train-cfg" ]]; then
        i=$2
        shift 2
    else
        shift 1
    fi
done

if [[ -z ${src_path} ]]; then
    echo "Error: --src-path is required"
    exit 1
elif [[ -z $i ]]; then
    echo "Error: --train-cfg is required"
    exit 1
fi

cd ${src_path}

train_path="../results/llama2/train-$i"
model_id="meta-llama/Llama-2-7b-chat-hf"
datacollator="llama2-chat"

python train.py \
    --model-id ${model_id} \
    --lora-cfg-path ../configs/train/lora.yaml \
    --datacollator ${datacollator} \
    --dataset harmbench \
    --utilityset alpaca \
    --trainer-cfg-path ../configs/train/trainer.yaml \
    --atker-cfg-path ../configs/train/atker/cfg-$i.yaml \
    --save-dir ${train_path} \
    --save-name train
