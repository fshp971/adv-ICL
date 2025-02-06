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
fi


cd ${src_path}

model_id="meta-llama/Meta-Llama-3-8B-Instruct"
datacollator="llama3-chat"

if [[ -z $i ]]; then

    save_path="../results/llama3/vanilla/eval-alpacaeval"

    python evaluate.py \
        --model-id ${model_id} \
        --datacollator ${datacollator} \
        --alpacaeval-cfg-path ../configs/eval/alpacaeval.yaml \
        --save-dir ${save_path} \
        --exp-type build-alpacaeval \
        --save-name alpacaeval

else

    train_path="../results/llama3/train-$i"
    save_path="${train_path}/eval-alpacaeval"

    python evaluate.py \
        --model-id ${model_id} \
        --lora-cfg-path ${train_path}/train_lora-cfg.yaml \
        --model-resume-path ${train_path}/train_fin-model/adapter_model.safetensors \
        --datacollator ${datacollator} \
        --alpacaeval-cfg-path ../configs/eval/alpacaeval.yaml \
        --save-dir ${save_path} \
        --exp-type build-alpacaeval \
        --save-name alpacaeval

fi
