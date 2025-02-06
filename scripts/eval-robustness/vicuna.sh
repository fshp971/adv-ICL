while [[ $# -gt 0 ]]; do
    if [[ $1 == "--src-path" ]]; then
        src_path=$2
        shift 2
    elif [[ $1 == "--train-cfg" ]]; then
        i=$2
        shift 2
    elif [[ $1 == "--eval-cfg" ]]; then
        k=$2
        shift 2
    elif [[ $1 == "--dataset" ]]; then
        dataset=$2
        shift 2
    elif [[ $1 == "--attack" ]]; then
        attack=$2
        shift 2
    else
        shift 1
    fi
done

if [[ -z ${src_path} ]]; then
    echo "Error: --src-path is required"
    exit 1
elif [[ -z $k ]]; then
    echo "Error: --eval-cfg is required"
    exit 1
elif [[ -z $dataset ]] || [[ $dataset != "harmbench-test50" ]] && [[ $dataset != "advbench-first50" ]]; then
    echo "Error: --dataset is required and should be either 'harmbench-test50' or 'advbench-first50'"
    exit 1
elif [[ -z $attack ]] || [[ $attack != "gcg" ]] && [[ $attack != "beast" ]]; then
    echo "Error: --attack is required and should be either 'gcg' or 'beast'"
    exit 1
fi

if [[ $dataset == "harmbench-test50" ]]; then
    ds_name="harmbench"
elif [[ $dataset == "advbench-first50" ]]; then
    ds_name="advbench"
fi


cd ${src_path}

model_id="lmsys/vicuna-7b-v1.5"
datacollator="vicuna-chat"

if [[ -z $i ]]; then

    save_path="../results/vicuna/vanilla/eval-${attack}/$k"

    python evaluate.py \
        --model-id ${model_id} \
        --dataset ${dataset} \
        --datacollator ${datacollator} \
        --evalset-cfg-path ../configs/eval/evalset.yaml \
        --atker-cfg-path ../configs/eval/${attack}/cfg-$k.yaml \
        --save-dir ${save_path} \
        --exp-type build-evalset \
        --save-name build-${ds_name}

else

    train_path="../results/vicuna/train-$i"
    save_path="${train_path}/eval-${attack}/$k"

    python evaluate.py \
        --model-id ${model_id} \
        --lora-cfg-path ${train_path}/train_lora-cfg.yaml \
        --model-resume-path ${train_path}/train_fin-model/adapter_model.safetensors \
        --dataset ${dataset} \
        --datacollator ${datacollator} \
        --evalset-cfg-path ../configs/eval/evalset.yaml \
        --atker-cfg-path ../configs/eval/${attack}/cfg-$k.yaml \
        --save-dir ${save_path} \
        --exp-type build-evalset \
        --save-name build-${ds_name}

fi

python evaluate.py \
    --judger-cfg-path ../configs/eval/judge.yaml \
    --evalset-path ${save_path}/build-${ds_name}_evalset.json \
    --save-dir ${save_path} \
    --exp-type judge-evalset \
    --save-name judge-${ds_name}
