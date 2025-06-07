while [[ $# -gt 0 ]]; do
    if [[ $1 == "--src-path" ]]; then
        src_path=$2
        shift 2
    elif [[ $1 == "--eval-cfg" ]]; then
        k=$2
        shift 2
    elif [[ $1 == "--dataset" ]]; then
        dataset=$2
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
fi

if [[ $dataset == "harmbench-test50" ]]; then
    ds_name="harmbench"
elif [[ $dataset == "advbench-first50" ]]; then
    ds_name="advbench"
fi

cd ${src_path}

lengths=("5" "10" "20" "40" "60" "80" "100" "120")
advsfx_len=${lengths[$((k-1))]}

python evaluate_amplegcg.py \
    --dataset ${dataset} \
    --exp-type build-pre-evalset \
    --amplegcg-sfx-len ${advsfx_len} \
    --save-dir ../results/amplegcg \
    --save-name pre-evalset-${ds_name}-${k}
