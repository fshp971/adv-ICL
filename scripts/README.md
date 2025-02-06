# Tutorial on Running Scripts

[Scripts](./) can be divided into three categories:

- [scripts/train](./train) stores all scripts for adversarial training.
- [scripts/eval-robustness](./eval-robustness) stores all scripts for robust evaluation.
- [scripts/eval-utility](./eval-utility) stores all scripts for utility evaluation.

Here we present a brief tutorial on how to run experiments on Vicuna-7B-v1.5 model with these scripts.

**Step 0:** Enter the script folder:

```bash
cd ./scripts
```

**Step 1:** Perform adversarial training:

```bash
bash train/vicuna.sh --src-path ../src --train-cfg 1
```

where `--train-cfg` should be set within `{1,2,3,4,5,6}`.

**Step 2:** Generate harmful responses via jailbreak attacks and calculate attack success rate (ASR) based on induced harmful responses with the [LLM judger from Harmbench](https://huggingface.co/cais/HarmBench-Llama-2-13b-cls).

```bash
# for vanilla pre-trained model
bash eval-robustness/vicuna.sh --src-path ../src --eval-cfg 1 --dataset harmbench-test50 --attack gcg

# for adversarially trained model
bash eval-robustness/vicuna.sh --src-path ../src --train-cfg 1 --eval-cfg 1 --dataset harmbench-test50 --attack gcg
```

where `--train-cfg` should be empty or be set within `{1,2,3,4,5,6}`, `--eval-cfg` should be set within `{1,2,3,4,5,6,7,8}`, `--dataset` should be set within `{"harmbench-test50", "advbench-first50"}`, and `--attack` should be set within `{"gcg", "beast"}`.

**Step 3:** Synthesize benign responses for [AlpacaEval2](https://github.com/tatsu-lab/alpaca_eval) utility evaluation:

```bash
# for vanilla pre-trained model
# Generated responses for the utility evaluation will be saved in:
# "./results/vicuna/vanilla/eval-alpacaeval/alpacaeval_alpacaeval.json"
bash eval-utility/vicuna.sh --src-path ../src

# for vanilla pre-trained model
# Generated responses for the utility evaluation will be saved in:
# "./results/vicuna/train-1/eval-alpacaeval/alpacaeval_alpacaeval.json"
bash eval-utility/vicuna.sh --src-path ../src --train-cfg 1
```

where `--train-cfg` should be empty or be set within `{1,2,3,4,5,6}`.

The utility evaluation is then performed based on the generated JSON file. See the [official tutorial](https://github.com/tatsu-lab/alpaca_eval?tab=readme-ov-file#evaluating-a-model) of AlpacaEval2 for detailed instructions.