import argparse, os, torch, numpy as np, json
# import pandas as pd

import utils, attacks, adv_trainers


def get_args():
    parser = argparse.ArgumentParser()

    utils.add_shared_args(parser)
    utils.add_eval_args(parser)

    return parser.parse_args()


def build_alpacaeval(args, logger):
    default_device = "cuda:0"

    lora_cfg = None
    if args.lora_cfg_path is not None:
        lora_cfg = utils.load_yaml(args.lora_cfg_path, f"{args.save_dir}/{args.save_name}_lora-cfg.yaml")
    model, tokenizer = utils.get_model(args.model_id, default_device, lora_cfg)
    if args.model_resume_path is not None:
        logger.info(f"resume model from {args.model_resume_path}")
        state_dict = utils.load_state_dict(args.model_resume_path, mode="safetensors")
        model.load_state_dict(state_dict)
        del state_dict

    alpacaeval_cfg = utils.load_yaml(args.alpacaeval_cfg_path)

    if "datacollator_config" not in alpacaeval_cfg.keys():
        alpacaeval_cfg["datacollator_config"] = {}
    alpacaeval_cfg["datacollator_config"]["output_raw"] = True

    if args.datacollator is not None:
        alpacaeval_cfg["datacollator_config"]["name"] = args.datacollator

    utils.save_yaml(alpacaeval_cfg, f"{args.save_dir}/{args.save_name}_alpacaeval-cfg.yaml")

    alpacaeval_cfg = adv_trainers.AlpacaEvalConfig(**alpacaeval_cfg)

    alpacaeval_set = adv_trainers.build_alpacaeval(
        eval_config=alpacaeval_cfg,
        model=model,
        tokenizer=tokenizer,
        device=default_device,
    )

    json_path = os.path.join(f"{args.save_dir}/{args.save_name}_alpacaeval.json")
    logger.info(f"saving alpacaeval-output to `{json_path}`")
    with open(json_path, "w") as f:
        json.dump(alpacaeval_set, f, indent=2)


def build_evalset(args, logger):
    default_device = "cuda:0"

    lora_cfg = None
    if args.lora_cfg_path is not None:
        lora_cfg = utils.load_yaml(args.lora_cfg_path, f"{args.save_dir}/{args.save_name}_lora-cfg.yaml")
    model, tokenizer = utils.get_model(args.model_id, default_device, lora_cfg)
    if args.model_resume_path is not None:
        logger.info(f"resume model from {args.model_resume_path}")
        state_dict = utils.load_state_dict(args.model_resume_path, mode="safetensors")
        model.load_state_dict(state_dict)
        del state_dict

    dataset = utils.get_dataset(args.dataset)

    evalset_cfg = utils.load_yaml(args.evalset_cfg_path)

    # ======== start to override evalset_cfg

    if "datacollator_config" not in evalset_cfg.keys():
        evalset_cfg["datacollator_config"] = {}
    evalset_cfg["datacollator_config"]["output_raw"] = True

    if args.datacollator is not None:
        evalset_cfg["datacollator_config"]["name"] = args.datacollator

    atker_cfg = utils.load_yaml(args.atker_cfg_path)
    if atker_cfg is not None:
        evalset_cfg["attacker_config"] = atker_cfg

    # ======== finished overriding evalset_cfg

    utils.save_yaml(evalset_cfg, f"{args.save_dir}/{args.save_name}_evalset-cfg.yaml")

    evalset_cfg = adv_trainers.EvalsetConfig(**evalset_cfg)

    evalset = adv_trainers.build_evalset(
        eval_config=evalset_cfg,
        model=model,
        tokenizer=tokenizer,
        device=default_device,
        dataset=dataset,
    )

    # df = pd.DataFrame(evalset)

    # csv_path = os.path.join(f"{args.save_dir}/{args.save_name}_evalset.csv")
    # logger.info(f"saving evalset to `{csv_path}`")
    # df.to_csv(csv_path, index=False)

    json_path = os.path.join(f"{args.save_dir}/{args.save_name}_evalset.json")
    logger.info(f"saving adv-evalset to `{json_path}`")
    with open(json_path, "w") as f:
        json.dump(evalset, f, indent=2)


def judge_evalset(args, logger):
    default_device = "cuda:0"

    judger_cfg = utils.load_yaml(args.judger_cfg_path, f"{args.save_dir}/{args.save_name}_judger-cfg.yaml")
    judger_cfg["device"] = default_device
    judger_cfg = adv_trainers.JudgerConfig(**judger_cfg)
    judger = adv_trainers.build_judger(judger_cfg)

    # evalset = pd.read_csv(args.evalset_path)
    with open(args.evalset_path, "r") as f:
        evalset = json.load(f)
    logger.info(f"loaded evalset from {args.evalset_path}")

    # evalset = [
    #     {"prompt": prompt, "generation": generation}
    #     for prompt, generation in zip(evalset["prompt"].tolist(), evalset["generation"].tolist())
    # ]

    attack_success_rate = judger.judge(evalset)
    logger.info("attack success rate: {:.3%}".format(attack_success_rate))


def main(args, logger):
    if args.exp_type == "build-alpacaeval":
        build_alpacaeval(args, logger)
    elif args.exp_type == "build-evalset":
        build_evalset(args, logger)
    elif args.exp_type == "judge-evalset":
        judge_evalset(args, logger)
    else:
        raise ValueError(f"unknown exp-type `{args.exp_type}`")


if __name__ == "__main__":
    args = get_args()
    logger = utils.generic_init(args)

    if args.random_seed is not None:
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)

    try:
        main(args, logger)
    except Exception as e:
        logger.exception('Unexpected exception! %s', e)
