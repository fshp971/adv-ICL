import argparse, os, torch, numpy as np

import utils, attacks, adv_trainers


def get_args():
    parser = argparse.ArgumentParser()

    utils.add_shared_args(parser)
    utils.add_train_args(parser)

    return parser.parse_args()


def main(args, logger):
    default_device = "cuda:0"

    lora_cfg = utils.load_yaml(args.lora_cfg_path, f"{args.save_dir}/{args.save_name}_lora-cfg.yaml")
    model, tokenizer = utils.get_model(args.model_id, default_device, lora_cfg)
    trainset = utils.get_dataset(args.dataset)

    utilityset = None
    if args.utilityset is not None:
        utilityset = utils.get_dataset(args.utilityset)

    trainer_cfg = utils.load_yaml(args.trainer_cfg_path)

    # ======== start to override trainer_cfg

    if "datacollator_config" not in trainer_cfg.keys():
        trainer_cfg["datacollator_config"] = {}
    if args.datacollator is not None:
        trainer_cfg["datacollator_config"]["name"] = args.datacollator

    atker_cfg = utils.load_yaml(args.atker_cfg_path)
    if atker_cfg is not None:
        trainer_cfg["attacker_config"] = atker_cfg

    # ======== finished overriding trainer_cfg

    utils.save_yaml(trainer_cfg, f"{args.save_dir}/{args.save_name}_trainer-cfg.yaml")

    trainer_cfg = adv_trainers.AdvTrainerConfig(**trainer_cfg)

    trainer = adv_trainers.build_trainer(
        trainer_config=trainer_cfg,
        model=model,
        tokenizer=tokenizer,
        device=default_device,
        trainset=trainset,
        utilityset=utilityset,
        logger=logger,
    )

    # try to resume training
    model_state_dict, trainer_state_dict = None, None
    if args.model_resume_path is not None:
        logger.info(f"resume model from {args.model_resume_path}")
        model_state_dict = utils.load_state_dict(args.model_resume_path, mode="safetensors")
    if args.trainer_resume_path is not None:
        logger.info(f"resume trainer from {args.trainer_resume_path}")
        trainer_state_dict = utils.load_state_dict(args.trainer_resume_path, mode="pytorch")
    trainer.load_state_dict(model_state_dict, trainer_state_dict)
    del model_state_dict, trainer_state_dict

    trainer.train()

    model_state_dict, trainer_state_dict = trainer.state_dict()
    # utils.save_state_dict(model_state_dict, path=f"{args.save_dir}/{args.save_name}_fin-model.safetensors", mode="safetensors")

    model.save_pretrained(f"{args.save_dir}/{args.save_name}_fin-model")
    utils.save_state_dict(trainer_state_dict, path=f"{args.save_dir}/{args.save_name}_fin-trainer.pkl", mode="pytorch")


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
