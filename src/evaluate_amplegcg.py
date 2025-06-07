import argparse, os, torch, numpy as np, json

import utils, adv_trainers


def get_args():
    parser = argparse.ArgumentParser()

    utils.add_shared_args(parser)

    parser.add_argument("--exp-type", type=str, default=None,
                        choices=["build-pre-evalset", "build-evalset"])
    parser.add_argument("--model-resume-path", type=str, default=None)
    parser.add_argument("--pre-evalset-path", type=str, default=None)
    parser.add_argument("--evalset-cfg-path", type=str, default=None)
    parser.add_argument("--amplegcg-sfx-len", type=int, default=0)

    return parser.parse_args()


def build_pre_evalset(args, logger):
    """ a part of the code is from
        https://huggingface.co/osunlp/AmpleGCG-plus-llama2-sourced-vicuna-7b13b-guanaco-7b13b
    """

    from transformers import AutoModelForCausalLM,AutoTokenizer,GenerationConfig

    prompt = "### Query:{qqq} ### Prompt:"
    device = "cuda:0"

    model_name = "osunlp/AmpleGCG-plus-llama2-sourced-vicuna-7b13b-guanaco-7b13b"

    # model_name = "osunlp/AmpleGCG-llama2-sourced-llama2-7b-chat"

    mmm = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.bfloat16, device_map="auto")
    ttt = AutoTokenizer.from_pretrained(model_name)
    ttt.padding_side = "left"
    if not ttt.pad_token:
        ttt.pad_token = ttt.eos_token
    gen_kwargs = {"pad_token_id":ttt.pad_token_id, "eos_token_id":ttt.eos_token_id, "bos_token_id":ttt.bos_token_id}

    num_beams = 50
    gen_config = {
        "do_sample": False,
        "max_new_tokens": args.amplegcg_sfx_len,
        "min_new_tokens": args.amplegcg_sfx_len,
        "diversity_penalty": 1.0,
        "num_beams": num_beams,
        "num_beam_groups": num_beams,
        "num_return_sequences": num_beams,
    }

    gen_config = GenerationConfig(**gen_kwargs,**gen_config)

    dataset = utils.get_dataset(args.dataset)
    pre_evalset = []
    for ii in range(len(dataset)):
        data = dataset[ii]
        qqq = data["prompt"]
        input_ids = ttt(prompt.format(qqq = qqq),return_tensors='pt',padding= True).to(device)
        output = mmm.generate(**input_ids,generation_config = gen_config)
        output = output[:,input_ids["input_ids"].shape[-1]:]
        adv_suffixes = ttt.batch_decode(output,skip_special_tokens= True)
        adv_sfx_id = np.random.randint(len(adv_suffixes))
        pre_evalset.append({
            "prompt": data["prompt"],
            "target": data["target"],
            "adv_input": data["prompt"] + " " + adv_suffixes[adv_sfx_id],
        })
        logger.info(f"[{ii+1}/{len(dataset)}] generated")

    json_path = os.path.join(f"{args.save_dir}/{args.save_name}.json")
    logger.info(f"saving pre_evalset to `{json_path}`")
    with open(json_path, "w") as f:
        json.dump(pre_evalset, f, indent=2)


def build_evalset(args, logger):
    default_device = "cuda:0"
    device = default_device

    lora_cfg = None
    if args.lora_cfg_path is not None:
        lora_cfg = utils.load_yaml(args.lora_cfg_path, f"{args.save_dir}/{args.save_name}_lora-cfg.yaml")
    model, tokenizer = utils.get_model(args.model_id, default_device, lora_cfg)
    if args.model_resume_path is not None:
        logger.info(f"resume model from {args.model_resume_path}")
        state_dict = utils.load_state_dict(args.model_resume_path, mode="safetensors")
        model.load_state_dict(state_dict)
        del state_dict

    with open(args.pre_evalset_path) as f:
        dataset = json.load(f)

    evalset_cfg = utils.load_yaml(args.evalset_cfg_path)

    # ======== start to override evalset_cfg

    if "datacollator_config" not in evalset_cfg.keys():
        evalset_cfg["datacollator_config"] = {}
    evalset_cfg["datacollator_config"]["output_raw"] = True

    if args.datacollator is not None:
        evalset_cfg["datacollator_config"]["name"] = args.datacollator

    # ======== finished overriding evalset_cfg

    utils.save_yaml(evalset_cfg, f"{args.save_dir}/{args.save_name}_evalset-cfg.yaml")

    evalset_cfg = adv_trainers.EvalsetConfig(**evalset_cfg)

    datacollator = adv_trainers.data.build_datacollator(
        evalset_cfg.datacollator_config,
        tokenizer=tokenizer,
        is_adv=True,
        # is_adv=False,
    )

    bs = evalset_cfg.batch_size
    evalset = []
    for be in range(0, len(dataset), bs):
        ed = min(be+bs, len(dataset))
        batch = [dataset[i] for i in range(be,ed)]
        inp_batch = [{"prompt": sample["adv_input"], "target": ""} for sample in batch]

        # manually implement left-padding
        inputs = datacollator(inp_batch)
        inputs_ids = torch.cat([inputs["prompt_ids"], inputs["harmful_ids"]], dim=1)
        inputs_mask = torch.cat([inputs["prompt_mask"], torch.ones_like(inputs["harmful_ids"])], dim=1)
        # inputs_ids = inputs["input_ids"]
        # inputs_mask = inputs["input_mask"]

        inputs_ids = inputs_ids.to(device)
        inputs_mask = inputs_mask.to(device)

        outputs_text_list = []
        for k in range(evalset_cfg.per_response_num):
            outputs_ids = model.generate(
                inputs=inputs_ids,
                attention_mask=inputs_mask,
                do_sample=True,
                max_new_tokens=evalset_cfg.max_new_tokens,
            ) # shape: [B, L + max_new_tokens]
            outputs_text = tokenizer.batch_decode(outputs_ids, skip_special_tokens=True)
            outputs_text_list.append(outputs_text)

        inputs_text = tokenizer.batch_decode(inputs_ids, skip_special_tokens=True)

        B = len(inputs_text)
        for b in range(B):
            for k in range(evalset_cfg.per_response_num):
                evalset.append({
                    "prompt": batch[b]["prompt"],
                    "target": batch[b]["target"],
                    "adv_input": inputs_text[b],
                    "generation": outputs_text_list[k][b][len(inputs_text[b]) : ],
                })

    json_path = os.path.join(f"{args.save_dir}/{args.save_name}_evalset.json")
    logger.info(f"saving adv-evalset to `{json_path}`")
    with open(json_path, "w") as f:
        json.dump(evalset, f, indent=2)


def main(args, logger):
    if args.exp_type == "build-pre-evalset":
        build_pre_evalset(args, logger)
    elif args.exp_type == "build-evalset":
        build_evalset(args, logger)
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
