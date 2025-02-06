from dataclasses import dataclass
from typing import Union, Sequence
import os, copy, torch, transformers, datasets

import adv_trainers


@dataclass
class DatacollatorConfig:
    name: str
    benign_text: Union[str, Sequence[str]] = None
    # pad_id: int = 0
    output_raw: bool = False


class DatasetWrapper:
    def __init__(
        self,
        dataset: datasets.Dataset,
        prompt_name: str,
        target_name: str,
    ):
        self.dataset = dataset
        self.prompt_name = prompt_name
        self.target_name = target_name

    def __getitem__(self, idx: int):
        res = self.dataset[idx]
        return {
            "prompt": res[self.prompt_name],
            "target": res[self.target_name],
        }

    def __len__(self):
        return len(self.dataset)


class ChatDatacollator:
    def __init__(
        self,
        prompt_prefix: str,
        target_prefix: str,
        tokenizer: transformers.PreTrainedTokenizerBase,
    ):
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.encode(tokenizer.eos_token, add_special_tokens=False)[0]

        self.prompt_prefix = prompt_prefix
        self.target_prefix = target_prefix
        self.prompt_prefix_ids = self.tokenizer.encode(
            self.prompt_prefix, add_special_tokens=True, return_tensors="pt")[0]
        self.target_prefix_ids = self.tokenizer.encode(
            self.target_prefix, add_special_tokens=False, return_tensors="pt")[0]

    def __call__(self, features):
        batch = {}

        prompt_ids_list = [
            torch.cat([
                self.prompt_prefix_ids,
                self.tokenizer.encode(feat["prompt"], add_special_tokens=False, return_tensors="pt")[0],
                self.target_prefix_ids,
            ]) for feat in features
        ]

        target_ids_list = [
            self.tokenizer.encode(feat["target"] + self.tokenizer.eos_token, add_special_tokens=False, return_tensors="pt")[0]
            for feat in features
        ]

        input_len = max(
            (len(p_ids) + len(t_ids)) for p_ids, t_ids in zip(prompt_ids_list, target_ids_list)
        )
        input_ids = torch.full((len(features), input_len), fill_value=self.pad_id, dtype=torch.int64)
        input_mask = torch.zeros_like(input_ids)

        for i, (p_ids, t_ids) in enumerate(zip(prompt_ids_list, target_ids_list)):
            inp_ids = torch.cat([p_ids, t_ids])
            input_ids[i, : len(inp_ids)] = inp_ids
            input_mask[i, len(p_ids) : len(inp_ids)] = 1

        return {
            "input_ids" : input_ids,
            "input_mask": input_mask,
        }


class AdvDatacollator:
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizerBase,
        benign_text: Union[str, Sequence[str]] = None,
        output_raw: bool = False,
    ):
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.encode(tokenizer.eos_token, add_special_tokens=False)[0]
        self.output_raw = output_raw

        if benign_text is None:
            self.benign_text = None
            self.benign_ids = None
        else:
            if isinstance(benign_text, str):
                benign_text = (benign_text,)
            self.benign_text = [txt.lstrip().rstrip() for txt in benign_text]
            self.benign_ids = [
                tokenizer.encode(txt, add_special_tokens=False, return_tensors="pt")[0] for txt in self.benign_text
            ]

    def _get_indices_and_msks(self, features, add_begin_token: bool = True):
        indices = {
            "prompt": [],
            "harmful": [],
            "benign": None if (self.benign_text is None) else [],
            "raw_prompt": [] if self.output_raw else None,
            "raw_harmful": [] if self.output_raw else None,
        }
        msks = copy.deepcopy(indices)

        for i, feat in enumerate(features):
            prompt = feat["prompt"].lstrip().rstrip()
            prompt = self.tokenizer.encode(prompt, add_special_tokens=add_begin_token, return_tensors="pt")[0]
            indices["prompt"].append(prompt)
            msks["prompt"].append(torch.ones(len(prompt), dtype=torch.int64))

            harmful = feat["target"].lstrip().rstrip()
            harmful = self.tokenizer.encode(harmful, add_special_tokens=False, return_tensors="pt")[0]
            indices["harmful"].append(harmful)
            msks["harmful"].append(torch.ones(len(harmful), dtype=torch.int64))

            if self.output_raw:
                indices["raw_prompt"].append(indices["prompt"][-1])
                msks["raw_prompt"].append(msks["prompt"][-1])

                indices["raw_harmful"].append(indices["harmful"][-1])
                msks["raw_harmful"].append(msks["harmful"][-1])

        if self.benign_ids is not None:
            bidx = torch.randint(0, len(self.benign_ids), (len(features),))
            for i in range(len(features)):
                benign = self.benign_ids[bidx[i].item()]
                indices["benign"].append(benign)
                msks["benign"].append(torch.ones(len(benign), dtype=torch.int64))

        return indices, msks

    def __call__(self, features):
        indices, msks = self._get_indices_and_msks(features)
        bs = len(indices["prompt"])
        batch = {}

        prompt_len = max(len(ids) for ids in indices["prompt"])
        prompt_ids = torch.full((bs, prompt_len), fill_value=self.pad_id, dtype=torch.int64)
        prompt_mask = torch.zeros_like(prompt_ids)

        for i, (ids, mask) in enumerate(zip(indices["prompt"], msks["prompt"])):
            prompt_ids[i, -len(ids) :] = ids
            prompt_mask[i, -len(ids) :] = mask

        batch["prompt_ids"] = prompt_ids
        batch["prompt_mask"] = prompt_mask

        harmful_len = max(len(ids) for ids in indices["harmful"])
        harmful_ids = torch.full((bs, harmful_len), fill_value=self.pad_id, dtype=torch.int64)
        harmful_mask = torch.zeros_like(harmful_ids)

        for i, (ids, mask) in enumerate(zip(indices["harmful"], msks["harmful"])):
            harmful_ids[i, : len(ids)] = ids
            harmful_mask[i, : len(ids)] = mask

        batch["harmful_ids"] = harmful_ids
        batch["harmful_mask"] = harmful_mask

        if self.benign_text is not None:
            benign_len = max(len(ids) for ids in indices["benign"])
            benign_ids = torch.full((bs, benign_len), fill_value=self.pad_id, dtype=torch.int64)
            benign_mask = torch.zeros_like(benign_ids)

            for i, (ids, mask) in enumerate(zip(indices["benign"], msks["benign"])):
                benign_ids[i, : len(ids)] = ids
                benign_mask[i, : len(ids)] = mask

            batch["benign_ids"]  = benign_ids
            batch["benign_mask"] = benign_mask

        if self.output_raw:
            raw_prompt_len = max(len(ids) for ids in indices["raw_prompt"])
            raw_prompt_ids = torch.full((bs, raw_prompt_len), fill_value=self.pad_id, dtype=torch.int64)
            raw_prompt_mask = torch.zeros_like(raw_prompt_ids)

            for i, (ids, mask) in enumerate(zip(indices["raw_prompt"], msks["raw_prompt"])):
                raw_prompt_ids[i, -len(ids) :] = ids
                raw_prompt_mask[i, -len(ids) :] = mask

            batch["raw_prompt_ids"] = raw_prompt_ids
            batch["raw_prompt_mask"] = raw_prompt_mask

            raw_harmful_len = max(len(ids) for ids in indices["raw_harmful"])
            raw_harmful_ids = torch.full((bs, raw_harmful_len), fill_value=self.pad_id, dtype=torch.int64)
            raw_harmful_mask = torch.zeros_like(raw_harmful_ids)

            for i, (ids, mask) in enumerate(zip(indices["raw_harmful"], msks["raw_harmful"])):
                raw_harmful_ids[i, : len(ids)] = ids
                raw_harmful_mask[i, : len(ids)] = mask

            batch["raw_harmful_ids"] = raw_harmful_ids
            batch["raw_harmful_mask"] = raw_harmful_mask

        return batch


class ChatAdvDatacollator(AdvDatacollator):
    def __init__(
        self,
        prompt_prefix: str,
        target_prefix: str,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.prompt_prefix = prompt_prefix
        self.target_prefix = target_prefix

        self.prompt_prefix_ids = self.tokenizer.encode(
            self.prompt_prefix, add_special_tokens=True, return_tensors="pt")[0]
        self.target_prefix_ids = self.tokenizer.encode(
            self.target_prefix, add_special_tokens=False, return_tensors="pt")[0]

        self.prompt_prefix_mask = torch.ones((len(self.prompt_prefix_ids),), dtype=torch.int64)
        self.target_prefix_mask = torch.zeros((len(self.target_prefix_ids),), dtype=torch.int64)

    def _get_indices_and_msks(self, features):
        indices, msks = super()._get_indices_and_msks(features, add_begin_token=False)

        indices["prompt"] = [torch.cat([self.prompt_prefix_ids, ids]) for ids in indices["prompt"]]
        msks["prompt"] = [torch.cat([self.prompt_prefix_mask, mask]) for mask in msks["prompt"]]

        indices["harmful"] = [torch.cat([self.target_prefix_ids, ids]) for ids in indices["harmful"]]
        msks["harmful"] = [torch.cat([self.target_prefix_mask, mask]) for mask in msks["harmful"]]

        if self.benign_text is not None:
            indices["benign"] = [torch.cat([self.target_prefix_ids, ids]) for ids in indices["benign"]]
            msks["benign"] = [torch.cat([self.target_prefix_mask, mask]) for mask in msks["benign"]]

        return indices, msks


class VicunaDatacollator(ChatDatacollator):
    def __init__(self, *args, **kwargs):
        super().__init__(
            prompt_prefix="USER:",
            target_prefix="\nASSISTANT:",
            *args, **kwargs,
        )

class Llama2Datacollator(ChatDatacollator):
    def __init__(self, *args, **kwargs):
        super().__init__(
            prompt_prefix="[INST]",
            target_prefix="[/INST]",
            *args, **kwargs,
        )

class Llama3Datacollator(ChatDatacollator):
    def __init__(self, *args, **kwargs):
        super().__init__(
            prompt_prefix="<|start_header_id|>user<|end_header_id|>\n\n",
            target_prefix="<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n",
            *args, **kwargs,
        )

class Qwen2Datacollator(ChatDatacollator):
    def __init__(self, *args, **kwargs):
        sys_prompt = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
        super().__init__(
            prompt_prefix=(sys_prompt + "<|im_start|>user\n"),
            target_prefix="<|im_end|>\n<|im_start|>assistant\n",
            *args, **kwargs,
        )


class VicunaAdvDatacollator(ChatAdvDatacollator):
    def __init__(self, *args, **kwargs):
        super().__init__(
            prompt_prefix="USER:",
            target_prefix="\nASSISTANT:",
            *args, **kwargs,
        )

class Llama2AdvDatacollator(ChatAdvDatacollator):
    def __init__(self, *args, **kwargs):
        super().__init__(
            prompt_prefix="[INST]",
            target_prefix="[/INST]",
            *args, **kwargs,
        )

class Llama3AdvDatacollator(ChatAdvDatacollator):
    def __init__(self, *args, **kwargs):
        super().__init__(
            prompt_prefix="<|start_header_id|>user<|end_header_id|>\n\n",
            target_prefix="<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n",
            *args, **kwargs,
        )

class Qwen2AdvDatacollator(ChatAdvDatacollator):
    def __init__(self, *args, **kwargs):
        sys_prompt = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
        super().__init__(
            prompt_prefix=(sys_prompt + "<|im_start|>user\n"),
            target_prefix="<|im_end|>\n<|im_start|>assistant\n",
            *args, **kwargs,
        )


__datacollator_zoo__ = {
    "vicuna-chat" : VicunaDatacollator,
    "llama2-chat" : Llama2Datacollator,
    "llama3-chat" : Llama3Datacollator,
    "qwen2-chat"  : Qwen2Datacollator,
}

__adv_datacollator_zoo__ = {
    # "vanilla"     : AdvDatacollator,
    "vicuna-chat" : VicunaAdvDatacollator,
    "llama2-chat" : Llama2AdvDatacollator,
    "llama3-chat" : Llama3AdvDatacollator,
    "qwen2-chat"  : Qwen2AdvDatacollator,
}


def build_datacollator(config: DatacollatorConfig, is_adv: bool = True, **kwargs):
    if is_adv:
        return __adv_datacollator_zoo__[config.name](
            benign_text=config.benign_text,
            output_raw=config.output_raw,
            **kwargs,
        )

    else:
        return __datacollator_zoo__[config.name](
            **kwargs,
        )


class Loader:
    def __init__(self, dataset, batch_size, train=True, collate_fn=None, num_workers=0):
        self.loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size, shuffle=train, drop_last=train,
            collate_fn=collate_fn, num_workers=num_workers,
        )
        self.iterator = None

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)

    def __next__(self):
        if self.iterator is None:
            self.iterator = iter(self.loader)

        try:
            samples = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            samples = next(self.iterator)

        return samples
