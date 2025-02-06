import torch, transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig
from typing import Union, Tuple


def named_peft_parameters(model, peft_name: str = None):
    if peft_name is None:
        return
    for name, pp in model.named_parameters():
        spt = name.rsplit(".", 2)
        if len(spt) == 3 and spt[1] == peft_name:
            yield name, pp


def peft_parameters(model, peft_name: str):
    for _, pp in named_peft_parameters(model, peft_name):
        yield pp


def peft_state_dict(model, peft_name: str):
    state_dict = dict()
    for name, pp in named_peft_parameters(model, peft_name):
        state_dict[name] = pp
    return state_dict


def load_peft_state_dict(model, peft_name: str, state_dict: dict):
    for name, pp in model.named_parameters():
        spt = name.rsplit(".", 2)
        if len(spt) == 3 and spt[1] == peft_name:
            name_new = f"{spt[0]}.{spt[2]}"

            pp_new = state_dict.get(name, None)
            if pp_new is None:
                pp_new = state_dict.get(name_new, None)

            if pp_new is not None:
                pp.data.copy_(pp_new.data)


class PEFTWrapper:
    def __init__(self, model, peft_name: str = None):
        self.model = model
        self.peft_name = peft_name

    @property
    def vocab_size(self):
        return self.model.vocab_size

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def save_pretrained(self, *args, **kwargs):
        self.model.save_pretrained(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def to(self, *args, **kwargs):
        self.model.to(*args, **kwargs)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def named_parameters(self):
        return named_peft_parameters(self.model, self.peft_name)

    def parameters(self):
        return peft_parameters(self.model, self.peft_name)

    def state_dict(self) -> dict:
        return peft_state_dict(self.model, self.peft_name)

    def load_state_dict(self, state_dict):
        load_peft_state_dict(self.model, self.peft_name, state_dict)


def get_model(
        model_id: str,
        device: Union[str, torch.device] = "cpu",
        lora_cfg: dict = None
    ) -> Tuple[Union[PEFTWrapper, transformers.PreTrainedModel], transformers.PreTrainedTokenizerBase]:

    # model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map=device)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if lora_cfg is not None:
        adapter_name = lora_cfg.pop("adapter_name", "default")
        lora_config = LoraConfig(**lora_cfg)
        model = get_peft_model(model, lora_config, adapter_name=adapter_name)
        model = PEFTWrapper(model, adapter_name)

    return model, tokenizer
