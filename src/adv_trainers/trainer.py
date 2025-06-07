from typing import Union, Dict, Sequence, Any
from dataclasses import dataclass
import copy, logging, torch, transformers

import attacks

from .utils import (
    build_optimizer,
    build_scheduler,
)
from .data import (
    DatasetWrapper,
    DatacollatorConfig,
    Loader,
    build_datacollator,
)


@dataclass
class AdvTrainerConfig:
    name: str
    train_steps: int
    report_freq: int
    batch_size: int
    optimizer_config: Dict
    scheduler_config: Dict
    datacollator_config: DatacollatorConfig
    attacker_config: Dict = None
    utility_batch_size: int = None
    utility_split_bs: int = None
    utility_max_length: int = None
    adv_alpha: float = 1.0
    utility_alpha: float = 1.0

    def __post_init__(self):
        self.datacollator_config = DatacollatorConfig(**self.datacollator_config)


class AdvTrainerBase:
    def __init__(
        self,
        trainer_config: AdvTrainerConfig,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizerBase,
        device: Union[str, torch.device],
        trainset: DatasetWrapper,
        utilityset: DatasetWrapper = None,
        logger: logging.Logger = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.trainset = trainset
        self.logger = logger

        trainer_config = copy.deepcopy(trainer_config)

        self.train_steps = trainer_config.train_steps
        self.current_step = 0

        self.report_freq = trainer_config.report_freq

        self.optimizer_config = trainer_config.optimizer_config
        self.optimizer = build_optimizer(
            parameters=model.parameters(), **self.optimizer_config)

        self.scheduler_config = trainer_config.scheduler_config
        self.scheduler = build_scheduler(
            optimizer=self.optimizer, **self.scheduler_config)

        self.attacker_config = trainer_config.attacker_config
        self.attacker = None
        if self.attacker_config is not None:
            self.attacker = attacks.build_attacker(**self.attacker_config)

        self.trainset = trainset
        self.train_datacollator = build_datacollator(
            trainer_config.datacollator_config,
            tokenizer=self.tokenizer,
            is_adv=True,
        )
        self.trainloader = Loader(
            dataset=trainset,
            batch_size=trainer_config.batch_size,
            train=True,
            collate_fn=self.train_datacollator,
            num_workers=0,
        )

        self.utilityset = None

        if utilityset is not None:
            self.utilityset = utilityset
            self.utility_datacollator = build_datacollator(
                trainer_config.datacollator_config,
                tokenizer=self.tokenizer,
                is_adv=False,
            )
            self.utilityloader = Loader(
                dataset=utilityset,
                batch_size=trainer_config.utility_batch_size,
                train=True,
                collate_fn=self.utility_datacollator,
                num_workers=0,
            )
            self.utility_split_bs = trainer_config.utility_split_bs
            self.utility_max_length = trainer_config.utility_max_length

            self.adv_alpha = trainer_config.adv_alpha
            self.utility_alpha = trainer_config.utility_alpha

    def train(self, next_steps: int = -1):
        if next_steps < 0:
            next_steps = self.train_steps
        next_steps = min(next_steps, self.train_steps - self.current_step)

        for i in range(next_steps):
            self.current_step += 1

            inputs = next(self.trainloader)
            inputs = {k : v.to(self.device) for k, v in inputs.items()}

            advsfx_ids = None
            if self.attacker is not None:
                temp = self.attacker.attack_embeds(
                    model        = self.model,
                    tokenizer    = self.tokenizer,
                    message_ids  = inputs["prompt_ids"],
                    message_mask = inputs["prompt_mask"],
                    target_ids   = inputs["harmful_ids"],
                    target_mask  = inputs["harmful_mask"],
                    device       = self.device,
                )
                if temp is None:
                    advsfx_ids = None
                    advsfx_embeds = None
                elif len(temp.shape) == 2:
                    advsfx_ids = temp
                    advsfx_embeds = None
                elif len(temp.shape) == 3:
                    advsfx_ids = None
                    advsfx_embeds = temp
                else:
                    raise RuntimeError
                advsfx_inputs = {"ids": advsfx_ids, "embeds": advsfx_embeds}

            utility_inputs = None
            if self.utilityset is not None:
                utility_inputs = next(self.utilityloader)
                utility_inputs = {k : v.to(self.device) for k, v in utility_inputs.items()}

            losses = self._train_one_epoch(inputs, advsfx_inputs, utility_inputs)

            if (self.logger is not None) and ((self.current_step) % self.report_freq == 0):
                self.logger.info(f"Step: [{self.current_step}/{self.train_steps}]")
                for k, v in losses.items():
                    self.logger.info("{}: {:.6f}".format(k, v))
                    # self.logger.info("training loss: {:.6f}".format(loss))

    def _train_one_epoch(self, inputs: torch.Tensor, advsfx_inputs: dict, utility_inputs: dict = None) -> dict:
        raise NotImplementedError(f"Please implement _compute_loss_and_backward() method for {self.__class__}")

    def state_dict(self) -> Dict[str, Any]:
        trainer_state_dict = {
            "current_step" : self.current_step,
            "optimizer"    : self.optimizer.state_dict(),
            "scheduler"    : self.scheduler.state_dict(),
        }
        return self.model.state_dict(), trainer_state_dict

    def load_state_dict(
        self,
        model_state_dict: Dict[str, torch.Tensor] = None,
        trainer_state_dict: Dict[str, Any] = None,
    ):
        if model_state_dict is not None:
            self.model.load_state_dict(model_state_dict)

        if trainer_state_dict is not None:
            self.current_step = trainer_state_dict["current_step"]
            self.optimizer.load_state_dict(trainer_state_dict["optimizer"])
            self.scheduler.load_state_dict(trainer_state_dict["scheduler"])


class AdvTrainerSFT(AdvTrainerBase):
    def _train_one_epoch(self, inputs: dict, advsfx_inputs: dict, utility_inputs: dict = None) -> dict:
        self.model.train()
        self.optimizer.zero_grad()

        advsfx_ids = advsfx_inputs["ids"]
        advsfx_embeds = advsfx_inputs["embeds"]

        if advsfx_ids is not None:
            input_ids = torch.cat([inputs["prompt_ids"], advsfx_ids, inputs["benign_ids"]], dim=1)
            input_mask = torch.cat([inputs["prompt_mask"], torch.ones_like(advsfx_ids), torch.ones_like(inputs["benign_ids"])], dim=1)
            input_embeds = self.model.get_input_embeddings()(input_ids)
        elif advsfx_embeds is not None:
            input_embeds = torch.cat([
                self.model.get_input_embeddings()(inputs["prompt_ids"]),
                advsfx_embeds,
                self.model.get_input_embeddings()(inputs["benign_ids"]),
            ], dim=1)
            input_mask = torch.cat([
                inputs["prompt_mask"],
                torch.ones(advsfx_embeds.shape[:2], dtype=torch.int64, device=advsfx_embeds.device),
                torch.ones_like(inputs["benign_ids"])
            ], dim=1)
        else:
            input_ids = torch.cat([inputs["prompt_ids"], inputs["benign_ids"]], dim=1)
            input_mask = torch.cat([inputs["prompt_mask"], torch.ones_like(inputs["benign_ids"])], dim=1)
            input_embeds = self.model.get_input_embeddings()(input_ids)

        target_ids = inputs["benign_ids"]
        target_mask = inputs["benign_mask"]
        tar_len = target_mask.shape[-1]

        # logits = self.model(input_ids=input_ids, attention_mask=input_mask).logits
        logits = self.model(inputs_embeds=input_embeds, attention_mask=input_mask).logits
        logits = logits[:, -(tar_len+1) : -1]
        logps = logits.log_softmax(dim=-1)
        logps = torch.gather(logps, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)

        loss = -(logps * target_mask).sum() / target_mask.sum()
        alpha_loss = loss * self.adv_alpha
        alpha_loss.backward()
        loss = loss.item()

        if utility_inputs is not None:
            u_bs = len(utility_inputs["input_ids"])
            u_cnt = utility_inputs["input_mask"].sum()
            sp_bs = self.utility_split_bs
            if sp_bs is None:
                sp_bs = u_bs

            utility_loss = 0
            for be in range(0, u_bs, sp_bs):
                ed = min(u_bs, be + sp_bs)

                u_ids = utility_inputs["input_ids"][be:ed]
                u_mask = utility_inputs["input_mask"][be:ed]

                if self.utility_max_length is not None:
                    max_L = self.utility_max_length
                    if u_ids.shape[1] > max_L:
                        u_ids = u_ids[:, : max_L]
                        u_mask = u_mask[:, : max_L]

                u_logits = self.model(input_ids=u_ids).logits
                u_logps = u_logits.log_softmax(dim=-1)
                u_logps = torch.gather(u_logps[:, :-1], dim=-1, index=u_ids[:, 1:].unsqueeze(-1)).squeeze(-1) # shape: [ed-be, L-1]
                u_mask = u_mask[:, 1:]

                u_loss = -(u_logps * u_mask).sum() / u_cnt
                u_alpha_loss = u_loss * self.utility_alpha
                u_alpha_loss.backward()
                utility_loss += u_loss.item()

        self.optimizer.step()
        self.scheduler.step()

        return {
            "align_loss"   : loss,
            "utility_loss" : utility_loss,
        }


__adv_trainer_zoo__ = {
    "sft": AdvTrainerSFT,
    # "dpo": AdvTrainerDPO,
}

def build_trainer(trainer_config: AdvTrainerConfig, **kwargs):
    return __adv_trainer_zoo__[trainer_config.name](
        trainer_config=trainer_config,
        **kwargs,
    )
