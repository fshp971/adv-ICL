from typing import Union, Sequence, Dict
from tqdm import tqdm
import copy
import torch, transformers

from .utils import ExpandablePrefixCache, AttackerBase


class ContiEmbed(AttackerBase):
    """ A "batch inputs" implementation of the continuous embedding attack
    """
    def __init__(
        self,
        suffix_length: int,
        steps: int,
        step_size: float,
        radius: float,
        norm_type: str,
        kv_cache: bool = False,
        disable_tqdm: bool = False,
    ):
        super().__init__(suffix_length)

        self.steps = steps
        self.step_size = step_size
        self.radius = radius

        assert norm_type in ["l-infty", "l2"]
        self.norm_type = norm_type

        self.kv_cache = kv_cache
        self.disable_tqdm = disable_tqdm

    def attack_embeds(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizerBase,
        message_ids: torch.Tensor,
        message_mask: torch.Tensor,
        target_ids: torch.Tensor,
        target_mask: torch.Tensor,
        device: Union[str, torch.device],
    ) -> torch.Tensor:

        if self.suffix_length == 0:
            return None

        # close model autograd for potential speed up
        reqs_grad = []
        for pp in model.parameters():
            reqs_grad.append(pp.requires_grad)
            pp.requires_grad = False
        model.eval()

        # prepare parameters
        vocab_size = model.vocab_size
        embedding_layer = model.get_input_embeddings()
        B = len(message_ids)

        sfx_len = self.suffix_length

        message_embeds = embedding_layer(message_ids)
        msg_len = message_embeds.shape[1]

        embed_dim = message_embeds.shape[-1]
        dtype = message_embeds.dtype

        target_embeds = embedding_layer(target_ids)
        tar_len = target_embeds.shape[1]

        L = msg_len + sfx_len + tar_len

        sfx_ids = torch.tensor(tokenizer.encode(" x" * (sfx_len + 5), add_special_tokens=False), dtype=torch.int64, device=device)
        sfx_ids = sfx_ids[: sfx_len].unsqueeze(0).expand(B, -1).clone()
        # sfx_embeds = torch.nn.functional.one_hot(sfx_ids, num_classes=vocab_size).to(dtype)
        sfx_embeds = embedding_layer(sfx_ids)

        if self.norm_type == "l-infty":
            advsfx_embeds = (torch.rand((B, sfx_len, embed_dim), dtype=dtype, device=device) * 2 - 1) * self.radius
            # advsfx_embeds = (torch.rand((B, sfx_len, vocab_size), dtype=dtype, device=device) * 2 - 1) * self.radius
        elif self.norm_type == "l2":
            advsfx_embeds = (torch.rand((B, sfx_len, embed_dim), dtype=dtype, device=device) * 2 - 1) * (self.radius / (self.steps + 1e-8))
            # advsfx_embeds = (torch.rand((B, sfx_len, vocab_size), dtype=dtype, device=device) * 2 - 1) * (self.radius / (self.steps + 1e-8))
        advsfx_embeds += sfx_embeds
        self._clip_embeds_(advsfx_embeds, sfx_embeds)
        # self._clip_embeds_(advsfx_embeds)

        input_mask = torch.cat([message_mask, torch.ones((B, sfx_len), dtype=torch.int64, device=device), target_mask], dim=1)

        if self.kv_cache:
            # step 0: build kv_cache to acclerate gradients/scores calculations
            raw_cache = model(inputs_embeds=message_embeds, attention_mask=message_mask)["past_key_values"]
            cache = ExpandablePrefixCache(L)
            for ii, (key_states, value_states) in enumerate(raw_cache):
                cache.initialize_layer(key_states, value_states, ii)
            del raw_cache, key_states, value_states

        pbar = tqdm(range(self.steps), disable=self.disable_tqdm)

        # for step in range(self.steps):
        advsfx_embeds.requires_grad_()
        for step in pbar:

            if self.kv_cache:
                cache.set_expand(None)
                input_embeds = torch.cat([
                    # torch.matmul(advsfx_embeds, embedding_layer.weight.data),
                    advsfx_embeds,
                    target_embeds,
                ], dim=1)
                out_logits = model(
                    inputs_embeds=input_embeds, attention_mask=input_mask,
                    past_key_values=cache, use_cache=True,
                )["logits"]
            else:
                raise NotImplementedError

            out_logits = out_logits[:, -(tar_len+1) : -1]
            out_logps = out_logits.log_softmax(dim=-1)
            out_logps = torch.gather(out_logps, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
            loss = -(out_logps * target_mask).mean()
            gd = torch.autograd.grad(loss, [advsfx_embeds])[0]

            # self._clip_gd_(gd)
            advsfx_embeds.data.add_(gd, alpha= -self.step_size)
            self._clip_embeds_(advsfx_embeds, sfx_embeds)
            # self._clip_embeds_(advsfx_embeds)

            # # ==== debug begin ====
            # print(f"loss = {loss.item()}", flush=True)
            # # ===== debug end =====

        # re-open model autograd
        for pp, rq_gd in zip(model.parameters(), reqs_grad):
            pp.requires_grad = rq_gd

        return advsfx_embeds.data
        # return torch.matmul(advsfx_embeds.data, embedding_layer.weight.data)

    @torch.no_grad()
    def _clip_embeds_(self, advsfx_embeds, sfx_embeds):
    # def _clip_embeds_(self, advsfx_embeds):
        # shape: [B, L, dim]
        advsfx_embeds -= sfx_embeds
        if self.norm_type == "l-infty":
            advsfx_embeds.clamp_(-self.radius, self.radius)
        elif self.norm_type == "l2":
            norm = (advsfx_embeds ** 2).sum(dim=-1).sqrt().unsqueeze(-1) # shape: [B, L, 1]
            mask = (norm >= self.radius)
            advsfx_embeds.data = (mask^True) * advsfx_embeds + mask * advsfx_embeds / (norm+1e-8) * self.radius
        advsfx_embeds += sfx_embeds

    # @torch.no_grad()
    # def _clip_gd_(self, gd):
    #     # shape: [B, L, dim]
    #     if self.norm_type == "l-infty":
    #         gd.sign_()
    #     elif self.norm_type == "l2":
    #         norm = (gd ** 2).sum(dim=-1).sqrt().unsqueeze(-1) # shape: [B, L, 1]
    #         gd.data /= (norm + 1e-8)
