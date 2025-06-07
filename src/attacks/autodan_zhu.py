from typing import Union, Sequence, Dict
from tqdm import tqdm
import copy
import torch, transformers

from .utils import ExpandablePrefixCache, AttackerBase


class AutoDANZhu(AttackerBase):
    """
    A "batch inputs" implementation of the Zhu's AutoDAN Jailbreaking Attack.

    Reference:
        [AutoDAN: Interpretable Gradient-Based Adversarial Attacks on Large Language Models](https://arxiv.org/pdf/2310.15140)
    """
    def __init__(
        self,
        suffix_length: int,
        iter_steps: int,
        obj_w1: float,
        obj_w2: float,
        topb: int,
        temperature: float,
        batch_size: int = -1,
        kv_cache: bool = False,
        disable_tqdm: bool = False,
    ):
        super().__init__(suffix_length)

        self.suffix_length = suffix_length
        self.iter_steps = iter_steps
        self.obj_w1 = obj_w1
        self.obj_w2 = obj_w2
        self.topb = topb
        self.temperature = temperature
        self.batch_size = batch_size
        self.kv_cache = kv_cache
        self.disable_tqdm = disable_tqdm

    def _single_token_optimization(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizerBase,
        pfx_ids: torch.Tensor,
        pfx_mask: torch.Tensor,
        pfx_offset: torch.Tensor,
        pfx_cache: Union[ExpandablePrefixCache, None], # kv-cache
        new_token: Union[torch.Tensor, None], # shape: [B,]
        target_ids: torch.Tensor,
        target_mask: torch.Tensor,
        target_offset: torch.Tensor,
        device: Union[str, torch.device],
    ) -> [torch.Tensor, torch.Tensor]:

        vocab_size = model.vocab_size
        embedding_layer = model.get_input_embeddings()
        B = pfx_ids.shape[0]

        if new_token is None:
            new_token = torch.zeros((B, 1), dtype=torch.int64, device=device) + int(tokenizer.encode("x", add_special_tokens=False)[0])

        pfx_len = pfx_ids.shape[1]

        target_embeds = embedding_layer(target_ids)
        tar_len = target_ids.shape[1]

        # build onehot embedding for (batched) new_token
        new_token_onehot = torch.nn.functional.one_hot(new_token, num_classes=vocab_size).to(embedding_layer.weight.dtype)
        new_token_onehot.requires_grad = True

        # build input embeds and mask that will be feeded into causal-LM
        new_token_embeds = torch.matmul(new_token_onehot, embedding_layer.weight.data)
        input_embeds = torch.cat([new_token_embeds, target_embeds], dim=-2)
        input_mask = torch.cat([
            pfx_mask,
            torch.ones((B, 1), dtype=torch.int64, device=device),
            torch.ones_like(target_mask),
        ], dim=1)

        if pfx_cache is not None:
            pfx_cache.set_expand(None)
            pfx_cache.set_clip(None)
            pfx_cache.set_pseudo_cache_len(None)
            out_logits = model(
                inputs_embeds=input_embeds, attention_mask=input_mask,
                past_key_values=pfx_cache, use_cache=True,
            )["logits"]
            out_logits = out_logits[:, -(tar_len+1) : -1]
            out_logps = out_logits.log_softmax(dim=-1)
            out_logps = torch.gather(out_logps, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
            loss = -((out_logps * target_mask).sum(dim=-1) / (target_mask.sum(dim=-1) + 1e-8)).mean()
            gd = torch.autograd.grad(loss, new_token_onehot)[0].data

            pfx_cache.set_expand(None)
            pfx_cache.set_clip(None)
            pfx_cache.set_pseudo_cache_len(pfx_len-1)
            out_logits = model(
                input_ids=pfx_ids[:, (pfx_len-1) : pfx_len], attention_mask=pfx_mask,
                past_key_values=pfx_cache, use_cache=True,
            )["logits"] # shape: [B, 1, vocab_size]

            pfx_next_token_logps = out_logits.log_softmax(dim=-1) # shape: [B, 1, vocab_size]

            # scr = gd
            # scr = gd * (self.obj_w1 * 10) - pfx_next_token_logps
            scr = gd * self.obj_w1 - pfx_next_token_logps

        else:
            raise NotImplementedError

        topb = self.topb
        topb_ids = scr.topk(topb, dim=-1, largest=False).indices # shape: [B, 1, topb]
        topb_ids = topb_ids.squeeze(1).permute([1, 0]).unsqueeze(-1) # shape: [topb, B, 1]
        # print(topb_ids.shape, " ", topb_ids[0].shape, " ", (topb_ids[0]-new_token), flush=True)

        scores = []

        batch_size = topb if self.batch_size == -1 else self.batch_size
        for be in range(0, topb, batch_size):
            ed = min(be + batch_size, topb)

            cand_ids = torch.cat([
                pfx_ids.unsqueeze(0).expand(ed-be, -1, -1), # shape: [(ed-be), B, tar_len]
                topb_ids[be:ed], # shape: [(ed-be), B, 1]
                target_ids.unsqueeze(0).expand(ed-be, -1, -1), # shape: [(ed-be), B, tar_len]
            ], dim=-1).view((ed-be) * B, -1).contiguous() # shape: [(ed-be) * B, length]

            cand_mask = input_mask.unsqueeze(0).expand(ed-be, -1, -1).reshape((ed-be) * B, -1)

            if self.kv_cache:
                pfx_cache.set_expand(ed-be)
                pfx_cache.set_clip(None)
                pfx_cache.set_pseudo_cache_len(None)

                out_logits = model(
                    input_ids=cand_ids[:, pfx_len : ], attention_mask=cand_mask,
                    past_key_values=pfx_cache, use_cache=True,
                )["logits"] # shape: [(ed-be) * B, 1 + tar_len, vocab_size]
                out_logits = out_logits[:, -(tar_len+1) : -1]
                out_logps = out_logits.log_softmax(dim=-1) # shape: [(ed-be) * B, tar_len, vocab_size]
                out_logps = torch.gather(out_logps, dim=-1, index=cand_ids[:, -tar_len : ].unsqueeze(-1)).squeeze(-1)
                temp_cand_mask = cand_mask[:, -tar_len : ]
                r_obj = -(out_logps * temp_cand_mask).sum(dim=-1) / (temp_cand_mask.sum(dim=-1) + 1e-8)

                out_logps = pfx_next_token_logps
                out_logps = out_logps.unsqueeze(0).expand(ed-be, B, -1, -1).contiguous() # shape: [(ed-be), B, 1, vocab_size]
                out_logps = torch.gather(out_logps, dim=-1, index=topb_ids[be:ed].unsqueeze(-1)) # shape: [(ed-be), B, 1, 1]
                out_logps = out_logps.view(-1)
                r_int = -out_logps

            else:
                raise NotImplementedError

            # r_fltr = 0
            cand_pfx_offset = pfx_offset.unsqueeze(0).expand(ed-be, -1).reshape(-1)
            cand_tar_offset = target_offset.unsqueeze(0).expand(ed-be, -1).reshape(-1)
            fltr_msk = self._get_filter_mask(tokenizer, cand_ids, cand_pfx_offset, cand_tar_offset)
            r_fltr = fltr_msk * 1e6

            scores.append(r_obj * self.obj_w2 + r_int + r_fltr)

        scores = torch.cat(scores).view(topb, B)

        # # ======= for debugging start =======
        # print(scores.min(dim=0), flush=True)
        # # ======== for debugging end ========

        # ind = scores.argmin(dim=0, keepdim=True)
        # res = torch.gather(topb_ids.squeeze(-1), dim=0, index=ind).permute([1, 0])
        # print(res, flush=True)

        scores = (-scores / (self.temperature + 1e-6)).permute([1,0]).softmax(dim=-1) # shape: [B, topb]
        ind = torch.multinomial(scores, num_samples=1).permute([1,0]) # shape: [1, B]
        res = torch.gather(topb_ids.squeeze(-1), dim=0, index=ind).permute([1, 0]) # shape: [B, 1]

        # # ======= for debugging start =======
        # print(res, flush=True)
        # # ======== for debugging end ========

        return res

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

        # pfx_embeds = embedding_layer(message_ids)
        pfx_ids = message_ids
        pfx_mask = message_mask
        pfx_len = message_mask.shape[1]
        pfx_offset = self._get_prefix_offset_from_mask(message_mask)

        # target_embeds = embedding_layer(target_ids)
        tar_len = target_ids.shape[1]
        target_offset = self._get_suffix_offset_from_mask(target_mask)

        pfx_cache = None
        if self.kv_cache:
            # step 0: build kv_cache to acclerate gradients/scores calculations
            raw_cache = model(input_ids=pfx_ids, attention_mask=pfx_mask)["past_key_values"]
            pfx_cache = ExpandablePrefixCache(pfx_len + sfx_len + tar_len)
            for ii, (key_states, value_states) in enumerate(raw_cache):
                pfx_cache.initialize_layer(key_states, value_states, ii)
            del raw_cache, key_states, value_states

        pbar = tqdm(range(self.suffix_length), disable=self.disable_tqdm)

        for step in pbar:
            new_token = None
            for ii in range(self.iter_steps):
                # shape: [B, 1]
                new_token = self._single_token_optimization(
                    model         = model,
                    tokenizer     = tokenizer,
                    pfx_ids       = pfx_ids,
                    pfx_mask      = pfx_mask,
                    pfx_offset    = pfx_offset,
                    pfx_cache     = pfx_cache,
                    new_token     = new_token,
                    target_ids    = target_ids,
                    target_mask   = target_mask,
                    target_offset = target_offset,
                    device        = device,
                )

            pfx_ids = torch.cat([pfx_ids, new_token], dim=-1)
            pfx_mask = torch.cat([pfx_mask, torch.ones((B,1), dtype=torch.int64, device=device)], dim=-1)
            pfx_len += 1
            if self.kv_cache:
                pfx_cache.set_expand(None)
                pfx_cache.set_clip(pfx_len)
                pfx_cache.set_pseudo_cache_len(None)
                _ = model(input_ids=new_token, attention_mask=pfx_mask, past_key_values=pfx_cache, use_cache=True)
                del _
                pfx_cache.set_clip(None)

            # # ======= for debugging start =======
            # pfx_cache.set_pseudo_cache_len(pfx_len-1)
            # input_ids = torch.cat([pfx_ids, target_ids], dim=-1)
            # input_mask = torch.cat([pfx_mask, torch.ones_like(target_mask)], dim=-1)
            # out_logits = model(
            #     input_ids=input_ids[:, -(tar_len+1) : ], attention_mask=input_mask,
            #     past_key_values=pfx_cache, use_cache=True,
            # )["logits"]
            # pfx_cache.set_pseudo_cache_len(None)
            # out_logits = out_logits[:, -(tar_len+1) : -1]
            # out_logps = out_logits.log_softmax(dim=-1)
            # out_logps = torch.gather(out_logps, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
            # loss = -(out_logps * target_mask).mean(dim=-1)
            # print("loss = ", loss, flush=True)
            # # ======== for debugging end ========

        # re-open model autograd
        for pp, rq_gd in zip(model.parameters(), reqs_grad):
            pp.requires_grad = rq_gd

        # print(f"message_ids[0] = {message_ids[0].tolist()}")
        # print(f"advsfx_ids[0] = {advsfx_ids[0].tolist()}")
        # print(f"target_ids[0] = {target_ids[0].tolist()}")

        advsfx_ids = pfx_ids[:, -self.suffix_length : ]
        return advsfx_ids
