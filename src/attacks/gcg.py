from typing import Union, Sequence, Dict
from tqdm import tqdm
import copy
import torch, transformers

from .utils import ExpandablePrefixCache, AttackerBase


class GCG(AttackerBase):
    """
    A "batch inputs" implementation of the GCG Jailbreaking Attack.

    Reference:
        [Universal and Transferable Adversarial Attacks on Aligned Language Models](https://arxiv.org/pdf/2307.15043)
    """
    def __init__(
        self,
        suffix_length: int,
        steps: int,
        topk: int,
        search_width: int,
        batch_size: int = -1,
        width_bs: int = -1,
        kv_cache: bool = False,
        disable_tqdm: bool = False,
    ):
        super().__init__(suffix_length)

        self.steps = steps
        self.topk = topk
        self.search_width = search_width
        self.batch_size = batch_size
        self.width_bs = width_bs
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

        topk = self.topk
        sfx_len = self.suffix_length
        width = self.search_width
        grad_bs = B if self.batch_size == -1 else self.batch_size
        qry_bs = width if self.width_bs == -1 else self.width_bs

        message_embeds = embedding_layer(message_ids)
        msg_len = message_embeds.shape[1]
        message_offset = self._get_prefix_offset_from_mask(message_mask)

        target_embeds = embedding_layer(target_ids)
        tar_len = target_embeds.shape[1]
        target_offset = self._get_suffix_offset_from_mask(target_mask)

        L = msg_len + sfx_len + tar_len

        # build ids and mask for advsfx
        # advsfx_ids = torch.tensor(tokenizer.encode("ab" * sfx_len, add_special_tokens=False), dtype=torch.int64, device=device)
        advsfx_ids = torch.tensor(tokenizer.encode(" x" * (sfx_len + 5), add_special_tokens=False), dtype=torch.int64, device=device)
        advsfx_ids = advsfx_ids[: sfx_len].unsqueeze(0).expand(B, -1).clone()

        advsfx_mask = torch.ones((B, sfx_len), dtype=torch.int64, device=device)

        # mask that will be used when calculating gradients
        input_mask = torch.cat([message_mask, advsfx_mask, torch.ones_like(target_mask)], dim=1) # shape: [B, L]

        cand_message_offset = message_offset.unsqueeze(0).expand(width, -1).reshape(width*B).contiguous() # shape [width*B,]
        cand_target_offset = target_offset.unsqueeze(0).expand(width, -1).reshape(width*B).contiguous() # shape [width*B,]

        if self.kv_cache:
            # step 0: build kv_cache to acclerate gradients/scores calculations
            raw_cache = model(inputs_embeds=message_embeds, attention_mask=message_mask)["past_key_values"]
            cache = [ExpandablePrefixCache(L) for ii in range(0, B, grad_bs)]
            for ii, (key_states, value_states) in enumerate(raw_cache):
                for kk, be in enumerate(range(0, B, grad_bs)):
                    ed = min(be+grad_bs, B)
                    cache[kk].initialize_layer(key_states[be:ed], value_states[be:ed], ii)
            del raw_cache, key_states, value_states

        pbar = tqdm(range(self.steps), disable=self.disable_tqdm)

        # for step in range(self.steps):
        for step in pbar:
            # step 1: calculate topk scores based on onehot gradients
            topk_ids = []

            for ii, be in enumerate(range(0, B, grad_bs)):
                ed = min(be + grad_bs, B)

                # top-k scores will be calculated based on gradients of adv_onehot
                adv_onehot = torch.nn.functional.one_hot(advsfx_ids[be:ed], num_classes=vocab_size).to(embedding_layer.weight.dtype)
                adv_onehot.requires_grad = True

                # build input embeds and mask that will be feeded into causal-LM
                adv_embeds = torch.matmul(adv_onehot, embedding_layer.weight.data)
                inp_mask = input_mask[be:ed]
                if self.kv_cache:
                    cache[ii].set_expand(None)
                    inp_embeds = torch.cat([adv_embeds, target_embeds[be:ed]], dim=1)
                    out_logits = model(
                        inputs_embeds=inp_embeds, attention_mask=inp_mask,
                        past_key_values=cache[ii], use_cache=True,
                    )["logits"]
                else:
                    inp_embeds = torch.cat([message_embeds[be:ed], adv_embeds, target_embeds[be:ed]], dim=1)
                    out_logits = model(inputs_embeds=inp_embeds, attention_mask=inp_mask, use_cache=False)["logits"]

                out_logits = out_logits[:, -(tar_len+1) : -1]
                out_logps = out_logits.log_softmax(dim=-1)
                out_logps = torch.gather(out_logps, dim=-1, index=target_ids[be:ed].unsqueeze(-1)).squeeze(-1)
                loss = -(out_logps * target_mask[be:ed]).mean()

                # print("step = {}, loss = {:.3f}".format(step, loss.item()))

                gd = torch.autograd.grad(loss, adv_onehot)[0]
                topk_ids.append(gd.topk(topk, dim=-1, largest=False).indices)

            topk_ids = torch.cat(topk_ids) # shape: [B, sfx_len, topk]

            # step 2: sampling and scoring candidates
            values = topk_ids.unsqueeze(0).expand(width, -1, -1, -1) # shape: [width, B, sfx_len, topk]
            indices = torch.randint(0, topk, (width, B, sfx_len, 1), device=device)
            values = torch.gather(values, dim=-1, index=indices).squeeze(-1) # shape: [width, B, sfx_len]
            indices = torch.randint(0, sfx_len, (width, B, 1), device=device) # selected seq positions; shape: [width, B, 1]
            values = torch.gather(values, dim=-1, index=indices) # selected adv token ids; shape: [width, B, 1]

            cand_advsfx_ids = advsfx_ids.unsqueeze(0).expand(width, -1, -1).clone() # shape: [width, B, sfx_len]
            cand_advsfx_ids.scatter_(dim=-1, index=indices, src=values)

            score = []
            for ii, be in enumerate(range(0, width, qry_bs)):
                ed = min(be + qry_bs, width)
                cand_advsfx_embeds = embedding_layer(cand_advsfx_ids[be:ed]) # shape: [ed-be, B, sfx_len, embed_dim]

                scr_rows = []

                for kk, be2 in enumerate(range(0, B, grad_bs)):
                    ed2 = min(be2 + grad_bs, B)

                    cand_embeds = torch.cat([
                        message_embeds[be2:ed2].unsqueeze(0).expand(ed-be, -1, -1, -1),
                        cand_advsfx_embeds[:, be2:ed2],
                        target_embeds[be2:ed2].unsqueeze(0).expand(ed-be, -1, -1, -1),
                    ], dim=-2).view((ed-be)*(ed2-be2), L, -1) # shape: [(ed-be)*(ed2-be2), L, embed_dim]

                    cand_mask = input_mask[be2:ed2].unsqueeze(0).expand(ed-be, -1, -1).reshape((ed-be)*(ed2-be2), -1).contiguous()
                    if self.kv_cache:
                        cache[kk].set_expand(ed-be)
                        out_logits = model(
                            inputs_embeds=cand_embeds[:, msg_len:], attention_mask=cand_mask,
                            past_key_values=cache[kk], use_cache=True,
                        )["logits"]
                    else:
                        out_logits = model(inputs_embeds=cand_embeds, attention_mask=cand_mask, use_cache=False)["logits"]

                    cand_target_ids = target_ids[be2:ed2].unsqueeze(0).expand(ed-be, -1, -1) # shape: [ed-be, ed2-be2, tar_len]
                    cand_target_mask = target_mask[be2:ed2].unsqueeze(0).expand(ed-be, -1, -1) # shape: [ed-be, ed2-be2, tar_len]

                    out_logits = out_logits[:, -(tar_len+1) : -1]
                    out_logits = out_logits.view(ed-be, ed2-be2, *out_logits.shape[1:]) # shape: [ed-be, ed2-be2, tar_len, embed_dim]

                    out_logps = out_logits.log_softmax(dim=-1)
                    out_logps = torch.gather(out_logps, dim=-1, index=cand_target_ids.unsqueeze(-1)).squeeze(-1) # shape: [ed-be, ed2-be2, tar_len]

                    scr = -(out_logps * cand_target_mask).mean(dim=-1) # shape: [ed-be, ed2-be2]
                    scr_rows.append(scr)

                scr_rows = torch.cat(scr_rows, dim=1)
                score.append(scr_rows)

            score = torch.cat(score, dim=0).view(width*B)

            cand_ids = torch.cat([
                message_ids.unsqueeze(0).expand(width, -1, -1),
                cand_advsfx_ids,
                target_ids.unsqueeze(0).expand(width, -1, -1),
            ], dim=-1) # shape: [width, B, L]
            cand_ids = cand_ids.view(width*B, L)
            fltr_msk = self._get_filter_mask(tokenizer, cand_ids, cand_message_offset, cand_target_offset)

            score = (score + fltr_msk * 1e6).view(width, B) # shape: [width, B]
            ind = score.argmin(dim=0, keepdim=True) # shape: [1, B]

            # print(f"score.min(dim=0) = {score.min(dim=0)[0].tolist()}")
            # print(f"score.min(dim=0)[0].max() = {score.min(dim=0)[0].max()}")
            # only for debugging
            # assert score.min(dim=0)[0].max().item() < 1e5
            fin_fltr_msk = (score.min(dim=0).values > 1e5).unsqueeze(-1) # shape: [B, 1]
            advsfx_ids_old = advsfx_ids.clone()

            upd_indices = torch.gather(indices.squeeze(-1), dim=0, index=ind).squeeze(0).unsqueeze(-1)
            upd_values = torch.gather(values.squeeze(-1), dim=0, index=ind).squeeze(0).unsqueeze(-1)
            advsfx_ids.scatter_(dim=-1, index=upd_indices, src=upd_values)

            advsfx_ids = advsfx_ids * (~fin_fltr_msk) + advsfx_ids_old * fin_fltr_msk

        # re-open model autograd
        for pp, rq_gd in zip(model.parameters(), reqs_grad):
            pp.requires_grad = rq_gd

        # print(f"message_ids[0] = {message_ids[0].tolist()}")
        # print(f"advsfx_ids[0] = {advsfx_ids[0].tolist()}")
        # print(f"target_ids[0] = {target_ids[0].tolist()}")

        return advsfx_ids
