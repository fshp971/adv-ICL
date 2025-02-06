from typing import Optional, Union, Sequence, Dict, Any, Tuple, List
import copy
import transformers, torch


class PrefixCache(transformers.Cache):
    def __init__(self, max_length: int):
        super().__init__()
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

        self.position: List[int] = []
        self._max_length = max_length

    def initialize_layer(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ):
        # both key_state and value_state should be in the shape: [B, head_num, L, embed_dim]
        assert len(key_states.shape) == 4 and len(value_states.shape) == 4

        while len(self.key_cache) <= layer_idx:
            self.key_cache.append([])
            self.value_cache.append([])
            self.position.append(0)

        self.key_cache[layer_idx] = key_states.clone()
        self.value_cache[layer_idx] = value_states.clone()
        self.position[layer_idx] =key_states.shape[-2]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # assert self.position[layer_idx] + key_states.shape[-2] == self._max_length

        key_states = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
        value_states = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
        return key_states, value_states

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        return self.position[layer_idx]

    def get_max_cache_shape(self) -> Optional[int]:
        # return None
        raise NotImplementedError

    def reorder_cache(self, beam_idx: torch.LongTensor):
        raise NotImplementedError


class ExpandablePrefixCache(PrefixCache):
    def __init__(self, max_length: int):
        super().__init__(max_length)
        self.expand_num = None

    def set_expand(self, expand_num: int = None):
        self.expand_num = expand_num

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # assert self.position[layer_idx] + key_states.shape[-2] == self._max_length

        key_cache = self.key_cache[layer_idx]
        value_cache = self.value_cache[layer_idx]

        if self.expand_num is not None:
            newB = self.expand_num * key_cache.shape[0]
            assert newB == key_states.shape[0]

            key_cache = key_cache.unsqueeze(0).expand(self.expand_num, *(len(key_cache.shape)*[-1,]))
            key_cache = key_cache.reshape(newB, *key_cache.shape[2:])

            value_cache = value_cache.unsqueeze(0).expand(self.expand_num, *(len(value_cache.shape)*[-1,]))
            value_cache = value_cache.reshape(newB, *value_cache.shape[2:])

        key_states = torch.cat([key_cache, key_states], dim=-2)
        value_states = torch.cat([value_cache, value_states], dim=-2)

        return key_states, value_states


class AttackerBase:
    """
    A "batch inputs" implementation of the GCG Jailbreaking Attack.

    Reference:
        [Universal and Transferable Adversarial Attacks on Aligned Language Models](https://arxiv.org/pdf/2307.15043)
    """
    def __init__(self, suffix_length: int):
        self.suffix_length = suffix_length

    def attack(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizerBase,
        message: Union[str, Sequence[Dict], Sequence[Union[str, Sequence[Dict]]]],
        target: Union[str, Sequence[str]],
        device: Union[str, torch.device],
    ):
        if (
            isinstance(message, str)
            or (isinstance(message, Sequence) and isinstance(message[0], Dict))
        ):
            message = [message,]
        if isinstance(target, str):
            target = [target,]
        assert len(message) == len(target)

        if isinstance(message[0], Sequence) and isinstance(message[0][0], Dict):
            raise NotImplementedError("Chatting-inputs-processing still has bugs")

        pad_id = tokenizer.encode(tokenizer.eos_token, add_special_tokens=False)[0]

        message = copy.deepcopy(message)
        target = copy.deepcopy(target)

        ids_mask_dict = self._build_ids_and_mask(tokenizer, message, target, device=device, pad_id=pad_id)
        advsfx_ids = self.attack_embeds(model, tokenizer, device=device, **ids_mask_dict)

        adv_message = []
        adv_suffix = []
        for msg, a_ids in zip(message, advsfx_ids):
            a_text = tokenizer.decode(a_ids)
            if isinstance(msg, str):
                msg += a_text
            else:
                msg[-1]["content"] += a_text
            adv_message.append(msg)
            adv_suffix.append(a_text)

        return {
            "adv_message": adv_message,
            "adv_suffix": adv_suffix,
        }

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
        raise NotImplementedError(f"Please implement attack_embeds() method for {self.__class__}")

    @staticmethod
    def _build_ids_and_mask(
        tokenizer: transformers.PreTrainedTokenizerBase,
        message: Sequence[Union[str, Sequence[Dict]]],
        target: Sequence[str],
        device: Union[str, torch.device],
        pad_id: int,
    ) -> Dict[str, Union[None, torch.Tensor]]:

        placeholder = "{advsfx}"
        before_list, after_list = [], []
        for x in message:
            if isinstance(x, str):
                before = tokenizer.encode(x, add_special_tokens=True)
                after = []
            else:
                if "content" not in x[-1].keys():
                    x[-1]["content"] = ""
                old_content = x[-1]["content"]
                x[-1]["content"] += placeholder
                text = tokenizer.apply_chat_template(x, tokenize=False, add_generation_prompt=True)
                x[-1]["content"] = old_content
                before, after = text.split(placeholder)
                before = tokenizer.encode(before, add_special_tokens=False)
                after = tokenizer.encode(after, add_special_tokens=False)

            before_list.append(before)
            after_list.append(after)

        B = len(message)
        before_len = max(len(ids) for ids in before_list)
        after_len = max(len(ids) for ids in after_list)

        message_ids = torch.full((B, before_len), fill_value=pad_id, dtype=torch.int64, device=device)
        message_mask = torch.zeros_like(message_ids)
        for ii, ids in enumerate(before_list):
            message_ids[ii, -len(ids) : ] = torch.tensor(ids, device=device)
            message_mask[ii, -len(ids) : ] = 1

        target_list = [tokenizer.encode(text, add_special_tokens=False) for text in target]
        tar_len = max(len(ids) for ids in target_list)
        target_ids = torch.full((B, after_len+tar_len), fill_value=pad_id, dtype=torch.int64, device=device)
        target_mask = torch.zeros_like(target_ids)
        for ii, (aft_ids, tar_ids) in enumerate(zip(after_list, target_list)):
            aft_ids = torch.tensor(aft_ids, dtype=torch.int64, device=device)
            tar_ids = torch.tensor(tar_ids, dtype=torch.int64, device=device)
            ids = torch.cat([aft_ids, tar_ids])
            target_ids[ii, : len(ids)] = ids
            target_mask[ii, len(aft_ids) : len(ids)] = 1

        return {
            "message_ids"  : message_ids,
            "message_mask" : message_mask,
            "target_ids"   : target_ids,
            "target_mask"  : target_mask,
        }

    @staticmethod
    def _get_prefix_offset_from_mask(mask: torch.Tensor):
        # mask should be in the shape [..., L]
        L = mask.shape[-1]
        filled_mask = (mask.cumsum(dim=-1) > 0)
        offset = L - filled_mask.sum(dim=-1)
        return offset

    @staticmethod
    def _get_suffix_offset_from_mask(mask: torch.Tensor):
        # mask should be in the shape [..., L]
        L = mask.shape[-1]
        filled_mask = ((mask.sum(dim=-1, keepdim=True) - mask.cumsum(dim=-1) + mask) > 0)
        offset = L - filled_mask.sum(dim=-1)
        return offset

    @staticmethod
    def _get_filter_mask(
        tokenizer: transformers.PreTrainedTokenizerBase,
        ids: torch.Tensor,
        prefix_offset: torch.Tensor,
        suffix_offset: torch.Tensor,
    ):
        B, L = len(ids), ids.shape[-1]
        device = ids.device
        re_ids = ids.clone()
        for i in range(B):
            pfx_off = prefix_offset[i].item()
            sfx_off = suffix_offset[i].item()

            txt = tokenizer.decode(ids[i, pfx_off+1 : L-sfx_off])
            rid = torch.tensor(tokenizer.encode(txt, add_special_tokens=False), device=device)
            if pfx_off+1 + len(rid) > L:
                continue

            re_ids[i, pfx_off+1 : L-sfx_off] = 0
            re_ids[i, pfx_off+1 : pfx_off+1 + len(rid)] = rid

        mask = ((ids != re_ids).sum(dim=-1) != 0)

        return mask

    @torch.no_grad()
    @staticmethod
    def _verify_loss(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizerBase,
        message: Union[str, Sequence[Dict]],
        target: str,
        device: Union[str, torch.device],
    ):
        if isinstance(message, str):
            m_ids = tokenizer(message, add_special_tokens=True).input_ids
        else:
            m_ids = tokenizer.apply_chat_template(message, add_generation_prompt=True)
        t_ids = tokenizer(target, add_special_tokens=False).input_ids

        m_ids = torch.tensor(m_ids, device=device)
        t_ids = torch.tensor(t_ids, device=device)
        ids = torch.cat([m_ids, t_ids])
        out_logits = model(input_ids=ids.unsqueeze(0)).logits.squeeze(0)

        out_logits = out_logits[-(len(t_ids)+1) : -1]
        out_logps = out_logits.log_softmax(-1)
        out_logps = torch.gather(out_logps, dim=-1, index=t_ids.unsqueeze(-1)).squeeze(-1)
        loss = -out_logps.mean()

        return loss.item()
