try:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
except ModuleNotFoundError:
    from einops import rearrange
    from transformers.modeling_flash_attention_utils import (
        _pad_input as pad_input,
        _unpad_input as unpad_input,
    )

    def index_first_axis(tensor, indices):
        return tensor[indices]
