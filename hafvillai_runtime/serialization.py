"""Llama-3 chat serialization shared by training and inference."""

import torch

IGNORE_INDEX = -100


class ChatSerializer:
    """Use an explicit shared format because the base Llama tokenizer has no chat_template."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.bos = self._special("<|begin_of_text|>")
        self.eot = self._special("<|eot_id|>")
        self._special("<|start_header_id|>")
        self._special("<|end_header_id|>")
        self.headers = {role: tokenizer.encode(
            f"<|start_header_id|>{role}<|end_header_id|>\n\n", add_special_tokens=False
        ) for role in ("system", "user", "assistant")}

    def _special(self, text):
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) != 1 or self.tokenizer.convert_ids_to_tokens(ids[0]) != text:
            raise ValueError(f"Токенизатор не поддерживает {text}")
        return ids[0]

    def encode(self, messages):
        ids, labels = [self.bos], [IGNORE_INDEX]
        for m in messages:
            header = self.headers[m["role"]]
            body = self.tokenizer.encode(m["content"], add_special_tokens=False) + [self.eot]
            ids.extend(header)
            labels.extend([IGNORE_INDEX] * len(header))
            ids.extend(body)
            labels.extend(body if m["role"] == "assistant" else [IGNORE_INDEX] * len(body))
        return torch.tensor(ids, dtype=torch.int32), torch.tensor(labels, dtype=torch.int32)

    def prompt(self, messages):
        """Build an inference prompt in the same format, ending with an assistant header."""
        ids, _ = self.encode(messages)
        return ids.tolist() + self.headers["assistant"]
