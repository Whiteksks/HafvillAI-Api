"""Descriptive repetition and output-region metrics, not semantic evaluation."""

import re

TAGS = ("system_thinking", "role_thinking", "role_action")


def tagged_spans(text, tag, *, body=False):
    pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL)
    return [match.span(1 if body else 0) for match in pattern.finditer(text)]


def token_kinds(text, offsets):
    """System planning takes precedence over nested role tags.

    Tag delimiters are included in their regions. A token belongs to a region
    when its character offsets overlap the corresponding block.
    """
    spans = {tag: tagged_spans(text, tag) for tag in TAGS}
    return [next((tag for tag in TAGS if any(start < b and end > a for a, b in spans[tag])),
                 "other") for start, end in offsets]


def visible_text(text):
    """Approximate visible reply without closed planning or thought blocks.

    An unclosed block hides the remaining text in this diagnostic view.
    The original response is always saved in full. <system> is handled as an
    observed generated block, not as the training serialization format.
    """
    result = text
    unfinished = False
    for tag in ("system_thinking", "system", "role_thinking"):
        result = re.sub(rf"<{tag}>.*?</{tag}>", "", result, flags=re.DOTALL)
        opening = result.find(f"<{tag}>")
        if opening >= 0:
            result = result[:opening]
            unfinished = True
    return re.sub(r"<\|[^>]+\|>", "", result).strip(), unfinished


def repetition_metrics(words):
    grams = [tuple(words[i:i + 4]) for i in range(max(0, len(words) - 3))]
    duplicate = 0.0 if not grams else 1 - len(set(grams)) / len(grams)
    longest, period = 0, None
    # For each period, count consecutive matches between shifted words.
    # This is linear in word count for a fixed maximum period of 16.
    for size in range(1, min(16, len(words) // 2) + 1):
        run = 0
        for i in range(size, len(words)):
            run = run + 1 if words[i] == words[i - size] else 0
            if run >= size:
                length = size * (1 + run // size)
                if length > longest:
                    longest, period = length, size
    return {"words": len(words), "duplicate_4gram_fraction": round(duplicate, 6),
            "longest_consecutive_repeat_words": longest, "repeat_period_words": period}


def text_metrics(text, expected_mentions=()):
    visible, unfinished = visible_text(text)
    words = re.findall(r"\w+(?:['’]\w+)?", text.lower())
    visible_words = re.findall(r"\w+(?:['’]\w+)?", visible.lower())
    markers = ("<system_thinking>", "<system>", "<system info>", "<screen>",
               "Context Analysis:", "Character Analysis", "Drafting the response:")
    return {"all_text": repetition_metrics(words), "visible_text": repetition_metrics(visible_words),
            "planning_marker_present": any(marker in text for marker in markers),
            "unfinished_thinking_block": unfinished, "visible_chars": len(visible),
            "expected_mentions_in_visible_text": {
                term: bool(re.search(rf"\b{re.escape(term)}\b", visible, re.IGNORECASE))
                for term in expected_mentions}}


def output_budget(prompt_tokens, max_length, max_new_tokens=None):
    if not 1 <= prompt_tokens < max_length <= 2048:
        raise ValueError("Нужен непустой вход короче max_length; общий лимит не более 2048. Вход не обрезан.")
    if max_new_tokens is not None and max_new_tokens < 1:
        raise ValueError("max_new_tokens должен быть положительным")
    return min(max_length - prompt_tokens, max_new_tokens or max_length)


def validate_cases(cases):
    if not isinstance(cases, list) or not cases:
        raise ValueError("Нужен непустой список тестов")
    seen = set()
    for case in cases:
        name = case.get("id", "")
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", name) or name in seen:
            raise ValueError("id теста должен быть уникальным безопасным именем")
        seen.add(name)
        if case.get("format") == "raw":
            if not isinstance(case.get("prompt"), str) or not case["prompt"]:
                raise ValueError("raw требует непустой prompt")
        elif case.get("format") == "chat":
            messages = case.get("messages")
            if not isinstance(messages, list) or not messages or messages[-1].get("role") != "user":
                raise ValueError("chat требует messages с последним user")
            for i, message in enumerate(messages):
                if message.get("role") not in {"system", "user", "assistant"}:
                    raise ValueError("Неизвестная роль сообщения")
                if not isinstance(message.get("content"), str) or not message["content"]:
                    raise ValueError("content должен быть непустой строкой")
                if message["role"] == "system" and i != 0:
                    raise ValueError("system разрешён только первым")
        else:
            raise ValueError("format должен быть raw или chat")
    return cases
