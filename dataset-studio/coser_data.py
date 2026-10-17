"""Pure conversion of CoSER scenes to thoughtless, bounded dialogue examples."""

from collections import Counter
import re


def remove_blocks(text, opening="[", closing="]"):
    """Remove balanced nested blocks, rejecting malformed delimiters."""
    depth, result = 0, []
    for char in text:
        if char == opening:
            depth += 1
        elif char == closing:
            if not depth:
                raise ValueError("Unmatched closing delimiter")
            depth -= 1
        elif depth == 0:
            result.append(char)
    if depth:
        raise ValueError("Unclosed delimiter")
    return " ".join("".join(result).split())


def clean_utterances(conversation):
    utterances = conversation.get("dialogues")
    if not isinstance(utterances, list) or not utterances:
        raise ValueError("Missing dialogues")
    result = []
    for index, item in enumerate(utterances):
        name, message = item.get("character"), item.get("message")
        if not isinstance(name, str) or not name.strip() or not isinstance(message, str):
            raise ValueError("Invalid character/message")
        if name == "Environment":
            continue
        if re.search(r"<\||</?(?:system|role|think)", name + message, re.I):
            raise ValueError("Unexpected control/planning markup")
        content = remove_blocks(message)
        speech = remove_blocks(content, "(", ")")
        if content:
            result.append({"character": name, "content": content, "speech": speech,
                           "utterance_index": index})
    if len({item["character"] for item in result}) < 2:
        raise ValueError("Need two speaking characters")
    return result


def make_system(book, character, plot, conversation):
    descriptions = [c.get("description", "") for c in plot.get("key_characters", [])
                    if c.get("name", c.get("character")) == character]
    description = next((d.strip() for d in descriptions if isinstance(d, str) and d.strip()), "")
    scenario = conversation.get("scenario", "")
    if not isinstance(scenario, str):
        raise ValueError("Invalid scenario")
    return (f"You are {character} from {book}.\n"
            + (f"Character: {description}\n" if description else "")
            + (f"Scene: {scenario.strip()}\n" if scenario.strip() else "")
            + f"Reply only as {character}. Use speech and optional actions in parentheses. End after your turn.")


def character_exchanges(utterances, character):
    """Each pair ends with a complete assistant turn; original order is preserved."""
    turns = []
    for item in utterances:
        role = "assistant" if item["character"] == character else "user"
        text = f"{item['character']}: {item['content']}"
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"] += "\n" + (item["content"] if role == "assistant" else text)
            turns[-1]["end"] = item["utterance_index"]
        else:
            turns.append({"role": role, "content": text, "start": item["utterance_index"],
                          "end": item["utterance_index"]})
    if turns and turns[0]["role"] == "assistant":
        turns.insert(0, {"role": "user", "content": "Begin the conversation.",
                         "start": turns[0]["start"], "end": turns[0]["start"]})
    return [turns[i:i + 2] for i in range(0, len(turns) - 1, 2)]


def bounded_examples(system, exchanges, serializer, *, max_length=2048, max_reply=256):
    """Partition at whole exchange boundaries; never cut tokens or messages."""
    prefix = [{"role": "system", "content": system}]
    current, start, end = list(prefix), None, None
    rejected = Counter()
    result = []

    def flush():
        if len(current) > 1:
            result.append({"messages": list(current), "utterance_start": start, "utterance_end": end})

    for pair in exchanges:
        messages = [{"role": m["role"], "content": m["content"]} for m in pair]
        reply_size = len(serializer.tokenizer.encode(messages[1]["content"], add_special_tokens=False)) + 1
        if reply_size > max_reply or len(serializer.encode(prefix + messages)[0]) > max_length:
            flush()
            current, start, end = list(prefix), None, None
            rejected["oversized_exchange"] += 1
            continue
        if len(serializer.encode(current + messages)[0]) > max_length:
            flush()
            current, start, end = list(prefix), None, None
        if start is None:
            start = pair[0]["start"]
        end = pair[1]["end"]
        current.extend(messages)
    flush()
    return result, rejected
