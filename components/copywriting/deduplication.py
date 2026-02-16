import json
import math
from openai import OpenAI
from langflow.custom import Component
from langflow.io import StrInput, FloatInput, DropdownInput, SecretStrInput, MultilineInput, Output
from langflow.schema.message import Message


def _char_ngrams(text, n=3):
    text = text.lower().strip()
    if len(text) < n:
        return {text}
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def _jaccard(set_a, set_b):
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def _copy_to_text(copy_dict):
    return " ".join(copy_dict.values())


def _cosine_similarity(vec_a, vec_b):
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def deduplicate_ngram(copies, threshold=0.85, ngram_size=3):
    if not copies:
        return copies

    ngram_sets = [_char_ngrams(_copy_to_text(c), ngram_size) for c in copies]
    selected = [0]

    for i in range(1, len(copies)):
        is_duplicate = False
        for j in selected:
            similarity = _jaccard(ngram_sets[i], ngram_sets[j])
            if similarity >= threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            selected.append(i)

    return [copies[i] for i in selected]


def deduplicate_embedding(copies, client, threshold=0.85, model="text-embedding-3-small"):
    if not copies:
        return copies

    texts = [_copy_to_text(c) for c in copies]

    response = client.embeddings.create(input=texts, model=model)
    embeddings = [item.embedding for item in response.data]

    selected = [0]

    for i in range(1, len(copies)):
        is_duplicate = False
        for j in selected:
            similarity = _cosine_similarity(embeddings[i], embeddings[j])
            if similarity >= threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            selected.append(i)

    return [copies[i] for i in selected]


def deduplicate_copies(copies, strategy="ngram", threshold=0.85, client=None):
    if strategy == "embedding" and client is not None:
        return deduplicate_embedding(copies, client, threshold=threshold)
    return deduplicate_ngram(copies, threshold=threshold)


class DiverseSubsetSelector(Component):
    display_name = "Diverse Subset Selector"
    description = "Removes near-duplicate copies using n-gram or embedding-based similarity"
    icon = "filter"

    inputs = [
        MultilineInput(
            name="copies_json",
            display_name="Copies JSON",
            required=True,
            info="JSON array of copy objects",
        ),
        DropdownInput(
            name="strategy",
            display_name="Strategy",
            options=["ngram", "embedding"],
            value="ngram",
            info="ngram: fast, no API calls | embedding: higher quality, uses OpenAI embeddings",
        ),
        FloatInput(
            name="similarity_threshold",
            display_name="Similarity Threshold",
            value=0.85,
            info="Copies above this similarity are considered duplicates (0.0-1.0)",
        ),
        SecretStrInput(
            name="api_key",
            display_name="OpenAI API Key",
            required=False,
            info="Required only for embedding strategy",
        ),
        StrInput(
            name="embedding_model",
            display_name="Embedding Model",
            value="text-embedding-3-small",
            info="Only used with embedding strategy",
        ),
    ]

    outputs = [
        Output(display_name="Deduplicated Copies", name="deduplicated", method="select"),
    ]

    def select(self) -> Message:
        copies = json.loads(self.copies_json)

        client = None
        if self.strategy == "embedding" and self.api_key:
            client = OpenAI(api_key=self.api_key)

        result = deduplicate_copies(
            copies,
            strategy=self.strategy,
            threshold=self.similarity_threshold,
            client=client,
        )

        output = {
            "original_count": len(copies),
            "deduplicated_count": len(result),
            "removed": len(copies) - len(result),
            "copies": result,
        }

        return Message(text=json.dumps(output, indent=2))
