def format_guidelines_for_prompt(guidelines):
    parts = []

    voice = guidelines.get("voice_attributes", [])
    if voice:
        parts.append(f"Voice attributes: {', '.join(voice)}")

    formality = guidelines.get("formality", "")
    if formality:
        parts.append(f"Formality: {formality}")

    structure = guidelines.get("sentence_structure", {})
    if structure.get("preferred"):
        parts.append(f"Preferred sentence structure: {structure['preferred']}")
    if structure.get("avoid"):
        parts.append(f"Avoid in sentence structure: {structure['avoid']}")

    vocab = guidelines.get("vocabulary", {})
    if vocab.get("preferred_words"):
        parts.append(f"Preferred vocabulary: {', '.join(vocab['preferred_words'])}")
    if vocab.get("avoid_words"):
        parts.append(f"Vocabulary to avoid: {', '.join(vocab['avoid_words'])}")
    if vocab.get("register"):
        parts.append(f"Language register: {vocab['register']}")

    emotional = guidelines.get("emotional_register", "")
    if emotional:
        parts.append(f"Emotional register: {emotional}")

    punctuation = guidelines.get("punctuation_preferences", "")
    if punctuation:
        parts.append(f"Punctuation preferences: {punctuation}")

    examples = guidelines.get("examples", {})
    if examples.get("on_brand"):
        parts.append("On-brand examples:")
        for ex in examples["on_brand"]:
            parts.append(f"  - {ex}")
    if examples.get("off_brand"):
        parts.append("Off-brand examples (avoid writing like this):")
        for ex in examples["off_brand"]:
            parts.append(f"  - {ex}")

    return "\n".join(parts)
