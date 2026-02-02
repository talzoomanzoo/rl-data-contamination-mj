class Claude:
    def __init__(self, model_path, max_tokens=1024):
        raise RuntimeError(
            "Anthropic models are not supported in this repo. "
            "Use a local HF model or add an Anthropic client."
        )
