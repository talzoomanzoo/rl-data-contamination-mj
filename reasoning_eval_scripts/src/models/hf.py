import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class CasualLM:
    def __init__(self, model_path, arch=None, use_vllm=False, max_tokens=1024):
        self.model_path = model_path
        self.arch = arch
        self.use_vllm = use_vllm
        self.max_tokens = max_tokens

        if use_vllm:
            from vllm import LLM  # type: ignore

            self.model = LLM(model=model_path)
            self.tokenizer = None
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype="auto", device_map="auto"
            )
            self.model.eval()

    def query(self, prompt, temperature=0.0):
        if self.use_vllm:
            raise RuntimeError("Use batch_decode_vllm with vLLM-enabled models.")

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
            )

        gen_ids = outputs[0][input_ids.shape[-1] :]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True)
