"""
Document reader / Question-answering system.
Uses HuggingFace models - supports extractive QA and generative LLM.
"""

from typing import List, Optional

from transformers import pipeline


def _get_qa_device(use_gpu: bool) -> int:
    """Return device id: 0 for GPU, -1 for CPU."""
    if not use_gpu:
        return -1
    try:
        import torch
        return 0 if torch.cuda.is_available() else -1
    except ImportError:
        return -1


class ExtractiveReader:
    """
    Extractive QA: select span from context (e.g., BERT-based).
    Good for short, factual answers.
    """

    def __init__(
        self,
        model_name: str = "deepset/roberta-base-squad2",
        device: int = -1,
    ):
        self.qa_pipeline = pipeline(
            "question-answering",
            model=model_name,
            tokenizer=model_name,
            device=device,
        )

    def answer(
        self,
        question: str,
        context: str,
        top_k: int = 1,
    ) -> str:
        if not context.strip():
            return ""
        result = self.qa_pipeline(
            question=question,
            context=context,
            top_k=top_k,
            max_answer_len=100,
        )
        if isinstance(result, list):
            result = result[0] if result else {}
        return result.get("answer", "").strip()


class GenerativeReader:
    """
    Generative QA: use LLM to generate answer from context.
    Supports HuggingFace models (e.g., Llama2, Mistral) via transformers.
    You can also integrate Ollama/llama.cpp for local inference.
    """

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        use_4bit: bool = False,
        device_map: str = "auto",
    ):
        """
        Load generative model. For 7B+ models on GPU, use_4bit recommended for memory.
        Requires bitsandbytes when use_4bit=True. device_map='auto' uses GPU when available.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        load_kwargs = {"device_map": device_map}
        if use_4bit:
            try:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype="float16",
                )
            except ImportError:
                pass  # fallback to full precision
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    def _build_prompt(self, question: str, context: str) -> str:
        return f"""Based on the following context, answer the question directly and concisely without any introductory phrases, explanations, or pleasantries. Do not output the reasoning steps, do not generate follow-up questions.
        If the information is not provided in the context, output exactly 'Not found' and nothing else. 

Context:
{context}

Question: {question}

Answer:"""

    # Phrases to strip from the start of model output (add more as needed)
    ANSWER_PREFIXES_TO_STRIP = [
        "Based on the context provided, ",
        "Based on the context, ",
        "According to the context, ",
    ]

    def _clean_answer(self, raw: str) -> str:
        """Remove known preamble phrases from the model output."""
        text = raw.strip()
        for prefix in self.ANSWER_PREFIXES_TO_STRIP:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
                break
        return text

    def answer(
        self,
        question: str,
        context: str,
        max_new_tokens: int = 64,
    ) -> str:
        prompt = self._build_prompt(question, context[:4000])  # Truncate long context
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return self._clean_answer(generated)


class SimpleReader:
    """
    Placeholder reader that returns empty or first-sentence heuristic.
    Use this for debugging pipeline before connecting a real QA model.
    """

    def answer(self, question: str, context: str, **kwargs) -> str:
        if not context:
            return ""
        # Placeholder: return first sentence (you would replace with real model)
        first_sent = context.split(".")[0].strip()
        return first_sent[:100] if first_sent else ""
