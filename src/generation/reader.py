"""
Document reader / Question-answering system.
Uses HuggingFace models - supports extractive QA and generative LLM.
"""

from typing import List, Optional, Tuple

from transformers import pipeline
from transformers.generation import StoppingCriteria, StoppingCriteriaList


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
        return f"""You are a factual QA system. The context below may contain multiple separate passages (split by ---). Use only the single most relevant passage to answer. Give one short, direct answer. Do not combine or mix information from different passages. No intro phrases, no reasoning, no follow-up questions.

Context:
{context}

Question: {question}

Answer:"""

    # Stop generation when model starts repeating prompt-like text or next question
    STOP_STRINGS = (
        "\n\nQuestion:",
        "\nQuestion:",
        "Given the following context",
        "Given the context",
        "Given the text",
        "Given the passage",
        "answer the following questions",
        "Answer the following questions",
        "Answer the following question",
        "the following questions:",
        "the following question:",
    )

    class _StopStringsCriteria(StoppingCriteria):
        def __init__(self, tokenizer, prompt_length: int, stop_strings: Tuple[str, ...]):
            self.tokenizer = tokenizer
            self.prompt_length = prompt_length
            self.stop_strings = stop_strings

        def __call__(self, input_ids, scores) -> bool:
            # Decode only the generated part (after prompt)
            gen_ids = input_ids[0][self.prompt_length:]
            if gen_ids.shape[0] == 0:
                return False
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            return any(s in text for s in self.stop_strings)

    # Phrases to strip from the start of model output (add more as needed)
    ANSWER_PREFIXES_TO_STRIP = [
        "Based on the context provided, ",
        "Based on the context, ",
        "According to the context, ",
        "Given the text, ",
        "Given the passage, ",
        "Based on the text, ",
    ]

    # All phrases that indicate start of garbage (truncate before earliest)
    TRUNCATE_AT_PHRASES = (
        "\n\nQuestion:",
        "\nQuestion:",
        "Given the following context",
        "Given the context",
        "Given the text",
        "Given the passage",
        "answer the following questions",
        "Answer the following questions",
        "Answer the following question",
        "the following questions:",
        "the following question:",
    )

    def _clean_answer(self, raw: str) -> str:
        """Remove known preamble phrases and truncate at spurious suffixes."""
        text = raw.strip()
        # Strip leading preamble phrases
        for prefix in self.ANSWER_PREFIXES_TO_STRIP:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
                break
        # Cut at earliest occurrence of any prompt-like / continuation phrase
        cut_at = len(text)
        for phrase in self.TRUNCATE_AT_PHRASES:
            idx = text.find(phrase)
            if idx != -1 and idx < cut_at:
                cut_at = idx
        text = text[:cut_at].strip()
        return text

    def answer(
        self,
        question: str,
        context: str,
        max_new_tokens: int = 64,
    ) -> str:
        prompt = self._build_prompt(question, context[:4000])  # Truncate long context
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_length = inputs["input_ids"].shape[1]
        stopping_criteria = StoppingCriteriaList([
            self._StopStringsCriteria(self.tokenizer, prompt_length, self.STOP_STRINGS)
        ])
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria,
        )
        generated = self.tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
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
