from __future__ import annotations
from mlx_lm import load, generate
from dotenv import load_dotenv
import os

load_dotenv()

class Generator:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._model_id = os.getenv(
            "GEN_MODEL",
            "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
        )

    def load_model(self):
        print(f"Loading generator: {self._model_id}")
        self.model, self.tokenizer = load(self._model_id)
        print("Generator ready.")

    def answer(self, query: str, context_chunks: list[dict], max_tokens: int = 512) -> str:
        if self.model is None:
            raise RuntimeError("Generator not loaded — call load_model() first")

        context = "\n\n".join([
            f"[{i+1}] {chunk['text']}"
            for i, chunk in enumerate(context_chunks)
        ])

        prompt = f"""[INST] Answer the question using only the context below.
If the answer is not in the context, say "I don't have enough information."

Context:
{context}

Question: {query} [/INST]"""

        response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
        )
        return response.strip()

generator = Generator()
