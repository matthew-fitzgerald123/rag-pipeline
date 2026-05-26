from __future__ import annotations
import asyncio
import json
from queue import Empty, Queue
from threading import Event
from typing import AsyncIterator
from mlx_lm import load, generate, stream_generate
from dotenv import load_dotenv
import os

load_dotenv()


def _build_prompt(query: str, context_chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[{i+1}] {chunk['text']}" for i, chunk in enumerate(context_chunks)
    )
    return (
        f"[INST] Answer the question using only the context below.\n"
        f"If the answer is not in the context, say \"I don't have enough information.\"\n\n"
        f"Context:\n{context}\n\nQuestion: {query} [/INST]"
    )


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
            raise RuntimeError("Generator not loaded -- call load_model() first")
        prompt = _build_prompt(query, context_chunks)
        response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
        )
        return response.strip()

    def _stream_into_queue(
        self,
        prompt: str,
        max_tokens: int,
        q: Queue,
        done: Event,
    ):
        try:
            for chunk in stream_generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
            ):
                q.put(chunk.text)
        finally:
            done.set()

    async def answer_stream(
        self,
        query: str,
        context_chunks: list[dict],
        max_tokens: int = 512,
    ) -> AsyncIterator[str]:
        if self.model is None:
            raise RuntimeError("Generator not loaded -- call load_model() first")

        prompt = _build_prompt(query, context_chunks)
        q: Queue = Queue()
        done = Event()

        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, self._stream_into_queue, prompt, max_tokens, q, done)

        while not (done.is_set() and q.empty()):
            try:
                token = await loop.run_in_executor(None, q.get, True, 0.05)
                yield json.dumps({"token": token})
            except Empty:
                continue
        yield "[DONE]"


generator = Generator()
