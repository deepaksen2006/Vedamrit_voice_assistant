
from ollama import chat
import time

MODEL = "qwen3:4b"

messages = [
    {
        "role": "system",
        "content": (
            "You are VedAmrit, a friendly Hindi AI voice assistant. "
            "Always answer in natural Hindi using one or two short sentences. "
            "Never claim to eat, sleep, or have human experiences. "
            "Do not translate the user's question or explain your reasoning."
        )
    },
    {
        "role": "user",
        "content": "खाना खाया?\n/no_think"
    }
]

start = time.perf_counter()

response = chat(
    model=MODEL,
    messages=messages,
    think=False,
    options={
        "temperature": 0.3,
        "num_predict": 300,
        "repeat_penalty": 1.15
    }
)

print("\nCONTENT:")
print(repr(response.message.content))

print("\nTHINKING:")
print(repr(getattr(response.message, "thinking", None)))

print("\nSTOP REASON:")
print(getattr(response, "done_reason", None))

print("\nTIME:")
print(f"{time.perf_counter() - start:.2f} seconds")
