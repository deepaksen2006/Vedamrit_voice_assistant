from ollama import chat
import time

MODEL = "qwen3:4b"

SYSTEM_PROMPT = """
तुम VedAmrit हो, एक हिंदी AI वॉइस असिस्टेंट।
हमेशा सरल, स्वाभाविक हिंदी में जवाब दो।
जवाब अधिकतम दो छोटे वाक्यों में दो।
तुम AI हो। तुम खाना नहीं खाते।
उपयोगकर्ता की बात ध्यान से समझो।
बातचीत का संदर्भ याद रखो।
"""

messages = [
    {"role": "system", "content": SYSTEM_PROMPT}
]

print("VedAmrit Qwen3:4b Test")
print("Type exit to quit.\n")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() == "exit":
        break

    messages.append({
        "role": "user",
        "content": user_input
    })

    start = time.perf_counter()

    response = chat(
        model=MODEL,
        messages=messages,
        think=False,
        options={
            "temperature": 0.4,
            "num_predict": 100,
            "repeat_penalty": 1.15,
            "num_ctx": 2048
        }
    )

    elapsed = time.perf_counter() - start
    answer = response.message.content.strip()

    print(f"VedAmrit: {answer}")
    print(f"Response time: {elapsed:.2f} seconds\n")

    messages.append({
        "role": "assistant",
        "content": answer
    })

    # Keep the system prompt and recent conversation.
    messages = [messages[0]] + messages[1:][-10:]
