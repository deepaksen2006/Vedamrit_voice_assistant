from ollama import chat

MODEL = "qwen3:1.7b"

SYSTEM_PROMPT = """
You are VedAmrit, a helpful multilingual AI assistant.

Rules:
1. Respond in the user's preferred language.
2. Speak clearly and simply.
3. You can explain general concepts.
4. Do not invent medical evidence or citations.
5. Do not prescribe medicines or dosages.
6. For Ayurvedic medical claims, explain that
   verified knowledge retrieval is not yet connected.
7. Ask for clarification if the question is unclear.
"""

def ask_assistant(question):
    response = chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        options={"temperature": 0.3},
    )

    return response["message"]["content"]


if __name__ == "__main__":
    print("VedAmrit AI Assistant")
    print("Type 'exit' to stop.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        try:
            answer = ask_assistant(question)
            print("\nVedAmrit:", answer, "\n")
        except Exception as error:
            print("Error:", error)
