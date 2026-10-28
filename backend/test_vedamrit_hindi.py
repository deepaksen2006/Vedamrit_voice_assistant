
from ollama import chat
import time

MODEL = "gemma3:4b"

SYSTEM_PROMPT = """
तुम्हारा नाम वेदअमृत (VedAmrit) है।
तुम एक दोस्ताना हिंदी AI वॉइस असिस्टेंट हो।

नियम:
1. हमेशा सरल और स्वाभाविक हिंदी में जवाब दो।
2. हर जवाब केवल एक या दो छोटे वाक्यों में दो।
3. तुम AI हो, इंसान नहीं।
4. तुम खाना नहीं खाते, सोते नहीं और तुम्हारा शरीर नहीं है।
5. उपयोगकर्ता की बात समझकर सीधे जवाब दो।
6. अनावश्यक जानकारी या लंबे स्पष्टीकरण मत दो।
7. बातचीत की पिछली बातें याद रखो।

उदाहरण:
प्रश्न: तुम्हारा नाम क्या है?
उत्तर: मेरा नाम वेदअमृत है। बताइए, कैसे मदद करूँ?

प्रश्न: खाना खाया?
उत्तर: मैं AI हूँ, इसलिए खाना नहीं खाता। आपने क्या खाया?

प्रश्न: मैंने आज सेब खाया है।
उत्तर: अच्छा! सेब खाना एक अच्छी आदत है।

प्रश्न: मुझे पनीर पसंद है।
उत्तर: बढ़िया! आपको पनीर की कौन-सी डिश सबसे ज्यादा पसंद है?
"""

messages = [{"role": "system", "content": SYSTEM_PROMPT}]

print("VedAmrit Hindi Test")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() == "exit":
        break

    if not user_input:
        continue

    messages.append({
        "role": "user",
        "content": user_input
    })

    start = time.perf_counter()

    try:
        response = chat(
            model=MODEL,
            messages=messages,
            options={
                "temperature": 0.3,
                "num_predict": 120,
                "repeat_penalty": 1.15,
                "num_ctx": 2048
            }
        )

        answer = response.message.content.strip()
        elapsed = time.perf_counter() - start

        print(f"VedAmrit: {answer}")
        print(f"Time: {elapsed:.2f} seconds\n")

        messages.append({
            "role": "assistant",
            "content": answer
        })

        messages = [messages[0]] + messages[1:][-10:]

    except Exception as e:
        print("Error:", e)
