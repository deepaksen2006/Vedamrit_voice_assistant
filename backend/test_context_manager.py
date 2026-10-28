
from context_manager import VedAmritContextManager

memory = VedAmritContextManager()

memory.remember(
    "मेरा पेट दर्द कर रहा है",
    "क्या आपको मतली या उल्टी भी हो रही है?"
)

memory.remember(
    "मुझे पनीर पसंद है",
    "बढ़िया! आपको कौन-सी डिश पसंद है?"
)

tests = [
    "मुझे क्या पसंद है?",
    "पनीर से क्या बना सकते हैं?",
]

for question in tests:
    context = memory.get_context(question)

    print("\nQuestion:", question)
    print("Context messages:", len(context))

    for message in context:
        print(message["role"], ":", message["content"])

