
import re


class VedAmritContextManager:
    """
    Phase 4.1 - VedAmrit Context Manager

    Features:
    - Topic-based conversation filtering
    - Preference recall
    - Follow-up question detection
    - Short-term conversation memory
    - Prevention of unrelated context mixing
    """

    def __init__(self, max_exchanges=5):
        self.max_exchanges = max_exchanges
        self.exchanges = []

        # Follow-up phrases
        self.followup_words = (
            "उसके", "उसका", "उसकी", "उससे",
            "इसके", "इसका", "इसकी",
            "वही", "उसी", "पहले",
            "अभी", "फिर",
            "इसके बारे में",
            "और बताओ",
            "क्या कहा था",
            "क्या बताया था",
            "याद है",
            "मैंने क्या",
            "मुझे क्या",
            "कौन सा शहर",
            "कौन-सा शहर",
        )

        # Topic keywords
        self.topics = {
            "health": (
                "पेट", "दर्द", "बुखार",
                "सिरदर्द", "उल्टी",
                "मतली", "बीमारी",
                "तबीयत", "स्वास्थ्य",
                "लक्षण", "खांसी",
                "सर्दी",
            ),

            "food": (
                "खाना", "खाया",
                "सेब", "पनीर",
                "फल", "सब्जी",
                "नाश्ता", "दाल",
                "रोटी", "चावल",
            ),

            "travel": (
                "यात्रा", "उदयपुर",
                "जयपुर", "शहर",
                "घूमने", "ट्रिप",
            ),

            "routine": (
                "सुबह", "उठना",
                "नींद", "दिनचर्या",
                "व्यायाम", "योग",
            ),
        }

        # User preference statements
        self.preference_phrases = (
            "पसंद है",
            "पसंद करता",
            "पसंद करती",
            "अच्छा लगता",
            "अच्छी लगती",
        )

        # Questions asking about preferences
        self.preference_questions = (
            "मुझे क्या पसंद",
            "मेरी पसंद",
            "मुझे क्या अच्छा लगता",
            "मुझे क्या अच्छी लगती",
        )

    # ----------------------------------------
    # Normalize Hindi text
    # ----------------------------------------

    def normalize(self, text):
        return re.sub(
            r"\s+",
            " ",
            text.strip().lower()
        )

    # ----------------------------------------
    # Detect conversation topic
    # ----------------------------------------

    def detect_topic(self, text):
        text = self.normalize(text)

        for topic, words in self.topics.items():
            if any(word in text for word in words):
                return topic

        return "general"

    # ----------------------------------------
    # Detect follow-up question
    # ----------------------------------------

    def is_followup(self, text):
        text = self.normalize(text)

        return any(
            word in text
            for word in self.followup_words
        )

    # ----------------------------------------
    # Detect user preference statement
    # ----------------------------------------

    def is_preference(self, text):
        text = self.normalize(text)

        return any(
            phrase in text
            for phrase in self.preference_phrases
        )

    # ----------------------------------------
    # Detect preference recall question
    # ----------------------------------------

    def is_preference_question(self, text):
        text = self.normalize(text)

        return any(
            phrase in text
            for phrase in self.preference_questions
        )

    # ----------------------------------------
    # Convert exchanges into Ollama messages
    # ----------------------------------------

    def _to_messages(self, exchanges):
        messages = []

        for item in exchanges:
            messages.append({
                "role": "user",
                "content": item["user"],
            })

            messages.append({
                "role": "assistant",
                "content": item["assistant"],
            })

        return messages

    # ----------------------------------------
    # Retrieve relevant context
    # ----------------------------------------

    def get_context(self, question):
        question = self.normalize(question)

        if not self.exchanges:
            return []

        current_topic = self.detect_topic(question)

        # CASE 1:
        # User asks about their preferences
        if self.is_preference_question(question):

            selected = [
                item
                for item in self.exchanges
                if self.is_preference(item["user"])
            ]

            selected = selected[-self.max_exchanges:]

            return self._to_messages(selected)

        # CASE 2:
        # Follow-up question
        if self.is_followup(question):

            if current_topic != "general":

                selected = [
                    item
                    for item in self.exchanges
                    if item["topic"] == current_topic
                ]

            else:
                # For an unspecified follow-up, use only
                # the most recent exchange instead of
                # mixing all previous topics.
                selected = self.exchanges[-1:]

            selected = selected[-self.max_exchanges:]

            return self._to_messages(selected)

        # CASE 3:
        # Same topic as the previous exchange
        previous = self.exchanges[-1]

        if (
            current_topic != "general"
            and current_topic == previous["topic"]
        ):
            return self._to_messages([previous])

        # CASE 4:
        # New or unrelated question
        return []

    # ----------------------------------------
    # Save conversation exchange
    # ----------------------------------------

    def remember(self, user_text, assistant_text):
        user_text = user_text.strip()
        assistant_text = assistant_text.strip()

        if not user_text or not assistant_text:
            return

        exchange = {
            "user": user_text,
            "assistant": assistant_text,
            "topic": self.detect_topic(user_text),
        }

        self.exchanges.append(exchange)

        # Limit short-term memory
        self.exchanges = self.exchanges[
            -self.max_exchanges:
        ]

    # ----------------------------------------
    # Clear conversation history
    # ----------------------------------------

    def reset(self):
        self.exchanges.clear()

    # ----------------------------------------
    # Debugging helper
    # ----------------------------------------

    def show_memory(self):
        print("\n========== VEDAMRIT MEMORY ==========")

        if not self.exchanges:
            print("No conversation history.")
            return

        for index, item in enumerate(
            self.exchanges,
            start=1
        ):
            print(f"\nExchange {index}")
            print("Topic:", item["topic"])
            print("User:", item["user"])
            print("Assistant:", item["assistant"])

        print("=====================================\n")
