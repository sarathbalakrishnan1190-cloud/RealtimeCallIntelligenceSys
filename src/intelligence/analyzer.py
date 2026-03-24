import os
from groq import Groq

class LLMAnalyzer:
    def __init__(self):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.system_prompt = """
        You are an expert Call Center Intelligence AI. 
        Read the given transcript and provide a structured JSON response.
        Your response MUST be exclusively valid JSON with these exact keys:
        1. "Summary": A 1-2 sentence summary of the call.
        2. "Sentiment": One word describing the customer's sentiment (e.g., Angry, Happy, Neutral, Confused).
        3. "ActionItems": A list of bullet points of next steps discussed in the call.
        Do NOT wrap the JSON in markdown code blocks. Output pure JSON.
        """

    def analyze(self, transcript_text: str) -> str:
        if not transcript_text:
            return "Error: No transcript provided."

        print("\n--- Sending to Intelligence Layer (Groq) ---")

        try:
            completion = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Transcript for analysis:\n\n{transcript_text}"}
                ],
                temperature=0,
            )

            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error during Groq LLM Analysis: {e}")
            return None
