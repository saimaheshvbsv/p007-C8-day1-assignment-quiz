import json
import os
import re
from typing import Dict, List

from openai import OpenAI


def get_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key

    api_key = input("Enter your Gemini API key: ").strip()
    if not api_key:
        raise ValueError("No API key provided.")
    return api_key


def build_client() -> OpenAI:
    api_key = get_api_key()
    return OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
    )


def extract_json(text: str) -> Dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL | re.IGNORECASE)
        if match:
            cleaned = match.group(1).strip()
    return json.loads(cleaned)


def generate_questions(topic: str, client: OpenAI, count: int = 10) -> List[Dict]:
    prompt = f"""
You are a quiz master creating a multiple-choice quiz on the topic: {topic}.
Generate exactly {count} questions. Each question must have:
- question: a clear question text
- options: a dictionary with exactly 4 keys: A, B, C, D and their values as strings
- correct_option: one of A, B, C, D

Rules:
- Make the questions accurate and educational.
- Ensure there is exactly one correct answer for each question.
- Keep the difficulty appropriate for a general audience.
- Return valid JSON only in this format:
{{
  "questions": [
    {{
      "question": "...",
      "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
      "correct_option": "A"
    }}
  ]
}}
"""

    response = client.chat.completions.create(
        model="gemini-2.5-flash-lite",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=4000,
    )

    content = response.choices[0].message.content
    if not content:
        raise ValueError("The model returned an empty response.")

    data = extract_json(content)
    if "questions" not in data or not isinstance(data["questions"], list):
        raise ValueError("The response did not contain a valid questions list.")

    return data["questions"]


def run_quiz(topic: str) -> None:
    client = build_client()
    questions = generate_questions(topic, client)

    score = 0
    print(f"\nStarting quiz on: {topic}")
    print("Answer each question with A, B, C, or D.\n")

    for index, question in enumerate(questions, start=1):
        print(f"Question {index}/{len(questions)}")
        print(question["question"])

        options = question["options"]
        for key in ["A", "B", "C", "D"]:
            print(f"{key}. {options[key]}")

        while True:
            answer = input("Your answer (A/B/C/D): ").strip().upper()
            if answer in {"A", "B", "C", "D"}:
                break
            print("Please enter only A, B, C, or D.")

        correct_option = question["correct_option"].upper()
        if answer == correct_option:
            print("Correct! ✅")
            score += 1
        else:
            print(f"Incorrect ❌ The correct answer is {correct_option}. {options[correct_option]}")

        print("-" * 40)

    print(f"\nQuiz complete! Final score: {score}/{len(questions)}")
    if score == len(questions):
        print("Excellent! You got every question right.")
    elif score >= (len(questions) * 0.7):
        print("Great job! You did very well.")
    elif score >= (len(questions) * 0.5):
        print("Nice effort! Keep practicing.")
    else:
        print("A good start. Try another round on the same topic.")


def main() -> None:
    try:
        topic = input("Enter a quiz topic: ").strip()
        if not topic:
            print("Topic cannot be empty.")
            return

        run_quiz(topic)
    except Exception as exc:
        print(f"An error occurred: {exc}")
        print("Check that your Gemini API key is valid and that the model name is available.")


if __name__ == "__main__":
    main()
