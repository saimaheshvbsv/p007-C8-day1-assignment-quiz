import json
import os
import re
from typing import Dict, List

from flask import Flask, render_template, request, redirect, url_for
from openai import OpenAI

app = Flask(__name__)

MODEL_NAME = "gemini-2.5-flash-lite"


def build_client(api_key: str) -> OpenAI:
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


def generate_questions(topic: str, api_key: str, count: int = 10) -> List[Dict]:
    topic = topic.strip()
    if not topic:
        raise ValueError("Enough information is not available to frame a quiz.")

    client = build_client(api_key)
    prompt = f"""
You are a quiz master creating a multiple-choice quiz on the topic: {topic}.
Generate exactly {count} questions. Each question must have:
- question: a clear question text
- options: a dictionary with exactly 4 keys: A, B, C, D and their values as strings
- correct_option: one of A, B, C, D

Rules:
- Only generate a quiz if the topic is sufficiently specific and knowledge-based.
- If the topic is too vague, too broad, or not a proper subject with enough factual content, return a JSON object with: {{"error": "Enough information is not available to frame a quiz."}}
- Do not invent facts or create questions for undefined topics.
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
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=4000,
    )

    content = response.choices[0].message.content
    if not content:
        raise ValueError("Enough information is not available to frame a quiz.")

    data = extract_json(content)
    if "error" in data:
        raise ValueError(str(data["error"]))
    if "questions" not in data or not isinstance(data["questions"], list):
        raise ValueError("Enough information is not available to frame a quiz.")
    if len(data["questions"]) < count:
        raise ValueError("Enough information is not available to frame a quiz.")

    return data["questions"][:count]


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        topic = request.form.get("topic", "").strip()
        if not topic:
            return render_template("index.html", error="Please enter a quiz topic.")

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            return render_template("index.html", error="Missing API key. Set GEMINI_API_KEY in the environment.")

        try:
            questions = generate_questions(topic, api_key)
            return render_template(
                "quiz.html",
                topic=topic,
                questions=questions,
                current_index=0,
                score=0,
                question=questions[0],
                answers={},
                total_questions=len(questions),
            )
        except ValueError as exc:
            return render_template("index.html", error=str(exc))

    return render_template("index.html", error=None)


@app.route("/quiz", methods=["POST"])
def quiz():
    topic = request.form.get("topic", "")
    current_index = int(request.form.get("current_index", 0))
    score = int(request.form.get("score", 0))
    questions_json = request.form.get("questions_json", "[]")

    try:
        questions = json.loads(questions_json)
    except Exception:
        return render_template("index.html", error="Unable to load quiz data.")

    selected_option = request.form.get("selected_option", "")
    if selected_option:
        current_question = questions[current_index]
        if selected_option == current_question["correct_option"].upper():
            score += 1

    next_index = current_index + 1
    if next_index >= len(questions):
        return render_template(
            "result.html",
            topic=topic,
            score=score,
            total_questions=len(questions),
        )

    return render_template(
        "quiz.html",
        topic=topic,
        questions=questions,
        current_index=next_index,
        score=score,
        question=questions[next_index],
        total_questions=len(questions),
        answers={},
    )


@app.route("/new", methods=["GET"])
def new_quiz():
    return redirect(url_for("home"))


if __name__ == "__main__":
    app.run(debug=True)
