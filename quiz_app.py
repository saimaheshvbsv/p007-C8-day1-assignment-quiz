import json
import os
import re
import tkinter as tk
from tkinter import messagebox
from typing import Dict, List

from openai import OpenAI


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


class QuizApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("AI Quiz Master")
        self.root.geometry("900x650")
        self.root.minsize(800, 600)

        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
        self.topic = ""
        self.questions: List[Dict] = []
        self.current_index = 0
        self.score = 0

        self.setup_ui()

    def setup_ui(self):
        self.root.configure(bg="#eef4ff")

        self.card = tk.Frame(self.root, bg="#ffffff", padx=30, pady=25)
        self.card.pack(fill="both", expand=True, padx=25, pady=25)

        title = tk.Label(
            self.card,
            text="AI Quiz Master",
            font=("Segoe UI", 24, "bold"),
            bg="#ffffff",
            fg="#1e3a8a",
        )
        title.pack(pady=(0, 20))

        self.setup_panel = tk.Frame(self.card, bg="#ffffff")
        self.setup_panel.pack(fill="x")

        tk.Label(self.setup_panel, text="Quiz Topic", font=("Segoe UI", 11, "bold"), bg="#ffffff", fg="#1f2937").pack(anchor="w")
        self.topic_entry = tk.Entry(self.setup_panel, width=80, font=("Segoe UI", 11), bd=1, relief="solid")
        self.topic_entry.pack(fill="x", pady=(5, 20))

        self.start_button = tk.Button(
            self.setup_panel,
            text="Start Quiz",
            command=self.start_quiz,
            width=18,
            height=2,
            bg="#2563eb",
            fg="#ffffff",
            font=("Segoe UI", 11, "bold"),
            relief="flat",
            cursor="hand2",
            bd=0,
            padx=12,
        )
        self.start_button.pack(anchor="center")

        self.quiz_panel = tk.Frame(self.card, bg="#ffffff")
        self.quiz_panel.pack(fill="both", expand=True)
        self.quiz_panel.pack_forget()

        self.quiz_canvas = tk.Canvas(self.quiz_panel, bg="#ffffff", highlightthickness=0)
        self.quiz_scrollbar = tk.Scrollbar(self.quiz_panel, orient="vertical", command=self.quiz_canvas.yview)
        self.quiz_canvas.configure(yscrollcommand=self.quiz_scrollbar.set)

        self.quiz_canvas.pack(side="left", fill="both", expand=True)
        self.quiz_scrollbar.pack(side="right", fill="y")

        self.quiz_content = tk.Frame(self.quiz_canvas, bg="#ffffff")
        self.quiz_canvas.create_window((0, 0), window=self.quiz_content, anchor="nw", width=760)

        self.quiz_content.bind("<Configure>", self.on_quiz_content_configure)
        self.quiz_canvas.bind("<Configure>", self.on_canvas_resize)

        self.question_label = tk.Label(
            self.quiz_content,
            text="",
            justify="left",
            wraplength=720,
            font=("Segoe UI", 17, "bold"),
            bg="#ffffff",
            fg="#111827",
        )
        self.question_label.pack(anchor="w", pady=(10, 15), fill="x")

        self.option_buttons = {}
        for option in ["A", "B", "C", "D"]:
            btn = tk.Button(
                self.quiz_content,
                text="",
                command=lambda letter=option: self.submit_answer(letter),
                width=90,
                height=2,
                anchor="w",
                justify="left",
                bg="#f8fafc",
                fg="#0f172a",
                font=("Segoe UI", 11),
                relief="solid",
                bd=1,
                pady=8,
                activebackground="#e2e8f0",
                activeforeground="#0f172a",
            )
            btn.pack(fill="x", pady=5)
            self.option_buttons[option] = btn

        self.result_label = tk.Label(
            self.quiz_content,
            text="",
            font=("Segoe UI", 12, "bold"),
            bg="#ffffff",
            fg="#16a34a",
            wraplength=720,
            justify="left",
        )
        self.result_label.pack(anchor="w", pady=(18, 10), fill="x")

        self.action_buttons_frame = tk.Frame(self.quiz_content, bg="#ffffff")
        self.action_buttons_frame.pack(fill="x", pady=(8, 10))

        self.next_button = tk.Button(
            self.action_buttons_frame,
            text="Next Question",
            command=self.next_question,
            width=18,
            height=2,
            bg="#10b981",
            fg="#ffffff",
            font=("Segoe UI", 11, "bold"),
            relief="flat",
            cursor="hand2",
            bd=0,
        )
        self.next_button.pack(anchor="center")
        self.next_button.pack_forget()

        self.restart_button = tk.Button(
            self.action_buttons_frame,
            text="Start New Quiz",
            command=self.reset_to_setup,
            width=18,
            height=2,
            bg="#2563eb",
            fg="#ffffff",
            font=("Segoe UI", 11, "bold"),
            relief="flat",
            cursor="hand2",
            bd=0,
        )
        self.restart_button.pack_forget()

        self.exit_button = tk.Button(
            self.action_buttons_frame,
            text="Exit Quiz",
            command=self.exit_quiz,
            width=18,
            height=2,
            bg="#ef4444",
            fg="#ffffff",
            font=("Segoe UI", 11, "bold"),
            relief="flat",
            cursor="hand2",
            bd=0,
        )
        self.exit_button.pack_forget()

    def on_quiz_content_configure(self, event):
        self.quiz_canvas.configure(scrollregion=self.quiz_canvas.bbox("all"))

    def on_canvas_resize(self, event):
        self.quiz_canvas.itemconfig(self.quiz_canvas.find_all()[0], width=event.width - 20)

    def start_quiz(self):
        topic = self.topic_entry.get().strip()

        if not topic:
            messagebox.showerror("Missing topic", "Please enter a topic for the quiz.")
            return

        api_key = self.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            messagebox.showerror("Missing API key", "Please set the GEMINI_API_KEY environment variable before starting the quiz.")
            return

        try:
            self.api_key = api_key
            os.environ["GEMINI_API_KEY"] = api_key
            self.topic = topic
            self.questions = generate_questions(topic, api_key)
            self.current_index = 0
            self.score = 0
            self.setup_panel.pack_forget()
            self.quiz_panel.pack(fill="both", expand=True)
            self.show_question()
        except Exception as exc:
            messagebox.showerror("Quiz generation failed", str(exc))
            self.topic = topic
            self.result_label.config(text="")

    def show_question(self):
        self.next_button.pack_forget()
        self.restart_button.pack_forget()
        self.exit_button.pack_forget()
        self.result_label.config(text="")

        for btn in self.option_buttons.values():
            btn.pack(fill="x", pady=5)
            btn.config(text="", state="normal", bg="#f8fafc", fg="#0f172a")

        if self.current_index >= len(self.questions):
            self.finish_quiz()
            return

        question = self.questions[self.current_index]
        options = question["options"]

        self.question_label.config(text=f"Question {self.current_index + 1}/{len(self.questions)}\n\n{question['question']}", justify="left")

        for letter in ["A", "B", "C", "D"]:
            btn = self.option_buttons[letter]
            btn.config(text=f"{letter}. {options[letter]}", bg="#f8fafc", fg="#0f172a", state="normal")

    def submit_answer(self, selected_option: str):
        if self.current_index >= len(self.questions):
            return

        question = self.questions[self.current_index]
        correct_option = question["correct_option"].upper()
        options = question["options"]

        for letter in ["A", "B", "C", "D"]:
            btn = self.option_buttons[letter]
            if letter == correct_option:
                btn.config(bg="#bbf7d0", fg="#14532d")
            elif letter == selected_option:
                btn.config(bg="#fecaca", fg="#7f1d1d")
            btn.config(state="disabled")

        if selected_option == correct_option:
            self.score += 1
            self.result_label.config(text="Correct! ✅", fg="#16a34a")
        else:
            self.result_label.config(
                text=f"Incorrect ❌ The correct answer is {correct_option}: {options[correct_option]}",
                fg="#dc2626",
            )

        self.next_button.pack(anchor="center", pady=(5, 10))

    def next_question(self):
        self.current_index += 1
        self.show_question()

    def finish_quiz(self):
        self.question_label.config(
            text=f"Quiz Complete!\n\nTopic: {self.topic}\n\nFinal Score: {self.score}/{len(self.questions)}",
            justify="center",
            fg="#111827",
        )
        self.result_label.config(text=self.get_score_message(), fg="#1d4ed8")

        for btn in self.option_buttons.values():
            btn.pack_forget()
            btn.config(text="", state="disabled")

        self.next_button.pack_forget()
        self.restart_button.pack(side="left", padx=8, pady=6)
        self.exit_button.pack(side="left", padx=8, pady=6)

    def reset_to_setup(self):
        self.current_index = 0
        self.score = 0
        self.questions = []
        self.question_label.config(text="")
        self.result_label.config(text="")
        self.restart_button.pack_forget()
        self.exit_button.pack_forget()

        for btn in self.option_buttons.values():
            btn.pack_forget()
            btn.config(text="", state="normal")

        self.quiz_panel.pack_forget()
        self.setup_panel.pack(fill="x")

    def exit_quiz(self):
        self.root.destroy()

    def get_score_message(self) -> str:
        total = len(self.questions)
        if total == 0:
            return "No questions were generated."
        if self.score == total:
            return "Excellent! You got every question correct."
        if self.score >= total * 0.7:
            return "Great job! You did very well."
        if self.score >= total * 0.5:
            return "Nice effort! Keep practicing."
        return "A good start. Try another round on the same topic."


def main() -> None:
    root = tk.Tk()
    app = QuizApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
