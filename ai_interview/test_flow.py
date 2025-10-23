import os
from typing import TypedDict
from langchain.prompts import PromptTemplate
from langchain_google_vertexai import ChatVertexAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# === Environment Config ===
SERVICE_ACCOUNT_KEY = os.path.abspath("C:/Users/SreejaTadakanti/OneDrive - Javaji Enterprises/Desktop/AI_ML_Impacteers/ai-ml-xooper/xooper/xooper.json")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_KEY
PROJECT_ID = "xooper-450012"
LOCATION = "us-central1"

# === LLM Setup ===
def get_llm():
    return ChatVertexAI(
        model="gemini-2.0-flash-001",
        project=PROJECT_ID,
        location=LOCATION,
        temperature=0.2,
        convert_system_message_to_human=True,
    )

llm = get_llm()

# === Prompts ===

intent_prompt = PromptTemplate.from_template("""
You are an expert intent classifier for technical interviews.

Classify the candidate's response into exactly one of these intents:

1. WELCOME - Greeting or acknowledging the start of the interview
2. ANSWER - Substantial answer to the question
3. CLARIFY - Asking for clarification or rephrasing
4. SKIP - Wants to skip the question
5. OFFTOPIC - Talking about unrelated topics
6. FEEDBACK - Giving or asking feedback
7. END - Wants to end the interview

Current Question:
{question}

Candidate Response:
{response}

Output ONLY the intent name in uppercase: WELCOME, ANSWER, CLARIFY, SKIP, OFFTOPIC, FEEDBACK, END
""")

answer_evaluation_prompt = PromptTemplate.from_template("""
You are an technical evaluator. Evaluate score of the candidate's response to the question out of 10

Criteria:
- Relevance to the question
- Depth of knowledge
                                                        
                                                        

Output ONLY the evaluation score as:
evaluation_score: <score>

Question:
{question}

Answer:
{response}
""")

clarify_prompt = PromptTemplate.from_template("""
You are a helpful technical interviewer. A candidate asked for clarification.

Your task is to rephrase the original question in simpler terms, using at most 50 words.

Do not answer the question or give examples.

Original Question: "{question}"
Candidate’s Response: "{response}"

Rephrased Question:
""")


followup_prompt = PromptTemplate.from_template("""
Generate a single follow-up question to probe deeper understanding.

Original Question:
{question}

Candidate's Answer:
{response}

Follow-up Question:
""")

# === Typing ===
class InterviewState(TypedDict):
    transcript: str
    intent: str
    response: str
    question: str
    question_index: int
    clarify_count: int
    is_followup: bool
    phase: str

# === Questions ===
DYNAMIC_QUESTIONS = [
    "Explain the difference between Python's list and tuple.",
    "Describe how you would optimize a slow database query.",
    "What is the CAP theorem and how does it affect distributed systems design?",
    "Walk me through your approach to debugging a production outage.",
    "How would you design a rate limiter for an API?",
]

# === Initialization ===
def initialize_state() -> InterviewState:
    return InterviewState(
        transcript="",
        intent="",
        response="👋 Welcome to the interview! To begin, please introduce yourself.",
        question="",
        question_index=0,
        clarify_count=0,
        is_followup=False,
        phase="INTRO",
    )

# === Intent Classification ===
def classify_intent(state: InterviewState) -> InterviewState:
    prompt = intent_prompt.format(
        question=state["question"] or "(Introduction phase)",
        response=state["transcript"]
    )
    
    result = llm.invoke(prompt).content.strip().upper()
    print("[DEBUG] Classified Intent:", result)
    state["intent"] = result
    return state

# === Answer Evaluation ===
def evaluate_answer(state: InterviewState) -> int:
    prompt = answer_evaluation_prompt.format(
        question=state["question"],
        response=state["transcript"]
    )
    
    evaluation = llm.invoke(prompt).content.strip()
    print("[DEBUG] Raw Evaluation Response:", evaluation)
    if evaluation.lower().startswith("evaluation_score:"):
        try:
            score = int(evaluation.split(":")[1].strip())
            print("[DEBUG] Parsed Evaluation Score:", score)
            return score
        except ValueError:
            print("[DEBUG] Failed to parse score.")
            return 0
    print("[DEBUG] Invalid evaluation response format.")
    return 0

# === Follow-up Generation ===
def generate_followup_question(state: InterviewState) -> str:
    prompt = followup_prompt.format(
        question=state["question"],
        response=state["transcript"]
    )
    followup = llm.invoke(prompt).content.strip()
    print("[DEBUG] Follow-up Question Generated:", followup)
    return followup

# === Clarification Generation ===
def generate_clarification(state: InterviewState) -> str:
    prompt = clarify_prompt.format(
        question=state["question"],
        response=state["transcript"]
    )
    clarification = llm.invoke(prompt).content.strip()
    print("[DEBUG] Clarification Generated:", clarification)
    return clarification

# === Process Intent ===
def process_intent(state: InterviewState) -> InterviewState:
    intent = state["intent"]
    idx = state.get("question_index", 0)
    total = len(DYNAMIC_QUESTIONS)
    phase = state.get("phase", "TECHNICAL")

    def move_to_next_question(s):
        next_idx = s["question_index"] + 1
        if next_idx >= total:
            s.update({
                "response": "✅ Interview completed.",
                "question": "",
                "phase": "COMPLETED"
            })
        else:
            q = DYNAMIC_QUESTIONS[next_idx]
            s.update({
                "response": f"Question {next_idx+1}/{total}: {q}",
                "question": q,
                "question_index": next_idx,
                "phase": "TECHNICAL",
                "clarify_count": 0,
                "is_followup": False,
            })
        return s

    print("[DEBUG] Processing Intent:", intent, "Phase:", phase)

    if intent == "END":
        state["response"] = "✅ Interview ended. Thank you!"
        return state

    if phase == "INTRO":
        state.update({
            "response": "Thanks for introducing yourself. Could you describe your recent projects?",
            "phase": "PROJECTS",
        })
        return state

    if phase == "PROJECTS":
        state.update({
            "response": "Great. What was your main contribution to these projects?",
            "phase": "PROJECTS_FOLLOWUP",
        })
        return state

    if phase == "PROJECTS_FOLLOWUP":
        q = DYNAMIC_QUESTIONS[0]
        state.update({
            "response": f"Thanks! Let's start the technical questions.\n\nQuestion 1/{total}: {q}",
            "question": q,
            "question_index": 0,
            "phase": "TECHNICAL",
            "clarify_count": 0,
            "is_followup": False,
        })
        return state

    if phase == "TECHNICAL":
        if intent == "ANSWER":
            if state.get("is_followup", False):
                return move_to_next_question(state)
            score = evaluate_answer(state)
            if score >= 5:
                followup = generate_followup_question(state)
                state.update({
                    "response": f"Good. Follow-up: {followup}",
                    "question": followup,
                    "is_followup": True,
                })
            else:
                state = move_to_next_question(state)
            return state

        if intent == "CLARIFY":
            if state["clarify_count"] < 1:
                clarification = generate_clarification(state)
                state.update({
                    "response": f"Clarification: {clarification}\n\nPlease answer again.",
                    "clarify_count": state["clarify_count"] + 1,
                })
            else:
                state = move_to_next_question(state)
            return state

        if intent == "SKIP":
            return move_to_next_question(state)

        if intent in ("OFFTOPIC", "FEEDBACK"):
            state["response"] = "Please keep responses relevant to the question."
            return state

        state["response"] = "Sorry, I didn't understand. Could you rephrase?"
        return state

    state["response"] = "I'm not sure how to handle that response."
    return state

# === LangGraph Setup ===
def build_graph() -> StateGraph:
    checkpointer = MemorySaver()
    builder = StateGraph(state_schema=InterviewState)
    builder.add_node("classify_intent", classify_intent)
    builder.add_node("process_intent", process_intent)
    builder.set_entry_point("classify_intent")
    builder.add_edge("classify_intent", "process_intent")
    builder.add_edge("process_intent", END)
    return builder.compile(checkpointer=checkpointer)

# === Main ===
def main():
    print("=== AI Interview Test ===")
    state = initialize_state()
    graph = build_graph()

    while True:
        print("\nBot:", state["response"])
        user_input = input("You: ").strip()
        if not user_input:
            print("Empty input. Exiting.")
            break

        state_input = {
            "transcript": user_input,
            "intent": "",
            "response": "",
            "question": state["question"],
            "question_index": state["question_index"],
            "clarify_count": state["clarify_count"],
            "is_followup": state["is_followup"],
            "phase": state["phase"],
        }

        result = graph.invoke(state_input, config={"configurable": {"thread_id": "test"}})
        state.update(result)

        print("[DEBUG] Updated State:", state)

        if state["phase"] == "COMPLETED":
            print("\n✅ Interview completed.")
            break

if __name__ == "__main__":
    main()
