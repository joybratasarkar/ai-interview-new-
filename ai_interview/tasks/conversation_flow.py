import os, json, logging, random
from typing import TypedDict, Dict, List, Any, Annotated
from apps.ai_interview.celery_app import celery_app
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_google_vertexai import ChatVertexAI

# ----------------- Configuration -----------------
os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", os.path.abspath("xooper.json"))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_llm() -> ChatVertexAI:
    return ChatVertexAI(
        model="gemini-2.0-flash-001",
        project="xooper-450012",
        temperature=0.7,
        convert_system_message_to_human=True,
    )

llm = get_llm()
global_checkpointer = MemorySaver()

# ----------------- Types & State -----------------
typestate = Annotated[List, add_messages]
class InterviewState(TypedDict):
    messages: typestate
    room_id: str
    user_input: str
    bot_response: str
    question_idx: int
    followup_count: int
    open_chat_turns: int
    in_followup: bool
    done: bool
    last_question_context: str
    current_expression: str
    last_answer_score: int
    phase: str
    should_advance: bool
    consent_allowed: bool

# ----------------- Questions & Limits -----------------
CORE_QS = [
    "Walk me through your most recent work experience.",
    "What programming languages are you most comfortable with?",
    "Tell me about a recent project you're proud of.",
    "How do you approach solving new challenges?"
]
MAX_Q_LEN, MAX_FU_LEN, MAX_EXPR_LEN = 80, 60, 30

# ----------------- Node Implementations -----------------

def greet_node(s: InterviewState):
    return {
        "bot_response": "Hello! I’m your AI interviewer—may I begin?",
        "phase": "CHECK_CONSENT"
    }


def check_consent_node(s: InterviewState):
    resp = llm.invoke(
        f"Determine if the user consents to start the interview.\nUser: {s['user_input']}\nReturn exactly YES or NO."
    )
    ans = resp.content.strip().upper()
    if ans == "YES":
        return {"consent_allowed": True, "phase": "ASK_Q", "bot_response": "Great! Let's get started."}
    return {"consent_allowed": False, "phase": "CHECK_CONSENT", "bot_response": "No problem! Take your time. May I begin when you're ready?"}


def ask_q_node(s: InterviewState):
    q = CORE_QS[s['question_idx']]
    if len(q) > MAX_Q_LEN:
        q = q[:MAX_Q_LEN-3] + "..."
    return {
        "bot_response": q,
        "in_followup": False,
        "followup_count": 0,
        "last_question_context": q,
        "phase": "DETECT_INTENT"
    }


def detect_intent_node(s: InterviewState):
    prompt = (
        f"Classify input as ANSWER or CHAT.\nQuestion: {s['last_question_context']}\nUser: {s['user_input']}"
    )
    resp = llm.invoke(prompt)
    if resp.content.strip().upper() == 'CHAT':
        return {"phase": "OPEN_CHAT", "open_chat_turns": 0}
    return {"phase": "SCORE_ROUTE"}


def score_route_node(s: InterviewState):
    resp = llm.invoke(
        f"Rate answer 0-10.\nQuestion: {s['last_question_context']}\nAnswer: {s['user_input']}"
    )
    try:
        score = int(resp.content.strip())
    except:
        score = max(0, min(10, len(s['user_input']) // 20))
    expr = random.choice(["Excellent!", "Good to know.", "I see.", "Alright.", "Impressive!"])
    follow = (score >= 7 and s['followup_count'] < 1)
    return {"last_answer_score": score, "current_expression": expr, "phase": "FOLLOWUP" if follow else "ADVANCE"}


def followup_node(s: InterviewState):
    resp = llm.invoke(
        f"Ask a short follow-up.\nQuestion: {s['last_question_context']}\nAnswer: {s['user_input']}"
    )
    fu = resp.content.strip()[:MAX_FU_LEN]
    return {
        "followup_count": s['followup_count'] + 1,
        "bot_response": f"{s['current_expression']} {fu}",
        "phase": "DETECT_INTENT"
    }


def open_chat_node(s: InterviewState):
    turn = s['open_chat_turns'] + 1
    if turn < 3:
        prompt = f"Answer off-topic briefly: {s['user_input']}"
        phase = END
    else:
        prompt = f"Answer: {s['user_input']}. Then say 'Let's stick to the interview.'"
        phase = "ADVANCE"
    resp = llm.invoke(prompt)
    return {"bot_response": resp.content.strip(), "open_chat_turns": turn, "phase": phase}


def advance_node(s: InterviewState):
    idx = s['question_idx'] + 1
    if idx >= len(CORE_QS):
        return {"done": True, "bot_response": f"{s['current_expression']} Thank you! This concludes our interview.", "phase": END}
    return {"question_idx": idx, "phase": "ASK_Q"}

# ----------------- Graph Construction -----------------
class NaturalInterviewManager:
    def __init__(self):
        builder = StateGraph(InterviewState)
        # Entry from START directly to greet
        builder.add_edge(START, "greet")
        # greet -> check_consent
        builder.add_edge("greet", "check_consent")
        # consent loop: check_consent -> ask_q or back to check_consent
        builder.add_conditional_edges(
            "check_consent",
            lambda s: "ask_q" if s.get("consent_allowed", False) else "check_consent",
            {"ask_q": "ask_q", "check_consent": "check_consent"}
        )
        # ask_q -> detect_intent
        builder.add_edge("ask_q", "detect_intent")
        # detect_intent -> score_route or open_chat
        builder.add_conditional_edges(
            "detect_intent",
            lambda s: "open_chat" if s.get("phase") == "OPEN_CHAT" else "score_route",
            {"open_chat": "open_chat", "score_route": "score_route"}
        )
        # score_route -> followup or advance
        builder.add_conditional_edges(
            "score_route",
            lambda s: "followup" if s.get("phase") == "FOLLOWUP" else "advance",
            {"followup": "followup", "advance": "advance"}
        )
        # followup -> detect_intent
        builder.add_edge("followup", "detect_intent")
        # open_chat -> advance or stay in open_chat
        builder.add_conditional_edges(
            "open_chat",
            lambda s: "advance" if s.get("phase") == "ADVANCE" else "open_chat",
            {"open_chat": "open_chat", "advance": "advance"}
        )
        # advance -> ask_q or END
        builder.add_conditional_edges(
            "advance",
            lambda s: "ask_q" if not s.get("done", False) else END,
            {"ask_q": "ask_q", END: END}
        )

        self.graph = builder.compile(checkpointer=global_checkpointer)

natural_manager = NaturalInterviewManager()

@celery_app.task(name="apps.ai_interview.tasks.natural_interview_flow.process_natural_interview", bind=True)
def process_natural_interview(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    data = payload if isinstance(payload, dict) else json.loads(payload)
    state: InterviewState = {
        'messages': [],
        'room_id': data.get('room_id', ''),
        'user_input': data.get('text', ''),
        'bot_response': '',
        'question_idx': data.get('question_idx', 0),
        'followup_count': data.get('followup_count', 0),
        'open_chat_turns': data.get('open_chat_turns', 0),
        'in_followup': data.get('in_followup', False),
        'done': data.get('done', False),
        'last_question_context': data.get('last_question_context', ''),
        'current_expression': data.get('current_expression', ''),
        'last_answer_score': data.get('last_answer_score', 0),
        'phase': data.get('phase', ''),
        'should_advance': data.get('should_advance', False),
        'consent_allowed': data.get('consent_allowed', False)
    }
    return natural_manager.graph.invoke(state, config={'thread_id': state['room_id']})
