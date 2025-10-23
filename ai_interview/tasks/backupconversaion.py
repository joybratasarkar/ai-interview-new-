import os
import json
import logging
from typing import Dict, List, Any, TypedDict, Annotated
from apps.ai_interview.celery_app import celery_app

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

# LLM Setup
os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", os.path.abspath("xooper.json"))
from langchain_google_vertexai import ChatVertexAI

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def get_llm() -> ChatVertexAI:
    return ChatVertexAI(
        model="gemini-2.0-flash-001",
        project="xooper-450012",
        temperature=0.7,
        convert_system_message_to_human=True,
    )

llm = get_llm()
memory = MemorySaver()

# State definition
typestate = Annotated[List, add_messages]
class InterviewState(TypedDict):
    messages: typestate
    room_id: str
    user_input: str
    bot_response: str
    question_idx: int
    followup_asked: bool
    done: bool
    consent_given: bool

QUESTIONS = [
    "Tell me about your recent work experience.",
    "What programming languages do you know?",
    "Describe a project you're proud of.",
    "How do you handle challenges?"
]

# Node functions
def greeting_node(state: InterviewState):
    """Initial greeting"""
    return {
        "bot_response": "Hello! I'm your AI interviewer. May I begin our interview?"
    }

def intro_node(state: InterviewState):
    """Introduction after consent"""
    return {
        "bot_response": "Great! Let's start with our first question.",
        "consent_given": True
    }

def ask_question_node(state: InterviewState):
    """Ask interview question"""
    idx = state.get('question_idx', 0)
    if idx >= len(QUESTIONS):
        return {"done": True, "bot_response": "Thank you! Interview complete."}
    
    question = QUESTIONS[idx]
    return {
        "bot_response": question,
        "followup_asked": False
    }

def evaluate_answer_node(state: InterviewState):
    """Evaluate answer and decide on follow-up"""
    answer = state['user_input']
    question = QUESTIONS[state['question_idx']]
    
    # Simple scoring
    score_prompt = f"Rate this answer 1-10: Question: {question} Answer: {answer}. Just return the number."
    try:
        resp = llm.invoke(score_prompt)
        score = int(resp.content.strip())
    except:
        score = 5
    
    # Decide follow-up
    needs_followup = score >= 6 and not state.get('followup_asked', False)
    
    if needs_followup:
        followup_prompt = f"Ask a short follow-up question about: {answer}"
        try:
            resp = llm.invoke(followup_prompt)
            followup = resp.content.strip()
        except:
            followup = "Can you tell me more about that?"
        
        return {
            "bot_response": f"Good! {followup}",
            "followup_asked": True
        }
    else:
        # Advance to next question
        return {
            "question_idx": state['question_idx'] + 1,
            "followup_asked": False
        }

def chat_node(state: InterviewState):
    """Handle open chat"""
    user_question = state['user_input']
    
    # Answer their question
    chat_prompt = f"Answer this question briefly and friendly: {user_question}"
    try:
        resp = llm.invoke(chat_prompt)
        answer = resp.content.strip()
    except:
        answer = "That's a good question!"
    
    # Return to interview
    return {
        "bot_response": f"{answer} Now let's continue with our interview.",
        "question_idx": state['question_idx'] + 1,
        "followup_asked": False
    }

# Routing functions
def should_continue_after_greeting(state: InterviewState):
    """Route after greeting based on user input"""
    user_input = state.get('user_input', '').lower()
    if 'yes' in user_input or 'sure' in user_input or 'ok' in user_input:
        return "intro"
    return "greeting"

def is_answer_or_chat(state: InterviewState):
    """Determine if user input is answering question or open chat"""
    user_input = state['user_input']
    current_q = QUESTIONS[state['question_idx']] if state['question_idx'] < len(QUESTIONS) else ""
    
    # Simple intent detection
    prompt = f"Is this answering the question '{current_q}'? User said: '{user_input}'. Answer 'yes' or 'no'."
    try:
        resp = llm.invoke(prompt)
        is_answer = 'yes' in resp.content.lower()
        return "evaluate" if is_answer else "chat"
    except:
        return "evaluate"  # Default to answer

def should_continue_after_eval(state: InterviewState):
    """Route after evaluation - either ask question or end"""
    if state.get('done', False):
        return END
    if state.get('followup_asked', False):
        return END  # Wait for followup response
    return "ask_question"

# Simple LangGraph implementation
class NaturalInterviewManager:
    def __init__(self):
        builder = StateGraph(InterviewState)
        
        # Add nodes
        builder.add_node("greeting", greeting_node)
        builder.add_node("intro", intro_node) 
        builder.add_node("ask_question", ask_question_node)
        builder.add_node("evaluate", evaluate_answer_node)
        builder.add_node("chat", chat_node)
        
        # Start with greeting
        builder.set_entry_point("greeting")
        
        # Greeting -> directly to intro (skip consent check)
        builder.add_edge("greeting", "intro")
        
        # After intro -> ask question -> wait
        builder.add_edge("intro", "ask_question") 
        builder.add_edge("ask_question", END)
        
        # After evaluation -> ask next question or end
        builder.add_conditional_edges(
            "evaluate",
            lambda s: END if s.get('done', False) or s.get('followup_asked', False) else "ask_question"
        )
        
        # After chat -> ask next question
        builder.add_edge("chat", "ask_question")
        
        self.graph = builder.compile(checkpointer=memory)
    
    def process_turn(self, state: InterviewState):
        """Process a single turn based on current state"""
        # If no consent yet and no user input, start with greeting
        if not state.get('consent_given', False) and not state.get('user_input', '').strip():
            return self.graph.invoke(state, config={'thread_id': state['room_id']}, 
                                   starting_node="greeting")
        
        # If no consent yet, check greeting response
        if not state.get('consent_given', False) and state.get('user_input'):
            consent_result = should_continue_after_greeting(state)
            if consent_result == "intro":
                return self.graph.invoke(state, config={'thread_id': state['room_id']}, 
                                       starting_node="intro")
            else:
                return self.graph.invoke(state, config={'thread_id': state['room_id']},
                                       starting_node="greeting")
        
        # If consent given and user input, route to evaluate or chat
        if state.get('consent_given', False) and state.get('user_input'):
            route = is_answer_or_chat(state)
            return self.graph.invoke(state, config={'thread_id': state['room_id']},
                                   starting_node=route)
        
        # Default: start from beginning
        return self.graph.invoke(state, config={'thread_id': state['room_id']})

natural_manager = NaturalInterviewManager()

@celery_app.task(name="apps.ai_interview.tasks.natural_interview_flow.process_natural_interview", bind=True)
def process_natural_interview(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Process interview turn"""
    try:
        data = payload if isinstance(payload, dict) else json.loads(payload)
        
        state: InterviewState = {
            'messages': [],
            'room_id': data.get('room_id', ''),
            'user_input': data.get('text', ''),
            'bot_response': '',
            'question_idx': data.get('question_idx', 0),
            'followup_asked': data.get('followup_asked', False),
            'done': data.get('done', False),
            'consent_given': data.get('consent_given', False)
        }
        print(f"Processing state: {payload}")
        
        # Handle different scenarios based on input and state
        user_text = payload.get('text', '').strip()
        print(f"Processing interview turn: {user_text}")
        
        if user_text.lower() == 'start':
            # Start command - begin with greeting flow
            result = natural_manager.graph.invoke(state, config={'thread_id': state['room_id']})
        elif not state.get('consent_given', False):
            # User responded to greeting - set consent and move to intro+question
            state['consent_given'] = True
            intro_result = intro_node(state)
            state.update(intro_result)
            question_result = ask_question_node(state)
            result = {**intro_result, **question_result}
        else:
            # User provided an answer - evaluate it
            eval_result = evaluate_answer_node(state)
            if eval_result.get('followup_asked', False):
                # Just return the follow-up question
                result = eval_result
            elif eval_result.get('done', False):
                # Interview complete
                result = eval_result
            else:
                # Move to next question
                state.update(eval_result)
                question_result = ask_question_node(state)
                result = {**eval_result, **question_result}
        logger.info(f"Interview result: {result.get('bot_response', '')}")
        return result
        
    except Exception as e:
        logger.error(f"Error in interview flow: {e}")
        return {
            'bot_response': 'Sorry, there was an issue. Let me restart.',
            'error': str(e)
        }
