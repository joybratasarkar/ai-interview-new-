import os
import json
import logging
import time
import asyncio
from typing import Dict, List, Any, TypedDict, Annotated, Literal
from apps.ai_interview.celery_app import celery_app
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage, SystemMessage

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

memory = MemorySaver()

# State definition
typestate = Annotated[List, add_messages]
class InterviewState(TypedDict):
    messages: typestate
    room_id: str
    user_input: str
    bot_response: str
    question_idx: int
    done: bool
    phase: str  # greeting, intro, questions, complete
    follow_up_count: int  # track follow-ups for current question
    conversation_history: str  # track conversation context for natural flow

# Interview phases
GREETING_MESSAGE = "Hello! Welcome to your AI interview session. I'm excited to get to know you better today."

INTRO_MESSAGE = """Before we begin with the main questions, let me explain how this works:
- This is a friendly conversation, so feel free to be yourself
Are you ready to start the interview?"""

# Dynamic questions that adapt based on conversation
BASE_QUESTIONS = [
    "So, let's start with your background - tell me about your recent work experience.",
    "That's interesting! What programming languages and technologies are you most comfortable working with?",
    "I'd love to hear about a challenging project you've tackled - can you walk me through how you approached it?",
    "Every role has its tough moments - how do you typically handle difficult situations or challenges at work?",
    "Finally, I'm curious about what drives you - what motivates you in your professional development?"
]

def generate_contextual_question(question_idx: int, conversation_history: str = "") -> str:
    """Generate contextual questions based on previous answers"""
    try:
        if question_idx >= len(BASE_QUESTIONS):
            return "Thank you for completing the interview!"
            
        base_question = BASE_QUESTIONS[question_idx]
        
        # If no conversation history, return base question
        if not conversation_history.strip():
            return base_question
            
        # Generate contextual version based on conversation
        messages = [
            SystemMessage(content=f"""Make this question flow naturally from their previous answer. Reference what they mentioned if relevant.

Question: {base_question}

Examples:
- "Since you mentioned Python, what about..."
- "That's cool! Now tell me about..."  
- "Nice! So what..."

Keep it short and natural."""),
            HumanMessage(content=f"Previous: {conversation_history[-100:]}\n\nNatural version:")
        ]
        
        response = circuit_breaker.call_llm(messages, base_question)
        
        if response and response.content:
            contextual_question = response.content.strip()
            print(f"Generated contextual question: {contextual_question}")
            return contextual_question
            
        # Fallback to base question
        return base_question
        
    except Exception as e:
        logger.error(f"Error generating contextual question: {e}")
        return BASE_QUESTIONS[question_idx] if question_idx < len(BASE_QUESTIONS) else "Could you tell me more about your experience?"

# Use BASE_QUESTIONS for backward compatibility
QUESTIONS = BASE_QUESTIONS
os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", os.path.abspath("xooper.json"))


# Improved LLM setup with better error handling
def get_llm() -> ChatVertexAI:
    try:
        return ChatVertexAI(
            model="gemini-2.0-flash-001",
            project="xooper-450012",
            temperature=0.1,  # Lower temperature for faster, more predictable responses
            convert_system_message_to_human=True,
            timeout=5,  # Reduced timeout for faster responses
            max_output_tokens=100,  # Limit response length for speed
        )
    except Exception as e:
        logger.error(f"Error initializing LLM: {e}")
        raise

# Circuit Breaker Pattern for LLM Calls
class LLMCircuitBreaker:
    def __init__(self, failure_threshold=2, recovery_timeout=30, half_open_max_calls=1):  # Faster recovery
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.half_open_calls = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
    def get_state(self):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                self.half_open_calls = 0
                logger.info("Circuit breaker transitioning to HALF_OPEN")
        return self.state
    
    def call_llm(self, messages, fallback_response="I understand. Could you tell me more?"):
        """Call LLM with circuit breaker protection"""
        current_state = self.get_state()
        
        if current_state == "OPEN":
            logger.warning("Circuit breaker OPEN - using fallback response")
            return type('Response', (), {'content': fallback_response})()
        
        if current_state == "HALF_OPEN" and self.half_open_calls >= self.half_open_max_calls:
            logger.warning("Circuit breaker HALF_OPEN limit reached - using fallback")
            return type('Response', (), {'content': fallback_response})()
        
        try:
            if current_state == "HALF_OPEN":
                self.half_open_calls += 1
                
            # FIX: Call the actual LLM instance, not circuit_breaker recursively
            response = llm.invoke(messages)
            
            # Success - reset circuit breaker
            if current_state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
                logger.info("Circuit breaker CLOSED - LLM calls restored")
            
            return response
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            logger.error(f"LLM call failed ({self.failure_count}/{self.failure_threshold}): {e}")
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                logger.error(f"Circuit breaker OPEN - LLM calls suspended for {self.recovery_timeout}s")
            
            # Return fallback response
            return type('Response', (), {'content': fallback_response})()

# Initialize circuit breaker and LLM
circuit_breaker = LLMCircuitBreaker()
llm = get_llm()

# Improved classification function with better error handling
def classify_answer(question: str, answer: str) -> bool:
    """Returns True if answer is related to question, False if off-topic"""
    try:
        if not answer or not answer.strip():
            return False
            
        # Faster classification with shorter prompt
        messages = [
            SystemMessage(content="Answer YES if relevant to question, NO if off-topic."),
            HumanMessage(content=f"Q: {question}\nA: {answer}\nRelevant?")
        ]
        
        print(f"Classifying answer for: {question[:50]}...")
        response = circuit_breaker.call_llm(messages, "YES")
        
        if not response or not response.content:
            print("Empty response from LLM, defaulting to related")
            return True
            
        classification = response.content.upper().strip()
        print(f"Raw classification response: '{classification}'")
        
        # More robust parsing
        is_related = any(word in classification for word in ["YES", "RELATED", "TRUE"])
        is_unrelated = any(word in classification for word in ["NO", "UNRELATED", "FALSE", "OFF-TOPIC"])
        
        if is_related and not is_unrelated:
            result = True
        elif is_unrelated and not is_related:
            result = False
        else:
            # If unclear, default to related to keep interview flowing
            result = True
            
        print(f"Classification result: {classification} -> {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error in classification: {e}")
        print(f"Error in classification: {e}")
        return True  # Default to related if error

# Evaluative response generator for interviewer
def generate_evaluative_response(answer: str) -> str:
    """Generate evaluative responses that assess the candidate's answer"""
    try:
        if not answer or not answer.strip():
            return "I'd like to hear more about that."
            
        messages = [
            SystemMessage(content="""You are a professional AI interviewer conducting a job interview. Your role is to EVALUATE the candidate's responses, not just be friendly.

Generate a brief response (8-12 words) that shows you're assessing their answer. Examples:

For good answers:
- "That shows good problem-solving skills."
- "I can see you have solid experience there."
- "That demonstrates strong technical knowledge."
- "Good approach to handling challenges."

For average answers:
- "Can you be more specific about that?"
- "I'd like to understand your role better."
- "What was your specific contribution there?"

For unclear answers:
- "That's quite general - can you give details?"
- "I need more concrete examples from you."

Be professional but evaluative. You're assessing their suitability for a role."""),
            HumanMessage(content=f"Candidate answered: {answer}\n\nGive an evaluative interviewer response (8-12 words):")
        ]
        
        response = circuit_breaker.call_llm(messages, "I see. Can you elaborate on that?")
        
        if response and response.content:
            evaluation = response.content.strip()
            # Ensure reasonable length
            if len(evaluation.split()) <= 15:
                return evaluation
        
        # Fallback if LLM fails
        return "I see. Can you elaborate on that?"
        
    except Exception as e:
        logger.error(f"Error generating evaluative response: {e}")
        return "Tell me more about that."

# Real interviewer evaluation logic
def should_ask_followup(question: str, answer: str, follow_up_count: int) -> bool:
    """Evaluate like a real interviewer whether to ask follow-up questions"""
    try:
        # Max 2 follow-ups per question
        if follow_up_count >= 2:
            return False
            
        if not answer or len(answer.strip()) < 10:
            return True  # Too short, need more detail
            
        # Quick evaluation using LLM
        messages = [
            SystemMessage(content="""You're an interviewer evaluating if you need a follow-up question.

Ask follow-up if answer is:
- Too vague or generic
- Missing important details  
- Incomplete or surface-level
- Interesting but needs elaboration

Don't ask follow-up if answer is:
- Complete and detailed
- Clear and specific
- Already thorough

Answer only: FOLLOWUP or NEXT"""),
            HumanMessage(content=f"Q: {question}\nA: {answer}\n\nDecision:")
        ]
        
        response = circuit_breaker.call_llm(messages, "NEXT")
        decision = response.content.strip().upper() if response and response.content else "NEXT"
        
        should_followup = "FOLLOWUP" in decision
        print(f"Follow-up decision: {decision} -> {should_followup}")
        return should_followup
        
    except Exception as e:
        logger.error(f"Error in follow-up evaluation: {e}")
        # Default: ask follow-up if answer is short
        return len(answer.strip()) < 50

# Natural conversational response generator 
def generate_natural_conversation(question: str, answer: str, conversation_context: str = "") -> str:
    """Generate natural, flowing conversation responses like a real interviewer"""
    try:
        if not answer or not answer.strip():
            return "I'd love to hear more about that - could you share some details?"
        
        # Faster, more natural conversation generation
        messages = [
            SystemMessage(content="""You're a friendly interviewer. Respond naturally and conversationally to their answer. 

Examples:
- "That's interesting! Tell me more about..."
- "Nice! How did you..."  
- "Cool! What made you..."
- "Great! Can you walk me through..."

Keep it short, natural, and curious. No formal language."""),
            HumanMessage(content=f"They said: {answer}\n\nYour natural response:")
        ]
        
        print(f"Generating natural conversation for: {question[:50]}...")
        response = circuit_breaker.call_llm(messages, "That's interesting! Tell me more.")
        
        if response and response.content:
            natural_response = response.content.strip()
            print(f"Generated natural conversation: {natural_response}")
            return natural_response
        
        # Fallback if LLM fails
        return "That's really interesting. Could you tell me more about how you approached that?"
        
    except Exception as e:
        logger.error(f"Error generating natural conversation: {e}")
        print(f"Error generating natural conversation: {e}")
        return "That's a great point. Can you elaborate on that experience?"

def get_fallback_followup(question: str) -> str:
    """Get a contextual fallback follow-up based on the original question"""
    fallbacks = {
        "experience": "What was the most challenging aspect of that role?",
        "technologies": "Which of these technologies do you enjoy working with most?",
        "project": "What would you do differently if you had to do it again?",
        "challenges": "Can you give me a specific example?",
        "motivates": "How do you stay updated with industry trends?"
    }
    
    # Find the most relevant fallback based on keywords
    question_lower = question.lower()
    for keyword, fallback in fallbacks.items():
        if keyword in question_lower:
            return fallback
    
    # Default fallback
    return "Could you elaborate on that a bit more?"

# Intelligent off-topic response handler
def handle_offtopic_intelligently(user_input: str) -> str:
    """Generate intelligent, natural responses to off-topic input"""
    try:
        if not user_input or not user_input.strip():
            return "I hear you, but let's keep our focus on the interview."
        
        messages = [
            SystemMessage(content="""You are an intelligent, empathetic AI interviewer. The candidate just said something off-topic or unrelated to the interview question. 

Generate a natural, understanding response that:
1. Acknowledges what they said briefly
2. Gently redirects back to the interview 
3. Sounds conversational and human-like
4. Keeps the mood positive

Examples:
- "I hear you on that! But let's get back to talking about your experience."
- "That's interesting, but I'd love to focus on your professional background."
- "I understand! Let's circle back to the interview though."
- "Ya, I get that. Let's keep our conversation on your career journey."

Keep it under 15 words and sound natural."""),
            HumanMessage(content=f"Candidate said (off-topic): {user_input}\n\nGenerate a natural, understanding redirect:")
        ]
        
        print(f"Handling off-topic intelligently: {user_input[:50]}...")
        response = circuit_breaker.call_llm(messages, "I hear you! Let's keep our focus on the interview though.")
        
        if response and response.content:
            redirect = response.content.strip()
            print(f"Intelligent off-topic response: {redirect}")
            return redirect
            
        # Fallback
        return "I hear you! Let's keep our focus on the interview though."
        
    except Exception as e:
        logger.error(f"Error handling off-topic intelligently: {e}")
        print(f"Error handling off-topic intelligently: {e}")
        return "I understand! Let's get back to the interview."

# Interview flow routing function (unchanged)
def route_interview_flow(state: InterviewState) -> Literal["greeting", "intro", "ask_question", "process_answer", "complete"]:
    """Determine next step in interview based on current state"""
    user_input = state.get('user_input', '').strip()
    current_idx = state.get('question_idx', 0)
    phase = state.get('phase', 'greeting')
    
    print(f"Routing decision - user_input: '{user_input}', current_idx: {current_idx}, phase: {phase}, done: {state.get('done', False)}")
    
    # If interview is already marked as done, complete
    if state.get('done', False):
        return "complete"
    
    # Phase-based routing
    if phase == 'greeting':
        if user_input:
            return "greeting"  # User responded to greeting, move to intro
        else:
            return "greeting"  # Show greeting
    
    elif phase == 'intro':
        if user_input:
            return "intro"  # User ready, start questions
        else:
            return "intro"  # Show intro
    
    elif phase == 'questions':
        # If we're at or past the last question, complete the interview
        if current_idx >= len(QUESTIONS):
            return "complete"
        
        # If user provided input (answer), process the answer
        if user_input:
            print(f"User provided answer, processing answer")
            return "process_answer"
        
        # Otherwise, ask the current question
        print(f"No user input, asking current question")
        return "ask_question"
    
    # Default fallback
    return "greeting"

# Node functions (unchanged but with better error handling)
def greeting_node(state: InterviewState):
    """Show greeting message or move to intro if user already responded to greeting"""
    try:
        user_input = state.get('user_input', '').strip().lower()
        
        # If user says "start", show greeting message first
        if user_input == "start":
            return {
                "bot_response": GREETING_MESSAGE,
                "phase": "greeting",
                "done": False
            }
        elif user_input and user_input != "start":
            # User responded to greeting (not "start"), move to intro phase
            return {
                "bot_response": INTRO_MESSAGE,
                "phase": "intro",
                "done": False
            }
        else:
            # No user input, show initial greeting
            return {
                "bot_response": GREETING_MESSAGE,
                "phase": "greeting",
                "done": False
            }
    except Exception as e:
        logger.error(f"Error in greeting_node: {e}")
        return {
            "bot_response": GREETING_MESSAGE,
            "phase": "greeting",
            "done": False
        }

def intro_node(state: InterviewState):
    """Show introduction or move to questions if user is ready"""
    try:
        user_input = state.get('user_input', '').strip()
        
        if user_input:
            # User is ready, start with first question (question_idx should be 0)
            current_idx = 0  # Always start with first question after intro
            if current_idx >= len(QUESTIONS):
                return {"done": True, "bot_response": "Thank you! Interview complete.", "phase": "complete"}
            
            print(f"Intro complete - transitioning to questions phase with question_idx: {current_idx}")
            return {
                "bot_response": QUESTIONS[current_idx],
                "phase": "questions",
                "question_idx": current_idx,
                "follow_up_count": 0,  # Reset follow-up count for new question
                "done": False
            }
        else:
            # Show introduction
            return {
                "bot_response": INTRO_MESSAGE,
                "phase": "intro", 
                "done": False
            }
    except Exception as e:
        logger.error(f"Error in intro_node: {e}")
        return {
            "bot_response": INTRO_MESSAGE,
            "phase": "intro", 
            "done": False
        }

def ask_question_node(state: InterviewState):
    """Ask current question based on question_idx"""
    try:
        idx = state.get('question_idx', 0)
        if idx >= len(QUESTIONS):
            return {"done": True, "bot_response": "Thank you! Interview complete.", "phase": "complete"}
        
        return {
            "bot_response": QUESTIONS[idx],
            "phase": "questions",
            "done": False
        }
    except Exception as e:
        logger.error(f"Error in ask_question_node: {e}")
        return {
            "bot_response": "Let me ask you a question about your experience.",
            "phase": "questions",
            "done": False
        }

def process_answer_node(state: InterviewState):
    """Process candidate's answer - classify and decide follow-up or next question"""
    try:
        current_idx = state.get('question_idx', 0)
        user_answer = state.get('user_input', '').strip()
        follow_up_count = state.get('follow_up_count', 0)
        
        # Get current question
        if current_idx < len(QUESTIONS):
            current_question = QUESTIONS[current_idx]
        else:
            current_question = ""
        
        # Classify answer: Related or Off-topic
        is_related = classify_answer(current_question, user_answer)
        
        if not is_related:
            # Off-topic: Give intelligent response + move to next question
            intelligent_redirect = handle_offtopic_intelligently(user_answer)
            next_idx = current_idx + 1
            
            # Check if we've reached the end of questions
            if next_idx >= len(QUESTIONS):
                return {
                    "done": True, 
                    "bot_response": f"{intelligent_redirect} Actually, that wraps up our interview! Thank you for the great conversation. We'll be in touch soon with next steps.", 
                    "question_idx": next_idx,
                    "phase": "complete",
                    "follow_up_count": 0
                }
            
            # Combine intelligent response with next question
            full_response = f"{intelligent_redirect}\n\n{QUESTIONS[next_idx]}"
            
            return {
                "bot_response": full_response,
                "question_idx": next_idx,
                "phase": "questions", 
                "follow_up_count": 0,  # Reset follow-ups for next question
                "done": False
            }
        
        # Get conversation context for natural flow
        conversation_context = state.get('conversation_history', '')
        
        # Related answer: Evaluate like a real interviewer
        needs_followup = should_ask_followup(current_question, user_answer, follow_up_count)
        
        if needs_followup and follow_up_count < 2:
            # Generate natural conversational follow-up
            natural_response = generate_natural_conversation(current_question, user_answer, conversation_context)
            
            # Update conversation history
            updated_history = f"{conversation_context}\nQ: {current_question}\nA: {user_answer}\nInterviewer: {natural_response}"
            
            return {
                "bot_response": natural_response,
                "question_idx": current_idx,  # Stay on same question
                "phase": "questions",
                "follow_up_count": follow_up_count + 1,
                "conversation_history": updated_history,
                "done": False
            }
        
        # No follow-up needed OR max follow-ups reached - transition to next question like a real interviewer
        next_idx = current_idx + 1
        
        # Check if we've reached the end of questions
        if next_idx >= len(QUESTIONS):
            # Generate natural conclusion
            messages = [
                SystemMessage(content="You are a professional interviewer concluding the interview. Generate a natural, warm conclusion that thanks them and explains next steps. Keep it professional but personable."),
                HumanMessage(content=f"Candidate's last response: {user_answer}\n\nGenerate a natural interview conclusion:")
            ]
            
            response = circuit_breaker.call_llm(messages, "Thank you for a wonderful conversation! I really enjoyed learning about your experience and background. We'll be reviewing everything we discussed and will get back to you soon with next steps.")
            conclusion = response.content if response and response.content else "Thank you for a wonderful conversation! I really enjoyed learning about your experience and background. We'll be reviewing everything we discussed and will get back to you soon with next steps."
            
            return {
                "done": True, 
                "bot_response": conclusion,
                "question_idx": next_idx,
                "phase": "complete",
                "follow_up_count": 0,
                "conversation_history": f"{conversation_context}\nConclusion: {conclusion}"
            }
        
        # Generate natural transition to next question like a real interviewer
        messages = [
            SystemMessage(content="""You're an interviewer transitioning between topics. Create a smooth transition:

Examples:
- "That's great insight! Now let me ask about..."
- "Perfect! Moving on to..."  
- "Excellent! Now I'm curious about..."
- "Thanks for sharing that! Let's talk about..."

Keep it short and natural."""),
            HumanMessage(content=f"Last answer: {user_answer[-100:]}\n\nNext question: {QUESTIONS[next_idx]}\n\nTransition:")
        ]
        
        response = circuit_breaker.call_llm(messages, f"Great! Now let me ask: {QUESTIONS[next_idx]}")
        transition = response.content if response and response.content else f"Great! Now let me ask: {QUESTIONS[next_idx]}"
        
        # Update conversation history
        updated_history = f"{conversation_context}\nQ: {current_question}\nA: {user_answer}\nTransition: {transition}"
        
        # Return the next question with natural transition
        return {
            "bot_response": transition,
            "question_idx": next_idx,
            "phase": "questions",
            "follow_up_count": 0,  # Reset for new question
            "conversation_history": updated_history,
            "done": False
        }
        
    except Exception as e:
        logger.error(f"Error in process_answer_node: {e}")
        # Fallback: move to next question
        next_idx = state.get('question_idx', 0) + 1
        if next_idx >= len(QUESTIONS):
            return {
                "done": True,
                "bot_response": "Thank you for completing the interview!",
                "phase": "complete"
            }
        return {
            "bot_response": QUESTIONS[next_idx],
            "question_idx": next_idx,
            "phase": "questions",
            "follow_up_count": 0,
            "done": False
        }

def complete_interview_node(state: InterviewState):
    """Complete the interview"""
    return {
        "done": True,
        "bot_response": "Thank you for completing the interview! It was great getting to know you and learning about your experience. We'll be in touch soon with next steps.",
        "phase": "complete"
    }

# LangGraph implementation with conditional edges
builder = StateGraph(InterviewState)

# Add nodes
builder.add_node("greeting", greeting_node)
builder.add_node("intro", intro_node)
builder.add_node("ask_question", ask_question_node)
builder.add_node("process_answer", process_answer_node)
builder.add_node("complete", complete_interview_node)

# Set entry point - determine initial route based on state
builder.add_conditional_edges(START, route_interview_flow)

# All nodes end after execution (wait for next user input)
builder.add_edge("greeting", END)
builder.add_edge("intro", END)
builder.add_edge("ask_question", END)
builder.add_edge("process_answer", END)
builder.add_edge("complete", END)

graph = builder.compile(checkpointer=memory)

@celery_app.task(name="apps.ai_interview.tasks.natural_interview_flow.process_natural_interview", bind=True)
def process_natural_interview(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Process interview turn using LangGraph conditional flow"""
    try:
        data = payload if isinstance(payload, dict) else json.loads(payload)
        
        # Add comprehensive logging of received data
        logger.info(f"=== NATURAL INTERVIEW FLOW - RECEIVED DATA ===")
        logger.info(f"Raw payload type: {type(payload)}")
        logger.info(f"Raw payload: {payload}")
        logger.info(f"Parsed data: {data}")
        logger.info(f"Data keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
        
        # Specifically log the user input and state fields that matter for debugging
        user_input = data.get('text', '')
        question_idx = data.get('question_idx', 0)
        phase = data.get('phase', 'greeting')
        
        logger.info(f"KEY DEBUG INFO - user_input: '{user_input}', question_idx: {question_idx}, phase: '{phase}'")
        
        state: InterviewState = {
            'messages': [],
            'room_id': data.get('room_id', ''),
            'user_input': user_input,
            'bot_response': '',
            'question_idx': question_idx,
            'done': data.get('done', False),
            'phase': phase,
            'follow_up_count': data.get('follow_up_count', 0),
            'conversation_history': data.get('conversation_history', '')
        }
        
        logger.info(f"Constructed state: {state}")
        logger.info(f"=== END RECEIVED DATA LOG ===")
        
        user_text = data.get('text', '').strip()
        current_question_idx = data.get('question_idx', 0)
        
        print(f"Processing interview - User text: '{user_text}', Question index: {current_question_idx}, Phase: {state.get('phase', 'greeting')}")
        print(f"Full state: {state}")
        
        # Use LangGraph with conditional flow
        result = graph.invoke(state, config={'thread_id': state['room_id']})
        
        # Add the updated question_idx to response for WebSocket state management
        if 'question_idx' not in result and not result.get('done', False):
            result['question_idx'] = current_question_idx
            
        print(f"Interview result: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error in interview flow: {e}")
        return {
            'bot_response': 'Sorry, there was an issue. Let me restart.',
            'error': str(e),
            'question_idx': 0,
            'done': False
        }