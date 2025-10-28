import os
import json
import logging
import time
import re
import sys
import asyncio
import threading
import weakref
from typing import Dict, List, Any, TypedDict, Annotated, Literal, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
import uuid
from concurrent.futures import ThreadPoolExecutor

# Prevent bytecode generation for development
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True

from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage, SystemMessage

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

logger = logging.getLogger(__name__)

# Thread pool for background tasks
executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="interview_")

class InterviewPhase(Enum):
    GREETING = "greeting"
    INTRO = "intro"
    QUESTIONS = "questions"
    COMPLETE = "complete"

class IntentType(Enum):
    ANSWER = "Answer"
    END_INTERVIEW = "EndInterview"
    REPEAT_QUESTION = "RepeatQuestion"
    PREVIOUS_QUESTION = "PreviousQuestion"
    CLARIFY_QUESTION = "ClarifyQuestion"
    HESITATION = "Hesitation"
    SMALL_TALK = "SmallTalk"
    OFF_TOPIC = "OffTopic"

@dataclass
class SessionContext:
    """Encapsulates all session-specific data to prevent global state issues"""
    session_id: str
    role_title: str = "Software Developer"
    years_experience: str = "2-5 years"
    candidate_name: str = "Candidate"
    company_name: str = "Company"
    skills: List[str] = field(default_factory=list)
    questions: List[str] = field(default_factory=list)
    conversation_history: str = ""
    is_ready: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)

    def get_cache_key(self) -> str:
        return f"{self.session_id}_{self.role_title}_{self.years_experience}"

    def touch(self):
        """Update last accessed time"""
        self.last_accessed = datetime.now()

    def is_expired(self, timeout_minutes: int = 30) -> bool:
        """Check if session has expired"""
        return datetime.now() - self.last_accessed > timedelta(minutes=timeout_minutes)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['last_accessed'] = self.last_accessed.isoformat()
        return data

@dataclass
class FlowResponse:
    """Standard response format for better communication flow"""
    bot_response: str
    phase: InterviewPhase
    question_idx: int = 0
    follow_up_count: int = 0
    done: bool = False
    interview_status: str = "ongoing"
    last_question_asked: str = ""
    conversation_history: str = ""
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "bot_response": self.bot_response,
            "phase": self.phase.value if isinstance(self.phase, InterviewPhase) else self.phase,
            "question_idx": self.question_idx,
            "follow_up_count": self.follow_up_count,
            "done": self.done,
            "interview_status": "end" if self.done else self.interview_status,
            "last_question_asked": self.last_question_asked,
            "conversation_history": self.conversation_history
        }
        if self.error:
            result["error"] = self.error
        return result

# Centralized session management with proper cleanup
class SessionManager:
    def __init__(self):
        self._sessions: Dict[str, SessionContext] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task = None
        self._session_timeout_minutes = 30
        self._max_sessions = 1000  # Prevent memory explosion

    async def start_cleanup_task(self):
        """Start background cleanup task"""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())

    async def stop_cleanup_task(self):
        """Stop background cleanup task"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

    async def _periodic_cleanup(self):
        """Periodically clean up expired sessions"""
        while True:
            try:
                await asyncio.sleep(300)  # Clean up every 5 minutes
                await self._cleanup_expired_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in session cleanup: {e}")

    async def _cleanup_expired_sessions(self):
        """Remove expired sessions"""
        async with self._lock:
            expired_sessions = []
            for session_id, session in self._sessions.items():
                if session.is_expired(self._session_timeout_minutes):
                    expired_sessions.append(session_id)

            for session_id in expired_sessions:
                logger.info(f"Cleaning up expired session: {session_id}")
                del self._sessions[session_id]

    async def get_session(self, session_id: str, **kwargs) -> SessionContext:
        async with self._lock:
            # Enforce max sessions limit
            if len(self._sessions) >= self._max_sessions and session_id not in self._sessions:
                # Remove oldest session
                oldest_session_id = min(self._sessions.keys(),
                                       key=lambda k: self._sessions[k].last_accessed)
                logger.warning(f"Max sessions reached, removing oldest: {oldest_session_id}")
                del self._sessions[oldest_session_id]

            if session_id not in self._sessions:
                self._sessions[session_id] = SessionContext(
                    session_id=session_id,
                    role_title=kwargs.get('role_title', 'Software Developer'),
                    years_experience=kwargs.get('years_experience', '2-5 years'),
                    candidate_name=kwargs.get('candidate_name', 'Candidate'),
                    company_name=kwargs.get('company_name', 'Company')
                )

            # Update last accessed time
            self._sessions[session_id].touch()
            return self._sessions[session_id]

    async def update_session(self, session_id: str, **updates) -> None:
        async with self._lock:
            if session_id in self._sessions:
                session = self._sessions[session_id]
                for key, value in updates.items():
                    if hasattr(session, key):
                        setattr(session, key, value)
                session.touch()

    async def clear_session(self, session_id: str) -> None:
        async with self._lock:
            self._sessions.pop(session_id, None)
            logger.info(f"Cleared session: {session_id}")

    async def get_session_count(self) -> int:
        """Get current session count for monitoring"""
        async with self._lock:
            return len(self._sessions)

    async def get_session_info(self) -> Dict[str, Any]:
        """Get session statistics for monitoring"""
        async with self._lock:
            return {
                "total_sessions": len(self._sessions),
                "max_sessions": self._max_sessions,
                "timeout_minutes": self._session_timeout_minutes,
                "cleanup_running": self._cleanup_task is not None and not self._cleanup_task.done()
            }

session_manager = SessionManager()

# Per-session circuit breaker to prevent global state issues
class PerSessionCircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=30):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._session_states: Dict[str, Dict] = {}
        self._lock = asyncio.Lock()

    async def _get_session_state(self, session_id: str) -> Dict:
        async with self._lock:
            if session_id not in self._session_states:
                self._session_states[session_id] = {
                    "failure_count": 0,
                    "last_failure_time": None,
                    "state": "CLOSED",  # CLOSED, OPEN, HALF_OPEN
                    "llm_instance": None
                }
            return self._session_states[session_id]

    async def get_llm(self, session_id: str) -> ChatVertexAI:
        """Get LLM instance for session with lazy initialization"""
        state = await self._get_session_state(session_id)
        if state["llm_instance"] is None:
            state["llm_instance"] = ChatVertexAI(
                model_name="gemini-2.0-flash-lite-001",
                temperature=0.1,
                max_retries=1
            )
        return state["llm_instance"]

    async def get_circuit_state(self, session_id: str) -> str:
        state = await self._get_session_state(session_id)
        if state["state"] == "OPEN":
            if (state["last_failure_time"] and
                time.time() - state["last_failure_time"] > self.recovery_timeout):
                state["state"] = "HALF_OPEN"
        return state["state"]

    async def call_llm_async(self, session_id: str, messages: List,
                           fallback_response: str = "I understand. Could you tell me more?") -> str:
        """Async LLM call with per-session circuit breaker protection"""
        current_state = await self.get_circuit_state(session_id)

        if current_state == "OPEN":
            logger.warning(f"Circuit breaker OPEN for session {session_id}, using fallback")
            return fallback_response

        try:
            llm = await self.get_llm(session_id)
            response = llm.invoke(messages)

            # Success - reset circuit breaker
            state = await self._get_session_state(session_id)
            if current_state == "HALF_OPEN":
                state["state"] = "CLOSED"
                state["failure_count"] = 0
                logger.info(f"Circuit breaker CLOSED for session {session_id}")

            return response.content if response and response.content else fallback_response

        except Exception as e:
            logger.error(f"LLM call failed for session {session_id}: {e}")
            state = await self._get_session_state(session_id)
            state["failure_count"] += 1
            state["last_failure_time"] = time.time()

            if state["failure_count"] >= self.failure_threshold:
                state["state"] = "OPEN"
                logger.error(f"Circuit breaker OPEN for session {session_id} for {self.recovery_timeout}s")

            return fallback_response

    def call_llm(self, messages: List, fallback_response: str = "I understand. Could you tell me more?"):
        """Synchronous LLM call for backward compatibility"""
        try:
            llm = ChatVertexAI(
                model_name="gemini-2.0-flash-lite-001",
                temperature=0.1,
                max_retries=1
            )
            response = llm.invoke(messages)
            return response if response else type('Response', (), {'content': fallback_response})()
        except Exception as e:
            logger.error(f"Sync LLM call failed: {e}")
            return type('Response', (), {'content': fallback_response})()

    async def cleanup_session(self, session_id: str):
        """Clean up session state"""
        async with self._lock:
            if session_id in self._session_states:
                del self._session_states[session_id]
                logger.info(f"Cleaned up circuit breaker state for session {session_id}")

# Global circuit breaker instance
circuit_breaker = PerSessionCircuitBreaker()

# Helper functions for backward compatibility
def clear_global_variables():
    """Clear global variables (legacy compatibility)"""
    pass  # Session-based approach doesn't need global cleanup

def start_background_generation(role_title: str, years_experience: str, session_id: str):
    """Legacy function for background generation (now uses session manager)"""
    pass  # Session manager handles this automatically

async def get_skills_questions_hybrid(session_id: str, role_title: str, years_experience: str):
    """Hybrid approach that delegates to existing session manager"""
    return await get_skills_questions_for_session(session_id, role_title, years_experience)

# Robust question and skills generation with multiple fallback layers
class QuestionGenerator:
    """Handles question generation with robust fallback mechanisms"""

    def __init__(self, circuit_breaker: PerSessionCircuitBreaker):
        self.circuit_breaker = circuit_breaker
        self.skill_fallbacks = {
            "Software Developer": ["Programming", "Problem Solving", "System Design", "Code Review", "Testing", "Collaboration"],
            "Data Scientist": ["Data Analysis", "Machine Learning", "Statistics", "Python/R", "Visualization", "Research"],
            "Product Manager": ["Product Strategy", "Roadmap Planning", "Stakeholder Management", "Analytics", "User Research", "Leadership"],
            "UX Designer": ["User Research", "Prototyping", "Design Systems", "Usability Testing", "Wireframing", "Collaboration"],
            "DevOps Engineer": ["Infrastructure", "CI/CD", "Monitoring", "Cloud Platforms", "Automation", "Security"],
            "_default": ["Problem Solving", "Communication", "Technical Skills", "Analytical Thinking", "Collaboration", "Adaptability"]
        }

        self.question_templates = {
            "experience": "Tell me about your experience with {skill} in your {role} work.",
            "challenge": "Describe a challenging situation you've faced in your {role} role and how you handled it.",
            "collaboration": "Tell me about a time when you had to work with others to achieve a goal in your {role} position.",
            "learning": "Describe a situation where you had to learn something new quickly for your {role} work.",
            "project": "Walk me through a significant project you worked on as a {role}.",
            "problem_solving": "Tell me about a complex problem you solved in your {role} role."
        }

    async def generate_skills(self, session_id: str, role_title: str, years_experience: str) -> List[str]:
        """Generate skills with multiple fallback layers"""
        try:
            # Try LLM generation first
            messages = [
                SystemMessage(content=f"""Generate exactly 6 key skills for a {role_title} position requiring {years_experience} of experience.

Return format: ["Skill 1", "Skill 2", "Skill 3", "Skill 4", "Skill 5", "Skill 6"]
No markdown, no extra text, just the Python list."""),
                HumanMessage(content=f"Role: {role_title}, Experience: {years_experience}")
            ]

            response = await self.circuit_breaker.call_llm_async(
                session_id,
                messages,
                str(self.skill_fallbacks.get(role_title, self.skill_fallbacks["_default"]))
            )

            # Parse LLM response
            skills = self._parse_list_response(response)
            if skills and len(skills) >= 6:
                return skills[:6]

        except Exception as e:
            logger.error(f"Error generating skills for session {session_id}: {e}")

        # Fallback to predefined skills
        return self.skill_fallbacks.get(role_title, self.skill_fallbacks["_default"])

    async def generate_questions(self, session_id: str, role_title: str, years_experience: str, skills: List[str]) -> List[str]:
        """Generate questions with multiple fallback layers"""
        try:
            # Try LLM generation first
            skills_context = ", ".join(skills[:4])

            messages = [
                SystemMessage(content=f"""You are an expert interview designer. Generate EXACTLY 4 comprehensive interview questions for a {role_title} position requiring {years_experience} of experience.

Key skills to assess: {skills_context}

CRITICAL REQUIREMENTS:
1. Return EXACTLY 4 questions in Python list format: ["Question 1?", "Question 2?", "Question 3?", "Question 4?"]
2. Questions must be appropriate for {years_experience} experience level
3. Each question must target different aspects of the role
4. Questions should encourage detailed responses
5. Handle ANY role type (technical, creative, management, sales, etc.)

Generate 4 questions for: {role_title} with {years_experience} experience
Focus on: {skills_context}
Return only the Python list format."""),
                HumanMessage(content=f"Role: {role_title}\nExperience: {years_experience}\nKey Skills: {skills_context}\n\nGenerate exactly 4 interview questions:")
            ]

            fallback_questions = self._generate_template_questions(role_title, skills)
            response = await self.circuit_breaker.call_llm_async(
                session_id,
                messages,
                str(fallback_questions)
            )

            # Parse LLM response
            questions = self._parse_list_response(response)
            if questions and len(questions) >= 4:
                return questions[:4]

        except Exception as e:
            logger.error(f"Error generating questions for session {session_id}: {e}")

        # Fallback to template questions
        return self._generate_template_questions(role_title, skills)

    def _parse_list_response(self, response: str) -> Optional[List[str]]:
        """Safely parse list response from LLM"""
        if not response:
            return None

        try:
            # Find list boundaries
            if '[' in response and ']' in response:
                start = response.find('[')
                end = response.rfind(']') + 1
                list_str = response[start:end]

                # Safe evaluation
                import ast
                parsed_list = ast.literal_eval(list_str)

                if isinstance(parsed_list, list) and all(isinstance(item, str) for item in parsed_list):
                    return parsed_list

        except (ValueError, SyntaxError, IndexError) as e:
            logger.warning(f"Failed to parse list response: {e}")

        return None

    def _generate_template_questions(self, role_title: str, skills: List[str]) -> List[str]:
        """Generate questions using templates as fallback"""
        primary_skill = skills[0] if skills else "your field"

        return [
            self.question_templates["experience"].format(skill=primary_skill, role=role_title),
            self.question_templates["challenge"].format(role=role_title),
            self.question_templates["collaboration"].format(role=role_title),
            self.question_templates["learning"].format(role=role_title)
        ]

# Initialize question generator
question_generator = QuestionGenerator(circuit_breaker)

async def get_skills_questions_for_session(session_id: str, role_title: str, years_experience: str) -> tuple[List[str], List[str]]:
    """Get skills and questions for a specific session with robust fallbacks"""
    try:
        # Get or create session
        session_context = await session_manager.get_session(
            session_id,
            role_title=role_title,
            years_experience=years_experience
        )

        # Check if already generated and ready
        if session_context.skills and session_context.questions and session_context.is_ready:
            logger.info(f"Using cached skills/questions for session {session_id}")
            return session_context.skills, session_context.questions

        # Generate skills
        logger.info(f"Generating skills for session {session_id}")
        skills = await question_generator.generate_skills(session_id, role_title, years_experience)

        # Generate questions
        logger.info(f"Generating questions for session {session_id}")
        questions = await question_generator.generate_questions(session_id, role_title, years_experience, skills)

        # Update session
        await session_manager.update_session(
            session_id,
            skills=skills,
            questions=questions,
            is_ready=True
        )

        logger.info(f"Successfully generated {len(skills)} skills and {len(questions)} questions for session {session_id}")
        return skills, questions

    except Exception as e:
        logger.error(f"Error generating skills/questions for session {session_id}: {e}")

        # Emergency fallback
        fallback_skills = question_generator.skill_fallbacks.get(
            role_title,
            question_generator.skill_fallbacks["_default"]
        )
        fallback_questions = question_generator._generate_template_questions(role_title, fallback_skills)

        # Try to update session with fallbacks
        try:
            await session_manager.update_session(
                session_id,
                skills=fallback_skills,
                questions=fallback_questions,
                is_ready=True
            )
        except Exception:
            pass  # If even this fails, just return the fallbacks

        return fallback_skills, fallback_questions

async def clear_session_data(session_id: str):
    """Clear all session data and prevent memory leaks"""
    try:
        # Clear from SessionManager
        await session_manager.clear_session(session_id)

        # Clear circuit breaker state
        await circuit_breaker.cleanup_session(session_id)

        logger.info(f"Successfully cleared all data for session {session_id}")

    except Exception as e:
        logger.error(f"Error clearing session data for {session_id}: {e}")

def clean_bot_response(response: str) -> str:
    """Clean bot response by removing quotes, markdown, and extra whitespace"""
    if not response:
        return ""

    try:
        cleaned = str(response).strip()

        # Remove surrounding quotes (single or double) - handles nested quotes safely
        cleaned = re.sub(r'^["\'](.+)["\']$', r'\1', cleaned, flags=re.DOTALL)

        # Remove markdown formatting
        cleaned = re.sub(r'\*{1,3}([^*]*?)\*{1,3}', r'\1', cleaned)  # Bold/italic
        cleaned = re.sub(r'_{1,3}([^_]*?)_{1,3}', r'\1', cleaned)    # Underscore formatting
        cleaned = re.sub(r'`{1,3}([^`]*?)`{1,3}', r'\1', cleaned)    # Code formatting

        # Remove escaped quotes
        cleaned = re.sub(r'\\["\']', lambda m: m.group(0)[1:], cleaned)

        # Normalize whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)

        result = cleaned.strip()
        return result if result else "I understand. Let me continue with the interview."

    except Exception as e:
        logger.error(f"Error cleaning response: {e}")
        return "Let me continue with our interview."

# State definition with session isolation
typestate = Annotated[List, add_messages]

class InterviewState(TypedDict):
    messages: typestate
    room_id: str
    user_input: str
    bot_response: str
    question_idx: int
    done: bool
    phase: str
    follow_up_count: int
    conversation_history: str
    role_title: str
    years_experience: str
    last_question_asked: str
    candidate_name: str
    company_name: str
    session_context: Optional[SessionContext]
    # Session-specific data to prevent global state contamination
    session_skills: List[str]
    session_questions: List[str]

# Set up Google Cloud credentials
os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", os.path.abspath("xooper.json"))

# Monitoring and health check utilities
class InterviewMetrics:
    """Collect metrics for monitoring and debugging"""

    def __init__(self):
        self.metrics = {
            "total_sessions": 0,
            "active_sessions": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "circuit_breaker_opens": 0,
            "last_reset": datetime.now()
        }
        self._lock = asyncio.Lock()

    async def increment(self, metric: str, value: int = 1):
        async with self._lock:
            if metric in self.metrics:
                self.metrics[metric] += value

    async def set_metric(self, metric: str, value: Any):
        async with self._lock:
            self.metrics[metric] = value

    async def get_metrics(self) -> Dict[str, Any]:
        async with self._lock:
            # Add current session count
            self.metrics["active_sessions"] = await session_manager.get_session_count()
            return self.metrics.copy()

    async def reset_metrics(self):
        async with self._lock:
            self.metrics = {
                "total_sessions": 0,
                "active_sessions": 0,
                "successful_generations": 0,
                "failed_generations": 0,
                "circuit_breaker_opens": 0,
                "last_reset": datetime.now()
            }

# Global metrics instance
metrics = InterviewMetrics()

# Health check and startup procedures
async def startup_interview_service():
    """Initialize the interview service with proper setup"""
    try:
        # Start session cleanup task
        await session_manager.start_cleanup_task()
        logger.info("Interview service started successfully")

        # Initialize metrics
        await metrics.reset_metrics()

        return True
    except Exception as e:
        logger.error(f"Failed to start interview service: {e}")
        return False

async def shutdown_interview_service():
    """Cleanup on service shutdown"""
    try:
        # Stop cleanup task
        await session_manager.stop_cleanup_task()

        # Clean up thread pool
        executor.shutdown(wait=True)

        logger.info("Interview service shut down successfully")
    except Exception as e:
        logger.error(f"Error during service shutdown: {e}")

async def health_check() -> Dict[str, Any]:
    """Comprehensive health check"""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "session_manager": await session_manager.get_session_info(),
            "metrics": await metrics.get_metrics(),
            "memory_usage": {
                "session_count": await session_manager.get_session_count(),
                "max_sessions": 1000
            }
        }

        # Check if session count is approaching limit
        session_count = await session_manager.get_session_count()
        if session_count > 800:  # 80% of max
            health_status["warnings"] = ["High session count - approaching limit"]

        return health_status

    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Legacy compatibility wrapper - now delegates to QuestionGenerator
async def generate_key_skills_legacy(session_id: str, role_title: str, years_experience: str) -> List[str]:
    """Legacy wrapper for key skills generation"""
    return await question_generator.generate_skills(session_id, role_title, years_experience)






def get_greeting_message(role_title: str, candidate_name: str = "Candidate", company_name: str = "Company") -> str:
    """Generate greeting message with personalized candidate and company names"""
    return f"Hi {candidate_name}! I'm Chris, and I'll be your interviewer today for the {role_title} position at {company_name}. Thanks for joining me - I'm looking forward to our conversation. How are you doing today?"

async def get_intro_message(session_id: str, role_title: str, years_experience: str, candidate_name: str = "Candidate", company_name: str = "Company") -> str:
    """Generate intro message using session-specific data"""
    try:
        # Get skills from session
        session_context = await session_manager.get_session(session_id)
        skills = session_context.skills if session_context.skills else ["relevant technologies"]
        skills_text = ', '.join(skills[:3])

        messages = [
            SystemMessage(content=f"""
You are an AI assistant tasked with generating a professional yet conversational interview introduction for a candidate interviewing for a {role_title} position requiring {years_experience} of experience at {company_name}.

Candidate Name: {candidate_name}
Key Skills: {skills_text}

Your introduction should:
- Briefly introduce the company and the role
- Mention the key skills relevant to the role
- Maintain a casual, welcoming, and professional tone
- Ask the candidate if they're ready to begin
- Be concise (2–3 sentences)

OUTPUT FORMAT REQUIREMENTS:
- Return ONLY the introduction text without quotes
- NO markdown formatting (**, *, _, `)
- NO extra punctuation or formatting
- Direct, clean text output only

Now, generate the introduction.
"""),
            HumanMessage(content=f"Generate intro message for the {role_title} interview with skills: {skills_text}.")
        ]

        response = await circuit_breaker.call_llm_async(
            session_id,
            messages,
            f"Great! So let me give you some context - we're looking for someone with experience in {skills_text} and similar technologies. This will be a casual conversation - just think of it as a chat between colleagues. Ready to get started?"
        )

        return clean_bot_response(response)

    except Exception as e:
        logger.error(f"Error generating intro for session {session_id}: {e}")
        return f"Great! So let me give you some context - we're looking for someone with experience in relevant technologies. This will be a casual conversation - just think of it as a chat between colleagues. Ready to get started?"

# Intent classification prompt
intent_classification_prompt = """
You are an expert intent classifier for AI interview systems. Analyze the candidate's response in the context of the current question to determine their true intention.

CONTEXT ANALYSIS:
- Current Question: {current_question}
- Candidate Response: {user_answer}

TASK: Classify the candidate's intent by analyzing their response:
1. Analyze the user_answer content and keywords
2. Determine if they're answering the question or making a request
3. Consider the context of what was asked to understand their intent

Return ONLY one of these intent names:
RepeatQuestion, ClarifyQuestion, PreviousQuestion, NextQuestion, EndInterview, Hesitation, SmallTalk, OffTopic, Answer

CONTEXTUAL INTENT ANALYSIS:

1. **Answer** - Default when candidate provides substantive content related to the current question
   - They're sharing experiences, examples, or explanations relevant to what was asked
   - Even incomplete responses that attempt to address the question
   - Phrases like "I'm done", "that's all" when concluding their answer

2. **NextQuestion** - Only when explicitly requesting to move forward
   - Direct requests: "next question", "move to next", "let's move on", "skip this"
   - Must be clear they want to bypass current question, not just finishing their answer

3. **RepeatQuestion** - When they didn't hear or understand what was asked
   - "Can you repeat that?", "Sorry, I missed the question", "What was that again?"
   - Different from clarification - they need the question restated

4. **ClarifyQuestion** - When they heard the question but need explanation
   - "What do you mean by...", "Can you clarify...", "I don't understand..."
   - They understand words but need context or definition

5. **PreviousQuestion** - When wanting to return to earlier topics
   - "Can we go back to...", "I want to revisit...", "About the previous question..."

6. **EndInterview** - Only when clearly wanting to terminate entire interview
   - "I want to end the interview", "Can we wrap this up?", "I'd like to finish now"
   - NOT when just completing an answer ("I'm done with this question")

7. **Hesitation** - Thinking, processing, or nervous responses
   - "Umm...", "Let me think...", "Give me a moment...", "I'm not sure..."
   - Filler words while formulating response

8. **SmallTalk** - Personal/casual conversation unrelated to question
   - Weather, personal life, casual comments not addressing the interview question

9. **OffTopic** - Completely unrelated topics that don't fit SmallTalk
   - Random subjects with no connection to interview or current question

CLASSIFICATION LOGIC:
- If response contains content attempting to address the current question → Answer
- If response explicitly requests question navigation → NextQuestion/PreviousQuestion/RepeatQuestion
- If response seeks clarification about current question → ClarifyQuestion  
- If response indicates wanting to end entire interview → EndInterview
- If response contains thinking/processing indicators → Hesitation
- If response is casual conversation → SmallTalk
- If response is completely unrelated → OffTopic

CRITICAL: Always consider the current question context. A response that seems like "NextQuestion" might actually be "Answer" if they're concluding their response to the current question.

OUTPUT FORMAT REQUIREMENTS:
- Return ONLY the intent name without quotes
- NO markdown formatting (**, *, _, `)
- NO extra punctuation or formatting
- Direct, clean text output only

Return ONLY the intent name.
"""

async def classify_intent(session_id: str, question: str, answer: str, conversation_history: str = "") -> str:
    """Classify user intent using rule-based approach with LLM fallback"""
    if not answer or not answer.strip():
        return "Answer"

    answer_lower = answer.lower().strip()

    # Rule-based classification for better reliability
    if any(phrase in answer_lower for phrase in ["end interview", "finish interview", "stop interview", "wrap up", "i'm done", "want to end"]):
        return "EndInterview"

    if any(phrase in answer_lower for phrase in ["repeat", "say that again", "didn't hear", "missed that", "what was that"]):
        return "RepeatQuestion"

    if any(phrase in answer_lower for phrase in ["previous question", "go back", "last question", "before"]):
        return "PreviousQuestion"

    if any(phrase in answer_lower for phrase in ["clarify", "don't understand", "what do you mean", "explain", "confused"]):
        return "ClarifyQuestion"

    if any(phrase in answer_lower for phrase in ["umm", "uh", "let me think", "give me a moment", "not sure", "hmm"]):
        return "Hesitation"

    if any(phrase in answer_lower for phrase in ["weather", "nice day", "how are you", "weekend", "traffic"]) and len(answer_lower.split()) < 10:
        return "SmallTalk"

    # Try LLM classification as fallback for complex cases
    try:
        filled_prompt = intent_classification_prompt.format(
            current_question=question,
            user_answer=answer
        )

        intent_messages = [
            SystemMessage(content=filled_prompt),
            HumanMessage(content=f"Please classify the intent.")
        ]

        response = await circuit_breaker.call_llm_async(
            session_id,
            intent_messages,
            "Answer"
        )

        if response:
            llm_classification = response.strip()

            valid_intents = ["EndInterview", "RepeatQuestion", "PreviousQuestion", "ClarifyQuestion",
                           "OffTopic", "Hesitation", "SmallTalk", "Answer"]

            if llm_classification in valid_intents:
                return llm_classification

            for intent in valid_intents:
                if intent.lower() in llm_classification.lower():
                    return intent

        # Default to Answer for substantial responses
        return "Answer"

    except Exception as e:
        logger.error(f"Error in intent classification for session {session_id}: {e}")
        return "Answer"


def analyze_star_completeness(answer: str) -> dict:
    """Fast STAR analysis using simple logic"""
    if not answer or len(answer.strip()) < 20:
        return {"completeness": 0.1, "missing_elements": ["situation", "task", "action", "result"]}
    
    answer_lower = answer.lower()
    length = len(answer.split())
    
    has_context = any(word in answer_lower for word in ["when", "where", "project", "company", "time", "situation"])
    has_task = any(word in answer_lower for word in ["responsible", "needed", "goal", "task", "job"]) 
    has_action = any(word in answer_lower for word in ["i did", "i used", "i created", "i implemented", "my approach"])
    has_result = any(word in answer_lower for word in ["result", "outcome", "achieved", "improved", "success"])
    
    situation_score = 0.8 if has_context else 0.3
    task_score = 0.8 if has_task else 0.3
    action_score = 0.8 if has_action and length > 30 else 0.3
    result_score = 0.8 if has_result else 0.3
    
    completeness = (situation_score + task_score + action_score + result_score) / 4
    missing = [elem for elem, score in [("situation", situation_score), ("task", task_score), 
                                       ("action", action_score), ("result", result_score)] if score < 0.5]
    
    return {"completeness": completeness, "missing_elements": missing}

def detect_answer_patterns(answer: str) -> dict:
    """Fast quality check using simple heuristics"""
    if not answer or len(answer.strip()) < 10:
        return {"quality_ratio": 0.1}
    
    answer_lower = answer.lower()
    word_count = len(answer.split())
    
    red_indicators = ["wasn't my", "not my fault", "they didn't", "management failed", "team was"]
    red_count = sum(1 for indicator in red_indicators if indicator in answer_lower)
    
    green_indicators = ["i did", "i created", "i implemented", "i learned", "my responsibility", "i decided"]
    green_count = sum(1 for indicator in green_indicators if indicator in answer_lower)
    
    length_bonus = min(0.3, word_count / 100)
    
    quality_ratio = (green_count + length_bonus) / max(1, red_count + green_count + 0.5)
    
    return {"quality_ratio": min(1.0, quality_ratio)}

def should_ask_followup(question: str, answer: str, follow_up_count: int) -> bool:
    """Smart follow-up decision - more forgiving to maintain conversation flow"""
    if follow_up_count >= 2:
        return False
        
    if not answer or len(answer.strip()) < 15:
        return True
        
    star_analysis = analyze_star_completeness(answer)
    patterns = detect_answer_patterns(answer)
    
    # Be more forgiving - focus on conversation flow over strict STAR requirements
    if star_analysis["completeness"] < 0.2:  # More forgiving threshold
        return True
        
    if patterns["quality_ratio"] < 0.2:  # More forgiving threshold
        return True
        
    # Don't follow up just for length - respect natural conversation pace
    if len(answer.split()) < 15 and follow_up_count == 0:  # Only for very short answers
        return True
        
    # Allow natural progression - don't over-follow-up
        
    return False

async def generate_natural_transition(session_id: str, current_question: str, user_answer: str, next_question: str, role_title: str) -> str:
    """Generate natural, smooth transitions between interview questions"""
    try:
        messages = [
            SystemMessage(content=f"""You are a professional interviewer for a {role_title} position conducting a natural conversation flow.

Your task: Create a smooth, natural transition from the candidate's current answer to the next interview question.

Guidelines:
1. Briefly acknowledge their previous answer (show you were listening)
2. Create a natural bridge/connection to the next topic
3. Ask the next question naturally as part of the conversation
4. Sound conversational, not robotic or scripted
5. Keep the flow smooth and professional
6. Total response should be 25-40 words

Examples of natural transitions:
- "That's a great example of problem-solving. Speaking of challenges, [next question]"
- "I can see you have strong technical skills. Now I'd like to understand [next question]"
- "Thanks for sharing that experience. Let me shift gears a bit - [next question]"

OUTPUT FORMAT REQUIREMENTS:
- Return ONLY the transition + question without quotes
- NO markdown formatting (**, *, _, `)
- NO extra punctuation or formatting
- Direct, clean text output only

Generate a natural transition that flows smoothly from their answer to the next question."""),
            HumanMessage(content=f"Their answer to '{current_question}': {user_answer[:200]}...\n\nNext question to ask: {next_question}\n\nCreate a natural transition:")
        ]

        response = await circuit_breaker.call_llm_async(
            session_id,
            messages,
            f"That's interesting! Now let me ask: {next_question}"
        )

        return clean_bot_response(response)

    except Exception as e:
        logger.error(f"Error generating natural transition for session {session_id}: {e}")
        return f"Thank you for that insight. {next_question}"

async def generate_natural_conversation(session_id: str, question: str, answer: str, conversation_context: str = "", role_title: str = "", years_experience: str = "") -> str:
    """Generate professional interviewer responses using psychology tactics"""
    try:
        star_analysis = analyze_star_completeness(answer)
        patterns = detect_answer_patterns(answer)

        # Get skills from session
        session_context = await session_manager.get_session(session_id)
        key_skills = session_context.skills if session_context.skills else []
        skills_context = ", ".join(key_skills)

        psychology_context = ""
        if star_analysis["missing_elements"]:
            psychology_context += f"Missing STAR elements: {', '.join(star_analysis['missing_elements'])}. "
        if patterns["quality_ratio"] < 0.5:
            psychology_context += "Answer lacks specificity and ownership. "

        messages = [
            SystemMessage(content=f"""You are a professional interviewer for a {role_title} position conducting follow-up questions.

Key Skills Being Evaluated: {skills_context}

Current Analysis: {psychology_context}

CRITICAL INSTRUCTION: You are generating a FOLLOW-UP question to dig deeper into their CURRENT answer to this specific question: "{question}"

STRICT REQUIREMENTS:
- DO NOT ask any question that has already been asked in the conversation history
- DO NOT ask a completely new unrelated question
- DO NOT change the topic from their current answer
- DO probe deeper into their current response only
- DO ask for specifics about what they just mentioned
- DO clarify their role, actions, or results from their current answer
- ALWAYS reference something specific from their current answer

OUTPUT FORMAT REQUIREMENTS:
- Return ONLY the question text without quotes
- NO markdown formatting (**, *, _, `)
- NO extra punctuation or formatting
- Direct, clean text output only

Generate ONE follow-up question (15-25 words) that explores their current answer deeper."""),
            HumanMessage(content=f"CURRENT QUESTION: {question}\n\nCANDIDATE'S CURRENT ANSWER: {answer}\n\nFULL CONVERSATION HISTORY (Questions already asked - DO NOT REPEAT ANY OF THESE):\n{conversation_context if conversation_context else 'This is the first question'}\n\nGenerate a follow-up question that digs deeper ONLY into their current answer above.")
        ]

        response = await circuit_breaker.call_llm_async(
            session_id,
            messages,
            "That's interesting! Tell me more about that specific aspect."
        )

        return clean_bot_response(response)

    except Exception as e:
        logger.error(f"Error generating natural conversation for session {session_id}: {e}")
        return "That's a great point. Can you elaborate on that experience?"

# Interview flow routing function with session isolation
def route_interview_flow(state: InterviewState) -> Literal["greeting", "intro", "ask_question", "process_answer", "complete"]:
    """Determine next step in interview based on current state"""
    user_input = state.get('user_input', '').strip()
    current_idx = state.get('question_idx', 0)
    phase = state.get('phase','greeting')
    print('------------------------------------------------------------------------phase-------------------',phase)
    if state.get('done', False):
        return "complete"

    if phase == 'greeting':
        return "greeting"

    elif phase == 'intro':
        return "intro"

    elif phase == 'questions':
        # Use session-specific questions
        questions = state.get('session_questions', [])

        # Safety check - ensure we have questions to work with
        if not questions or len(questions) == 0:
            # Use fallback questions
            role_title = state.get('role_title', 'Software Developer')
            questions = [
                f"Tell me about your experience with {role_title} work.",
                "Describe a challenging situation you've faced.",
                "How do you handle working with teams?",
                "What are your career goals?"
            ]
            logger.warning(f"Using fallback questions for session {state.get('room_id')}")

        # Only complete if we've genuinely finished all available questions
        if current_idx >= len(questions):
            return "complete"

        if user_input:
            return "process_answer"

        return "ask_question"

    return "greeting"

# Node functions with session isolation
async def greeting_node(state: InterviewState):
    """Show greeting message or move to intro if user already responded to greeting"""
    try:
        user_input = state.get('user_input', '').strip().lower()
        print('------------------------------------------------------------------------user_input-------------------',user_input  )
        role_title = state.get('role_title', 'Software Developer')
        years_experience = state.get('years_experience', '2-5 years')
        candidate_name = state.get('candidate_name', 'Candidate')
        company_name = state.get('company_name', 'Company')
        session_id = state.get('room_id', 'default_session')

        # Ensure questions and skills are loaded properly
        try:
            print('------------------------------------------------------------------------test-------------------',state)
            
            skills, questions = await get_skills_questions_for_session(session_id, role_title, years_experience)
            # Update state with session-specific data
            state['session_skills'] = skills
            state['session_questions'] = questions
            print('------------------------------------------------------------------------test-------------------',state['session_questions'])
            await metrics.increment("successful_generations")
        except Exception as e:
            logger.error(f"Failed to load questions in greeting_node for session {session_id}: {e}")
            await metrics.increment("failed_generations")

        if user_input == "start":
            return {
                "bot_response": get_greeting_message(role_title, candidate_name, company_name),
                "phase": "intro",
                "done": False,
                "session_skills": state.get('session_skills', []),
                "session_questions": state.get('session_questions', [])
            }
        elif user_input != "start":
            intro_message = await get_intro_message(session_id, role_title, years_experience, candidate_name, company_name)
            return {
                "bot_response": intro_message,
                "phase": "intro",
                "done": False,
                "session_skills": state.get('session_skills', []),
                "session_questions": state.get('session_questions', [])
            }
    except Exception as e:
        logger.error(f"Error in greeting_node for session {state.get('room_id')}: {e}")
        return {
            "bot_response": "Hi! I'm Chris, and I'll be your interviewer today. Thanks for joining me - I'm looking forward to our conversation. How are you doing today?",
            "phase": "greeting",
            "done": False,
            "session_skills": [],
            "session_questions": []
        }

async def intro_node(state: InterviewState):
    """Show introduction or move to questions if user is ready"""
    try:
        current_idx = 0  # Always start with first question after intro
        session_id = state.get('room_id', 'default_session')
        role_title = state.get('role_title', 'Software Developer')
        years_experience = state.get('years_experience', '2-5 years')
        skills, questions = await get_skills_questions_for_session(session_id, role_title, years_experience)

        # questions = state.get('session_questions', [])
        # state['session_skills'] = skills
        # state['session_questions'] = questions
        # print(">>> intro_node invoked", questions)
        print('------------------------------------------------------------------------101-------------------',questions)
        if not questions or current_idx >= len(questions):
            return {"done": True, "bot_response": "Thank you! Interview complete.", "phase": "complete"}
    
        return {
            "bot_response": questions[current_idx],
            "phase": "questions",
            "question_idx": current_idx,
            "follow_up_count": 0,
            "last_question_asked": questions[current_idx],
            "done": False,
            "session_skills": state.get('session_skills', []),
            "session_questions": questions
        }

    except Exception as e:
        logger.error(f"Error in intro_node for session {state.get('room_id')}: {e}")
        session_id = state.get('room_id', 'default_session')
        role_title = state.get('role_title', 'Software Developer')
        years_experience = state.get('years_experience', '2-5 years')
        candidate_name = state.get('candidate_name', 'Candidate')
        company_name = state.get('company_name', 'Company')

        intro_message = await get_intro_message(session_id, role_title, years_experience, candidate_name, company_name)
        return {
            "bot_response": intro_message,
            "phase": "intro",
            "done": False,
            "session_skills": state.get('session_skills', []),
            "session_questions": state.get('session_questions', [])
        }

async def ask_question_node(state: InterviewState):
    """Ask current question based on question_idx"""
    try:
        print(">>> ask_question_node invoked, idx:", state.get('question_idx'))

        idx = state.get('question_idx', 0)
        questions = state.get('session_questions', [])

        if not questions or idx >= len(questions):
            return {"done": True, "bot_response": "Thank you! Interview complete.", "phase": "complete"}

        return {
            "bot_response": questions[idx],
            "question_idx": idx,
            "phase": "questions",
            "follow_up_count": 0,
            "last_question_asked": questions[idx],
            "done": False,
            "session_skills": state.get('session_skills', []),
            "session_questions": questions
        }
    except Exception as e:
        logger.error(f"Error in ask_question_node for session {state.get('room_id')}: {e}")
        return {
            "bot_response": "Let me ask you a question about your experience.",
            "phase": "questions",
            "done": False,
            "session_skills": state.get('session_skills', []),
            "session_questions": state.get('session_questions', [])
        }

# Intent-specific handler functions based on intent.flow specification

async def handle_end_interview_intent(state: InterviewState) -> Dict[str, Any]:
    """Handle EndInterview intent - complete interview immediately"""
    try:
        # Use dynamic role and experience from state
        role_title = state.get('role_title', 'this')
        
        messages = [
            SystemMessage(content=f"""You are a professional interviewer for a {role_title} position. The candidate has requested to end the interview early. 
            
Generate a professional, understanding response that:
1. Acknowledges their request respectfully
2. Thanks them for their time
3. Explains next steps professionally
4. Keeps it brief and positive (20-30 words)

OUTPUT FORMAT REQUIREMENTS:
- Return ONLY the conclusion text without quotes
- NO markdown formatting (**, *, _, `)
- NO extra punctuation or formatting
- Direct, clean text output only"""),
            HumanMessage(content="Candidate requested to end interview. Generate professional conclusion:")
        ]
        
        response = circuit_breaker.call_llm(messages, 
            "I understand. Thank you for your time today! We'll review our conversation and be in touch soon with next steps.")
        
        conclusion = response.content if response and response.content else \
            "I understand. Thank you for your time today! We'll review our conversation and be in touch soon with next steps."
        
        # Clear global variables on interview completion
        clear_global_variables()
        
        return {
            "done": True,
            "bot_response": conclusion,
            "phase": "complete",
            "interview_status": "end"
        }
        
    except Exception as e:
        logger.error(f"Error handling EndInterview intent: {e}")
        # Clear global variables on interview completion
        clear_global_variables()
        
        return {
            "done": True,
            "bot_response": "Thank you for your time today! We'll be in touch with next steps.",
            "phase": "complete",
            "interview_status": "end"
        }

def handle_repeat_question_intent(state: InterviewState) -> Dict[str, Any]:
    """Handle RepeatQuestion intent - repeat the actual last question asked (could be follow-up)"""
    try:
        current_idx = state.get('question_idx', 0)
        follow_up_count = state.get('follow_up_count', 0)

        # Get the actual last question asked (which could be a follow-up)
        last_question = state.get('last_question_asked', '')

        # Check if we're at the end of the interview
        role_title = state.get('role_title', '')
        years_experience = state.get('years_experience', '')
        questions = state.get('session_questions', [])
        
        if current_idx >= len(questions):
            # We've completed all questions
            repeat_response = "I'd be happy to repeat the question, but we've actually completed all the questions!"
        elif last_question and not any(completion_phrase in last_question.lower() for completion_phrase in ['thank you for completing', 'interview complete', 'wraps up our interview']):
            # Repeat the exact last question that was asked (but not completion messages)
            repeat_response = f"Of course! Let me repeat that question: {last_question}"
        else:
            # Fallback: get the base question if last_question_asked is not tracked
            if current_idx < len(questions):
                current_question = questions[current_idx]
                repeat_response = f"Of course! Let me repeat that question: {current_question}"
            else:
                repeat_response = "I'd be happy to repeat the question, but we've actually completed all the questions!"
        
        return {
            "bot_response": repeat_response,
            "question_idx": current_idx,  # Stay on same question
            "follow_up_count": follow_up_count,  # Maintain current follow-up count
            "phase": "questions",
            "done": False
        }
        
    except Exception as e:
        logger.error(f"Error handling RepeatQuestion intent: {e}")
        return {
            "bot_response": "Of course! Let me repeat that question for you.",
            "phase": "questions",
            "done": False
        }

def handle_previous_question_intent(state: InterviewState) -> Dict[str, Any]:
    """Handle PreviousQuestion intent - go back to previous question"""
    try:
        current_idx = state.get('question_idx', 0)

        if current_idx > 0:
            previous_idx = current_idx - 1

            # Use dynamic role and experience from state
            role_title = state.get('role_title', '')
            years_experience = state.get('years_experience', '')
            questions = state.get('session_questions', [])

            
            previous_question = questions[previous_idx]
            response = f"Sure, let's go back to the previous question: {previous_question}"
            
            return {
                "bot_response": response,
                "question_idx": previous_idx,
                "phase": "questions", 
                "follow_up_count": 0,  # Reset follow-ups
                "done": False
            }
        else:
            return {
                "bot_response": "We're actually on the first question, so there's no previous question to go back to. Let me repeat the current question instead.",
                "question_idx": current_idx,
                "phase": "questions",
                "done": False
            }
            
    except Exception as e:
        logger.error(f"Error handling PreviousQuestion intent: {e}")
        return {
            "bot_response": "Let me repeat the current question for you.",
            "phase": "questions",
            "done": False
        }

def handle_clarify_question_intent(state: InterviewState, user_input: str) -> Dict[str, Any]:
    """Handle ClarifyQuestion intent - explain/clarify current question"""
    try:
        current_idx = state.get('question_idx', 0)
        # Use dynamic role and experience from state
        role_title = state.get('role_title', '')
        years_experience = state.get('years_experience', '')
        questions = state.get('session_questions', [])
        
        if current_idx < len(questions):
            current_question = questions[current_idx]
            
            messages = [
                SystemMessage(content=f"""You are a professional interviewer for a {role_title} position. The candidate has asked for clarification on the current question.
                
Your task: Explain the question in a different way that's clearer and more specific.

Guidelines:
1. Break down what you're looking for in their answer
2. Give examples or context to help them understand
3. Be encouraging and supportive
4. Keep it concise but helpful (30-50 words)
5. Reference the role context when helpful

OUTPUT FORMAT REQUIREMENTS:
- Return ONLY the clarification text without quotes
- NO markdown formatting (**, *, _, `)
- NO extra punctuation or formatting
- Direct, clean text output only"""),
                HumanMessage(content=f"Current question: {current_question}\n\nCandidate's request for clarification: {user_input}\n\nProvide helpful clarification:")
            ]
            
            response = circuit_breaker.call_llm(messages, 
                f"Let me clarify that question. I'm looking for a specific example from your experience that demonstrates your skills. Think of a situation where you...")
            
            clarification = response.content if response and response.content else \
                f"Let me clarify that question. I'm looking for a specific example from your experience that demonstrates your skills in this area."
            
            return {
                "bot_response": clarification,
                "question_idx": current_idx,  # Stay on same question
                "phase": "questions",
                "done": False
            }
        else:
            return {
                "bot_response": "I'd be happy to clarify, but we've actually completed all the questions!",
                "phase": "questions",
                "done": False
            }
            
    except Exception as e:
        logger.error(f"Error handling ClarifyQuestion intent: {e}")
        return {
            "bot_response": "Let me clarify that question for you. I'm looking for a specific example from your experience.",
            "phase": "questions",
            "done": False
        }

def handle_hesitation_intent(state: InterviewState) -> Dict[str, Any]:
    """Handle Hesitation intent - provide gentle encouragement"""
    try:
        current_idx = state.get('question_idx', 0)
        
        encouragement_responses = [
            "Take your time! There's no rush. Think of a specific example from your experience.",
            "No worries at all. Let me give you a moment to think about it. Any example that comes to mind?",
            "That's perfectly fine! Take a moment to think. I'm looking for any relevant experience you might have.",
            "No problem! Sometimes it helps to think about recent projects or challenges you've worked on.",
            "Take your time! Even a simple example from your experience would be great to hear about."
        ]
        
        # Rotate through different encouragement messages
        encouragement = encouragement_responses[current_idx % len(encouragement_responses)]
        
        return {
            "bot_response": encouragement,
            "question_idx": current_idx,  # Stay on same question
            "phase": "questions",
            "done": False
        }
        
    except Exception as e:
        logger.error(f"Error handling Hesitation intent: {e}")
        return {
            "bot_response": "Take your time! Think of any relevant example from your experience.",
            "phase": "questions",
            "done": False
        }

def handle_smalltalk_intent(state: InterviewState, user_input: str) -> Dict[str, Any]:
    """Handle SmallTalk intent - polite redirect back to interview"""
    try:
        current_idx = state.get('question_idx', 0)

        role_title = state.get('role_title', 'this')
        
        messages = [
            SystemMessage(content=f"""You are a professional interviewer for a {role_title} position. The candidate has engaged in small talk instead of answering the interview question.

Your task: Politely acknowledge their comment and redirect back to the interview question.

Guidelines:
1. Briefly acknowledge what they said (show you were listening)
2. Smoothly transition back to the interview
3. Re-ask the current question or reference it
4. Stay professional but friendly
5. Keep it concise (15-25 words)

OUTPUT FORMAT REQUIREMENTS:
- Return ONLY the redirect text without quotes
- NO markdown formatting (**, *, _, `)
- NO extra punctuation or formatting
- Direct, clean text output only

Examples:
- "That's nice! Now, let's get back to the question about your technical experience..."
- "I appreciate that! For our interview though, I'd love to hear about..."
- "Interesting! Let's focus on the interview question though..."""),
            HumanMessage(content=f"Candidate's small talk: {user_input}\n\nGenerate polite redirect:")
        ]
        
        response = circuit_breaker.call_llm(messages, 
            "That's nice! Let's get back to the interview question though.")
        
        redirect = response.content if response and response.content else \
            "That's nice! Let's get back to the interview question though."
        
        # Get current question to re-ask
        role_title = state.get('role_title', '')
        years_experience = state.get('years_experience', '')
        questions = state.get('session_questions', [])

        
        if current_idx < len(questions):
            current_question = questions[current_idx]
            full_response = f"{redirect} {current_question}"
        else:
            full_response = redirect
        
        return {
            "bot_response": full_response,
            "question_idx": current_idx,  # Stay on same question
            "phase": "questions",
            "done": False
        }
        
    except Exception as e:
        logger.error(f"Error handling SmallTalk intent: {e}")
        return {
            "bot_response": "That's nice! Let's focus on the interview question though.",
            "phase": "questions", 
            "done": False
        }

# Intelligent off-topic response handler
def handle_offtopic_intelligently(user_input: str) -> str:
    """Generate intelligent, natural responses to off-topic input"""
    try:
        if not user_input or not user_input.strip():
            return "I hear you, but let's keep our focus on the interview."

        role_title = "this"  # Minimal fallback for off-topic handling
        
        messages = [
            SystemMessage(content=f"""You are a professional AI interviewer for a {role_title} position. The candidate has gone off-topic from the interview discussion. 

Your task: Generate a professional but understanding response that acknowledges their comment and smoothly redirects to the interview.

Response Requirements:
1. Brief acknowledgment of what they said (show you were listening)
2. Smooth transition back to interview topics
3. Maintain professional but friendly tone
4. Keep it concise (10-15 words)
5. Stay positive and encouraging

OUTPUT FORMAT REQUIREMENTS:
- Return ONLY the redirect text without quotes
- NO markdown formatting (**, *, _, `)
- NO extra punctuation or formatting
- Direct, clean text output only

Professional Redirect Examples:
- "I understand. Now, let's get back to discussing your technical experience."
- "That's interesting! Let's refocus on your background and experience."
- "I hear you. Let's continue with the interview and talk about your projects."
- "Got it! Now, back to learning about your professional experience."

Generate a natural, professional redirect that maintains interview flow."""),
            HumanMessage(content=f"Candidate's off-topic comment: {user_input}\n\nGenerate professional redirect:")
        ]
        
        response = circuit_breaker.call_llm(messages, "I hear you! Let's keep our focus on the interview though.")
        
        if response and response.content:
            redirect = response.content.strip()
            return redirect
            
        # Fallback
        return "I hear you! Let's keep our focus on the interview though."
        
    except Exception as e:
        logger.error(f"Error handling off-topic intelligently: {e}")
        print(f"Error handling off-topic intelligently: {e}")
        return "I understand! Let's get back to the interview."

async def process_answer_node(state: InterviewState):
    """Process candidate's answer using professional interviewer psychology"""
    try:
        session_id = state.get('room_id', 'default_session')
        current_idx = state.get('question_idx', 0)
        user_answer = state.get('user_input', '').strip()
        follow_up_count = state.get('follow_up_count', 0)
        
        role_title = state.get('role_title', 'Software Developer')
        years_experience = state.get('years_experience', '2-5 years')
        questions = state.get('session_questions', [])
        
        # Defensive: ensure questions array is valid before accessing
        if not questions or len(questions) == 0:
            logger.warning("Empty questions in process_answer_node, using fallback")
            questions = [
                f"Tell me about your experience with {role_title} work.",
                "Describe a challenging situation you've faced.",
                "How do you handle working with teams?",
                "What are your career goals?"
            ]
        
        if current_idx < len(questions):
            current_question = questions[current_idx]
        else:
            current_question = ""
        
        star_analysis = analyze_star_completeness(user_answer)
        patterns = detect_answer_patterns(user_answer)
        
        conversation_history = state.get('conversation_history', '')
        user_intent = await classify_intent(session_id, current_question, user_answer, conversation_history)
        
        # Route based on intent classification (as per intent.flow specification)
        if user_intent == "EndInterview":
            return await handle_end_interview_intent(state)

        elif user_intent == "RepeatQuestion":
            return handle_repeat_question_intent(state)

        elif user_intent == "PreviousQuestion":
            return handle_previous_question_intent(state)

        elif user_intent == "ClarifyQuestion":
            return handle_clarify_question_intent(state, user_answer)

        elif user_intent == "Hesitation":
            return handle_hesitation_intent(state)

        elif user_intent == "SmallTalk":
            return handle_smalltalk_intent(state, user_answer)
            
        elif user_intent == "OffTopic":
            # Off-topic: Give intelligent response + move to next question
            intelligent_redirect = handle_offtopic_intelligently(user_answer)
            next_idx = current_idx + 1
            
            # Check if we've reached the end of questions
            if next_idx >= len(questions):
                # Clear global variables on interview completion
                clear_global_variables()
                
                return {
                    "done": True, 
                    "bot_response": f"{intelligent_redirect} Actually, that wraps up our interview! Thank you for the great conversation. We'll be in touch soon with next steps.", 
                    "question_idx": next_idx,
                    "phase": "complete",
                    "follow_up_count": 0,
                    "interview_status": "end"
                }
            
            # Combine intelligent response with next question
            full_response = f"{intelligent_redirect}\n\n{questions[next_idx]}"
            
            return {
                "bot_response": full_response,
                "question_idx": next_idx,
                "phase": "questions", 
                "follow_up_count": 0,  # Reset follow-ups for next question
                "done": False
            }
        
        # Intent: "Answer" - Process as a legitimate answer to the interview question
        elif user_intent == "Answer":
            # This is the main interview flow path - process the answer professionally
            pass  # Continue to answer processing logic below
        else:
            # Fallback for unknown intents - treat as Answer
            logger.warning(f"Unknown intent '{user_intent}', treating as Answer")
        
        # Main answer processing logic
        conversation_context = state.get('conversation_history', '')
        needs_followup = should_ask_followup(current_question, user_answer, follow_up_count)
        if  needs_followup and follow_up_count < 2:
            natural_response = await generate_natural_conversation(session_id, current_question, user_answer, conversation_context, role_title, years_experience)
            
            updated_history = f"{conversation_context}\nQ: {current_question}\nA: {user_answer}\nInterviewer: {natural_response}"
            
            return {
                "bot_response": natural_response,
                "question_idx": current_idx,
                "phase": "questions",
                "follow_up_count": follow_up_count + 1,
                "conversation_history": updated_history,
                "last_question_asked": natural_response,
                "done": False
            }
        
        # Move to next question
        next_idx = current_idx + 1
        if next_idx >= len(questions):
            messages = [
                SystemMessage(content=f"""You are a professional interviewer concluding a {role_title} interview. Generate a natural, warm conclusion that thanks them and explains next steps.

OUTPUT FORMAT REQUIREMENTS:
- Return ONLY the conclusion text without quotes
- NO markdown formatting (**, *, _, `)
- NO extra punctuation or formatting
- Direct, clean text output only"""),
                HumanMessage(content=f"Generate a natural interview conclusion for the {role_title} position:")
            ]
            
            response = circuit_breaker.call_llm(messages, "Thank you for a wonderful conversation! I really enjoyed learning about your experience and background. We'll be reviewing everything we discussed and will get back to you soon with next steps.")
            conclusion = response.content if response and response.content else "Thank you for a wonderful conversation! I really enjoyed learning about your experience and background. We'll be reviewing everything we discussed and will get back to you soon with next steps."
            
            # Clear global variables on interview completion
            clear_global_variables()
            
            return {
                "done": True, 
                "bot_response": conclusion,
                "question_idx": next_idx,
                "phase": "complete",
                "follow_up_count": 0,
                "conversation_history": f"{conversation_context}\nConclusion: {conclusion}"
            }
        
        # Generate natural transition to next question
        # Defensive: ensure we have a valid next question before generating transition
        if next_idx < len(questions):
            transition_response = await generate_natural_transition(session_id, current_question, user_answer, questions[next_idx], role_title)
        else:
            # Fallback if somehow we're out of bounds
            logger.warning(f"Next question index {next_idx} out of bounds, using simple transition")
            transition_response = "Thank you for that answer. Let me ask you about something else."
        updated_history = f"{conversation_context}\nQ: {current_question}\nA: {user_answer}\nTransition: {transition_response}"
        
        return {
            "bot_response": transition_response,
            "question_idx": next_idx,
            "phase": "questions",
            "follow_up_count": 0,
            "conversation_history": updated_history,
            "last_question_asked": transition_response,
            "done": False
        }
        
    except Exception as e:
        logger.error(f"Error in process_answer_node: {e}")
        next_idx = state.get('question_idx', 0) + 1
        role_title = state.get('role_title', 'Software Developer')
        questions_fallback = state.get('session_questions', ["Tell me about your experience.", "Describe a challenge you've faced.", "How do you work with teams?", "What are your career goals?"])
        if next_idx >= len(questions_fallback):
            # Generate natural interview conclusion
            messages = [
                SystemMessage(content=f"""You are a professional interviewer concluding a {role_title} interview. Generate a natural, warm conclusion.

OUTPUT FORMAT REQUIREMENTS:
- Return ONLY the conclusion text without quotes
- NO markdown formatting (**, *, _, `)
- Direct, clean text output only"""),
                HumanMessage(content=f"Generate a natural interview conclusion for the {role_title} position:")
            ]
            
            response = circuit_breaker.call_llm(messages, "Thank you for a wonderful conversation! We'll be in touch with next steps.")
            conclusion = response.content if response and response.content else "Thank you for a wonderful conversation! We'll be in touch with next steps."
            
            # Clear global variables on interview completion
            clear_global_variables()
            return {
                "done": True,
                "bot_response": conclusion,
                "phase": "complete"
            }
async def clear_session_cache_async(session_id: str):
    """Async helper to clear all in-memory session data for a given session id."""
    try:
        # Clear session manager & circuit breaker state
        await clear_session_data(session_id)
        # If you have other caches (Redis keys, local dicts) clear them here too.
        logger.info(f"Cleared session cache (async) for session: {session_id}")
    except Exception as e:
        logger.error(f"Failed to clear session cache (async) for {session_id}: {e}")

def clear_session_cache(session_id: str):
    """
    Synchronous wrapper for websocket sync callers.
    This will schedule an async cleanup task and return immediately.
    """
    try:
        # schedule the async cleanup to avoid blocking callers
        asyncio.create_task(clear_session_cache_async(session_id))
        logger.info(f"Scheduled session cache clear for: {session_id}")
    except Exception as e:
        # As fallback, call sync clear (best-effort)
        logger.error(f"Failed to schedule clear_session_cache for {session_id}: {e}")
        # Try to run a blocking loop safely if there isn't one (rare)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # cannot run sync, just log
                logger.warning("Event loop already running; cleanup scheduled may not run immediately.")
            else:
                loop.run_until_complete(clear_session_cache_async(session_id))
        except Exception:
            pass

def complete_interview_node(state: InterviewState):
    """Complete the interview with professional assessment"""
    try:
        conversation_history = state.get('conversation_history', '')
        role_title = state.get('role_title', 'this')
        
        messages = [
            SystemMessage(content=f"""You are a senior interviewer concluding a {role_title} interview. 

Generate a professional, warm conclusion that:
1. Thanks them genuinely 
2. Mentions a specific strength you observed (be authentic)
3. Explains realistic next steps for the {role_title} role
4. Maintains professional confidence without making promises

OUTPUT FORMAT REQUIREMENTS:
- Return ONLY the conclusion text without quotes
- NO markdown formatting (**, *, _, `)
- NO extra punctuation or formatting
- Direct, clean text output only

Sound like a real hiring manager who was actively listening and evaluating. 
Keep it conversational but professional (30-40 words)."""),
            HumanMessage(content=f"Interview conversation: {conversation_history[-400:]}\n\nGenerate professional conclusion:")
        ]
        
        response = circuit_breaker.call_llm(messages, 
            f"Thank you for a great conversation! I enjoyed learning about your experience and approach to technical challenges. We'll review everything and get back to you within a few days about next steps.")
        
        conclusion = response.content if response and response.content else f"Thank you for completing the interview! It was great getting to know you and learning about your experience. We'll be reviewing your background against our requirements and will be in touch soon with next steps."
        
        # Clear global variables on interview completion
        clear_global_variables()
        
        return {
            "done": True,
            "bot_response": conclusion,
            "phase": "complete"
        }
        
    except Exception as e:
        logger.error(f"Error in complete_interview_node: {e}")
        # Clear global variables on interview completion
        clear_global_variables()
        
        return {
            "done": True,
            "bot_response": f"Thank you for completing the interview! We'll be in touch soon with next steps.",
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

# Set entry point

# print('-----------------------------------before adding conditional edges',state.get('phase'))
builder.add_conditional_edges(START, route_interview_flow)

# All nodes end after execution
builder.add_edge("greeting", END)
builder.add_edge("intro", END)
# builder.add_edge("intro", "ask_question")   # ensures intro and ask_question run in same invocation

# builder.add_edge("intro", "ask_question") 
builder.add_edge("ask_question", END)
builder.add_edge("process_answer", END)
builder.add_edge("complete", END)

graph = builder.compile()

class InterviewFlowService:
    """Service class to handle interview flow operations with SessionManager support"""
    
    def __init__(self):
        self.use_session_manager = True  # Enable SessionManager by default
    
    async def trigger_session_preparation(self, role_title: str, years_experience: str, session_id: str):
        """Prepare session data using SessionManager"""
        try:
            if self.use_session_manager:
                # Use new SessionManager approach
                await get_skills_questions_hybrid(session_id, role_title, years_experience)
            else:
                # Use legacy background generation
                start_background_generation(role_title, years_experience, session_id)
        except Exception as e:
            logger.error(f"Session preparation failed: {e}")
            # Fallback to legacy
            start_background_generation(role_title, years_experience, session_id)
    
    async def process_interview_flow(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process interview turn using LangGraph conditional flow"""
        print("🚀 === INTERVIEW FLOW SERVICE - PROCESS_INTERVIEW_FLOW ===" )
        print(f"🚀 Service method invoked with payload type: {type(payload)}")
        
        try:
            data = payload if isinstance(payload, dict) else json.loads(payload)
            
            # Add comprehensive logging of received data
            logger.info(f"=== INTERVIEW FLOW SERVICE - RECEIVED DATA ===")
            logger.info(f"Raw payload type: {type(payload)}")
            logger.info(f"Raw payload: {payload}")
            logger.info(f"Parsed data: {data}")
            logger.info(f"Data keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
            
            # Extract key fields
            user_input = data.get('text', '')
            question_idx = data.get('question_idx', 0)
            phase = data.get('phase', 'greeting')
            
            # Extract role and experience from payload (dynamic from WebSocket)
            role_title = data.get('role_title') or data.get('role_type') or 'Software Developer'
            years_experience = data.get('years_experience') or '2-5 years'
            candidate_name = data.get('candidate_name', 'Candidate')
            company_name = data.get('company_name', 'Company')
            session_id = data.get('room_id', 'default_session')
            
            # Immediately trigger session preparation (SessionManager)
            await self.trigger_session_preparation(role_title, years_experience, session_id)
            
            logger.info(f"KEY DEBUG INFO - user_input: '{user_input}', question_idx: {question_idx}, phase: '{phase}'")
            logger.info(f"ROLE DEBUG INFO - role_title: '{role_title}', years_experience: '{years_experience}'")
            
            state: InterviewState = {
                'messages': [],
                'room_id': data.get('room_id', ''),
                'user_input': user_input,
                'bot_response': '',
                'question_idx': question_idx,
                'done': data.get('done', False),
                'phase': phase,
                'follow_up_count': data.get('follow_up_count', 0),
                'conversation_history': data.get('conversation_history', ''),
                'role_title': role_title,
                'years_experience': years_experience,
                'candidate_name': candidate_name,
                'company_name': company_name,
                'last_question_asked': data.get('last_question_asked', '')
            }
            
            logger.info(f"Constructed state: {state}")
            logger.info(f"=== END RECEIVED DATA LOG ===")
            
            user_text = data.get('text', '').strip()
            current_question_idx = data.get('question_idx', 0)
            
            print(f"Processing interview - User text: '{user_text}', Question index: {current_question_idx}, Phase: {state.get('phase', 'greeting')}")
            print(f"Full state: {state}")
            
            # Use LangGraph with conditional flow
            # result = graph.invoke(state)
            result = await graph.ainvoke(state)

            # Add the updated question_idx to response for WebSocket state management
            if 'question_idx' not in result and not result.get('done', False):
                result['question_idx'] = current_question_idx
            
            # Clean bot response before sending to frontend
            if 'bot_response' in result and result['bot_response']:
                result['bot_response'] = clean_bot_response(result['bot_response'])
            
            # Add interview status for frontend
            if result.get('done', False):
                result['interview_status'] = 'end'
            else:
                result['interview_status'] = 'ongoing'
                
            
            return result
            
        except Exception as e:
            logger.error(f"Error in interview flow service: {e}")
            return {
                'bot_response': 'Sorry, there was an issue. Let me restart.',
                'error': str(e),
                'question_idx': 0,
                'done': False,
                'interview_status': 'error'
            }

# Global service instance
interview_flow_service = InterviewFlowService()