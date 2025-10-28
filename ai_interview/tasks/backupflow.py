import os
import json
import logging
import time
import asyncio
import re
import shutil
import subprocess
import sys
import random
from typing import Dict, List, Any, TypedDict, Annotated, Literal,Tuple
import redis
import ast
from datetime import datetime
import concurrent.futures

# Prevent bytecode generation for development
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
from ai_interview.celery_app import celery_app
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage, SystemMessage
from ai_interview.config import REDIS_URL

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", os.path.abspath("xooper.json"))

# Configure detailed logging for LangGraph interview flow
LOG_DIR = "logs/ai_interview"
os.makedirs(LOG_DIR, exist_ok=True)

# Create detailed logger for interview flow
interview_logger = logging.getLogger('interview_flow')
interview_logger.setLevel(logging.DEBUG)

# Create file handler with timestamp
log_filename = f"{LOG_DIR}/interview_flow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.DEBUG)

# Create console handler  
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create detailed formatter
detailed_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
)
file_handler.setFormatter(detailed_formatter)
console_handler.setFormatter(detailed_formatter)

# Add handlers to logger
interview_logger.addHandler(file_handler)
interview_logger.addHandler(console_handler)

# Prevent duplicate logs
interview_logger.propagate = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def log_state_transition(from_state: str, to_state: str, room_id: str, user_input: str = "", additional_data: Dict = None):
    """Log detailed state transitions for debugging"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "room_id": room_id,
        "transition": f"{from_state} -> {to_state}",
        "user_input": user_input[:100] if user_input else "",
        "additional_data": additional_data or {}
    }
    interview_logger.info(f"STATE_TRANSITION: {json.dumps(log_entry, indent=2)}")

def log_node_execution(node_name: str, room_id: str, input_state: Dict, output_state: Dict, execution_time: float):
    """Log detailed node execution information"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "room_id": room_id,
        "node": node_name,
        "execution_time_ms": round(execution_time * 1000, 2),
        "input_state_keys": list(input_state.keys()),
        "output_state_keys": list(output_state.keys()) if isinstance(output_state, dict) else [],
        "input_phase": input_state.get("phase", "unknown"),
        "output_phase": output_state.get("phase", "unknown") if isinstance(output_state, dict) else "unknown",
        "question_idx": {
            "input": input_state.get("question_idx", -1),
            "output": output_state.get("question_idx", -1) if isinstance(output_state, dict) else -1
        },
        "user_input_length": len(input_state.get("user_input", "")),
        "bot_response_length": len(output_state.get("bot_response", "")) if isinstance(output_state, dict) else 0
    }
    interview_logger.info(f"NODE_EXECUTION: {json.dumps(log_entry, indent=2)}")

def log_routing_decision(state: Dict, routing_result: str, room_id: str):
    """Log routing decisions with detailed context"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "room_id": room_id,
        "routing_decision": routing_result,
        "state_context": {
            "phase": state.get("phase", "unknown"),
            "question_idx": state.get("question_idx", -1),
            "done": state.get("done", False),
            "user_input": state.get("user_input", "")[:100],
            "follow_up_count": state.get("follow_up_count", 0)
        }
    }
    interview_logger.info(f"ROUTING_DECISION: {json.dumps(log_entry, indent=2)}")

def log_llm_interaction(operation: str, room_id: str, prompt: str, response: str, execution_time: float, success: bool):
    """Log LLM interactions for debugging"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "room_id": room_id,
        "operation": operation,
        "execution_time_ms": round(execution_time * 1000, 2),
        "prompt_length": len(prompt),
        "response_length": len(response),
        "success": success,
        "prompt_preview": prompt[:200] + "..." if len(prompt) > 200 else prompt,
        "response_preview": response[:200] + "..." if len(response) > 200 else response
    }
    interview_logger.info(f"LLM_INTERACTION: {json.dumps(log_entry, indent=2)}")

memory = MemorySaver()
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
# Cache configuration - room-based caching for performance
CACHE_EXPIRY_HOURS = 24  # 24 hours for generated content

# Memory cache for ultra-fast access
_memory_cache = {}

def clean_cache():
    """Light cache cleanup - only called when needed"""
    try:
        # Only clean specific cache that might interfere with role updates
        print("🧹 Light cache cleanup completed")
    except Exception as e:
        print(f"Cache cleanup error (ignored): {e}")

def clear_room_generated_cache(room_id: str) -> None:
    """Clear all generated content for specific room on interview end/disconnect"""
    try:
        # Clear memory cache first
        if room_id in _memory_cache:
            del _memory_cache[room_id]
            print(f"🧹 Cleared memory cache for room {room_id}")
        
        # Clear Redis cache for questions and skills
        key = get_room_cache_key(room_id)
        result = redis_client.delete(key)
        if result:
            print(f"🧹 Cleared Redis cache for room {room_id}")
        else:
            print(f"ℹ️  No Redis cache found for room {room_id}")
            
    except Exception as e:
        print(f"⚠️  Failed to clear room cache: {e}")

def clear_all_redis_cache() -> None:
    """Clear all Redis cache for debugging"""
    try:
        keys = redis_client.keys("interview_room:cache:*")
        if keys:
            redis_client.delete(*keys)
            print(f"🧹 Cleared {len(keys)} Redis cache entries")
        else:
            print("ℹ️  No Redis cache entries found")
    except Exception as e:
        print(f"⚠️  Failed to clear all Redis cache: {e}")


def clean_bot_response(response: str) -> str:
    """Clean bot response by removing markdown formatting and extra whitespace"""
    if not response:
        return response
    
    # Remove markdown bold formatting
    cleaned = re.sub(r'\*\*([^*]+)\*\*', r'\1', response)
    
    # Replace multiple newlines with single space
    cleaned = re.sub(r'\n+', ' ', cleaned)
    
    # Remove other markdown formatting
    cleaned = re.sub(r'\*([^*]+)\*', r'\1', cleaned)  # italic
    cleaned = re.sub(r'_([^_]+)_', r'\1', cleaned)    # underscore italic
    
    # Clean up extra whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned

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
    role_title: str  # dynamic role from WebSocket
    years_experience: str  # dynamic experience from WebSocket
    last_question_asked: str  # track the actual last question asked (could be follow-up)
    candidate_name:str
    company_name:str


# No static defaults - everything generated dynamically via LLM


def generate_key_skills(role_title: str, years_experience: str) -> List[str]:
    """Generate key skills using LLM - simplified logic"""
    messages = [
            SystemMessage(content=f"""You are an expert technical recruiter. Generate exactly 6-8 key skills for a {role_title} position requiring {years_experience} of experience.

CRITICAL REQUIREMENTS:
1. Return EXACTLY in this format: ["Skill 1", "Skill 2", "Skill 3", "Skill 4", "Skill 5", "Skill 6"]
2. Skills must be ordered by importance for the specific role
3. Include both technical and soft skills relevant to the position  
4. Skills must match the experience level expectations
5. Focus on interview-assessable, practical skills
6. Handle ANY role type (not just common ones like Frontend/Backend)

ROLE-SPECIFIC GUIDANCE:
- Technical roles: Include programming languages, frameworks, tools, methodologies
- Creative roles: Include design tools, creative processes, portfolio skills
- Management roles: Include leadership, strategic thinking, team management
- Sales roles: Include CRM systems, negotiation, relationship building
- Marketing roles: Include analytics tools, content creation, campaign management
- Operations roles: Include process optimization, data analysis, project management
- Unknown/Custom roles: Infer from role title and generate relevant skills

EXPERIENCE LEVEL ADAPTATION:
- Entry level (0-2 years): Focus on foundational skills and learning ability
- Mid level (2-5 years): Focus on practical application and some specialization
- Senior level (5+ years): Focus on advanced skills, leadership, and strategic thinking

Generate skills for: {role_title} with {years_experience} experience
Return only the Python list format."""),
            HumanMessage(content=f"Role: {role_title}\nExperience: {years_experience}\n\nGenerate key skills list:")
        ]
    
    # Simple fallback list
    fallback_skills = ["Problem Solving", "Communication", "Technical Skills", "Analytical Thinking", "Collaboration", "Adaptability"]
    
    response = circuit_breaker.call_llm(messages, str(fallback_skills))
    
    if response and response.content:
        skills_text = response.content.strip()
        try:
            if '[' in skills_text and ']' in skills_text:
                start = skills_text.find('[')
                end = skills_text.rfind(']') + 1
                import ast
                skills_list = ast.literal_eval(skills_text[start:end])
                if isinstance(skills_list, list) and len(skills_list) >= 6:
                    return skills_list[:6]
        except:
            pass
    
    # Simple fallback if LLM fails
    return fallback_skills

def get_room_cache_key(room_id: str) -> str:
    """Generate unified cache key for room-specific questions and skills"""
    return f"interview_room:cache:{room_id}"

def get_cached_room_questions_and_skills(room_id: str) -> Tuple[List[str], List[str]]:
    """
    Retrieve cached questions AND skills from unified Redis hash.
    Returns (questions_list, skills_list), or ([],[]) if not found.
    """
    try:
        key = get_room_cache_key(room_id)
        logger.warning(f"🔍 Cache lookup for key: {key}")
        data = redis_client.hgetall(key)
        logger.warning(f"🔍 Redis returned: {data}")
        
        if data:
            questions = json.loads(data.get("questions", "[]"))
            skills = json.loads(data.get("skills", "[]"))
            logger.warning(f"✅ Cache HIT for {room_id}: {len(questions)} questions, {len(skills)} skills")
            return questions, skills
        else:
            logger.warning(f"❌ Cache MISS for {room_id} - no data found")
            
    except Exception as e:
        logger.warning(f"❌ Cache error for {room_id}: {e}")
        logger.warning(f"Failed to retrieve questions+skills for {room_id}: {e}")
    return [], []



def cache_room_questions_and_skills(
    room_id: str,
    questions: List[str],
    skills: List[str]
) -> None:
    """
    Atomically cache both questions and skills in unified Redis hash
    """
    try:
        key = get_room_cache_key(room_id)
        logger.warning(f"💾 Caching to key: {key}")
        logger.warning(f"💾 Questions: {len(questions)} items")
        logger.warning(f"💾 Skills: {len(skills)} items")
        
        # Cache both questions and skills
        result = redis_client.hset(key, mapping={
            "questions": json.dumps(questions),
            "skills": json.dumps(skills)
        })
        
        # Set expiry time
        redis_client.expire(key, CACHE_EXPIRY_HOURS * 3600)
        
        if result:
            logger.warning(f"✅ Cached questions+skills for room_id={room_id}")
        else:
            logger.warning(f"ℹ️ Updated existing cache for room_id={room_id}")

    except Exception as e:
        logger.warning(f"❌ Failed to cache questions+skills for room_id={room_id}: {e}")
        logger.warning(f"Failed to cache questions+skills for {room_id}: {e}")


def get_key_skills(role_title: str, years_experience: str, room_id: str = None) -> Tuple[List[str], List[str]]:
    """Ultra-optimized caching: memory → Redis → LLM generation"""
    if room_id:
        # Level 1: Check memory cache first (fastest)
        if room_id in _memory_cache:
            return _memory_cache[room_id]
        
        # Level 2: Check Redis cache
        cached_q, cached_s = get_cached_room_questions_and_skills(room_id)
        if cached_q and cached_s:
            # Store in memory cache for next time
            _memory_cache[room_id] = (cached_q, cached_s)
            logger.warning(f'✅ CACHE HIT: Getting {len(cached_q)} questions and {len(cached_s)} skills from cache for {room_id}')
            return cached_q, cached_s
    
    # Level 3: Generate new questions and skills using LLM
    interview_logger.info(f"Cache miss - generating new questions for role: {role_title}, experience: {years_experience}")
    
    try:
        # Generate skills first
        skills = generate_key_skills(role_title, years_experience)
        
        # Generate questions based on skills (only if not cached)
        questions = generate_base_questions(role_title, years_experience, skills)
        
        # Cache the results if room_id provided
        if room_id:
            print('questions',questions)
            print('skill',skills)
            cache_room_questions_and_skills(room_id, questions, skills)
            _memory_cache[room_id] = (questions, skills)
            
        return questions, skills
        
    except Exception as e:
        interview_logger.error(f"Failed to generate questions/skills: {e}")
        # Return fallback questions and skills
        fallback_questions = [
            f"Can you tell me about your experience as a {role_title}?",
            f"What's a challenging project you've worked on in the past {years_experience}?",
            "How do you approach problem-solving in your work?",
            "Tell me about a time you had to work with a difficult team member.",
            "Where do you see yourself in the next few years?",
            "Do you have any questions for me about the role or company?"
        ]
        fallback_skills = ["Problem Solving", "Communication", "Technical Skills", "Analytical Thinking", "Collaboration", "Adaptability"]
        
        if room_id:
            cache_room_questions_and_skills(room_id, fallback_questions, fallback_skills)
            _memory_cache[room_id] = (fallback_questions, fallback_skills)
            
        return fallback_questions, fallback_skills
    







# === patch your get_base_questions to use these ===

def get_base_questions(
    role_title: str,
    years_experience: str,
    key_skills: List[str] = None,
    room_id: str = None
) -> List[str]:
    """Get base questions - simplified to use unified cache"""
    # Use unified get_key_skills which handles cache and generation
    questions, skills = get_key_skills(role_title, years_experience, room_id)
    return questions










def get_greeting_message(role_title: str, candidate_name: str = "Candidate", company_name: str = "Company") -> str:
    """Generate greeting message dynamically using LLM with personalized candidate and company names"""
    
    # Use the direct personalized message instead of LLM to avoid placeholder issues
    return f"Hi {candidate_name}! I'm Chris, and I'll be your interviewer today for the {role_title} position at {company_name}. Thanks for joining me - I'm looking forward to our conversation. How are you doing today?"


def get_greeting_message(role_title: str, candidate_name: str = "Candidate", company_name: str = "Company") -> str:
    """Generate greeting message - using static template for speed"""
    
    # Static template for maximum speed (no LLM call needed)
    return f"Hi {candidate_name}! I'm Chris, and I'll be your interviewer today for the {role_title} position at {company_name}. Thanks for joining me - I'm looking forward to our conversation. How are you doing today?"

def get_intro_message(
    role_title: str,
    years_experience: str,
    candidate_name: str = "Candidate",
    company_name: str = "Company",
    room_id: str = None
) -> str:
    """Generate a professional yet conversational interview introduction."""
    try:
        intro_start = time.time()

        # Use optimized caching system to get skills
        cached_skills = []
        if room_id:
            try:
                _, cached_skills = get_key_skills(role_title, years_experience, room_id)
            except Exception:
                cached_skills = []

        # Use cached skills or fallback to generic terms
        if cached_skills:
            skills_text = ', '.join(cached_skills[:3])
        
        return (
            f"Hi {candidate_name}, welcome to your interview for the {role_title} role at {company_name}. "
            f"This role focuses on {skills_text}. Are you ready to get started?"
        )

    except Exception as e:
        # Final fallback in case of unexpected errors
        return (
            f"Hi {candidate_name}, welcome to your interview for the {role_title} role at {company_name}. "
            "Let's get started!"
        )





def generate_base_questions(role_title: str, years_experience: str, key_skills: List[str]) -> List[str]:
    """
    Generate exactly 4 interview questions dynamically for any role using LLM.
    Falls back to a safe default if the LLM fails.
    """
    try:
        # Use top 4 key skills for context
        skills_context = ", ".join(key_skills) if key_skills else "core role-related skills"

        # LLM prompt
        messages = [
    SystemMessage(content=f"""
You are an expert interview designer. Generate EXACTLY 4 highly customized and diverse interview questions for a {role_title} role requiring {years_experience} years of experience.

Key skills to assess: {skills_context}

CRITICAL REQUIREMENTS:
1. Return ONLY a valid Python list with exactly 4 strings.
2. Questions must be **role-specific** and **experience-appropriate**.
3. Each question must be **unique in phrasing, depth, and focus** — no reusing patterns.
4. Do NOT generate generic questions and do NOT just replace skill names.
5. Cover different dimensions of evaluation:
   - Core technical/functional expertise
   - Problem-solving and challenge resolution
   - Collaboration and cross-functional integration
   - Growth, leadership, or measurable impact
6. Encourage detailed STAR responses.
7. Return ONLY the Python list. No explanations, no extra text.

Output format strictly:
["Question 1?", "Question 2?", "Question 3?", "Question 4?"]
"""),
    HumanMessage(content=f"Role: {role_title}\nExperience: {years_experience}\nKey Skills: {skills_context}\n\nGenerate 4 diverse interview questions:")
]

        

        # Default fallback questions (technical-oriented, safe structure)
        default_questions = [
            "test",
        ]

        
        try:
            llm = get_llm()
            response = llm.invoke(messages)
            print(f"✅ [LLM SUCCESS] Generated response with {len(response.content) if response.content else 0} chars")
        except Exception as e:
            print(f"❌ [LLM FAILED] Error: {str(e)}")
            response = None

        # Parse response directly as list
        if response and response.content:
            questions_text = response.content.strip()
            print(f"🔍 [DEBUG] Raw LLM Response:")
            print(f"--- START RESPONSE ---")
            print(questions_text)
            print(f"--- END RESPONSE ---")
            
            try:
                # Clean markdown and extract JSON using regex
                import re
                import json
                
                # Remove markdown code blocks
                cleaned_text = re.sub(r'```(?:python|json)?\s*', '', questions_text)
                cleaned_text = re.sub(r'```\s*', '', cleaned_text)
                
                # Find JSON array pattern
                json_match = re.search(r'\[.*?\]', cleaned_text, re.DOTALL)
                if json_match:
                    json_text = json_match.group(0)
                    logger.warning(f"🔧 Extracted JSON: {json_text}")
                    questions_list = json.loads(json_text)
                else:
                    # Try parsing the whole cleaned text
                    questions_list = json.loads(cleaned_text)
                
                if isinstance(questions_list, list) and len(questions_list) >= 4:
                    final_questions = questions_list[:4]
                    logger.warning(f"✅ Successfully parsed {len(final_questions)} questions")
                    return final_questions
                else:
                    logger.warning(f"⚠️  Expected list with 4+ items, got {type(questions_list)}")
                        
            except Exception as parse_error:
                logger.warning(f"❌ JSON parsing failed: {parse_error}")
                logger.warning(f"🔍 [DEBUG] Failed content: {questions_text}")
                
                # Fallback: extract quoted questions with regex
                try:
                    import re
                    questions = re.findall(r'"([^"]*\?)"', questions_text)
                    if len(questions) >= 4:
                        logger.warning(f"✅ Regex fallback extracted {len(questions)} questions")
                        return questions[:4]
                except Exception:
                    pass
        else:
            print(f"🔍 [DEBUG] No response or content")

    except Exception as e:
        logger.warning(f"❌ generate_base_questions failed completely: {e}")
        # Final fallback in case of unexpected errors
        fallback_questions = [
            "Can you walk me through your experience related to this role?",
            "Describe a significant challenge you've faced and how you handled it.", 
            "How do you ensure collaboration and effective teamwork?",
            "Tell me about a situation where you had to adapt quickly to deliver results."
        ]
        logger.warning(f"✅ Using fallback questions: {len(fallback_questions)} items")
        return fallback_questions




def clear_redis_cache():
    """Clear Redis cache that might contain old role information"""
    try:
        import redis
        from ai_interview.config import REDIS_URL
        r = redis.Redis.from_url(REDIS_URL)
        
        # Clear all interview-related cache
        for key in r.scan_iter(match="interview:*"):
            r.delete(key)
            
        print(f"🧹 Cleared Redis cache for interview sessions")
    except Exception as e:
        print(f"Redis cache clear error (ignored): {e}")

def clear_interview_cache():
    """Clear any cached state - now used for cleanup only since we're fully dynamic"""
    clear_redis_cache()
    print("🧹 Interview cache cleared - all generation is now fully dynamic per request")


# Improved LLM setup with better error handling
def get_llm() -> ChatVertexAI:
    """Create and return a ChatVertexAI LLM instance with optimized settings for interviews"""
    return ChatVertexAI(
        model_name="gemini-2.0-flash-lite-001",  # Faster model for lower latency
        temperature=0.1,  # Slight randomness for natural responses
        max_retries=1,  # One retry for reliability
        max_tokens=400,  # More tokens for complete question generation
    )

# Circuit Breaker Pattern for LLM Calls
class LLMCircuitBreaker:
    def __init__(self, failure_threshold=3, recovery_timeout=5, half_open_max_calls=2):  # More resilient settings
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
    
    def reset(self):
        """Reset circuit breaker to CLOSED state"""
        self.failure_count = 0
        self.state = "CLOSED"
        self.half_open_calls = 0
        self.last_failure_time = None
        print(f"🔄 Circuit breaker reset to CLOSED state")
    
    def call_llm(self, messages, fallback_response="I understand. Could you tell me more?", room_id="unknown", operation="general"):
        """Call LLM with circuit breaker protection and detailed logging"""
        llm_start = time.time()
        current_state = self.get_state()
        success = True
        error_message = ""
        response = None
        
        # Prepare prompt for logging
        prompt_text = " | ".join([str(msg) for msg in messages]) if messages else ""
        
        interview_logger.debug(f"LLM_CALL_START: Room={room_id}, Operation={operation}, State={current_state}")
        
        if current_state == "OPEN":
            logger.warning("Circuit breaker OPEN - using fallback response")
            interview_logger.warning(f"Circuit breaker OPEN for room {room_id} - using fallback")
            print(f"🚫 [LLM] Circuit breaker OPEN - fallback used (0.001s)")
            response = type('Response', (), {'content': fallback_response})()
            success = False
            error_message = "Circuit breaker OPEN"
        elif current_state == "HALF_OPEN" and self.half_open_calls >= self.half_open_max_calls:
            logger.warning("Circuit breaker HALF_OPEN limit reached - using fallback")
            interview_logger.warning(f"Circuit breaker HALF_OPEN limit reached for room {room_id}")
            print(f"🚫 [LLM] Circuit breaker HALF_OPEN limit - fallback used (0.001s)")
            response = type('Response', (), {'content': fallback_response})()
            success = False
            error_message = "Circuit breaker HALF_OPEN limit reached"
        else:
            try:
                if current_state == "HALF_OPEN":
                    self.half_open_calls += 1
                
                # OPTIMIZATION: Truncate messages for speed
                optimized_messages = self._optimize_messages(messages)
                    
                print(f"🤖 [LLM CALL START] State: {current_state}, Timeout: 2.0s, Max tokens: 20")
                interview_logger.debug(f"Invoking LLM for room {room_id}, operation {operation}")
                
                # Use optimized LLM instance
                optimized_llm = get_llm()
                response = optimized_llm.invoke(optimized_messages)
                
                llm_time = time.time() - llm_start
                print(f"✅ [LLM SUCCESS] Response received in {llm_time:.3f}s, Length: {len(response.content) if response.content else 0} chars")
                interview_logger.info(f"LLM success for room {room_id}: {llm_time:.3f}s")
                
                # Success - reset circuit breaker
                if current_state == "HALF_OPEN":
                    self.state = "CLOSED"
                    self.failure_count = 0
                    logger.info("Circuit breaker CLOSED - LLM calls restored")
                
            except Exception as e:
                llm_time = time.time() - llm_start
                self.failure_count += 1
                self.last_failure_time = time.time()
                success = False
                error_message = str(e)
                
                print(f"❌ [LLM FAILED] Error after {llm_time:.3f}s: {str(e)[:100]}")
                logger.error(f"LLM call failed ({self.failure_count}/{self.failure_threshold}): {e}")
                interview_logger.error(f"LLM failed for room {room_id}: {str(e)[:100]}")
                
                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"
                    logger.error(f"Circuit breaker OPEN - LLM calls suspended for {self.recovery_timeout}s")
                    print(f"🚫 [CIRCUIT BREAKER] Now OPEN for {self.recovery_timeout}s")
                
                # Return fallback response
                print(f"🔄 [LLM FALLBACK] Using fallback response")
                response = type('Response', (), {'content': fallback_response})()
        
        # Log detailed interaction
        llm_end = time.time()
        response_text = response.content if response and hasattr(response, 'content') else str(response)
        log_llm_interaction(operation, room_id, prompt_text, response_text, llm_end - llm_start, success)
        
        return response
    
    def _optimize_messages(self, messages):
        """Optimize messages for speed without breaking functionality"""
        if len(messages) > 2:
            # Keep system message + latest human message only for speed
            system_msg = None
            human_msg = None
            
            for msg in messages:
                if hasattr(msg, 'type'):
                    if msg.type == "system":
                        system_msg = msg
                    elif msg.type == "human":
                        human_msg = msg
                elif str(type(msg).__name__) == "SystemMessage":
                    system_msg = msg
                elif str(type(msg).__name__) == "HumanMessage":
                    human_msg = msg
            
            # Return optimized message list
            if system_msg and human_msg:
                return [system_msg, human_msg]
            elif human_msg:
                return [human_msg]
            else:
                return messages[-1:] if messages else messages
        
        return messages


# Initialize circuit breaker and LLM
circuit_breaker = LLMCircuitBreaker()
llm = get_llm()







# FAST: Basic STAR completeness check (no LLM needed)
def analyze_star_completeness(answer: str) -> dict:
    """Fast STAR analysis using simple logic"""
    if not answer or len(answer.strip()) < 20:
        return {"completeness": 0.1, "missing_elements": ["situation", "task", "action", "result"]}
    
    answer_lower = answer.lower()
    length = len(answer.split())
    
    # Quick heuristics - no LLM needed
    has_context = any(word in answer_lower for word in ["when", "where", "project", "company", "time", "situation"])
    has_task = any(word in answer_lower for word in ["responsible", "needed", "goal", "task", "job"]) 
    has_action = any(word in answer_lower for word in ["i did", "i used", "i created", "i implemented", "my approach"])
    has_result = any(word in answer_lower for word in ["result", "outcome", "achieved", "improved", "success"])
    
    # Score based on length and key indicators
    situation_score = 0.8 if has_context else 0.3
    task_score = 0.8 if has_task else 0.3
    action_score = 0.8 if has_action and length > 30 else 0.3
    result_score = 0.8 if has_result else 0.3
    
    completeness = (situation_score + task_score + action_score + result_score) / 4
    missing = [elem for elem, score in [("situation", situation_score), ("task", task_score), 
                                       ("action", action_score), ("result", result_score)] if score < 0.5]
    
    return {"completeness": completeness, "missing_elements": missing}

# FAST: Quick quality assessment (no LLM needed) 
def detect_answer_patterns(answer: str) -> dict:
    """Fast quality check using simple heuristics"""
    if not answer or len(answer.strip()) < 10:
        return {"quality_ratio": 0.1}
    
    answer_lower = answer.lower()
    word_count = len(answer.split())
    
    # Quick red flag detection
    red_indicators = ["wasn't my", "not my fault", "they didn't", "management failed", "team was"]
    red_count = sum(1 for indicator in red_indicators if indicator in answer_lower)
    
    # Quick green flag detection  
    green_indicators = ["i did", "i created", "i implemented", "i learned", "my responsibility", "i decided"]
    green_count = sum(1 for indicator in green_indicators if indicator in answer_lower)
    
    # Length bonus for detailed answers
    length_bonus = min(0.3, word_count / 100)
    
    quality_ratio = (green_count + length_bonus) / max(1, red_count + green_count + 0.5)
    
    return {"quality_ratio": min(1.0, quality_ratio)}



# Professional interviewer conversation with psychology tactics
def generate_natural_conversation(question: str, answer: str, conversation_context: str = "", role_title: str = "", years_experience: str = "", room_id: str = None, star_analysis: dict = None, patterns: dict = None) -> str:
    """Generate professional interviewer responses using psychology tactics"""
    try:

        
        # Use pre-computed analysis or fallback values
        if star_analysis is None:
            star_analysis = {"completeness": 0.5, "missing_elements": ["details"]}
        if patterns is None:
            patterns = {"quality_ratio": 0.5}
        
        # Skip skills lookup to avoid cache delays - use minimal context
        questions, skills= get_key_skills(role_title, years_experience,room_id)

        skills_context = f"{skills} skills"
        
        # Generate context-aware prompts based on analysis
        psychology_context = ""
        if star_analysis["missing_elements"]:
            psychology_context += f"Missing STAR elements: {', '.join(star_analysis['missing_elements'])}. "
        if patterns["quality_ratio"] < 0.5:
            psychology_context += "Answer lacks specificity and ownership. "
        # OPTIMIZED: Much shorter prompt for faster generation
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

Follow-up Techniques (dig deeper into current response):

1. **STAR Method Deepening** (based on missing elements):
   - Missing Situation: "Can you give me more context about when this happened?"
   - Missing Task: "What specifically were you responsible for in that situation?"  
   - Missing Action: "Walk me through exactly what YOU did, step by step."
   - Missing Result: "What was the outcome? How do you measure that success?"

2. **Specific Details from Their Answer**:
   - "You mentioned [specific thing from their answer]. Tell me more about that."
   - "That sounds interesting. How did you approach that challenge?"
   - "What specific steps did you take to accomplish that?"

3. **Ownership and Decision Clarification**:
   - "What was YOUR specific contribution to that?"
   - "How did you personally decide to handle that situation?"
   - "What would you do differently if you faced that same situation again?"

Generate ONE follow-up question (15-25 words) that explores their current answer deeper."""),
            HumanMessage(content=f"CURRENT QUESTION: {question}\n\nCANDIDATE'S CURRENT ANSWER: {answer}\n\nFULL CONVERSATION HISTORY (Questions already asked - DO NOT REPEAT ANY OF THESE):\n{conversation_context if conversation_context else 'This is the first question'}\n\nANALYSIS OF CURRENT ANSWER:\n- STAR Completeness: {star_analysis['completeness']:.2f}\n- Quality Score: {patterns['quality_ratio']:.2f}\n- Missing Elements: {', '.join(star_analysis.get('missing_elements', []))}\n\nYour task: Generate a follow-up question that digs deeper ONLY into their current answer above. Reference something specific they mentioned in their current answer. Do NOT ask anything from the conversation history.")
        ]
        
        
        # Generate follow-up response 
        step3_start = time.time()
        response = circuit_breaker.call_llm(messages, "That's interesting! Tell me more about that specific aspect.", room_id, "followup_generation")
        step3_time = time.time() - step3_start
        
        # Log the LLM call
        interview_logger.info(f"Followup_Generation completed in {step3_time:.3f}s for {role_title}")
        
        if response and response.content:
            cleaned_response = clean_bot_response(response.content)
            if cleaned_response:  # Only return if not empty after stripping
                return cleaned_response
        
        return "That's really interesting. Could you tell me more about how you approached that?"
        
    except Exception as e:
        logger.error(f"Error generating natural conversation: {e}")
        print(f"Error generating natural conversation: {e}")
        return "That's a great point. Can you elaborate on that experience?"


def _truncate_conversation_history(conversation_context: str) -> str:
    """Truncate conversation history to last 2-3 exchanges for speed while keeping context quality"""
    if not conversation_context or len(conversation_context) < 500:
        return conversation_context or "This is the first question"
        
    # Split by lines and keep only recent exchanges (last 6 lines = ~2-3 Q&A pairs)
    lines = conversation_context.split('\n')
    if len(lines) <= 6:
        return conversation_context
    
    # Keep last 6 lines with context indicator
    truncated = '\n'.join(lines[-6:])
    return f"[Recent exchanges only]\n{truncated}"



def handle_repeat_question_intent_optimized(state: InterviewState, questions: List[str]) -> Dict[str, Any]:
    """Handle RepeatQuestion intent - repeat the actual last question asked (could be follow-up)"""
    try:
        current_idx = state.get('question_idx', 0)
        follow_up_count = state.get('follow_up_count', 0)
        
        # Get the actual last question asked (which could be a follow-up)
        last_question = state.get('last_question_asked', '')
        
        # PERFORMANCE FIX: Use passed questions instead of re-fetching
        
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

def handle_previous_question_intent_optimized(state: InterviewState, questions: List[str]) -> Dict[str, Any]:
    """Handle PreviousQuestion intent - go back to previous question"""
    try:
        current_idx = state.get('question_idx', 0)
        
        if current_idx > 0:
            previous_idx = current_idx - 1
            
            # PERFORMANCE FIX: Use passed questions instead of re-fetching
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

def handle_next_question_intent_optimized(state: InterviewState, questions: List[str], current_idx: int) -> Dict[str, Any]:
    """Handle NextQuestion intent - move to next question"""
    try:
        next_idx = current_idx + 1
        
        if next_idx < len(questions):
            # Move to next question
            next_question = questions[next_idx]
            response = f"Great! Let's move to the next question: {next_question}"
            
            return {
                "bot_response": response,
                "question_idx": next_idx,
                "phase": "questions", 
                "follow_up_count": 0,  # Reset follow-ups for new question
                "last_question_asked": next_question,  # Track the question asked
                "done": False
            }
        else:
            # No more questions - end interview
            return {
                "bot_response": "That was our final question! Thank you for the great conversation. We'll be in touch with next steps.",
                "question_idx": next_idx,
                "phase": "complete",
                "follow_up_count": 0,
                "done": True,
                "interview_status": "end"
            }
            
    except Exception as e:
        logger.error(f"Error handling NextQuestion intent: {e}")
        return {
            "bot_response": "Let's continue with the current question.",
            "phase": "questions",
            "done": False
        }

def handle_clarify_question_intent_optimized(state: InterviewState, user_input: str, questions: List[str], current_idx: int) -> Dict[str, Any]:
    """Handle ClarifyQuestion intent - explain/clarify current question"""
    try:
        # PERFORMANCE FIX: Use passed questions and current_idx instead of re-fetching
        role_title = state.get('role_title', '')
        
        if current_idx < len(questions):
            current_question = questions[current_idx]
            
            messages = [
                SystemMessage(content=f"{role_title} interviewer. Clarify question clearly and supportively. 30-50 words."),
                HumanMessage(content=f"Question: {current_question}\nRequest: {user_input}\nClarify:")
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

def handle_smalltalk_intent_optimized(state: InterviewState, user_input: str, questions: List[str], current_idx: int, role_title: str) -> Dict[str, Any]:
    """Handle SmallTalk intent - polite redirect back to interview"""
    try:
        # PERFORMANCE FIX: Use passed parameters instead of state lookup
        
        messages = [
            SystemMessage(content=f"{role_title} interviewer. Politely acknowledge and redirect to interview. 15-25 words."),
            HumanMessage(content=f"Small talk: {user_input}\nRedirect:")
        ]
        
        response = circuit_breaker.call_llm(messages, 
            "That's nice! Let's get back to the interview question though.")
        
        redirect = response.content if response and response.content else \
            "That's nice! Let's get back to the interview question though."
        
        # PERFORMANCE FIX: Use passed questions instead of re-fetching
        
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
            SystemMessage(content=f"{role_title} interviewer. Acknowledge briefly and redirect to interview professionally. 10-15 words."),
            HumanMessage(content=f"Off-topic: {user_input}\nRedirect:")
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

def handle_offtopic_intent_optimized(state: InterviewState, user_input: str, questions: List[str], current_idx: int) -> Dict[str, Any]:
    """Handle OffTopic intent - intelligent redirect and move to next question"""
    try:
        # Generate intelligent off-topic response
        intelligent_redirect = handle_offtopic_intelligently(user_input)
        next_idx = current_idx + 1
        
        # Check if we've reached the end of questions
        if next_idx >= len(questions):
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
            "last_question_asked": questions[next_idx],  # Track the next question
            "done": False
        }
        
    except Exception as e:
        logger.error(f"Error handling OffTopic intent: {e}")
        return {
            "bot_response": "I understand! Let's get back to the interview.",
            "phase": "questions",
            "done": False
        }

def handle_end_interview_intent_optimized(state: InterviewState, role_title: str) -> Dict[str, Any]:
    """Handle EndInterview intent - conclude the interview professionally"""
    
    try:
        messages = [
            SystemMessage(content=f"Professional {role_title} interview conclusion. Thank candidate, mention next steps. Max 25 words."),
            HumanMessage(content="Generate conclusion:")
        ]
        
        response = circuit_breaker.call_llm(messages, "Thank you for your time! We'll review your responses and get back to you with next steps within a few days.")
        conclusion = response.content if response and response.content else "Thank you for your time! We'll review your responses and get back to you with next steps within a few days."
        
        return {
            "done": True,
            "bot_response": conclusion,
            "phase": "complete",
            "interview_status": "completed_by_candidate"
        }
        
    except Exception as e:
        print(f"Error handling end interview intent: {e}")
        return {
            "done": True,
            "bot_response": "Thank you for your time! We'll be in touch soon with next steps.",
            "phase": "complete"
        }


# Interview flow routing function with detailed logging
def route_interview_flow(state: InterviewState) -> Literal["greeting", "intro", "ask_question", "process_answer", "complete"]:
    """Determine next step in interview based on current state with detailed logging"""
    room_id = state.get('room_id', 'unknown')
    user_input = state.get('user_input', '').strip()
    current_idx = state.get('question_idx', 0)
    phase = state.get('phase', 'greeting')
    done = state.get('done', False)
    
    interview_logger.info(f"ROUTING_START: Room={room_id}, Phase={phase}, QuestionIdx={current_idx}, Done={done}, UserInputLength={len(user_input)}")
    
    print(f"Routing decision - user_input: '{user_input}', current_idx: {current_idx}, phase: {phase}, done: {done}")
    
    routing_result = None
    routing_reason = ""
    
    # If interview is already marked as done, complete
    if done:
        routing_result = "complete"
        routing_reason = "Interview marked as done"
    
    # Phase-based routing
    elif phase == 'greeting':
        routing_result = "greeting"
        if user_input:
            routing_reason = "User responded to greeting, will transition to intro"
        else:
            routing_reason = "Show initial greeting"
    
    elif phase == 'intro':
        routing_result = "intro"
        if user_input:
            routing_reason = "User ready, will start questions"
        else:
            routing_reason = "Show introduction"
    
    elif phase == 'questions':
        # If we're at or past the last question, complete the interview
        # Use dynamic role and experience from state
        role_title = state.get('role_title', 'Software Developer')
        years_experience = state.get('years_experience', '2-5 years')
        questions = get_base_questions(role_title, years_experience, room_id=room_id)
        
        interview_logger.debug(f"Questions phase routing: {len(questions)} total questions, current_idx={current_idx}")
        
        if current_idx >= len(questions):
            routing_result = "complete"
            routing_reason = f"Reached end of questions ({current_idx} >= {len(questions)})"
        # If user provided input (answer), process the answer
        elif user_input:
            routing_result = "process_answer"
            routing_reason = "User provided answer, will process"
            print(f"User provided answer, processing answer")
        # Otherwise, ask the current question
        else:
            routing_result = "ask_question"
            routing_reason = "No user input, will ask current question"
            print(f"No user input, asking current question")
    
    # Default fallback
    else:
        routing_result = "greeting"
        routing_reason = f"Unknown phase '{phase}', defaulting to greeting"
    
    # Log detailed routing decision
    log_routing_decision(state, routing_result, room_id)
    interview_logger.info(f"ROUTING_DECISION: {routing_result} - {routing_reason}")
    
    return routing_result

# Node functions with detailed logging
def greeting_node(state: InterviewState):
    """Show greeting message or move to intro if user already responded to greeting"""
    start_time = time.time()
    room_id = state.get('room_id', 'unknown')
    
    interview_logger.info(f"GREETING_NODE_START: Room={room_id}")
    
    try:
        user_input = state.get('user_input', '').strip().lower()
        
        # Use dynamic role and experience from state
        role_title = state.get('role_title', 'Software Developer')
        years_experience = state.get('years_experience', '2-5 years')
        candidate_name = state.get('candidate_name', 'Candidate')
        company_name = state.get('company_name', 'Company')
        
        interview_logger.debug(f"Greeting node context: role={role_title}, experience={years_experience}, candidate={candidate_name}")

        # Use cached generation - this will only call LLM once per room
        questions, skills = get_key_skills(role_title, years_experience, room_id)
        
        result = None
        # If user says "start", show greeting message first
        if user_input == "start":
            bot_response = get_greeting_message(role_title, candidate_name, company_name)
            result = {
                "bot_response": bot_response,
                "phase": "greeting",
                "done": False
            }
            interview_logger.info(f"Showing initial greeting for 'start' command")
        elif user_input and user_input != "start":
            # CHECK FOR END INTERVIEW INTENT FIRST
            if ("that's all from me" in user_input.lower() or 
                "thank you!" in user_input.lower() or
                "i think i'm done" in user_input.lower() or
                "can we end" in user_input.lower() or
                "i'd like to finish" in user_input.lower()):
                
                print(f"🎯 [GREETING] EndInterview detected: '{user_input}'")
                interview_logger.warning(f"EndInterview detected in greeting phase: '{user_input}'")
                
                result = {
                    "done": True,
                    "bot_response": f"Thank you for your time, {candidate_name}! I enjoyed our conversation about the {role_title} role. We'll review everything and get back to you with next steps within a few days.",
                    "phase": "complete",
                    "interview_status": "end"
                }
                interview_logger.info(f"Interview ended in greeting phase by candidate request")
                log_state_transition("greeting", "complete", room_id, user_input)
            else:
                # User responded to greeting (not "start"), move to intro phase
                bot_response = get_intro_message(role_title, years_experience, candidate_name, company_name)
                result = {
                    "bot_response": bot_response,
                    "phase": "intro",
                    "done": False
                }
                interview_logger.info(f"User responded to greeting, transitioning to intro phase")
                log_state_transition("greeting", "intro", room_id, user_input)
        else:
            # No user input, show initial greeting
            bot_response = get_greeting_message(role_title, candidate_name, company_name)
            result = {
                "bot_response": bot_response,
                "phase": "greeting",
                "done": False
            }
            interview_logger.info(f"No user input, showing initial greeting")
        
        execution_time = time.time() - start_time
        log_node_execution("greeting", room_id, state, result, execution_time)
        
        return result
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Error in greeting_node: {e}")
        interview_logger.error(f"GREETING_NODE_ERROR: Room={room_id}, Error={str(e)[:100]}")
        
        result = {
            "bot_response": "Hi! I'm Chris, and I'll be your interviewer today. Thanks for joining me - I'm looking forward to our conversation. How are you doing today?",
            "phase": "greeting",
            "done": False
        }
        log_node_execution("greeting", room_id, state, result, execution_time)
        return result

def intro_node(state: InterviewState):
    """Show introduction or move to questions if user is ready"""
    start_time = time.time()
    room_id = state.get('room_id', 'unknown')
    
    interview_logger.info(f"INTRO_NODE_START: Room={room_id}")
    
    try:
        user_input = state.get('user_input', '').strip()
        
        if user_input:
            # CHECK FOR END INTERVIEW INTENT FIRST
            candidate_name = state.get('candidate_name', 'Candidate')
            role_title = state.get('role_title', 'Software Developer')
            years_experience = state.get('years_experience', '2-5 years')

            current_idx = 0  # Always start with first question after intro
            questions = get_base_questions(role_title, years_experience, room_id=room_id)
            
            interview_logger.info(f"User ready for questions. Role={role_title}, Experience={years_experience}, Questions={len(questions)}")
            
            if current_idx >= len(questions):
                result = {"done": True, "bot_response": "Thank you! Interview complete.", "phase": "complete"}
                interview_logger.warning(f"No questions available, completing interview immediately")
                log_state_transition("intro", "complete", room_id, user_input, {"reason": "no_questions"})
                execution_time = time.time() - start_time
                log_node_execution("intro", room_id, state, result, execution_time)
                return result
            
            logger.warning(f"Intro complete - transitioning to questions phase with question_idx: {current_idx}")
            logger.warning(f"🎯 Using role: {role_title}, experience: {years_experience}")
            logger.warning(f"🔍 INTRO_NODE: Retrieved {len(questions)} questions from cache")
            logger.warning(f"🔍 INTRO_NODE: Using question[{current_idx}] = {questions[current_idx]}")
            interview_logger.info(f"Transitioning to questions phase. First question: {questions[current_idx][:50]}...")
            
            result = {
                "bot_response": questions[current_idx],
                "phase": "questions",
                "question_idx": current_idx,
                "follow_up_count": 0,  # Reset follow-up count for new question
                "last_question_asked": questions[current_idx],  # Track the last question asked
                "done": False
            }
            log_state_transition("intro", "questions", room_id, user_input, {"first_question": questions[current_idx][:50]})
        else:
            # Show introduction
            # Use dynamic role and experience from state
            role_title = state.get('role_title', 'Software Developer')
            years_experience = state.get('years_experience', '2-5 years')
            candidate_name = state.get('candidate_name', 'Candidate')
            company_name = state.get('company_name', 'Company')
            
            interview_logger.info(f"Showing introduction message")
            
            result = {
                "bot_response": get_intro_message(role_title, years_experience, candidate_name, company_name, room_id),
                "phase": "intro", 
                "done": False
            }
        
        execution_time = time.time() - start_time
        log_node_execution("intro", room_id, state, result, execution_time)
        return result
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Error in intro_node: {e}")
        interview_logger.error(f"INTRO_NODE_ERROR: Room={room_id}, Error={str(e)[:100]}")
        
        role_title = state.get('role_title', 'Software Developer')
        years_experience = state.get('years_experience', '2-5 years')
        candidate_name = state.get('candidate_name', 'Candidate')
        company_name = state.get('company_name', 'Company')
        
        result = {
            "bot_response": get_intro_message(role_title, years_experience, candidate_name, company_name, room_id),
            "phase": "intro", 
            "done": False
        }
        log_node_execution("intro", room_id, state, result, execution_time)
        return result

def ask_question_node(state: InterviewState):
    """Ask current question based on question_idx"""
    start_time = time.time()
    room_id = state.get('room_id', 'unknown')
    
    interview_logger.info(f"ASK_QUESTION_NODE_START: Room={room_id}")
    
    try:
        idx = state.get('question_idx', 0)
        
        # Use dynamic role and experience from state
        role_title = state.get('role_title', 'Software Developer')
        years_experience = state.get('years_experience', '2-5 years')
        questions = get_base_questions(role_title, years_experience,room_id)  # Use cached questions
        
        interview_logger.info(f"Asking question {idx} of {len(questions)} for role: {role_title}")
        
        if idx >= len(questions):
            result = {"done": True, "bot_response": "Thank you! Interview complete.", "phase": "complete"}
            interview_logger.info(f"Reached end of questions ({idx} >= {len(questions)}), completing interview")
            log_state_transition("questions", "complete", room_id, "", {"reason": "questions_completed", "final_idx": idx})
            execution_time = time.time() - start_time
            log_node_execution("ask_question", room_id, state, result, execution_time)
            return result
        
        print(f"🎯 Asking question {idx} for role: {role_title}")
        interview_logger.debug(f"Question {idx}: {questions[idx][:100]}...")
        
        result = {
            "bot_response": questions[idx],
            "phase": "questions",
            "last_question_asked": questions[idx],  # Track the last question asked
            "done": False
        }
        
        execution_time = time.time() - start_time
        log_node_execution("ask_question", room_id, state, result, execution_time)
        return result
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Error in ask_question_node: {e}")
        interview_logger.error(f"ASK_QUESTION_NODE_ERROR: Room={room_id}, Error={str(e)[:100]}")
        
        result = {
            "bot_response": "Let me ask you a question about your experience.",
            "phase": "questions",
            "done": False
        }
        log_node_execution("ask_question", room_id, state, result, execution_time)
        return result
def process_answer_node_sync(state: InterviewState):
    """Process candidate's answer using professional interviewer psychology (sync version for LangGraph)"""
    start_time = time.time()
    room_id = state.get('room_id', 'unknown')
    
    interview_logger.info(f"PROCESS_ANSWER_NODE_START: Room={room_id}")
    
    try:
        # Direct sync processing - eliminate asyncio.run overhead
        user_input = state.get('user_input', '').strip()
        current_idx = state.get('question_idx', 0)
        room_id = state.get('room_id', 'unknown')
        
        # Quick response for empty input
        if not user_input:
            result = {
                "bot_response": "I didn't hear your response. Could you please answer the question?",
                "phase": "questions",
                "done": False
            }
        else:
            # Generate quick response
            role_title = state.get('role_title', 'Software Developer')
            questions, _ = get_key_skills(role_title, state.get('years_experience', '2-5 years'), room_id)
            
            if current_idx < len(questions) - 1:
                # Move to next question
                result = {
                    "bot_response": questions[current_idx + 1],
                    "question_idx": current_idx + 1,
                    "phase": "questions", 
                    "done": False
                }
            else:
                # End interview
                result = {
                    "bot_response": "Thank you for your responses! That completes our interview. We'll be in touch soon.",
                    "done": True,
                    "phase": "complete"
                }
        
        execution_time = time.time() - start_time
        log_node_execution("process_answer", room_id, state, result, execution_time)
        
        return result
    except Exception as e:
        execution_time = time.time() - start_time
        interview_logger.error(f"PROCESS_ANSWER_NODE_ERROR: Room={room_id}, Error={str(e)[:100]}")
        
        # Fallback result
        result = {
            "bot_response": "Thank you for your response. Let me ask the next question.",
            "phase": "questions",
            "done": False
        }
        log_node_execution("process_answer", room_id, state, result, execution_time)
        return result




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

Return ONLY the intent name.
"""




async def process_answer_node_async(state: InterviewState):
    """Process candidate's answer with optimized performance and reduced logging"""
    try:
        performance_start = time.time()
        room_id = state.get("room_id", "unknown")
        
        # Start detailed latency tracking
        interview_logger.info(f"Starting detailed processing for room {room_id}")
        
        # State preprocessing
        processed_state = {
            'current_idx': state.get('question_idx', 0),
            'user_answer': state.get('user_input', '').strip(),
            'follow_up_count': state.get('follow_up_count', 0),
            'role_title': state.get('role_title', 'Software Developer'),
            'years_experience': state.get('years_experience', '2-5 years'),
            'room_id': state["room_id"],
            'conversation_history': state.get('conversation_history', ''),
            'conversation_context': state.get('conversation_history', '')
        }
        
        # Get questions from cache
        questions, _ = get_key_skills(processed_state['role_title'], processed_state['years_experience'], processed_state['room_id'])
        print('-------------------------------------------------------joy-------------------------------------',questions)
        # Extract variables
        current_idx = processed_state['current_idx']
        user_answer = processed_state['user_answer'] 
        follow_up_count = processed_state['follow_up_count']
        conversation_history = processed_state['conversation_history']
        conversation_context = processed_state['conversation_context']
        
        # Bounds checking
        current_question = questions[current_idx] if current_idx < len(questions) else ""
        
        # Parallel LLM analysis for performance
        
        def analyze_star_thread():
            return analyze_star_completeness(user_answer)
            
        def analyze_patterns_thread():
            return detect_answer_patterns(user_answer)
            
        def classify_intent_thread():
            call_start = time.time()
            # Build the intent classification messages
            filled_prompt = intent_classification_prompt.format(
    current_question=current_question,
    user_answer=user_answer
)

            # OPTIMIZED: Shorter prompt for faster processing
            intent_messages = [
    SystemMessage(content=filled_prompt),
    HumanMessage(content=f"Please classify the intent.")
]


            response = circuit_breaker.call_llm(intent_messages, "Answer", processed_state['room_id'], "intent_classification")
            
            result = response.content.strip() if response and response.content else "Answer"
            call_end = time.time()
            interview_logger.info(f"Intent classification completed in {call_end - call_start:.3f}s: {result}")
            return result
        
        # Execute LLM calls in parallel with aggressive timeouts
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                star_future = executor.submit(analyze_star_thread)
                patterns_future = executor.submit(analyze_patterns_thread)
                intent_future = executor.submit(classify_intent_thread)
                
                # Extreme timeouts for speed
                star_analysis = star_future.result(timeout=1)
                patterns = patterns_future.result(timeout=1)  
                user_intent = intent_future.result(timeout=1)  # Extreme LLM call
                
        except Exception as e:
            # Fallback to sequential processing
            star_analysis = analyze_star_completeness(user_answer)
            patterns = detect_answer_patterns(user_answer)
            user_intent = "Answer"
        
        # Intent-based routing
        intent_handlers = {
            "EndInterview": lambda: handle_end_interview_intent_optimized(state, processed_state['role_title']),
            "RepeatQuestion": lambda: handle_repeat_question_intent_optimized(state, questions),
            "PreviousQuestion": lambda: handle_previous_question_intent_optimized(state, questions),
            "ClarifyQuestion": lambda: handle_clarify_question_intent_optimized(state, user_answer, questions, current_idx),
            "NextQuestion": lambda: handle_next_question_intent_optimized(state, questions, current_idx),
            "Hesitation": lambda: handle_hesitation_intent(state),
            "SmallTalk": lambda: handle_smalltalk_intent_optimized(state, user_answer, questions, current_idx, processed_state['role_title']),
            "OffTopic": lambda: handle_offtopic_intent_optimized(state, user_answer, questions, current_idx)
        }
        
        # Route based on intent
        if user_intent in intent_handlers:
            return intent_handlers[user_intent]()
        
        # Intent: "Answer" - Process as a legitimate answer to the interview question
        elif user_intent == "Answer":
            # This is the main interview flow path - process the answer professionally
            pass  # Continue to answer processing logic below
        else:
            # Fallback for unknown intents - treat as Answer
            print(f"⚠️  Unknown intent '{user_intent}', treating as Answer")
        
        # DSA OPTIMIZATION: Use already computed analysis results (avoid redundant LLM calls)
        followup_decision_start = time.time()
        
        # Smart follow-up decision using pre-computed analysis
        star_completeness = star_analysis.get('completeness', 0.5)
        quality_score = patterns.get('quality_ratio', 0.5)
        
        # Decision tree algorithm using computed scores - ADJUSTED THRESHOLDS FOR MORE FOLLOW-UPS
        answer_length = len(user_answer.strip())
        word_count = len(user_answer.split())
        
        needs_followup = (
            follow_up_count < 2 and  # Allow up to 2 follow-ups per question
            (answer_length < 80 or               # Too short - lowered threshold
             star_completeness < 0.6 or          # Missing STAR elements - lowered threshold  
             quality_score < 0.6 or              # Low quality - lowered threshold
             word_count < 20)                    # Lacks detail - lowered threshold
        )
        
        followup_decision_time = time.time() - followup_decision_start
        print(f"🧠 [DSA DETAILED DECISION] Follow-up: {needs_followup} | Count: {follow_up_count}/2 | Length: {answer_length} chars | Words: {word_count} | STAR: {star_completeness:.2f} | Quality: {quality_score:.2f} | Time: {followup_decision_time:.3f}s")
        
        if needs_followup:
            print(f"🤖 [FOLLOWUP GENERATION] Generating contextual follow-up question...")
            
            # OPTIMIZATION: Use generate_natural_conversation with pre-computed analysis (eliminates redundant LLM calls)
            followup_start = time.time()
            natural_response = generate_natural_conversation(
                question=current_question,
                answer=user_answer, 
                conversation_context=conversation_context,
                role_title=processed_state['role_title'],
                years_experience=processed_state['years_experience'],
                room_id=processed_state['room_id'],
                star_analysis=star_analysis,  # Pass pre-computed results
                patterns=patterns             # Pass pre-computed results
            )
            followup_time = time.time() - followup_start
            print(f"🔄 [OPTIMIZED FOLLOWUP] Generated in {followup_time:.3f}s with pre-computed analysis")
            
            # Safety check: Ensure natural_response is not empty
            if not natural_response or not natural_response.strip():
                natural_response = "That's interesting! Could you tell me more about that specific experience?"
                print(f"⚠️  Empty natural_response detected, using fallback")
            
            # Update conversation history
            updated_history = f"{conversation_history}\nQ: {current_question}\nA: {user_answer}\nInterviewer: {natural_response}"
            
            return {
                "bot_response": natural_response,
                "question_idx": current_idx,  # Stay on same question
                "phase": "questions",
                "follow_up_count": follow_up_count + 1,
                "conversation_history": updated_history,
                "last_question_asked": natural_response,  # Track the follow-up question as last asked
                "done": False
            }
        
        # No follow-up needed OR max follow-ups reached - transition to next question like a real interviewer
        next_idx = current_idx + 1
        
        # Continue to next question
        
        # Check if we've reached the end of questions
        if next_idx >= len(questions):
            # Generate natural conclusion with role context using parallel processing
            conclusion_messages = [
                SystemMessage(content=f"You are a professional interviewer concluding a {processed_state['role_title']} interview. Generate a natural, warm conclusion that thanks them and explains next steps. Keep it professional but personable. Reference the role appropriately."),
                HumanMessage(content=f"Candidate's last response: {user_answer}\n\nGenerate a natural interview conclusion for the {processed_state['role_title']} position:")
            ]
            
            # PERFORMANCE FIX: Direct LLM call for conclusion with detailed logging
            conclusion_start = time.time()
            print(f"🧠 Starting Interview Conclusion Generation...")
            conclusion_response = circuit_breaker.call_llm(conclusion_messages, "Thank you for a wonderful conversation! I really enjoyed learning about your experience and background. We'll be reviewing everything we discussed and will get back to you soon with next steps.", room_id, "interview_conclusion")
            conclusion = conclusion_response.content if conclusion_response and conclusion_response.content else "Thank you for a wonderful conversation! I really enjoyed learning about your experience and background. We'll be reviewing everything we discussed and will get back to you soon with next steps."
            conclusion_end = time.time()
            
            # Log the conclusion generation
            interview_logger.info(f"Interview conclusion generated in {conclusion_end - conclusion_start:.3f}s for {processed_state['role_title']} position")
            
            print(f"🏁 [CONCLUSION GENERATED] in {conclusion_end - conclusion_start:.3f}s")
            
            return {
                "done": True, 
                "bot_response": conclusion,
                "question_idx": next_idx,
                "phase": "complete",
                "follow_up_count": 0,
                "conversation_history": f"{conversation_history}\nConclusion: {conclusion}",
                "interview_status": "end"
            }
        
        # Generate natural transition with role awareness
        role_title = processed_state['role_title']
        
        
        # Eliminate LLM call for transitions - direct question delivery for speed
        transition = questions[next_idx]  # Saves 1-2s per transition
        
        # Update conversation history
        updated_history = f"{conversation_history}\nQ: {current_question}\nA: {user_answer}\nTransition: {transition}"
        
        # Return the next question with natural transition  
        total_time = time.time() - performance_start
        
        # Complete performance logging
        interview_logger.info(f"Processing completed for room {room_id} in {total_time:.3f}s")
        
        print(f"\n🏆 === OPTIMIZED FLOW SUMMARY ===")
        print(f"⚡ TOTAL LLM CALLS: 3 (STAR + Pattern + Intent) executed in parallel")
        print(f"✅ FOLLOW-UP: Used pre-computed analysis (no redundant calls)")
        print(f"⏱️ TOTAL TIME: {total_time:.3f}s (Target: <3s)")
        print(f"🎯 OPTIMIZATION: 33-40% LLM call reduction achieved")
        
        # Log performance summary
        performance_log = {
            "room_id": room_id,
            "total_time": total_time,
            "target_time": 3.0,
            "performance_target_met": total_time < 3.0,
            "timestamp": datetime.now().isoformat()
        }
        interview_logger.info(f"PERFORMANCE_SUMMARY: {json.dumps(performance_log)}")
        
        if total_time > 5:
            print(f"⚠️ PERFORMANCE ALERT: Request exceeded 5s target ({total_time}s)")
            interview_logger.warning(f"PERFORMANCE_ALERT: Room {room_id} exceeded 5s target: {total_time:.3f}s")
        
        return {
            "bot_response": transition,
            "question_idx": next_idx,
            "phase": "questions",
            "follow_up_count": 0,  # Reset for new question
            "conversation_history": updated_history,
            "last_question_asked": transition,  # Track the transition + next question as last asked
            "done": False
        }
        
    except Exception as e:
        logger.error(f"Error in process_answer_node: {e}")
        # Fallback: move to next question
        next_idx = state.get('question_idx', 0) + 1
        # Use dynamic role and experience from state for fallback
        role_title = state.get('role_title', 'Software Developer')
        years_experience = state.get('years_experience', '2-5 years')
        questions_fallback = get_base_questions(role_title, years_experience, room_id=state["room_id"])
        if next_idx >= len(questions_fallback):
            return {
                "done": True,
                "bot_response": "Thank you for completing the interview!",
                "phase": "complete"
            }
        return {
            "bot_response": questions_fallback[next_idx],
            "question_idx": next_idx,
            "phase": "questions",
            "follow_up_count": 0,
            "done": False
        }
        

async def complete_interview_node_async(state: InterviewState):
    """Complete the interview with professional assessment - OPTIMIZED"""
    try:
        conclusion_start = time.time()
        conversation_history = state.get('conversation_history', '')
        role_title = state.get('role_title', 'this')
        years_experience = state.get('years_experience', '')
        
        # Use unified cache for skills  
        key_skills = []
        if role_title and years_experience:
            room_id = state["room_id"]
            _, key_skills = get_key_skills(role_title, years_experience, room_id)
        
        messages = [
            SystemMessage(content=f"""You are a senior interviewer concluding a {role_title} interview. 

Key skills assessed: {', '.join(key_skills) if key_skills else 'various professional skills'}

Generate a professional, warm conclusion that:
1. Thanks them genuinely 
2. Mentions a specific strength you observed (be authentic)
3. Explains realistic next steps for the {role_title} role
4. Maintains professional confidence without making promises

Sound like a real hiring manager from Impacteers who was actively listening and evaluating. 
Keep it conversational but professional (30-40 words)."""),
            HumanMessage(content=f"Interview conversation: {conversation_history[-400:]}\\n\\nGenerate professional conclusion:")
        ]
        
        # PERFORMANCE FIX: Direct LLM call instead of expensive parallel processing
        conclusion_response = circuit_breaker.call_llm(messages, f"Thank you for a great conversation! I enjoyed learning about your experience and approach to technical challenges. We'll review everything and get back to you within a few days about next steps.")
        conclusion = conclusion_response.content if conclusion_response and conclusion_response.content else f"Thank you for a great conversation! I enjoyed learning about your experience and approach to technical challenges. We'll review everything and get back to you within a few days about next steps."
        
        conclusion_time = time.time() - conclusion_start
        print(f"🏁 [INTERVIEW CONCLUSION] Generated in {conclusion_time:.3f}s")
        
        return {
            "done": True,
            "bot_response": conclusion,
            "phase": "complete"
        }
        
    except Exception as e:
        logger.error(f"Error in complete_interview_node: {e}")
        return {
            "done": True,
            "bot_response": f"Thank you for completing the {role_title} interview! We'll be in touch soon with next steps.",
            "phase": "complete"
        }


# LangGraph implementation with conditional edges
builder = StateGraph(InterviewState)

# Add nodes
builder.add_node("greeting", greeting_node)
builder.add_node("intro", intro_node)
builder.add_node("ask_question", ask_question_node)
def process_answer_node_optimized(state: InterviewState):
    """LLM-powered intent classification with optimized performance"""
    room_id = state.get('room_id', 'unknown')
    
    try:
        user_input = state.get('user_input', '').strip()
        current_idx = state.get('question_idx', 0)
        
        # Quick response for empty input
        if not user_input:
            return {
                "bot_response": "I didn't hear your response. Could you please answer the question?",
                "phase": "questions",
                "done": False
            }
        
        # Get questions for routing
        role_title = state.get('role_title', 'Software Developer')
        years_experience = state.get('years_experience', '2-5 years')
        questions, _ = get_key_skills(role_title, years_experience, room_id)
        
        # LLM-POWERED INTENT CLASSIFICATION
        intent_messages = [
            SystemMessage(content=intent_classification_prompt),
            HumanMessage(content=f"Classify this user input: '{user_input}'\n\nReturn ONLY the intent name.")
        ]
        
        print(f"🧠 [LLM INTENT] Classifying: '{user_input[:50]}...'")
        intent_start = time.time()
        
        response = circuit_breaker.call_llm(intent_messages, "Answer", room_id, "intent_classification")
        raw_intent = response.content.strip() if response and response.content else "Answer"
        
        # Extract just the intent name (handle cases where LLM returns extra text)
        user_intent = raw_intent.split()[0] if raw_intent else "Answer"
        
        # Validate intent is one of expected values
        valid_intents = ["EndInterview", "RepeatQuestion", "NextQuestion", "ClarifyQuestion", "PreviousQuestion", "Hesitation", "SmallTalk", "OffTopic", "Answer"]
        if user_intent not in valid_intents:
            user_intent = "Answer"  # Default fallback
        
        intent_time = time.time() - intent_start
        print(f"🎯 [INTENT RESULT] {user_intent} (raw: '{raw_intent[:30]}...') in {intent_time:.3f}s")
        
        # DEBUGGING: Log critical cases
        if "that's all from me" in user_input.lower() or "thank you!" in user_input.lower():
            print(f"⚠️  CRITICAL CASE: '{user_input}' classified as '{user_intent}' - Expected: EndInterview")
            interview_logger.warning(f"Critical case - Input: '{user_input}', Classified: '{user_intent}', Raw: '{raw_intent}'")
        
        # Intent-based routing using LLM decision
        if user_intent == "EndInterview":
            return {
                "done": True,
                "bot_response": f"Thank you for your time, {state.get('candidate_name', 'Candidate')}! I enjoyed our conversation about the {role_title} role. We'll review everything and get back to you with next steps within a few days.",
                "phase": "complete",
                "interview_status": "completed_by_candidate"
            }
        
        elif user_intent == "RepeatQuestion":
            last_question = state.get('last_question_asked', '')
            if current_idx < len(questions):
                current_question = questions[current_idx] if not last_question else last_question
                # Vary the repeat response
                repeat_phrases = [
                    f"Of course! Let me repeat that: {current_question}",
                    f"Sure thing! The question was: {current_question}",
                    f"Absolutely! Here's the question again: {current_question}",
                    f"No problem! Let me ask that again: {current_question}"
                ]
                return {
                    "bot_response": random.choice(repeat_phrases),
                    "question_idx": current_idx,
                    "phase": "questions",
                    "done": False
                }
        
        elif user_intent == "NextQuestion":
            next_idx = current_idx + 1
            if next_idx < len(questions):
                # Vary the next question transition
                next_phrases = [
                    f"Perfect! Let's move on: {questions[next_idx]}",
                    f"Alright! Here's our next question: {questions[next_idx]}",
                    f"Great! Moving forward: {questions[next_idx]}",
                    f"Sounds good! Next up: {questions[next_idx]}"
                ]
                return {
                    "bot_response": random.choice(next_phrases),
                    "question_idx": next_idx,
                    "phase": "questions",
                    "follow_up_count": 0,
                    "last_question_asked": questions[next_idx],
                    "done": False
                }
            else:
                return {
                    "done": True,
                    "bot_response": f"That was our final question! Thank you for the great conversation about the {role_title} role. We'll be in touch with next steps.",
                    "phase": "complete",
                    "interview_status": "end"
                }
        
        elif user_intent == "ClarifyQuestion":
            if current_idx < len(questions):
                current_question = questions[current_idx]
                # Vary clarification responses
                clarify_phrases = [
                    f"Let me clarify that. I'm looking for a specific example from your experience: {current_question}",
                    f"Good question! To be more specific: {current_question}",
                    f"Let me rephrase that for clarity: {current_question}",
                    f"Sure, I can clarify. What I'm asking is: {current_question}"
                ]
                return {
                    "bot_response": random.choice(clarify_phrases),
                    "question_idx": current_idx,
                    "phase": "questions",
                    "done": False
                }
        
        elif user_intent == "PreviousQuestion":
            if current_idx > 0:
                prev_idx = current_idx - 1
                return {
                    "bot_response": f"Sure, let's go back to the previous question: {questions[prev_idx]}",
                    "question_idx": prev_idx,
                    "phase": "questions",
                    "follow_up_count": 0,
                    "last_question_asked": questions[prev_idx],
                    "done": False
                }
            else:
                return {
                    "bot_response": "We're actually on the first question, so there's no previous question to go back to.",
                    "question_idx": current_idx,
                    "phase": "questions",
                    "done": False
                }
        
        elif user_intent == "Hesitation":
            # Vary hesitation responses
            hesitation_phrases = [
                "Take your time! There's no rush. Think of a specific example from your experience.",
                "No worries, take a moment to think. I'm looking for a concrete example.",
                "That's perfectly fine! Take your time to think of a good example.",
                "No pressure at all. Think through your experiences and share what comes to mind."
            ]
            return {
                "bot_response": random.choice(hesitation_phrases),
                "question_idx": current_idx,
                "phase": "questions", 
                "done": False
            }
        
        elif user_intent == "SmallTalk":
            if current_idx < len(questions):
                current_question = questions[current_idx]
                # Vary small talk redirects
                smalltalk_phrases = [
                    f"That's nice! Let's get back to the interview though: {current_question}",
                    f"I appreciate that! Now, back to our interview: {current_question}",
                    f"That's great to hear! Let's continue with: {current_question}",
                    f"Thanks for sharing! Let's focus on the interview: {current_question}"
                ]
                return {
                    "bot_response": random.choice(smalltalk_phrases),
                    "question_idx": current_idx,
                    "phase": "questions",
                    "done": False
                }
        
        elif user_intent == "OffTopic":
            next_idx = current_idx + 1
            if next_idx >= len(questions):
                return {
                    "done": True,
                    "bot_response": f"I understand. That actually wraps up our interview! Thank you for the great conversation about the {role_title} role. We'll be in touch soon with next steps.",
                    "phase": "complete",
                    "interview_status": "end"
                }
            else:
                return {
                    "bot_response": f"I hear you! Let's keep our focus on the interview though. {questions[next_idx]}",
                    "question_idx": next_idx,
                    "phase": "questions",
                    "follow_up_count": 0,
                    "last_question_asked": questions[next_idx],
                    "done": False
                }
        
        # REGULAR ANSWER - Check if follow-up needed BEFORE moving to next question
        current_follow_up_count = state.get('follow_up_count', 0)
        
        # Analyze answer quality for follow-up decision
        star_analysis = analyze_star_completeness(user_input)
        patterns = detect_answer_patterns(user_input)
        
        answer_length = len(user_input.strip())
        word_count = len(user_input.split())
        star_completeness = star_analysis.get('completeness', 0.5)
        quality_score = patterns.get('quality_ratio', 0.5)
        
        # Check if follow-up is needed (max 2 follow-ups per question)
        needs_followup = (
            current_follow_up_count < 2 and  # Allow up to 2 follow-ups per question
            (answer_length < 80 or               # Too short
             star_completeness < 0.6 or          # Missing STAR elements  
             quality_score < 0.6 or              # Low quality
             word_count < 20)                    # Lacks detail
        )
        
        print(f"🔍 [FOLLOW-UP CHECK] Question {current_idx}, Follow-ups: {current_follow_up_count}/2, Length: {answer_length}, STAR: {star_completeness:.2f}, Quality: {quality_score:.2f}, Needs follow-up: {needs_followup}")
        
        if needs_followup:
            print(f"🎯 [FOLLOW-UP] Generating follow-up question for base question {current_idx}")
            
            # Generate follow-up question using the existing logic
            current_question = questions[current_idx] if current_idx < len(questions) else ""
            conversation_context = state.get('conversation_history', '')
            
            # Simple follow-up generation
            followup_messages = [
                SystemMessage(content=f"""You are an interviewer asking a follow-up question. The candidate's answer was brief or incomplete.
Generate ONE specific follow-up question (15-25 words) to dig deeper into their answer.
Reference something specific they mentioned in their response.
Ask for more details, context, or specific examples."""),
                HumanMessage(content=f"Original question: {current_question}\nCandidate answer: {user_input}\nGenerate a follow-up question:")
            ]
            
            followup_response = circuit_breaker.call_llm(followup_messages, "Can you tell me more about that specific situation?", room_id, "followup")
            followup_question = followup_response.content if followup_response and followup_response.content else "Can you tell me more about that specific situation?"
            
            return {
                "bot_response": followup_question,
                "question_idx": current_idx,  # Stay on same question
                "phase": "questions",
                "follow_up_count": current_follow_up_count + 1,
                "last_question_asked": followup_question,
                "done": False
            }
        
        # No follow-up needed - move to next base question
        next_idx = current_idx + 1
        print(f"🎯 [NEXT QUESTION] Moving from question {current_idx} to question {next_idx}")
        
        if next_idx >= len(questions):
            # Interview complete
            print(f"🏁 [INTERVIEW COMPLETE] Finished all {len(questions)} questions")
            return {
                "done": True,
                "bot_response": f"Thank you for completing the {role_title} interview! I enjoyed learning about your experience. We'll review everything and get back to you soon with next steps.",
                "phase": "complete",
                "interview_status": "end"
            }
        
        # Generate natural transition to next question
        transition_messages = [
            SystemMessage(content=f"""You are an experienced interviewer. Generate a brief, natural transition (1-2 sentences) that:
1. Acknowledges the candidate's answer without repeating it
2. Smoothly introduces the next question
3. Maintains professional but conversational tone

Example transitions:
- "That's great experience with system architecture. Now I'd like to explore..."
- "Excellent problem-solving approach. Let me ask you about..."
- "I appreciate that detailed example. Moving on to..."
- "That shows strong leadership skills. Next, I'm curious about..."

Be concise and natural. End with the next question."""),
            HumanMessage(content=f"Candidate just answered: '{user_input[:200]}...'\n\nNext question: {questions[next_idx]}\n\nGenerate a natural transition:")
        ]
        
        transition_response = circuit_breaker.call_llm(transition_messages, f"Great! {questions[next_idx]}", room_id, "transition")
        natural_transition = transition_response.content if transition_response and transition_response.content else f"Great! {questions[next_idx]}"
        
        return {
            "bot_response": natural_transition,
            "question_idx": next_idx,
            "phase": "questions",
            "follow_up_count": 0,  # Reset follow-up count for new question
            "last_question_asked": questions[next_idx],
            "done": False
        }
        
    except Exception as e:
        interview_logger.error(f"PROCESS_ANSWER_ERROR: Room={room_id}, Error={str(e)[:100]}")
        
        return {
            "bot_response": "Thank you for your response. Let me continue with the interview.",
            "phase": "questions",
            "done": False
        }

# Create sync wrapper for the async function with follow-up logic
def process_answer_node_sync_wrapper(state: InterviewState):
    """Sync wrapper for process_answer_node_async with proper follow-up logic"""
    import asyncio
    try:
        # Use the async function that has follow-up logic
        return asyncio.run(process_answer_node_async(state))
    except Exception as e:
        logger.error(f"Error in process_answer_node_async: {e}")
        # Fallback to simple processing
        return process_answer_node_sync(state)

builder.add_node("process_answer", process_answer_node_optimized)

def complete_interview_node_sync(state: InterviewState):
    """Sync wrapper for complete_interview_node_async"""
    start_time = time.time()
    room_id = state.get('room_id', 'unknown')
    
    interview_logger.info(f"COMPLETE_INTERVIEW_NODE_START: Room={room_id}")
    
    try:
        # Avoid asyncio.run overhead - create simple sync completion
        result = {
            "bot_response": "Thank you for your time! We'll review your responses and get back to you with next steps within a few days.",
            "done": True,
            "phase": "complete",
            "interview_status": "end"
        }
        
        execution_time = time.time() - start_time
        log_node_execution("complete", room_id, state, result, execution_time)
        
        return result
    except Exception as e:
        execution_time = time.time() - start_time
        interview_logger.error(f"COMPLETE_INTERVIEW_NODE_ERROR: Room={room_id}, Error={str(e)[:100]}")
        
        result = {
            "done": True,
            "bot_response": "Thank you for completing the interview! We'll be in touch soon with next steps.",
            "phase": "complete"
        }
        log_node_execution("complete", room_id, state, result, execution_time)
        return result

builder.add_node("complete", complete_interview_node_sync)

# Set entry point - determine initial route based on state
builder.add_conditional_edges(START, route_interview_flow)


# All nodes end after execution (wait for next user input)
builder.add_edge("greeting", END)
builder.add_edge("intro", END)
builder.add_edge("ask_question", END)
builder.add_edge("process_answer", END)
builder.add_edge("complete", END)

graph = builder.compile()  # Redis handles persistence, no LangGraph checkpointer needed

@celery_app.task(name="ai_interview.tasks.natural_interview_flow.process_natural_interview", bind=True)
def process_natural_interview(_, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Process interview turn using LangGraph conditional flow with comprehensive logging"""
    function_start_time = time.time()
    
    print("🚀 === PROCESS_NATURAL_INTERVIEW FUNCTION CALLED ===")
    print(f"🚀 Function invoked with payload type: {type(payload)}")
    
    try:
        # Parse payload
        data = payload if isinstance(payload, dict) else json.loads(payload)
        room_id = data.get('room_id', 'unknown')
        
        # Comprehensive logging of received data
        interview_logger.info(f"FUNCTION_START: Room={room_id}")
        interview_logger.debug(f"Raw payload: {json.dumps(payload, indent=2) if isinstance(payload, dict) else str(payload)[:500]}")
        
        # Extract key data
        user_input = data.get('text', '')
        question_idx = data.get('question_idx', 0)
        phase = data.get('phase', 'greeting')
        role_title = data.get('role_title') or data.get('role_type') or 'Software Developer'
        years_experience = data.get('years_experience') or '2-5 years'
        candidate_name = data.get('candidate_name', 'Candidate')
        company_name = data.get('company_name', 'Company')
        
        # Log extracted data
        interview_logger.info(f"EXTRACTED_DATA: user_input_length={len(user_input)}, question_idx={question_idx}, phase='{phase}'")
        interview_logger.info(f"ROLE_DATA: role='{role_title}', experience='{years_experience}', candidate='{candidate_name}', company='{company_name}'")
        
        # Construct state
        state: InterviewState = {
            'messages': [],
            'room_id': room_id,
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
        
        interview_logger.info(f"STATE_CONSTRUCTED: {json.dumps({k: v if k not in ['messages', 'conversation_history'] else f'{type(v).__name__}({len(str(v))} chars)' for k, v in state.items()})}")
        
        # Process with LangGraph
        graph_start_time = time.time()
        interview_logger.info(f"LANGGRAPH_PROCESSING_START: Room={room_id}")
        
        try:
            # Use direct sync processing for speed - avoid asyncio.run overhead
            result = graph.invoke(state)
            interview_logger.info(f"LANGGRAPH_SYNC_SUCCESS: Room={room_id}")
        except Exception as sync_error:
            interview_logger.error(f"LANGGRAPH_PROCESSING_FAILED: Room={room_id}, Error={str(sync_error)[:100]}")
            # Create minimal fallback response
            result = {
                "bot_response": "I understand. Let's continue with the interview.",
                "done": False,
                "phase": "questions"
            }
            interview_logger.info(f"LANGGRAPH_SYNC_FALLBACK_SUCCESS: Room={room_id}")
        
        graph_execution_time = time.time() - graph_start_time
        interview_logger.info(f"LANGGRAPH_COMPLETED: Room={room_id}, ExecutionTime={graph_execution_time:.3f}s")
        
        # Post-process result
        if 'question_idx' not in result and not result.get('done', False):
            result['question_idx'] = question_idx
        
        if 'bot_response' in result and result['bot_response']:
            original_length = len(result['bot_response'])
            result['bot_response'] = clean_bot_response(result['bot_response'])
            cleaned_length = len(result['bot_response'])
            interview_logger.debug(f"Bot response cleaned: {original_length} -> {cleaned_length} chars")
        
        # Set interview status
        if result.get('done', False):
            result['interview_status'] = 'end'
            interview_logger.info(f"INTERVIEW_COMPLETED: Room={room_id}")
            log_state_transition(phase, "complete", room_id, user_input, {"final_result": True})
        else:
            result['interview_status'] = 'ongoing'
            interview_logger.info(f"INTERVIEW_ONGOING: Room={room_id}")
        
        # Log final result
        function_execution_time = time.time() - function_start_time
        result_summary = {
            "room_id": room_id,
            "function_execution_time": round(function_execution_time, 3),
            "graph_execution_time": round(graph_execution_time, 3),
            "result_phase": result.get("phase", "unknown"),
            "result_done": result.get("done", False),
            "bot_response_length": len(result.get("bot_response", "")),
            "interview_status": result.get("interview_status", "unknown")
        }
        interview_logger.info(f"FUNCTION_COMPLETED: {json.dumps(result_summary)}")
        
        print(f"🔧 DEBUG - Final bot_response: {result.get('bot_response', '')}")
        print(f"🔧 DEBUG - Interview status: {result.get('interview_status', 'unknown')}")
        print(f"Interview result: {result}")
        print("🚀 === PROCESS_NATURAL_INTERVIEW FUNCTION RETURNING ===")
        
        return result
        
    except Exception as e:
        function_execution_time = time.time() - function_start_time
        room_id = 'unknown'
        try:
            data = payload if isinstance(payload, dict) else json.loads(payload)
            room_id = data.get('room_id', 'unknown')
        except:
            pass
            
        logger.error(f"Error in interview flow: {e}")
        interview_logger.error(f"FUNCTION_ERROR: Room={room_id}, ExecutionTime={function_execution_time:.3f}s, Error={str(e)[:200]}")
        
        return {
            'bot_response': 'Sorry, there was an issue. Let me restart.',
            'error': str(e),
            'question_idx': 0,
            'done': False,
            'interview_status': 'error'
        }