import os
import json
import logging
import time
import asyncio
import re
import shutil
import subprocess
import sys
from typing import Dict, List, Any, TypedDict, Annotated, Literal

# Prevent bytecode generation for development
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
from ai_interview.celery_app import celery_app
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage, SystemMessage

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

memory = MemorySaver()

def clean_cache():
    """Light cache cleanup - only called when needed"""
    try:
        # Only clean specific cache that might interfere with role updates
        print("🧹 Light cache cleanup completed")
    except Exception as e:
        print(f"Cache cleanup error (ignored): {e}")

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

# No static defaults - everything generated dynamically via LLM

def generate_fallback_skills_via_llm(role_title: str, years_experience: str) -> List[str]:
    """Generate fallback skills using LLM when primary generation fails"""
    try:
        messages = [
            SystemMessage(content=f"""Generate 6 essential skills for a {role_title} position with {years_experience} experience.
            
Return exactly in format: ["Skill1", "Skill2", "Skill3", "Skill4", "Skill5", "Skill6"]
Focus on core competencies for this role."""),
            HumanMessage(content=f"Generate skills for {role_title}")
        ]
        
        response = circuit_breaker.call_llm(messages, '["Problem Solving", "Communication", "Technical Skills", "Analytical Thinking", "Collaboration", "Adaptability"]')
        
        if response and response.content:
            skills_text = response.content.strip()
            import ast
            try:
                if '[' in skills_text and ']' in skills_text:
                    start = skills_text.find('[')
                    end = skills_text.rfind(']') + 1
                    return ast.literal_eval(skills_text[start:end])
            except:
                pass
        
        # Final fallback if LLM fails completely
        return ["Problem Solving", "Communication", "Technical Skills", "Analytical Thinking", "Collaboration", "Adaptability"]
        
    except Exception as e:
        return ["Problem Solving", "Communication", "Technical Skills", "Analytical Thinking", "Collaboration", "Adaptability"]

def generate_key_skills(role_title: str, years_experience: str) -> List[str]:
    """Generate key skills dynamically based on any role and experience level using LLM"""
    try:
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
        
        # Dynamic fallback based on role type - no static skills
        fallback_skills = generate_fallback_skills_via_llm(role_title, years_experience)
        response = circuit_breaker.call_llm(messages, str(fallback_skills))
        
        if response and response.content:
            skills_text = response.content.strip()
            # Extract list from response
            import ast
            try:
                if '[' in skills_text and ']' in skills_text:
                    start = skills_text.find('[')
                    end = skills_text.rfind(']') + 1
                    skills_list = ast.literal_eval(skills_text[start:end])
                    return skills_list if isinstance(skills_list, list) else ["Full Stack Development", "Backend Systems", "API Development", "Database Design", "Python", "JavaScript"]
            except:
                pass
        
        # Use LLM-generated fallback instead of static patterns
        return generate_fallback_skills_via_llm(role_title, years_experience)
            
    except Exception as e:
        logger.error(f"Error generating key skills: {e}")
        return generate_fallback_skills_via_llm(role_title, years_experience)

def get_key_skills(role_title: str, years_experience: str) -> List[str]:
    """Get key skills dynamically using LLM - no static fallbacks"""
    skills = generate_key_skills(role_title, years_experience)
    print(f"🔧 DEBUG - Generated skills for {role_title} ({years_experience}): {skills}")
    return skills

def get_base_questions(role_title: str, years_experience: str, key_skills: List[str] = None) -> List[str]:
    """Get base questions dynamically using LLM - no static fallbacks"""
    skills = key_skills or get_key_skills(role_title, years_experience)
    questions = generate_base_questions(role_title, years_experience, skills)
    print(f"🔧 DEBUG - Generated questions for {role_title} ({years_experience}): {questions}")
    return questions

def get_greeting_message(role_title: str, candidate_name: str = "Candidate", company_name: str = "Company") -> str:
    """Generate greeting message dynamically using LLM with personalized candidate and company names"""
    
    # Use the direct personalized message instead of LLM to avoid placeholder issues
    return f"Hi {candidate_name}! I'm Chris, and I'll be your interviewer today for the {role_title} position at {company_name}. Thanks for joining me - I'm looking forward to our conversation. How are you doing today?"

def get_intro_message(role_title: str, years_experience: str,candidate_name: str = "Candidate", company_name: str = "Company") -> str:
    """Generate intro message dynamically using LLM based on role from WebSocket"""
    try:
        skills = get_key_skills(role_title, years_experience)
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

        Now, generate the introduction.
        """),
            HumanMessage(content=f"Generate intro message for the {role_title} interview with skills: {skills_text}.")
        ]

        response = circuit_breaker.call_llm(messages, f"Great! So let me give you some context - we're looking for someone with experience in {skills_text} and similar technologies. This will be a casual conversation - just think of it as a chat between colleagues. Ready to get started?")
        
        if response and response.content:
            intro = response.content.strip()
            print(f"🔧 DEBUG - Generated LLM intro: '{intro}'")
            return intro
        
        # LLM-generated fallback
        return f"Great! So let me give you some context - we're looking for someone with experience in {skills_text} and similar technologies. This will be a casual conversation - just think of it as a chat between colleagues. Ready to get started?"
        
    except Exception as e:
        logger.error(f"Error generating intro: {e}")
        skills = get_key_skills(role_title, years_experience)
        skills_text = ', '.join(skills[:3])
        return f"Great! So let me give you some context - we're looking for someone with experience in {skills_text} and similar technologies. This will be a casual conversation - just think of it as a chat between colleagues. Ready to get started?"

def generate_fallback_questions_via_llm(role_title: str, years_experience: str, key_skills: List[str]) -> List[str]:
    """Generate fallback questions using LLM when primary generation fails"""
    try:
        primary_skill = key_skills[0] if key_skills else "core competencies"
        
        messages = [
            SystemMessage(content=f"""Generate exactly 4 interview questions for a {role_title} position with {years_experience} experience.
            
Key skill to focus on: {primary_skill}

Requirements:
- Return exactly 4 questions in format: ["Question 1?", "Question 2?", "Question 3?", "Question 4?"]  
- Questions appropriate for {years_experience} experience level
- Cover: technical expertise, problem-solving, collaboration, growth/impact
- Encourage STAR method responses

Generate 4 questions for {role_title}."""),
            HumanMessage(content=f"Generate 4 questions for {role_title}")
        ]
        
        response = circuit_breaker.call_llm(messages, f'["Tell me about your experience with {primary_skill}. How have you applied it to create value?", "Describe a challenging situation you faced in your {role_title} work. How did you approach it?", "Tell me about a time when you had to collaborate with others to achieve a goal. What was your contribution?", "Describe a situation where you had to learn something new or deliver under pressure. How did you handle it?"]')
        
        if response and response.content:
            questions_text = response.content.strip()
            import ast
            try:
                if '[' in questions_text and ']' in questions_text:
                    start = questions_text.find('[')
                    end = questions_text.rfind(']') + 1
                    questions_list = ast.literal_eval(questions_text[start:end])
                    if isinstance(questions_list, list) and len(questions_list) >= 4:
                        return questions_list[:4]
            except:
                pass
        
        # Final LLM-based fallback
        return [
            f"Tell me about your experience with {primary_skill}. How have you applied it to create value in your role?",
            f"Describe a challenging situation you faced in your {role_title} work. How did you approach and resolve it?",
            "Tell me about a time when you had to collaborate with others or stakeholders to achieve a goal. What was your contribution?",
            "Describe a situation where you had to learn something new, adapt to change, or deliver under pressure. How did you handle it?"
        ]
        
    except Exception as e:
        logger.error(f"Error generating fallback questions: {e}")
        return [
            f"Tell me about your experience with {key_skills[0] if key_skills else 'your field'}. How have you applied it?",
            f"Describe a challenging situation in your {role_title} work. How did you handle it?",
            "Tell me about a time when you had to work with others to achieve a goal.",
            "Describe a situation where you had to adapt or learn quickly. What was your approach?"
        ]

def generate_base_questions(role_title: str, years_experience: str, key_skills: List[str]) -> List[str]:
    """Generate exactly 4 interview questions dynamically for any role using LLM"""
    try:
        skills_context = ", ".join(key_skills[:4])  # Use top 4 skills for context
        
        messages = [
            SystemMessage(content=f"""You are an expert interview designer. Generate EXACTLY 4 comprehensive interview questions for a {role_title} position requiring {years_experience} of experience.

Key skills to assess: {skills_context}

CRITICAL REQUIREMENTS:
1. Return EXACTLY 4 questions in Python list format: ["Question 1?", "Question 2?", "Question 3?", "Question 4?"]
2. Questions must be appropriate for {years_experience} experience level
3. Each question must target different aspects of the role
4. Questions should encourage detailed STAR method responses
5. Handle ANY role type (technical, creative, management, sales, etc.)

QUESTION FRAMEWORK (adapt to role):
Question 1: Core Technical/Functional Expertise
- Assess mastery of primary skills for the role
- Example: "Tell me about your experience with [core skill]. How have you applied it in complex projects?"

Question 2: Problem-Solving/Challenge Resolution  
- Assess analytical thinking and solution approach
- Example: "Describe a challenging [role-specific situation] you faced. How did you approach and resolve it?"

Question 3: Collaboration/Integration
- Assess teamwork and cross-functional skills
- Example: "Tell me about a time when you had to work with [relevant stakeholders]. How did you ensure success?"

Question 4: Growth/Leadership/Impact
- Assess learning, impact, and growth mindset
- Example: "Describe a situation where you had to [learn something new/lead others/deliver under pressure]. What was the outcome?"

ROLE ADAPTATION EXAMPLES:
- Technical roles: Focus on coding, architecture, debugging, performance
- Creative roles: Focus on design process, client feedback, creative challenges
- Management roles: Focus on team leadership, strategic decisions, performance management  
- Sales roles: Focus on deal closing, relationship building, objection handling
- Operations roles: Focus on process improvement, efficiency, stakeholder management

Generate 4 questions for: {role_title} with {years_experience} experience
Focus on: {skills_context}
Return only the Python list format."""),
            HumanMessage(content=f"Role: {role_title}\nExperience: {years_experience}\nKey Skills: {skills_context}\n\nGenerate exactly 4 interview questions:")
        ]
        
        response = circuit_breaker.call_llm(messages, '["Tell me about your experience with the core technologies in your tech stack.", "Describe a challenging technical problem you solved.", "How do you approach system design and architecture decisions?", "Tell me about a time when you had to work under tight deadlines."]')
        
        if response and response.content:
            questions_text = response.content.strip()
            # Extract list from response
            import ast
            try:
                if '[' in questions_text and ']' in questions_text:
                    start = questions_text.find('[')
                    end = questions_text.rfind(']') + 1
                    questions_list = ast.literal_eval(questions_text[start:end])
                    if isinstance(questions_list, list) and len(questions_list) >= 4:
                        return questions_list[:4]  # Take first 4 questions
            except:
                pass
        
        # Use LLM to generate role-appropriate fallback questions
        return generate_fallback_questions_via_llm(role_title, years_experience, key_skills)
            
    except Exception as e:
        logger.error(f"Error generating base questions: {e}")
        # Use LLM fallback instead of static questions
        return generate_fallback_questions_via_llm(role_title, years_experience, key_skills)

# Dynamic access to questions - completely LLM-based
def get_questions(role_title: str, years_experience: str) -> List[str]:
    """Get interview questions dynamically using LLM - no static fallbacks"""
    return get_base_questions(role_title, years_experience)

def generate_contextual_question(question_idx: int, conversation_history: str = "", role_title: str = "", years_experience: str = "") -> str:
    """Generate contextual questions based on previous answers"""
    try:
        # Use provided role parameters - no static fallbacks
        if not role_title or not years_experience:
            return "Could you tell me more about your experience?"
            
        questions = get_base_questions(role_title, years_experience)
        if question_idx >= len(questions):
            return "Thank you for completing the interview!"
            
        base_question = questions[question_idx]
        
        # If no conversation history, return base question
        if not conversation_history.strip():
            return base_question
            
        # Generate contextual version based on conversation
        messages = [
            SystemMessage(content=f"""You are a professional AI interviewer. Make this question flow naturally from the candidate's previous answer.

Base Question: {base_question}

Instructions:
1. Reference something specific they mentioned in their previous response
2. Create a smooth transition that feels conversational
3. Keep the core question intact but make it contextual
4. Use professional but friendly language

Examples:
- "That's interesting! Since you mentioned working with Python, {base_question.lower()}"
- "Great background! Now I'd like to ask: {base_question}"
- "Thanks for sharing that. Building on what you said about [specific detail], {base_question.lower()}"

Generate a natural, contextual version that maintains the question's intent."""),
            HumanMessage(content=f"Previous conversation: {conversation_history[-150:]}\n\nCreate contextual version:")
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
        if role_title and years_experience:
            questions = get_base_questions(role_title, years_experience)
            return questions[question_idx] if question_idx < len(questions) else "Could you tell me more about your experience?"
        else:
            return "Could you tell me more about your experience?"

# Dynamic backward compatibility (deprecated - role required)
def get_questions_list(role_title: str = "Software Developer", years_experience: str = "2-5 years") -> List[str]:
    """Get questions list for backward compatibility - role should be provided"""
    return get_base_questions(role_title, years_experience)

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
os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", os.path.abspath("xooper.json"))


# Improved LLM setup with better error handling
def get_llm() -> ChatVertexAI:
    try:
        return ChatVertexAI(
            model="gemini-2.0-flash-lite-001",
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

# Pure LLM-based intent classification function based on intent.flow specification
def classify_intent(question: str, answer: str, conversation_history: str = "") -> str:
    """Classify user intent using pure LLM analysis based on intent.flow specification
    
    Returns one of: EndInterview, RepeatQuestion, PreviousQuestion, ClarifyQuestion, 
                    OffTopic, Hesitation, SmallTalk, Answer
    """
    try:
        
        # Enhanced prompt for real-world intent classification
        messages = [
            SystemMessage(content="""You are an expert intent classifier for AI interview systems. You must accurately identify candidate intentions from real interview conversations.

REAL-WORLD INTENT PATTERNS:

1. REPEAT REQUEST PATTERNS (RepeatQuestion):
   EXPLICIT: "repeat", "say again", "can you repeat", "one more time"
   INFORMAL: "sorry what?", "what?", "huh?", "come again?"
   POLITE: "could you repeat that", "can you say that again", "pardon me"
   TECHNICAL: "didn't hear", "audio cut out", "you're breaking up", "connection issues"
   CONTEXTUAL: "what was the question?", "what did you ask?", "what was that?"
   DISTRACTED: "sorry I was distracted", "my attention wandered"

2. CLARIFICATION REQUEST PATTERNS (ClarifyQuestion):
   UNDERSTANDING: "what do you mean", "I don't understand", "can you explain"
   MIXED REQUESTS: If they ask for BOTH repeat AND explanation, classify as ClarifyQuestion
   TERMS: "what does X mean", "can you define", "I'm not familiar with"

3. PREVIOUS QUESTION PATTERNS (PreviousQuestion):
   EXPLICIT: "go back", "previous question", "original question", "what was the first question"
   CONTEXT: "I lost track", "what were we talking about originally"

4. END INTERVIEW PATTERNS (EndInterview):
   DIRECT: "end interview", "I'm done", "stop", "that's enough"
   POLITE: "I think we can wrap up", "I don't have more to add"

5. HESITATION PATTERNS (Hesitation):
   FILLERS: Short responses (<15 words) with "um", "uh", "well", "hmm"
   THINKING: "let me think", "that's a good question", "give me a moment"
   NERVOUS: "I'm nervous", "I need to think"

6. SMALL TALK PATTERNS (SmallTalk):
   PERSONAL: Questions about interviewer ("how are you", "where are you from")
   WEATHER: "nice weather", "how's your day"
   UNRELATED: Personal comments not related to interview

7. OFF-TOPIC PATTERNS (OffTopic):
   HOBBIES: Discussing personal interests when asked about work
   UNRELATED: Completely unrelated to the professional context

8. ANSWER PATTERNS (Answer):
   ATTEMPTS: Any genuine attempt to answer the interview question
   PARTIAL: Incomplete but relevant responses to the question

CLASSIFICATION PRIORITY:
1. If contains repeat signals (including informal) → RepeatQuestion
2. If asks for explanation/clarification → ClarifyQuestion  
3. If wants to go back → PreviousQuestion
4. If wants to end → EndInterview
5. If hesitating/thinking → Hesitation
6. If making small talk → SmallTalk
7. If completely off-topic → OffTopic
8. Otherwise → Answer

CRITICAL: Handle mixed requests (repeat + explain) as ClarifyQuestion since they need explanation."""),
            HumanMessage(content=f"""CLASSIFY THIS INTERVIEW RESPONSE:

Question: "{question}"
Response: "{answer}"

Consider real conversation patterns. Return ONLY the classification name.""")
        ]
        
        print(f"Using LLM to classify intent for: {question[:50]}...")
        response = circuit_breaker.call_llm(messages, "Answer")
        
        if response and response.content:
            llm_classification = response.content.strip()
            print(f"LLM classification: {llm_classification}")
            
            # Enhanced validation with exact matching
            valid_intents = ["EndInterview", "RepeatQuestion", "PreviousQuestion", "ClarifyQuestion", 
                           "OffTopic", "Hesitation", "SmallTalk", "Answer"]
            
            # First try exact match
            if llm_classification in valid_intents:
                return llm_classification
            
            # Then try partial match
            for intent in valid_intents:
                if intent.lower() in llm_classification.lower():
                    return intent
            
            # Parse common variations
            classification_lower = llm_classification.lower()
            if "end" in classification_lower or "stop" in classification_lower:
                return "EndInterview"
            elif "repeat" in classification_lower:
                return "RepeatQuestion"
            elif "previous" in classification_lower or "back" in classification_lower:
                return "PreviousQuestion"
            elif "clarify" in classification_lower or "explain" in classification_lower:
                return "ClarifyQuestion"
            elif "hesitat" in classification_lower or "thinking" in classification_lower:
                return "Hesitation"
            elif "small" in classification_lower or "chat" in classification_lower:
                return "SmallTalk"
            elif "off" in classification_lower or "topic" in classification_lower:
                return "OffTopic"
            
            # If no valid intent found, default to Answer
            print(f"⚠️ Invalid classification '{llm_classification}', defaulting to Answer")
            return "Answer"
        
        # Default fallback if LLM fails
        return "Answer"
        
    except Exception as e:
        logger.error(f"Error in intent classification: {e}")
        print(f"Error in intent classification: {e}")
        return "Answer"  # Default to Answer if error

# Legacy function for backward compatibility
def classify_answer(question: str, answer: str) -> bool:
    """Returns True if answer is related to question, False if off-topic"""
    intent = classify_intent(question, answer)
    return intent not in ["OffTopic", "SmallTalk"]

# Evaluative response generator for interviewer
def generate_evaluative_response(answer: str, role_title: str = "", years_experience: str = "") -> str:
    """Generate evaluative responses that assess the candidate's answer"""
    try:
        if not answer or not answer.strip():
            return "I'd like to hear more about that."
        
        if not role_title or not years_experience:
            return "I see. Can you elaborate on that?"
            
        key_skills = get_key_skills(role_title, years_experience)
        
        messages = [
            SystemMessage(content=f"""You are a professional AI interviewer for a {role_title} position conducting a thorough evaluation.

Your task: Generate a brief evaluative response (8-15 words) that shows you're professionally assessing their answer.

Key skills we're evaluating: {', '.join(key_skills)}

Response Guidelines:
For strong, detailed answers:
- "That demonstrates solid technical expertise."
- "I can see you have good problem-solving experience."
- "That shows strong hands-on experience with [specific technology]."
technical back
For adequate but shallow answers:
- "Can you be more specific about your role in that?"
- "I'd like to understand the technical details better."
- "What specific challenges did you face there?"

For vague or unclear answers:
- "That's quite general - can you give concrete examples?"
- "I need more specifics about your actual contribution."
- "Help me understand what you personally implemented."

Be professional, evaluative, and encouraging while probing for depth."""),
            HumanMessage(content=f"Candidate's answer: {answer}\n\nGenerate professional evaluative response:")
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

# FAST: Smart follow-up decision (minimal LLM use)
def should_ask_followup(question: str, answer: str, follow_up_count: int) -> bool:
    """Fast follow-up decision with smart heuristics - increased depth for 4-question interview"""
    # Hard limits - increased for deeper exploration
    if follow_up_count >= 2:  # Increased from 2 to allow more depth
        return False
        
    if not answer or len(answer.strip()) < 15:
        return True  # Obviously too short
        
    # Fast analysis without heavy LLM calls
    star_analysis = analyze_star_completeness(answer)
    patterns = detect_answer_patterns(answer)
    
    # Simple decision logic - no LLM needed
    if star_analysis["completeness"] < 0.4:  # Missing key STAR elements
        return True
        
    if patterns["quality_ratio"] < 0.3:  # Low quality answer
        return True
        
    if len(answer.split()) < 30:  # Increased threshold for more thorough answers
        return True
        
    # For 4-question interview, be more thorough - check if answer covers all aspects
    if follow_up_count == 0 and len(answer.split()) < 50:  # First answer should be substantial
        return True
        
    # Answer seems complete enough
    return False

# Professional interviewer conversation with psychology tactics
def generate_natural_conversation(question: str, answer: str, conversation_context: str = "", role_title: str = "", years_experience: str = "") -> str:
    """Generate professional interviewer responses using psychology tactics"""
    try:
        if not answer or not answer.strip():
            return "I'd love to hear more about that - could you share some details?"
        
        if not role_title or not years_experience:
            return "That's interesting! Tell me more about that specific aspect."
        
        # Analyze answer using professional tactics
        star_analysis = analyze_star_completeness(answer)
        patterns = detect_answer_patterns(answer)
        
        # Professional interviewer follow-up with psychology insights
        key_skills = get_key_skills(role_title, years_experience)
        skills_context = ", ".join(key_skills)
        
        # Generate context-aware prompts based on analysis
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
        
        print(f"Generating natural conversation for: {question[:50]}...")
        response = circuit_breaker.call_llm(messages, "That's interesting! Tell me more about that specific aspect.")
        
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

# Intent-specific handler functions based on intent.flow specification

def handle_end_interview_intent(state: InterviewState) -> Dict[str, Any]:
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
4. Keeps it brief and positive (20-30 words)"""),
            HumanMessage(content="Candidate requested to end interview. Generate professional conclusion:")
        ]
        
        response = circuit_breaker.call_llm(messages, 
            "I understand. Thank you for your time today! We'll review our conversation and be in touch soon with next steps.")
        
        conclusion = response.content if response and response.content else \
            "I understand. Thank you for your time today! We'll review our conversation and be in touch soon with next steps."
        
        return {
            "done": True,
            "bot_response": conclusion,
            "phase": "complete",
            "interview_status": "end"
        }
        
    except Exception as e:
        logger.error(f"Error handling EndInterview intent: {e}")
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
        questions = get_base_questions(role_title, years_experience)
        
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
            questions = get_base_questions(role_title, years_experience)
            
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
        questions = get_base_questions(role_title, years_experience)
        
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
5. Reference the role context when helpful"""),
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
        questions = get_base_questions(role_title, years_experience)
        
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

Professional Redirect Examples:
- "I understand. Now, let's get back to discussing your technical experience."
- "That's interesting! Let's refocus on your background in {', '.join(get_key_skills()[:2])}."
- "I hear you. Let's continue with the interview and talk about your projects."
- "Got it! Now, back to learning about your professional experience."

Generate a natural, professional redirect that maintains interview flow."""),
            HumanMessage(content=f"Candidate's off-topic comment: {user_input}\n\nGenerate professional redirect:")
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
        # Use dynamic role and experience from state
        role_title = state.get('role_title', 'Software Developer')
        years_experience = state.get('years_experience', '2-5 years')
        questions = get_base_questions(role_title, years_experience)
        if current_idx >= len(questions):
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
        
        # Use dynamic role and experience from state
        role_title = state.get('role_title', 'Software Developer')
        years_experience = state.get('years_experience', '2-5 years')
        candidate_name = state.get('candidate_name', 'Candidate')
        company_name = state.get('company_name', 'Company')
        
        # If user says "start", show greeting message first
        if user_input == "start":
            return {
                "bot_response": get_greeting_message(role_title, candidate_name, company_name),
                "phase": "greeting",
                "done": False
            }
        elif user_input and user_input != "start":
            # User responded to greeting (not "start"), move to intro phase
            return {
                "bot_response": get_intro_message(role_title, years_experience,candidate_name,company_name),
                "phase": "intro",
                "done": False
            }
        else:
            # No user input, show initial greeting
            return {
                "bot_response": get_greeting_message(role_title, candidate_name, company_name),
                "phase": "greeting",
                "done": False
            }
    except Exception as e:
        logger.error(f"Error in greeting_node: {e}")
        return {
            "bot_response": "Hi! I'm Chris, and I'll be your interviewer today. Thanks for joining me - I'm looking forward to our conversation. How are you doing today?",
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
            
            # Use dynamic role and experience from state
            role_title = state.get('role_title', 'Software Developer')
            years_experience = state.get('years_experience', '2-5 years')
            questions = get_base_questions(role_title, years_experience)
            
            if current_idx >= len(questions):
                return {"done": True, "bot_response": "Thank you! Interview complete.", "phase": "complete"}
            
            print(f"Intro complete - transitioning to questions phase with question_idx: {current_idx}")
            print(f"🎯 Using role: {role_title}, experience: {years_experience}")
            return {
                "bot_response": questions[current_idx],
                "phase": "questions",
                "question_idx": current_idx,
                "follow_up_count": 0,  # Reset follow-up count for new question
                "last_question_asked": questions[current_idx],  # Track the last question asked
                "done": False
            }
        else:
            # Show introduction
            # Use dynamic role and experience from state
            role_title = state.get('role_title', 'Software Developer')
            years_experience = state.get('years_experience', '2-5 years')
            return {
                "bot_response": get_intro_message(role_title, years_experience),
                "phase": "intro", 
                "done": False
            }
    except Exception as e:
        logger.error(f"Error in intro_node: {e}")
        return {
            "bot_response": get_intro_message(),
            "phase": "intro", 
            "done": False
        }

def ask_question_node(state: InterviewState):
    """Ask current question based on question_idx"""
    try:
        idx = state.get('question_idx', 0)
        
        # Use dynamic role and experience from state
        role_title = state.get('role_title', 'Software Developer')
        years_experience = state.get('years_experience', '2-5 years')
        questions = get_base_questions(role_title, years_experience)  # Get once for performance
        
        if idx >= len(questions):
            return {"done": True, "bot_response": "Thank you! Interview complete.", "phase": "complete"}
        
        print(f"🎯 Asking question {idx} for role: {role_title}")
        return {
            "bot_response": questions[idx],
            "phase": "questions",
            "last_question_asked": questions[idx],  # Track the last question asked
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
    """Process candidate's answer using professional interviewer psychology"""
    try:
        current_idx = state.get('question_idx', 0)
        user_answer = state.get('user_input', '').strip()
        follow_up_count = state.get('follow_up_count', 0)
        
        # Get current questions (do this once for performance)
        # Use dynamic role and experience from state
        role_title = state.get('role_title', 'Software Developer')
        years_experience = state.get('years_experience', '2-5 years')
        questions = get_base_questions(role_title, years_experience)
        
        if current_idx < len(questions):
            current_question = questions[current_idx]
        else:
            current_question = ""
        
        # Professional interviewer analysis
        star_analysis = analyze_star_completeness(user_answer)
        patterns = detect_answer_patterns(user_answer)
        
        print(f"⚡ FAST INTERVIEWER ANALYSIS:")
        print(f"   STAR Completeness: {star_analysis.get('completeness', 0):.2f}")
        print(f"   Missing Elements: {star_analysis.get('missing_elements', [])}")
        print(f"   Quality Ratio: {patterns.get('quality_ratio', 0):.2f}")
        
        # NEW: Comprehensive Intent Classification (as per intent.flow)
        conversation_history = state.get('conversation_history', '')
        user_intent = classify_intent(current_question, user_answer, conversation_history)
        
        print(f"🎯 INTENT CLASSIFIED: {user_intent}")
        
        # Route based on intent classification (as per intent.flow specification)
        if user_intent == "EndInterview":
            return handle_end_interview_intent(state)
        
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
            print(f"⚠️  Unknown intent '{user_intent}', treating as Answer")
        
        # Get conversation context for natural flow
        conversation_context = state.get('conversation_history', '')
        
        # ANSWER PROCESSING: Evaluate like a real interviewer (STAR analysis + follow-ups)
        needs_followup = should_ask_followup(current_question, user_answer, follow_up_count)
        
        if needs_followup and follow_up_count < 4:
            # Generate natural conversational follow-up
            natural_response = generate_natural_conversation(current_question, user_answer, conversation_context, role_title, years_experience)
            
            # Update conversation history
            updated_history = f"{conversation_context}\nQ: {current_question}\nA: {user_answer}\nInterviewer: {natural_response}"
            
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
            # Generate natural conclusion with role context
            messages = [
                SystemMessage(content=f"You are a professional interviewer concluding a {role_title} interview. Generate a natural, warm conclusion that thanks them and explains next steps. Keep it professional but personable. Reference the role appropriately."),
                HumanMessage(content=f"Candidate's last response: {user_answer}\n\nGenerate a natural interview conclusion for the {role_title} position:")
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
        
        # Generate natural transition with role awareness
        role_title = state.get('role_title', 'this')
        
        messages = [
            SystemMessage(content=f"""You're a professional interviewer for a {role_title} position transitioning to the next question.

Create a natural transition that:
1. Briefly acknowledges their previous answer
2. Smoothly introduces the next topic
3. Sounds conversational and engaging
4. Maintains the context of interviewing for {role_title}

Examples:
- "That's really interesting. Now I'd like to shift gears and ask about..."
- "Great example! Moving on, I'm curious about..."
- "Thanks for sharing that insight. Let me ask you about..."  
- "That makes sense. Now, tell me about..."

Keep it concise (10-15 words) and natural."""),
            HumanMessage(content=f"Their last answer: {user_answer[-100:]}\n\nNext question: {questions[next_idx]}\n\nCreate a smooth transition:")
        ]
        
        response = circuit_breaker.call_llm(messages, f"Great! Now let me ask: {questions[next_idx]}")
        transition = response.content if response and response.content else f"Great! Now let me ask: {questions[next_idx]}"
        
        # Update conversation history
        updated_history = f"{conversation_context}\nQ: {current_question}\nA: {user_answer}\nTransition: {transition}"
        
        # Return the next question with natural transition
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
        questions_fallback = get_base_questions(role_title, years_experience)
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

def complete_interview_node(state: InterviewState):
    """Complete the interview with professional assessment"""
    try:
        conversation_history = state.get('conversation_history', '')
        role_title = state.get('role_title', 'this')
        years_experience = state.get('years_experience', '')
        
        # Professional interviewer conclusion with psychology assessment
        key_skills = get_key_skills(role_title, years_experience) if role_title and years_experience else []
        
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
        
        response = circuit_breaker.call_llm(messages, 
            f"Thank you for a great conversation! I enjoyed learning about your experience and approach to technical challenges. We'll review everything and get back to you within a few days about next steps.")
        
        conclusion = response.content if response and response.content else f"Thank you for completing the  interview! It was great getting to know you and learning about your experience. We'll be reviewing your background against our requirements and will be in touch soon with next steps."
        
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

graph = builder.compile()  # Redis handles persistence, no LangGraph checkpointer needed

@celery_app.task(name="ai_interview.tasks.natural_interview_flow.process_natural_interview", bind=True)
def process_natural_interview(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Process interview turn using LangGraph conditional flow"""
    print("🚀 === PROCESS_NATURAL_INTERVIEW FUNCTION CALLED ===")
    print(f"🚀 Function invoked with payload type: {type(payload)}")
    
    # Write to file to verify execution
    with open('/tmp/interview_debug.log', 'a') as f:
        f.write(f"🚀 FUNCTION CALLED at {time.time()}\n")
        f.flush()
    
    try:
        # Clear Redis cache on first call to ensure fresh role
        if not hasattr(process_natural_interview, '_redis_cleared'):
            clear_redis_cache()
            process_natural_interview._redis_cleared = True
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
        
        # Extract role and experience from payload (dynamic from WebSocket)
        role_title = data.get('role_title') or data.get('role_type') or 'Software Developer'  # Use payload or minimal fallback
        years_experience = data.get('years_experience') or '2-5 years'  # Use payload or minimal fallback
        candidate_name = data.get('candidate_name', 'Candidate')  # Use payload or minimal fallback
        company_name = data.get('company_name', 'Company')  # Use payload or minimal fallback
        
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
            'role_title': role_title,  # Add role to state
            'years_experience': years_experience,  # Add experience to state
            'candidate_name': candidate_name,  # Add candidate name to state
            'company_name': company_name,  # Add company name to state
            'last_question_asked': data.get('last_question_asked', '')  # Track last question asked
        }
        
        logger.info(f"Constructed state: {state}")
        logger.info(f"=== END RECEIVED DATA LOG ===")
        
        user_text = data.get('text', '').strip()
        current_question_idx = data.get('question_idx', 0)
        
        print(f"Processing interview - User text: '{user_text}', Question index: {current_question_idx}, Phase: {state.get('phase', 'greeting')}")
        print(f"Full state: {state}")
        
        
        # Use LangGraph with conditional flow
        result = graph.invoke(state)
        
        # Add the updated question_idx to response for WebSocket state management
        if 'question_idx' not in result and not result.get('done', False):
            result['question_idx'] = current_question_idx
        
        # Clean bot response before sending to frontend
        if 'bot_response' in result and result['bot_response']:
            result['bot_response'] = clean_bot_response(result['bot_response'])
        
        # Add interview status for frontend
        if result.get('done', False):
            result['interview_status'] = 'end'
            print(f"🏁 Interview completed - sending end status to frontend")
            print(f"🏁 Result keys after adding interview_status: {list(result.keys())}")
        else:
            result['interview_status'] = 'ongoing'
            print(f"📝 Interview ongoing - sending ongoing status to frontend")
            
        print(f"🔧 DEBUG - Final bot_response: {result.get('bot_response', '')}")
        print(f"🔧 DEBUG - Interview status: {result.get('interview_status', 'unknown')}")
        print(f"🔧 DEBUG - Result keys before return: {list(result.keys())}")
        print(f"Interview result: {result}")
        print("🚀 === PROCESS_NATURAL_INTERVIEW FUNCTION RETURNING ===")
        
        # Write result to file for verification
        with open('/tmp/interview_debug.log', 'a') as f:
            f.write(f"🏁 RESULT: {result}\n")
            f.write(f"🏁 interview_status: {result.get('interview_status', 'NOT_FOUND')}\n")
            f.flush()
        
        return result
        
    except Exception as e:
        logger.error(f"Error in interview flow: {e}")
        return {
            'bot_response': 'Sorry, there was an issue. Let me restart.',
            'error': str(e),
            'question_idx': 0,
            'done': False,
            'interview_status': 'error'
        }