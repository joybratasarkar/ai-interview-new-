# apps/ai_interview/config/interview_config.py
"""
Configuration for the natural interview system
"""

class InterviewConfig:
    """Configuration class for interview parameters"""
    
    # Core interview limits
    MAX_QUESTIONS = 6
    MAX_FOLLOWUPS_PER_QUESTION = 1
    
    # API Configuration (for future technical question integration)
    TECHNICAL_QUESTIONS_API_URL = "https://api.example.com/questions"
    API_TIMEOUT = 30  # seconds
    
    # Default question set (can be overridden by API)
    CORE_QUESTIONS = [
        "Can you briefly introduce yourself and tell me about your background?",
        "Walk me through your most recent work experience.", 
        "What programming languages and technologies are you most comfortable with?",
        "Tell me about a recent project you're particularly proud of.",
        "Describe your problem-solving approach when facing a new challenge.",
        "Where do you see yourself in the next few years?"
    ]
    
    # Question categories for API integration
    QUESTION_CATEGORIES = {
        "introduction": [
            "Can you briefly introduce yourself and tell me about your background?",
            "What motivated you to apply for this position?"
        ],
        "experience": [
            "Walk me through your most recent work experience.",
            "What has been your biggest professional achievement so far?"
        ],
        "technical": [
            "What programming languages and technologies are you most comfortable with?",
            "Tell me about a complex technical problem you solved recently."
        ],
        "projects": [
            "Tell me about a recent project you're particularly proud of.",
            "What's the most challenging technical project you've worked on?"
        ],
        "problem_solving": [
            "Describe your problem-solving approach when facing a new challenge.",
            "Tell me about a time when you had to think outside the box."
        ],
        "career": [
            "Where do you see yourself in the next few years?",
            "What motivates you professionally?"
        ]
    }
    
    @classmethod
    def get_questions_for_role(cls, job_role: str = "software_developer") -> list:
        """
        Get questions based on job role
        In the future, this will call an API
        """
        # TODO: Implement API integration
        # For now, return default questions
        return cls.CORE_QUESTIONS[:cls.MAX_QUESTIONS]
    
    @classmethod
    def customize_interview(cls, max_questions: int = None, max_followups: int = None, custom_questions: list = None):
        """
        Customize interview parameters
        """
        if max_questions is not None:
            cls.MAX_QUESTIONS = max_questions
        if max_followups is not None:
            cls.MAX_FOLLOWUPS_PER_QUESTION = max_followups
        if custom_questions is not None:
            cls.CORE_QUESTIONS = custom_questions[:cls.MAX_QUESTIONS]
    
    @classmethod
    def reset_to_defaults(cls):
        """Reset to default configuration"""
        cls.MAX_QUESTIONS = 6
        cls.MAX_FOLLOWUPS_PER_QUESTION = 1