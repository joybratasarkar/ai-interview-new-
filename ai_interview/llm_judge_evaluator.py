#!/usr/bin/env python3
"""
LLM Judge Evaluator for AI Interview System
Evaluates interview flow quality, responses, and candidate experience
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class InterviewMetrics:
    """Comprehensive interview evaluation metrics"""
    session_id: str
    role_title: str
    years_experience: str
    candidate_name: str

    # Flow metrics
    total_questions_asked: int = 0
    follow_up_questions: int = 0
    phase_transitions: List[str] = None
    conversation_turns: int = 0

    # Quality metrics
    question_relevance_scores: List[float] = None
    response_naturalness_scores: List[float] = None
    conversation_flow_score: float = 0.0

    # Performance metrics
    avg_response_time: float = 0.0
    session_duration: float = 0.0
    completion_rate: float = 0.0

    # Content analysis
    technical_depth_score: float = 0.0
    star_method_adherence: float = 0.0
    interviewer_professionalism: float = 0.0

    # Overall scores
    overall_interview_quality: float = 0.0
    candidate_experience_score: float = 0.0

    def __post_init__(self):
        if self.phase_transitions is None:
            self.phase_transitions = []
        if self.question_relevance_scores is None:
            self.question_relevance_scores = []
        if self.response_naturalness_scores is None:
            self.response_naturalness_scores = []

class LLMJudge:
    """Advanced LLM-based judge for evaluating interview quality"""

    def __init__(self):
        self.evaluation_history: Dict[str, InterviewMetrics] = {}

        # Evaluation criteria templates
        self.question_quality_prompt = """
You are an expert interview evaluator. Analyze the following interview question for quality.

Role: {role_title} ({years_experience} experience)
Question: "{question}"
Context: This is question #{question_num} in the interview

Evaluate on these criteria (score 1-10):
1. Relevance to role and experience level
2. Technical appropriateness
3. Clarity and understandability
4. Potential to elicit meaningful responses
5. Professional tone and phrasing

Provide scores and brief justification:
RELEVANCE: [score]
TECHNICAL: [score]
CLARITY: [score]
MEANINGFUL: [score]
PROFESSIONAL: [score]
OVERALL: [average score]

JUSTIFICATION: [2-3 sentences explaining the assessment]
"""

        self.response_naturalness_prompt = """
You are an expert conversation evaluator. Analyze this interviewer response for naturalness and professionalism.

Context: {role_title} interview, {years_experience} experience level
Candidate said: "{candidate_response}"
Interviewer responded: "{interviewer_response}"

Evaluate on these criteria (score 1-10):
1. Natural conversation flow
2. Professional tone
3. Appropriate acknowledgment of candidate's response
4. Logical transition to next topic
5. Human-like vs robotic language

Provide scores:
NATURALNESS: [score]
PROFESSIONAL: [score]
ACKNOWLEDGMENT: [score]
TRANSITION: [score]
HUMAN_LIKE: [score]
OVERALL: [average score]

FEEDBACK: [Brief improvement suggestions if any]
"""

        self.star_method_prompt = """
You are an expert behavioral interview evaluator. Analyze if this candidate response follows the STAR method.

Question: "{question}"
Candidate Response: "{response}"

Evaluate STAR method adherence (score 0-10 for each):
SITUATION: [score] - Did they describe the context/situation?
TASK: [score] - Did they explain their responsibility/goal?
ACTION: [score] - Did they detail specific actions taken?
RESULT: [score] - Did they share measurable outcomes?

OVERALL_STAR_SCORE: [average score]
COMPLETENESS: [percentage 0-100%]

ANALYSIS: [Brief assessment of what's missing or well-done]
"""

        self.overall_interview_prompt = """
You are a senior hiring manager evaluating the overall quality of this AI interview session.

INTERVIEW SUMMARY:
Role: {role_title} ({years_experience})
Duration: {duration} minutes
Questions Asked: {total_questions}
Follow-ups: {follow_ups}
Conversation Turns: {turns}

CONVERSATION FLOW:
{conversation_summary}

METRICS:
- Avg Question Relevance: {avg_question_score}/10
- Avg Response Naturalness: {avg_naturalness}/10
- Technical Depth: {technical_depth}/10
- STAR Method Adherence: {star_adherence}/10

Provide final evaluation (score 1-10):
INTERVIEW_QUALITY: [score] - Overall interview effectiveness
CANDIDATE_EXPERIENCE: [score] - How positive was the candidate experience
TECHNICAL_ASSESSMENT: [score] - Quality of technical evaluation
CONVERSATION_FLOW: [score] - Natural conversation progression
PROFESSIONALISM: [score] - Interviewer professionalism

OVERALL_SCORE: [average score]

SUMMARY: [3-4 sentences summarizing the interview quality and areas for improvement]
"""

    async def evaluate_question_quality(self, question: str, role_title: str, years_experience: str, question_num: int) -> Dict[str, Any]:
        """Evaluate the quality of an interview question"""
        try:
            # For now, return a comprehensive mock evaluation
            # In production, this would call an actual LLM

            # Simulate LLM thinking time
            await asyncio.sleep(0.1)

            # Analyze question characteristics
            technical_keywords = ['algorithm', 'design', 'scale', 'performance', 'architecture', 'implement', 'optimize']
            behavioral_keywords = ['tell me about', 'describe a time', 'how do you handle', 'experience with']

            question_lower = question.lower()
            is_technical = any(keyword in question_lower for keyword in technical_keywords)
            is_behavioral = any(keyword in question_lower for keyword in behavioral_keywords)

            # Base scores with some intelligent analysis
            relevance_score = 8.5 if (is_technical and 'developer' in role_title.lower()) else 7.0
            technical_score = 8.0 if is_technical else 6.5
            clarity_score = 8.5 if len(question.split()) < 20 else 7.0  # Shorter questions are clearer
            meaningful_score = 8.0 if (is_behavioral or is_technical) else 6.0
            professional_score = 9.0 if question.endswith('?') else 7.5

            overall_score = (relevance_score + technical_score + clarity_score + meaningful_score + professional_score) / 5

            return {
                "relevance": relevance_score,
                "technical": technical_score,
                "clarity": clarity_score,
                "meaningful": meaningful_score,
                "professional": professional_score,
                "overall": overall_score,
                "justification": f"Question demonstrates good {'technical' if is_technical else 'behavioral'} focus appropriate for {role_title} role. Clear phrasing with professional tone."
            }

        except Exception as e:
            logger.error(f"Error evaluating question quality: {e}")
            return {
                "relevance": 7.0, "technical": 7.0, "clarity": 7.0,
                "meaningful": 7.0, "professional": 7.0, "overall": 7.0,
                "justification": "Default evaluation due to processing error"
            }

    async def evaluate_response_naturalness(self, candidate_response: str, interviewer_response: str,
                                          role_title: str, years_experience: str) -> Dict[str, Any]:
        """Evaluate how natural and professional the interviewer's response is"""
        try:
            await asyncio.sleep(0.1)

            # Analyze response characteristics
            response_lower = interviewer_response.lower()

            # Check for natural conversation indicators
            acknowledgments = ['that\'s', 'great', 'interesting', 'i see', 'good', 'excellent', 'perfect']
            transitions = ['now', 'next', 'let\'s', 'moving on', 'speaking of', 'that brings']
            professional_phrases = ['thank you', 'i understand', 'let me', 'could you', 'would you']

            has_acknowledgment = any(phrase in response_lower for phrase in acknowledgments)
            has_transition = any(phrase in response_lower for phrase in transitions)
            is_professional = any(phrase in response_lower for phrase in professional_phrases)

            # Calculate scores
            naturalness_score = 8.5 if has_acknowledgment else 6.5
            professional_score = 9.0 if is_professional else 7.0
            acknowledgment_score = 8.0 if has_acknowledgment else 5.0
            transition_score = 8.0 if has_transition else 6.5
            human_like_score = 8.5 if (has_acknowledgment and not response_lower.startswith('next question')) else 6.0

            overall_score = (naturalness_score + professional_score + acknowledgment_score +
                           transition_score + human_like_score) / 5

            feedback = "Response shows natural conversation flow" if overall_score > 7.5 else "Could be more conversational and acknowledging"

            return {
                "naturalness": naturalness_score,
                "professional": professional_score,
                "acknowledgment": acknowledgment_score,
                "transition": transition_score,
                "human_like": human_like_score,
                "overall": overall_score,
                "feedback": feedback
            }

        except Exception as e:
            logger.error(f"Error evaluating response naturalness: {e}")
            return {
                "naturalness": 7.0, "professional": 7.0, "acknowledgment": 7.0,
                "transition": 7.0, "human_like": 7.0, "overall": 7.0,
                "feedback": "Default evaluation due to processing error"
            }

    async def evaluate_star_method(self, question: str, response: str) -> Dict[str, Any]:
        """Evaluate how well a candidate response follows STAR method"""
        try:
            await asyncio.sleep(0.1)

            response_lower = response.lower()

            # STAR indicators
            situation_indicators = ['when', 'at my previous', 'in my role', 'project', 'company', 'team']
            task_indicators = ['responsible', 'needed to', 'goal', 'objective', 'task', 'my job']
            action_indicators = ['i did', 'i used', 'i implemented', 'i created', 'my approach', 'i decided']
            result_indicators = ['result', 'outcome', 'achieved', 'improved', 'reduced', 'increased']

            situation_score = 8.0 if any(indicator in response_lower for indicator in situation_indicators) else 3.0
            task_score = 7.5 if any(indicator in response_lower for indicator in task_indicators) else 3.0
            action_score = 8.5 if any(indicator in response_lower for indicator in action_indicators) else 4.0
            result_score = 7.0 if any(indicator in response_lower for indicator in result_indicators) else 2.0

            overall_star_score = (situation_score + task_score + action_score + result_score) / 4
            completeness = min(100, (overall_star_score / 10) * 100)

            return {
                "situation": situation_score,
                "task": task_score,
                "action": action_score,
                "result": result_score,
                "overall_star_score": overall_star_score,
                "completeness": completeness,
                "analysis": f"Response demonstrates {int(completeness)}% STAR method adherence. {'Strong' if overall_star_score > 6 else 'Weak'} structure present."
            }

        except Exception as e:
            logger.error(f"Error evaluating STAR method: {e}")
            return {
                "situation": 5.0, "task": 5.0, "action": 5.0, "result": 5.0,
                "overall_star_score": 5.0, "completeness": 50.0,
                "analysis": "Default evaluation due to processing error"
            }

    async def evaluate_technical_depth(self, questions: List[str], responses: List[str],
                                     role_title: str) -> float:
        """Evaluate the technical depth of the interview"""
        try:
            await asyncio.sleep(0.1)

            # Technical keywords for different roles
            technical_keywords = {
                'software developer': ['algorithm', 'database', 'api', 'architecture', 'scalability', 'performance', 'design pattern'],
                'data scientist': ['machine learning', 'statistics', 'model', 'dataset', 'analysis', 'visualization', 'algorithm'],
                'devops': ['deployment', 'ci/cd', 'docker', 'kubernetes', 'infrastructure', 'monitoring', 'automation'],
                'product manager': ['user research', 'roadmap', 'metrics', 'stakeholder', 'priority', 'market', 'feature']
            }

            role_lower = role_title.lower()
            relevant_keywords = []
            for role_key, keywords in technical_keywords.items():
                if role_key in role_lower:
                    relevant_keywords = keywords
                    break

            if not relevant_keywords:
                relevant_keywords = technical_keywords['software developer']  # Default

            # Count technical content
            total_technical_mentions = 0
            total_content = " ".join(questions + responses).lower()

            for keyword in relevant_keywords:
                total_technical_mentions += total_content.count(keyword)

            # Score based on technical density
            technical_score = min(10.0, (total_technical_mentions / len(questions)) * 2)

            return max(4.0, technical_score)  # Minimum score of 4.0

        except Exception as e:
            logger.error(f"Error evaluating technical depth: {e}")
            return 6.0

    async def generate_overall_evaluation(self, metrics: InterviewMetrics,
                                        conversation_summary: str) -> Dict[str, Any]:
        """Generate comprehensive overall interview evaluation"""
        try:
            await asyncio.sleep(0.2)

            # Calculate averages
            avg_question_score = sum(metrics.question_relevance_scores) / len(metrics.question_relevance_scores) if metrics.question_relevance_scores else 7.0
            avg_naturalness = sum(metrics.response_naturalness_scores) / len(metrics.response_naturalness_scores) if metrics.response_naturalness_scores else 7.0

            # Base evaluation scores
            interview_quality = (avg_question_score + avg_naturalness + metrics.technical_depth_score) / 3
            candidate_experience = (avg_naturalness + metrics.interviewer_professionalism + metrics.conversation_flow_score) / 3
            technical_assessment = metrics.technical_depth_score
            conversation_flow = metrics.conversation_flow_score
            professionalism = metrics.interviewer_professionalism

            overall_score = (interview_quality + candidate_experience + technical_assessment +
                           conversation_flow + professionalism) / 5

            # Generate summary
            quality_level = "Excellent" if overall_score >= 8.5 else "Good" if overall_score >= 7.0 else "Fair" if overall_score >= 5.5 else "Needs Improvement"

            summary = f"This {quality_level.lower()} interview session for {metrics.role_title} demonstrated {'strong' if interview_quality > 7.5 else 'adequate'} technical assessment capabilities. "
            summary += f"The conversation flow was {'natural and engaging' if conversation_flow > 7.5 else 'functional but could be more engaging'}. "
            summary += f"Candidate experience was {'positive' if candidate_experience > 7.0 else 'acceptable'} with good professional tone maintained throughout."

            if overall_score < 7.0:
                summary += " Areas for improvement include more natural conversation transitions and deeper technical questioning."

            return {
                "interview_quality": interview_quality,
                "candidate_experience": candidate_experience,
                "technical_assessment": technical_assessment,
                "conversation_flow": conversation_flow,
                "professionalism": professionalism,
                "overall_score": overall_score,
                "summary": summary,
                "quality_level": quality_level
            }

        except Exception as e:
            logger.error(f"Error generating overall evaluation: {e}")
            return {
                "interview_quality": 7.0, "candidate_experience": 7.0, "technical_assessment": 7.0,
                "conversation_flow": 7.0, "professionalism": 7.0, "overall_score": 7.0,
                "summary": "Default evaluation due to processing error",
                "quality_level": "Good"
            }

    async def evaluate_complete_interview(self, session_id: str, conversation_log: List[Dict],
                                        role_title: str, years_experience: str,
                                        candidate_name: str = "Test Candidate") -> InterviewMetrics:
        """Perform comprehensive evaluation of a complete interview session"""

        logger.info(f"🧠 Starting comprehensive interview evaluation for session {session_id}")

        # Initialize metrics
        metrics = InterviewMetrics(
            session_id=session_id,
            role_title=role_title,
            years_experience=years_experience,
            candidate_name=candidate_name
        )

        start_time = time.time()

        # Process conversation log
        questions = []
        candidate_responses = []
        interviewer_responses = []
        follow_ups = 0

        for i, entry in enumerate(conversation_log):
            if entry.get('type') == 'interviewer_question':
                questions.append(entry['content'])
                metrics.total_questions_asked += 1

                # Evaluate question quality
                question_eval = await self.evaluate_question_quality(
                    entry['content'], role_title, years_experience, len(questions)
                )
                metrics.question_relevance_scores.append(question_eval['overall'])

            elif entry.get('type') == 'candidate_response':
                candidate_responses.append(entry['content'])

                # Evaluate STAR method if it's a substantial response
                if len(entry['content'].split()) > 10:
                    star_eval = await self.evaluate_star_method(
                        questions[-1] if questions else "", entry['content']
                    )
                    if not hasattr(metrics, 'star_scores'):
                        metrics.star_scores = []
                    metrics.star_scores = getattr(metrics, 'star_scores', [])
                    metrics.star_scores.append(star_eval['overall_star_score'])

            elif entry.get('type') == 'interviewer_response':
                interviewer_responses.append(entry['content'])

                # Check if it's a follow-up
                if 'follow' in entry.get('context', '').lower():
                    follow_ups += 1

                # Evaluate response naturalness
                prev_candidate = candidate_responses[-1] if candidate_responses else ""
                naturalness_eval = await self.evaluate_response_naturalness(
                    prev_candidate, entry['content'], role_title, years_experience
                )
                metrics.response_naturalness_scores.append(naturalness_eval['overall'])

            elif entry.get('type') == 'phase_transition':
                metrics.phase_transitions.append(entry['phase'])

        metrics.follow_up_questions = follow_ups
        metrics.conversation_turns = len(conversation_log)

        # Calculate technical depth
        metrics.technical_depth_score = await self.evaluate_technical_depth(
            questions, candidate_responses, role_title
        )

        # Calculate STAR method adherence
        if hasattr(metrics, 'star_scores') and metrics.star_scores:
            metrics.star_method_adherence = sum(metrics.star_scores) / len(metrics.star_scores)
        else:
            metrics.star_method_adherence = 6.0  # Default for non-behavioral questions

        # Calculate conversation flow score
        metrics.conversation_flow_score = sum(metrics.response_naturalness_scores) / len(metrics.response_naturalness_scores) if metrics.response_naturalness_scores else 7.0

        # Calculate interviewer professionalism
        metrics.interviewer_professionalism = metrics.conversation_flow_score * 0.9  # Slightly lower than flow

        # Performance metrics
        metrics.session_duration = time.time() - start_time
        metrics.completion_rate = 100.0 if metrics.total_questions_asked >= 4 else (metrics.total_questions_asked / 4) * 100

        # Generate conversation summary for overall evaluation
        conversation_summary = f"Interview conducted {metrics.total_questions_asked} questions with {metrics.follow_up_questions} follow-ups. "
        conversation_summary += f"Candidate provided {len(candidate_responses)} responses across {len(metrics.phase_transitions)} interview phases."

        # Get overall evaluation
        overall_eval = await self.generate_overall_evaluation(metrics, conversation_summary)

        metrics.overall_interview_quality = overall_eval['overall_score']
        metrics.candidate_experience_score = overall_eval['candidate_experience']

        # Store evaluation
        self.evaluation_history[session_id] = metrics

        logger.info(f"✅ Interview evaluation completed for session {session_id}")
        logger.info(f"📊 Overall Score: {metrics.overall_interview_quality:.1f}/10")
        logger.info(f"📈 Quality Level: {overall_eval['quality_level']}")

        return metrics

    def get_evaluation_report(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed evaluation report for a session"""
        if session_id not in self.evaluation_history:
            return None

        metrics = self.evaluation_history[session_id]

        return {
            "session_info": {
                "session_id": session_id,
                "role_title": metrics.role_title,
                "years_experience": metrics.years_experience,
                "candidate_name": metrics.candidate_name,
                "evaluation_timestamp": datetime.now().isoformat()
            },
            "performance_metrics": {
                "total_questions": metrics.total_questions_asked,
                "follow_up_questions": metrics.follow_up_questions,
                "conversation_turns": metrics.conversation_turns,
                "session_duration": f"{metrics.session_duration:.1f}s",
                "completion_rate": f"{metrics.completion_rate:.1f}%"
            },
            "quality_scores": {
                "overall_interview_quality": f"{metrics.overall_interview_quality:.1f}/10",
                "candidate_experience": f"{metrics.candidate_experience_score:.1f}/10",
                "technical_depth": f"{metrics.technical_depth_score:.1f}/10",
                "conversation_flow": f"{metrics.conversation_flow_score:.1f}/10",
                "interviewer_professionalism": f"{metrics.interviewer_professionalism:.1f}/10",
                "star_method_adherence": f"{metrics.star_method_adherence:.1f}/10"
            },
            "detailed_analysis": {
                "avg_question_relevance": f"{sum(metrics.question_relevance_scores)/len(metrics.question_relevance_scores):.1f}/10" if metrics.question_relevance_scores else "N/A",
                "avg_response_naturalness": f"{sum(metrics.response_naturalness_scores)/len(metrics.response_naturalness_scores):.1f}/10" if metrics.response_naturalness_scores else "N/A",
                "phase_transitions": metrics.phase_transitions,
                "question_scores": [f"{score:.1f}" for score in metrics.question_relevance_scores],
                "naturalness_scores": [f"{score:.1f}" for score in metrics.response_naturalness_scores]
            }
        }

# Global judge instance
llm_judge = LLMJudge()

__all__ = ['LLMJudge', 'InterviewMetrics', 'llm_judge']