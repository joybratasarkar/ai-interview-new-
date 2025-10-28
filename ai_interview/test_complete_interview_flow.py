#!/usr/bin/env python3
"""
Complete Interview Flow Unit Test with LLM Judge Evaluation
Tests the entire AI interview system from start to finish with realistic scenarios
"""

import asyncio
import json
import logging
import time
import uuid
import pytest
import sys
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import traceback
import websockets
from unittest.mock import AsyncMock, MagicMock

# Add the app directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from services.interview_flow_service import interview_flow_service, session_manager, startup_interview_service
    from llm_judge_evaluator import llm_judge
    from sde2_interview_simulator import sde2_simulator, CandidateProfile
    logger.info("Successfully imported all components")
except ImportError as e:
    logger.error(f"Import failed: {e}")
    logger.error("Make sure you're running this from the ai_interview directory")
    sys.exit(1)

class WebSocketSimulator:
    """Simulates WebSocket connection for testing"""

    def __init__(self, test_mode=True):
        self.test_mode = test_mode
        self.connection_active = False
        self.messages_sent = []
        self.messages_received = []

    async def connect(self, uri):
        """Simulate WebSocket connection"""
        self.connection_active = True
        logger.info(f"Simulated WebSocket connection to {uri}")
        return self

    async def send(self, message):
        """Simulate sending message"""
        if not self.connection_active:
            raise ConnectionError("WebSocket connection not active")

        if self.test_mode:
            self.messages_sent.append(message)
            logger.debug(f"Simulated WebSocket send: {str(message)[:100]}...")
            # In test mode, process through interview service directly
            try:
                if isinstance(message, str):
                    payload = json.loads(message)
                else:
                    payload = message

                # Validate payload structure
                if not isinstance(payload, dict):
                    return {"error": "Invalid payload format"}

                # Process through interview flow service
                result = await interview_flow_service.process_interview_flow(payload)

                # Ensure result is a dictionary
                if not isinstance(result, dict):
                    result = {"error": "Invalid response format", "raw_response": str(result)}

                # Simulate receiving the response
                response = json.dumps(result)
                self.messages_received.append(response)
                return result
            except json.JSONDecodeError as e:
                error_msg = f"JSON decode error: {e}"
                logger.error(error_msg)
                return {"error": error_msg}
            except Exception as e:
                error_msg = f"WebSocket simulation error: {e}"
                logger.error(error_msg)
                return {"error": error_msg}
        else:
            # Real WebSocket connection would go here
            raise NotImplementedError("Real WebSocket not implemented in test mode")

    async def recv(self):
        """Simulate receiving message"""
        if self.messages_received:
            return self.messages_received.pop(0)
        return None

    async def close(self):
        """Simulate closing connection"""
        self.connection_active = False
        logger.info("Simulated WebSocket connection closed")

class InterviewFlowTester:
    """Comprehensive tester for the complete interview flow"""

    def __init__(self):
        self.test_results = {}
        self.session_ids = []
        self.websocket_sim = WebSocketSimulator(test_mode=True)

    async def setup_test_environment(self):
        """Set up the test environment"""
        logger.info("Setting up test environment...")

        try:
            # Initialize the interview service
            startup_result = await startup_interview_service()
            if startup_result:
                logger.info("Interview service initialized")

                # Initialize WebSocket simulator
                await self.websocket_sim.connect("ws://localhost:8003/ws/interview")
                logger.info("WebSocket simulator connected")
                return True
            else:
                logger.error("Failed to initialize interview service")
                return False
        except Exception as e:
            logger.error(f"Failed to set up test environment: {e}")
            return False

    async def cleanup_test_environment(self):
        """Clean up test environment"""
        logger.info("Cleaning up test environment...")

        try:
            # Close WebSocket simulation
            await self.websocket_sim.close()

            # Clean up all test sessions
            for session_id in self.session_ids:
                await session_manager.clear_session(session_id)

            logger.info(f"Cleaned up {len(self.session_ids)} test sessions")
            self.session_ids.clear()
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

    async def test_single_interview_flow(self, candidate_type: str = "experienced_backend") -> Dict[str, Any]:
        """Test a complete interview flow with a simulated candidate"""

        session_id = f"test_interview_{uuid.uuid4().hex[:8]}"
        self.session_ids.append(session_id)

        logger.info(f"Starting interview flow test for {candidate_type}")
        logger.info(f"Session ID: {session_id}")

        test_result = {
            "session_id": session_id,
            "candidate_type": candidate_type,
            "status": "running",
            "start_time": time.time(),
            "conversation_log": [],
            "flow_metrics": {},
            "evaluation_metrics": {},
            "errors": []
        }

        try:
            # Get candidate profile
            candidate = sde2_simulator.candidate_profiles[candidate_type]

            # Define the role
            role_title = "Senior Software Developer"
            years_experience = candidate.years_experience
            candidate_name = candidate.name
            company_name = "TechInnovate"

            # Test Phase 1: Greeting
            logger.info("Phase 1: Testing greeting flow")

            greeting_payload = {
                "text": "start",
                "room_id": session_id,
                "phase": "greeting",
                "question_idx": 0,
                "role_title": role_title,
                "years_experience": years_experience,
                "candidate_name": candidate_name,
                "company_name": company_name,
                "done": False,
                "follow_up_count": 0,
                "conversation_history": ""
            }

            # Send through WebSocket simulation
            greeting_result = await self.websocket_sim.send(greeting_payload)

            if "error" in greeting_result:
                raise Exception(f"Greeting phase failed: {greeting_result['error']}")

            test_result["conversation_log"].append({
                "type": "interviewer_question",
                "content": greeting_result["bot_response"],
                "phase": "greeting",
                "timestamp": time.time()
            })

            logger.info(f"Greeting: {greeting_result['bot_response'][:100]}...")

            # Test Phase 2: Intro transition
            logger.info("Phase 2: Testing intro transition")

            # Simulate candidate response to greeting
            candidate_greeting = "I'm doing great, thank you! I'm excited to be here and discuss this opportunity."

            intro_payload = {
                "text": candidate_greeting,
                "room_id": session_id,
                "phase": greeting_result.get("phase", "greeting"),
                "question_idx": greeting_result.get("question_idx", 0),
                "role_title": role_title,
                "years_experience": years_experience,
                "candidate_name": candidate_name,
                "company_name": company_name,
                "done": greeting_result.get("done", False),
                "follow_up_count": greeting_result.get("follow_up_count", 0),
                "conversation_history": greeting_result.get("conversation_history", "")
            }

            # Send through WebSocket simulation
            intro_result = await self.websocket_sim.send(intro_payload)

            test_result["conversation_log"].extend([
                {
                    "type": "candidate_response",
                    "content": candidate_greeting,
                    "timestamp": time.time()
                },
                {
                    "type": "interviewer_question",
                    "content": intro_result["bot_response"],
                    "phase": "intro",
                    "timestamp": time.time()
                }
            ])

            logger.info(f"Intro: {intro_result['bot_response'][:100]}...")

            # Test Phase 3: Main interview questions
            logger.info("Phase 3: Testing main interview questions")

            current_state = intro_result
            question_count = 0
            max_questions = 5

            while not current_state.get("done", False) and question_count < max_questions:
                # Simulate candidate ready response if in intro
                if current_state.get("phase") == "intro":
                    ready_response = "Absolutely, I'm ready to get started!"

                    ready_payload = {
                        "text": ready_response,
                        "room_id": session_id,
                        **{k: v for k, v in current_state.items() if k not in ["bot_response"]}
                    }

                    # Send through WebSocket simulation
                    current_state = await self.websocket_sim.send(ready_payload)

                    test_result["conversation_log"].extend([
                        {
                            "type": "candidate_response",
                            "content": ready_response,
                            "timestamp": time.time()
                        },
                        {
                            "type": "interviewer_question",
                            "content": current_state["bot_response"],
                            "phase": "questions",
                            "timestamp": time.time(),
                            "question_number": question_count + 1
                        }
                    ])

                    question_count += 1
                    logger.info(f"Question {question_count}: {current_state['bot_response'][:100]}...")
                    continue

                # Generate realistic candidate response
                current_question = current_state.get("bot_response", "")
                if current_question and not current_state.get("done", False):

                    # Generate candidate response using simulator
                    candidate_response = sde2_simulator.generate_realistic_response(
                        current_question, candidate
                    )

                    test_result["conversation_log"].append({
                        "type": "candidate_response",
                        "content": candidate_response,
                        "timestamp": time.time(),
                        "question_number": question_count
                    })

                    # Process candidate response
                    response_payload = {
                        "text": candidate_response,
                        "room_id": session_id,
                        **{k: v for k, v in current_state.items() if k not in ["bot_response"]}
                    }

                    # Send through WebSocket simulation
                    current_state = await self.websocket_sim.send(response_payload)

                    if current_state.get("done", False):
                        # Interview completed
                        test_result["conversation_log"].append({
                            "type": "interviewer_response",
                            "content": current_state["bot_response"],
                            "phase": "complete",
                            "timestamp": time.time(),
                            "context": "conclusion"
                        })
                        logger.info(f"Interview completed: {current_state['bot_response'][:100]}...")
                        break
                    else:
                        # Continue with next question or follow-up
                        test_result["conversation_log"].append({
                            "type": "interviewer_question",
                            "content": current_state["bot_response"],
                            "phase": "questions",
                            "timestamp": time.time(),
                            "question_number": question_count + 1 if current_state.get("follow_up_count", 0) == 0 else question_count,
                            "context": "follow_up" if current_state.get("follow_up_count", 0) > 0 else "main_question"
                        })

                        if current_state.get("follow_up_count", 0) == 0:
                            question_count += 1

                        logger.info(f"{'Follow-up' if current_state.get('follow_up_count', 0) > 0 else f'Question {question_count}'}: {current_state['bot_response'][:100]}...")

                    # Add small delay to simulate realistic conversation
                    await asyncio.sleep(0.1)

            # Calculate flow metrics
            test_result["flow_metrics"] = {
                "total_questions": question_count,
                "total_turns": len(test_result["conversation_log"]),
                "completion_status": "completed" if current_state.get("done", False) else "partial",
                "final_phase": current_state.get("phase", "unknown"),
                "session_duration": time.time() - test_result["start_time"]
            }

            # Test Phase 4: LLM Judge Evaluation
            logger.info("Phase 4: Running LLM judge evaluation")

            evaluation_metrics = await llm_judge.evaluate_complete_interview(
                session_id=session_id,
                conversation_log=test_result["conversation_log"],
                role_title=role_title,
                years_experience=years_experience,
                candidate_name=candidate_name
            )

            test_result["evaluation_metrics"] = evaluation_metrics
            test_result["status"] = "completed"
            test_result["end_time"] = time.time()

            logger.info(f"Interview flow test completed successfully")
            logger.info(f"Overall Quality Score: {evaluation_metrics.overall_interview_quality:.1f}/10")

            return test_result

        except Exception as e:
            logger.error(f"Interview flow test failed: {e}")
            logger.error(f"Error details: {traceback.format_exc()}")

            test_result["status"] = "failed"
            test_result["end_time"] = time.time()
            test_result["errors"].append({
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": time.time()
            })

            return test_result

    async def test_multiple_candidate_types(self) -> Dict[str, Any]:
        """Test interview flow with multiple candidate types"""

        logger.info("Testing multiple candidate types...")

        candidate_types = [
            "experienced_backend",
            "fullstack_balanced",
            "tech_heavy",
            "concise_practical"
        ]

        results = {}

        for candidate_type in candidate_types:
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing candidate type: {candidate_type}")
            logger.info(f"{'='*60}")

            result = await self.test_single_interview_flow(candidate_type)
            results[candidate_type] = result

            # Add delay between tests
            await asyncio.sleep(1)

        return results

    async def test_edge_cases(self) -> Dict[str, Any]:
        """Test edge cases and error scenarios"""

        logger.info("Testing edge cases...")

        edge_case_results = {}

        try:
            # Test 1: Invalid payload
            logger.info("Testing invalid payload handling...")

            invalid_payload = {
                "invalid_field": "test",
                "room_id": f"edge_test_{uuid.uuid4().hex[:8]}"
            }

            # Send through WebSocket simulation
            result = await self.websocket_sim.send(invalid_payload)
            edge_case_results["invalid_payload"] = {
                "status": "handled" if "error" not in result else "error",
                "response": result
            }

            # Test 2: Empty text input
            logger.info("Testing empty text input...")

            empty_payload = {
                "text": "",
                "room_id": f"edge_test_{uuid.uuid4().hex[:8]}",
                "phase": "greeting",
                "role_title": "Software Developer",
                "years_experience": "2-5 years"
            }

            # Send through WebSocket simulation
            result = await self.websocket_sim.send(empty_payload)
            edge_case_results["empty_text"] = {
                "status": "handled",
                "response": result
            }

            # Test 3: Very long input
            logger.info("Testing very long input...")

            long_text = "This is a very long response that goes on and on. " * 50
            long_payload = {
                "text": long_text,
                "room_id": f"edge_test_{uuid.uuid4().hex[:8]}",
                "phase": "questions",
                "role_title": "Software Developer",
                "years_experience": "2-5 years"
            }

            # Send through WebSocket simulation
            result = await self.websocket_sim.send(long_payload)
            edge_case_results["long_input"] = {
                "status": "handled",
                "response_length": len(result.get("bot_response", "")),
                "processed": True
            }

            logger.info("Edge case testing completed")

        except Exception as e:
            logger.error(f"Edge case testing failed: {e}")
            edge_case_results["error"] = str(e)

        return edge_case_results

    async def generate_comprehensive_report(self, test_results: Dict[str, Any]) -> str:
        """Generate comprehensive test report"""

        report = []
        report.append("AI INTERVIEW SYSTEM - COMPREHENSIVE TEST REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Summary statistics
        total_tests = 0
        successful_tests = 0
        failed_tests = 0

        for test_name, result in test_results.items():
            if isinstance(result, dict):
                total_tests += 1
                if result.get("status") == "completed":
                    successful_tests += 1
                elif result.get("status") == "failed":
                    failed_tests += 1

        report.append("TEST SUMMARY")
        report.append("-" * 30)
        report.append(f"Total Tests: {total_tests}")
        report.append(f"Successful: {successful_tests}")
        report.append(f"Failed: {failed_tests}")
        report.append(f"Success Rate: {(successful_tests/total_tests*100):.1f}%" if total_tests > 0 else "N/A")
        report.append("")

        # Detailed results for each candidate type
        for test_name, result in test_results.items():
            if not isinstance(result, dict) or "candidate_type" not in result:
                continue

            report.append(f"CANDIDATE TYPE: {result['candidate_type'].upper()}")
            report.append("-" * 40)

            # Basic metrics
            flow_metrics = result.get("flow_metrics", {})
            report.append(f"Status: {result.get('status', 'unknown')}")
            report.append(f"Questions Asked: {flow_metrics.get('total_questions', 'N/A')}")
            report.append(f"Conversation Turns: {flow_metrics.get('total_turns', 'N/A')}")
            report.append(f"Completion: {flow_metrics.get('completion_status', 'N/A')}")
            report.append(f"Duration: {flow_metrics.get('session_duration', 0):.1f}s")

            # Evaluation metrics
            eval_metrics = result.get("evaluation_metrics")
            if eval_metrics:
                report.append("")
                report.append("QUALITY SCORES:")
                report.append(f"  Overall Interview Quality: {eval_metrics.overall_interview_quality:.1f}/10")
                report.append(f"  Candidate Experience: {eval_metrics.candidate_experience_score:.1f}/10")
                report.append(f"  Technical Depth: {eval_metrics.technical_depth_score:.1f}/10")
                report.append(f"  Conversation Flow: {eval_metrics.conversation_flow_score:.1f}/10")
                report.append(f"  Interviewer Professionalism: {eval_metrics.interviewer_professionalism:.1f}/10")
                report.append(f"  STAR Method Adherence: {eval_metrics.star_method_adherence:.1f}/10")

            # Errors if any
            errors = result.get("errors", [])
            if errors:
                report.append("")
                report.append("ERRORS:")
                for error in errors:
                    report.append(f"  - {error.get('error', 'Unknown error')}")

            report.append("")

        # Overall assessment
        if successful_tests > 0:
            avg_quality = 0
            quality_count = 0

            for result in test_results.values():
                if isinstance(result, dict) and "evaluation_metrics" in result:
                    eval_metrics = result["evaluation_metrics"]
                    avg_quality += eval_metrics.overall_interview_quality
                    quality_count += 1

            if quality_count > 0:
                avg_quality /= quality_count

                report.append("OVERALL ASSESSMENT")
                report.append("-" * 30)
                report.append(f"Average Quality Score: {avg_quality:.1f}/10")

                if avg_quality >= 8.0:
                    assessment = "EXCELLENT - System performing at high quality"
                elif avg_quality >= 7.0:
                    assessment = "GOOD - System meets quality standards"
                elif avg_quality >= 6.0:
                    assessment = "ACCEPTABLE - System functional with room for improvement"
                else:
                    assessment = "NEEDS IMPROVEMENT - System requires optimization"

                report.append(f"Quality Level: {assessment}")
                report.append("")

        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 30)

        if successful_tests == total_tests:
            report.append("All tests passed successfully")
            report.append("System is ready for production use")
            report.append("Continue monitoring with real user interactions")
        else:
            report.append("Some tests failed - review error logs")
            report.append("Address failures before production deployment")

        if quality_count > 0 and avg_quality < 7.0:
            report.append("Consider improving conversation naturalness")
            report.append("Enhance technical question relevance")
            report.append("Optimize response generation quality")

        return "\n".join(report)

async def run_comprehensive_test_suite():
    """Run the complete test suite"""

    logger.info("Starting comprehensive AI interview system test suite")

    tester = InterviewFlowTester()

    try:
        # Setup
        if not await tester.setup_test_environment():
            logger.error("Failed to set up test environment")
            return False

        # Run tests
        all_results = {}

        # Test 1: Multiple candidate types
        logger.info("\n" + "="*60)
        logger.info("PHASE 1: Testing Multiple Candidate Types")
        logger.info("="*60)

        candidate_results = await tester.test_multiple_candidate_types()
        all_results.update(candidate_results)

        # Test 2: Edge cases
        logger.info("\n" + "="*60)
        logger.info("PHASE 2: Testing Edge Cases")
        logger.info("="*60)

        edge_results = await tester.test_edge_cases()
        all_results["edge_cases"] = edge_results

        # Generate comprehensive report
        logger.info("\n" + "="*60)
        logger.info("PHASE 3: Generating Report")
        logger.info("="*60)

        report = await tester.generate_comprehensive_report(all_results)

        # Save report
        report_file = f"interview_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(report)

        # Print report
        print("\n" + report)
        print(f"\nReport saved to: {report_file}")

        # Cleanup
        await tester.cleanup_test_environment()

        logger.info("Comprehensive test suite completed successfully")
        return True

    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        logger.error(f"Error details: {traceback.format_exc()}")

        try:
            await tester.cleanup_test_environment()
        except:
            pass

        return False

if __name__ == "__main__":
    # Run the comprehensive test
    success = asyncio.run(run_comprehensive_test_suite())
    sys.exit(0 if success else 1)