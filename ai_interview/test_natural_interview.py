# Test script to demonstrate the natural interviewer system

import json
import asyncio
from ai_interview.tasks.natural_interviewer import ConversationManager

async def demo_natural_interview():
    """
    Demonstrate how the natural interviewer works
    """
    print("🎯 NATURAL AI INTERVIEWER DEMO")
    print("="*50)
    
    conversation_manager = ConversationManager()
    room_id = "demo_room"
    
    # Simulate a natural interview conversation
    candidate_responses = [
        "Hi Alex! I'm doing well, thank you. I'm excited about this opportunity. I'm a software engineer with about 5 years of experience, primarily working with Python and machine learning. I applied because I'm really interested in your company's work on AI-powered healthcare solutions.",
        
        "Sure! I recently led a project where we built a recommendation system for a e-commerce platform. We had about 2 million users and needed to provide real-time product recommendations. The main challenge was handling the scale and ensuring sub-100ms response times.",
        
        "We used a hybrid approach - collaborative filtering with matrix factorization for the core recommendations, and then a real-time layer using Redis for caching and feature serving. The tricky part was dealing with the cold start problem for new users and products.",
        
        "The cold start was interesting - for new users, we built a quick onboarding flow that captured their preferences through a few questions and then used content-based filtering initially. For new products, we used product metadata and category-based recommendations until we had enough interaction data.",
        
        "The results were really good! We improved click-through rates by 23% and conversion rates by 15%. But more importantly, we reduced the average response time from 300ms to 80ms, which significantly improved the user experience."
    ]
    
    interviewer_responses = []
    
    for i, response in enumerate(candidate_responses):
        print(f"\n--- EXCHANGE {i+1} ---")
        print(f"🗣️  CANDIDATE: {response}")
        print("\n🤔 Processing...")
        
        # Process the response
        result = conversation_manager.process_candidate_response(room_id, response)
        
        interviewer_response = result["interviewer_response"]
        interviewer_responses.append(interviewer_response)
        
        print(f"🤖 INTERVIEWER: {interviewer_response}")
        print(f"📊 STAGE: {result['stage']}")
        
        if result.get('analysis'):
            analysis = result['analysis']
            print(f"📈 ANALYSIS:")
            print(f"   - Content Quality: {analysis.get('content_quality', 'N/A')}")
            print(f"   - Key Points: {', '.join(analysis.get('key_points_mentioned', [])[:3])}")
            print(f"   - Action: {analysis.get('suggested_action', 'N/A')}")
        
        # Add some delay to make it feel more realistic
        await asyncio.sleep(1)
    
    print("\n" + "="*50)
    print("🎯 DEMO COMPLETE")
    print(f"📈 Total exchanges: {len(candidate_responses)}")
    print(f"🎪 Final stage: {result.get('stage', 'unknown')}")
    
    # Show how it builds context
    print("\n🧠 CONTEXT BUILDING EXAMPLE:")
    print("Notice how each response:")
    print("✅ References specific things the candidate mentioned")
    print("✅ Asks relevant follow-ups based on their actual answers")
    print("✅ Shows genuine interest in their work")
    print("✅ Builds a complete picture of their experience")
    
    return interviewer_responses

def compare_with_old_system():
    """
    Show how old system would respond vs new system
    """
    print("\n🆚 COMPARISON: OLD vs NEW SYSTEM")
    print("="*50)
    
    candidate_input = "I built a microservices architecture using Docker and Kubernetes for a fintech company. We had issues with service discovery and load balancing that I solved using Istio service mesh."
    
    print(f"📝 CANDIDATE SAYS: {candidate_input}")
    print()
    
    print("❌ OLD SYSTEM RESPONSE:")
    print("   'Good. What is your experience with cloud platforms?'")
    print("   (Generic, ignores what they just said)")
    print()
    
    print("✅ NEW SYSTEM RESPONSE:")
    print("   'That's a sophisticated architecture! Microservices with Kubernetes can definitely")
    print("   present service discovery challenges. I'm curious about your experience with Istio -")
    print("   what specific service discovery and load balancing issues were you facing before")
    print("   implementing the service mesh, and how did Istio solve them?'")
    print("   (Specific, shows listening, asks intelligent follow-up)")

if __name__ == "__main__":
    print("🚀 Testing Natural Interview System")
    
    # Run the demo
    asyncio.run(demo_natural_interview())
    
    # Show comparison
    compare_with_old_system()
    
    print("\n🎯 This is how REAL interviewers conduct interviews!")
    print("💬 Natural, curious, context-aware, and genuinely interested in the candidate's experience.")