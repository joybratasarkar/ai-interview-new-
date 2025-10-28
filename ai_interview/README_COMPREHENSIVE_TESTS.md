# AI Interview Comprehensive Testing Suite

This comprehensive testing suite validates the AI interview service across 5 professional roles with complete intent coverage and realistic timing patterns.

## 🎯 Test Overview

### Roles Tested
1. **Software Developer** (2-5 years) - Alex Johnson at TechCorp Solutions
2. **Data Scientist** (3-7 years) - Dr. Sarah Chen at DataVision Analytics  
3. **Product Manager** (5-10 years) - Michael Rodriguez at InnovatePM Inc
4. **DevOps Engineer** (4-8 years) - Emily Zhang at CloudOps Dynamics
5. **UX Designer** (2-6 years) - Jordan Taylor at DesignFirst Studios

### Intent Coverage
- ✅ **Answer** - Technical, behavioral, and situational responses
- ✅ **RepeatQuestion** - Clarification requests for unclear questions
- ✅ **ClarifyQuestion** - Asking for more context or details
- ✅ **Hesitation** - Natural thinking pauses and processing time
- ✅ **SmallTalk** - Casual conversation and rapport building
- ✅ **OffTopic** - Handling tangential or unrelated responses
- ✅ **EndInterview** - Natural interview conclusion

### Key Features
- **Static First Questions**: First 2 responses use predefined content (no LLM calls)
- **Realistic Timing**: Human-like response delays (2-7 seconds based on complexity)
- **Role-Specific Context**: Tailored technical knowledge for each profession
- **Comprehensive Reporting**: Detailed analysis with conversation logs
- **Performance Metrics**: Response times, success rates, error tracking

## 🚀 Quick Start

### Prerequisites
```bash
# Install required packages
pip install websockets asyncio

# Ensure AI interview service is running
./apps/run.sh ai_interview
# OR
docker-compose up ai_interview
```

### Running Tests
```bash
# Basic usage (localhost:8003)
python run_comprehensive_tests.py

# Custom host/port
python run_comprehensive_tests.py --host localhost --port 8003

# Skip connection checks
python run_comprehensive_tests.py --no-connection-check

# Custom output directory
python run_comprehensive_tests.py --output-dir my_test_results
```

## 📊 Test Flow

### Phase 1: Static Questions (First 2 Questions)
- **Greeting**: Standard professional greeting
- **Introduction**: Role-specific background with dynamic context

### Phase 2: Dynamic Interview
- Multiple **Answer** intents with technical/behavioral questions
- **RepeatQuestion** and **ClarifyQuestion** for communication testing
- **Hesitation** with realistic thinking delays
- **SmallTalk** and **OffTopic** for conversation flow
- Natural pauses between interactions (2-5 seconds)

### Phase 3: Interview Conclusion  
- **EndInterview** intent with natural closure
- State cleanup and final metrics collection

## 📈 Expected Results

### Performance Benchmarks
- **Response Time**: < 5 seconds average
- **Success Rate**: > 90% for stable connections
- **Intent Recognition**: 100% coverage of all 7 intent types
- **Conversation Flow**: Natural progression with appropriate timing

### Report Generation
- **Markdown Report**: Comprehensive analysis with metrics
- **CSV Summary**: Tabular data for analysis
- **JSON Logs**: Detailed conversation transcripts
- **Performance Charts**: Response time and success rate analysis

## 🔧 Configuration

### WebSocket Settings
```python
# Default configuration
WEBSOCKET_URL = "ws://localhost:8003/ai/interview/ws"
RESPONSE_TIMEOUT = 30.0  # seconds
CONNECTION_TIMEOUT = 10.0  # seconds
```

### Timing Configuration
```python
# Realistic response delays by intent
TIMING_SETTINGS = {
    "Answer": (2.0, 5.0),        # Technical responses
    "Hesitation": (3.0, 7.0),    # Thinking time
    "RepeatQuestion": (0.5, 1.0), # Quick clarification
    "SmallTalk": (0.5, 1.5),     # Casual conversation
    # ... more configurations
}
```

### Role Customization
```python
# Add new roles in test_comprehensive_interview_scenarios.py
new_role = TestScenario(
    role_title="Your Role",
    years_experience="X-Y years", 
    candidate_name="Candidate Name",
    company_name="Company Name",
    expected_intents=["Answer", "Hesitation", "EndInterview"],
    test_duration_minutes=15
)
```

## 🐛 Troubleshooting

### Common Issues

**Connection Refused**
```bash
# Check if service is running
curl -I http://localhost:8003/health
# Start the service
./apps/run.sh ai_interview
```

**WebSocket Timeout**
```bash
# Test WebSocket manually
wscat -c ws://localhost:8003/ai/interview/ws/test_client
```

**Missing Dependencies**
```bash
pip install websockets asyncio json pathlib
```

### Debug Mode
```bash
# Add debug logging
export PYTHONPATH=.
python -c "import logging; logging.basicConfig(level=logging.DEBUG)"
python run_comprehensive_tests.py
```

## 📋 Sample Output

```
🚀 AI Interview Comprehensive Test Runner
==================================================
Target service: localhost:8003
WebSocket URL: ws://localhost:8003/ai/interview/ws
Output directory: test_reports

🔍 Checking if AI interview service is running...
✅ Service is running
🔍 Testing WebSocket connection...
✅ WebSocket connection successful

📋 Test plan:
   1. Software Developer (2-5 years)
      👤 Alex Johnson at TechCorp Solutions
      🎯 7 intents, ~15min
   [... more roles ...]

⏱️  Estimated total time: 86 minutes

▶️  Start comprehensive testing? [y/N]: y

🚀 Starting comprehensive tests...
==================================================
[... test execution ...]

🎉 Testing Complete!
==================================================
⏱️  Total time: 28.3 minutes
📈 Scenarios: 5 completed
💬 Total exchanges: 47
✅ Success rate: 94.2%
⚡ Avg response: 2.34s
🎯 Intents tested: 7 (Answer, ClarifyQuestion, EndInterview, Hesitation, OffTopic, RepeatQuestion, SmallTalk)

📋 Reports generated:
   📄 Main report: test_reports/ai_interview_comprehensive_report_20250830_143022.md
   📊 CSV summary: test_reports/ai_interview_test_summary_20250830_143022.csv
   📝 Conversation logs: 5 files
```

## 🔍 Report Analysis

### Markdown Report Contents
- Executive summary with key metrics
- Detailed scenario results for each role
- Intent coverage analysis  
- Performance benchmarks and recommendations
- Technical observations and system reliability

### CSV Data Fields
- `scenario_id`, `role_title`, `duration_minutes`
- `total_exchanges`, `intents_tested`, `success_rate`
- `avg_response_time`, `min_response_time`, `max_response_time`
- `error_count`, `errors` (detailed error descriptions)

### JSON Conversation Logs
- Complete conversation transcripts
- Timing data for each exchange
- Intent classification results
- AI response analysis and metrics

## 🎯 Success Criteria

### ✅ Passing Tests
- All 5 roles complete successfully
- All 7 intents tested and recognized
- Average response time < 5 seconds  
- Success rate > 90%
- No critical errors or timeouts

### ❌ Failing Tests
- Connection failures or timeouts
- Intent misclassification (< 80% accuracy)
- Excessive response times (> 10 seconds)
- Critical errors in conversation flow
- Missing or incomplete responses

---

**Generated by**: AI Interview Comprehensive Test Suite  
**Last Updated**: 2025-08-30  
**Version**: 1.0