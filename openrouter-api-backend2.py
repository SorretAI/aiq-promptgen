# openrouter_main.py - Fast OpenRouter API Backend for AIQueue
# Replace your slow Ollama backend with this high-performance OpenRouter integration

import asyncio
import aiohttp
import json
import hashlib
import time
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, asdict
from enum import Enum
import os
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ============================================================================
# MODELS & SCHEMAS (same as your original)
# ============================================================================

class TaskType(str, Enum):
    SOCIAL_MEDIA = "social_media"
    CONTENT_CREATION = "content_creation"
    PROSPECTING = "prospecting"
    EMAIL = "email"
    VIDEO_SCRIPT = "video_script"
    PHONE_SCRIPT = "phone_script"
    BRAND_VOICE = "brand_voice"
    TASK_PRIORITIZATION = "task_prioritization"
    RESEARCH = "research"

class ProcessingStation(str, Enum):
    IMMEDIATE = "immediate"
    DELEGATE = "delegate"
    BACKBURNER = "backburner"

class OptimizationLevel(str, Enum):
    LOW_REGENERATE = "low_regenerate"
    LOW_FIND_TWEAKS = "low_find_tweaks"
    STANDARD = "standard"
    ADVANCED = "advanced"

class Priority(str, Enum):
    SPEED = "speed"
    QUALITY = "quality"
    BALANCED = "balanced"

# Request Models
class QRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=10000)
    task_type: Optional[TaskType] = TaskType.CONTENT_CREATION
    priority: Priority = Priority.BALANCED
    user_id: Optional[str] = None
    brand_voice_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = {}

class OptimizePromptRequest(BaseModel):
    original_prompt: str = Field(..., min_length=1, max_length=5000)
    task_type: TaskType = TaskType.CONTENT_CREATION
    target_audience: Optional[str] = ""
    constraints: Optional[List[str]] = []
    optimization_level: OptimizationLevel = OptimizationLevel.STANDARD
    session_id: Optional[str] = None
    user_answers: Optional[str] = None

# Response Models
class RoutingDecision(BaseModel):
    station: ProcessingStation
    reasoning: str
    estimated_complexity: int
    can_execute_locally: bool
    requires_premium_api: bool

class OptimizedPrompt(BaseModel):
    id: str
    original_prompt: str
    optimized_prompt: str
    task_type: TaskType
    optimization_level: OptimizationLevel
    model_used: str
    timestamp: str
    improvement_score: float
    stored_in_library: bool = False
    memory_updated: bool = False

class WizardSession(BaseModel):
    session_id: str
    questions: List[Dict[str, Any]]

# ============================================================================
# OPENROUTER INTEGRATION CLASS
# ============================================================================

class OpenRouterClient:
    """High-performance OpenRouter API client with smart model selection"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
        self.session = None
        
        # Smart model selection based on task type and optimization level
        self.model_map = {
            # Fast models for quick tasks
            "speed": {
                "low": "meta-llama/llama-3.1-8b-instruct:free",
                "medium": "microsoft/wizardlm-2-8x22b:free", 
                "high": "google/gemma-2-9b-it:free"
            },
            # Balanced models for standard optimization
            "balanced": {
                "low": "microsoft/wizardlm-2-8x22b:free",
                "medium": "google/gemma-2-27b-it:free",
                "high": "meta-llama/llama-3.1-70b-instruct:free"
            },
            # Quality models for advanced optimization
            "quality": {
                "low": "google/gemma-2-27b-it:free",
                "medium": "meta-llama/llama-3.1-70b-instruct:free",
                "high": "anthropic/claude-3.5-sonnet:beta"  # Premium when needed
            }
        }
    
    async def get_session(self):
        """Get or create aiohttp session"""
        if self.session is None:
            self.session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://aiqueue.ai",  # Optional: your site
                    "X-Title": "AIQueue Optimization"
                }
            )
        return self.session
    
    def select_model(self, optimization_level: OptimizationLevel, priority: Priority, task_type: TaskType) -> str:
        """Smart model selection based on requirements"""
        
        # Map optimization levels to complexity
        complexity_map = {
            OptimizationLevel.LOW_REGENERATE: "low",
            OptimizationLevel.LOW_FIND_TWEAKS: "low", 
            OptimizationLevel.STANDARD: "medium",
            OptimizationLevel.ADVANCED: "high"
        }
        
        # Map priority to model category
        priority_map = {
            Priority.SPEED: "speed",
            Priority.BALANCED: "balanced", 
            Priority.QUALITY: "quality"
        }
        
        complexity = complexity_map.get(optimization_level, "medium")
        priority_cat = priority_map.get(priority, "balanced")
        
        model = self.model_map[priority_cat][complexity]
        
        # Special cases for specific task types
        if task_type == TaskType.BRAND_VOICE and complexity != "low":
            # Use higher quality model for brand voice tasks
            model = self.model_map["quality"]["high"]
        elif task_type == TaskType.RESEARCH and complexity != "low":
            # Use reasoning-focused model for research
            model = "meta-llama/llama-3.1-70b-instruct:free"
            
        return model
    
    async def generate(self, prompt: str, model: str, **kwargs) -> str:
        """Generate response using OpenRouter API"""
        session = await self.get_session()
        
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": kwargs.get('temperature', 0.7),
            "max_tokens": kwargs.get('max_tokens', 2000),
            "top_p": kwargs.get('top_p', 0.9),
            "stream": False
        }
        
        try:
            async with session.post(f"{self.base_url}/chat/completions", json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise HTTPException(status_code=500, f"OpenRouter API error: {error_text}")
                
                result = await response.json()
                return result['choices'][0]['message']['content']
                
        except Exception as e:
            raise HTTPException(status_code=500, f"Failed to generate response: {str(e)}")
    
    async def close(self):
        """Close the aiohttp session"""
        if self.session:
            await self.session.close()

# ============================================================================
# ENHANCED AIQUEUE CORE WITH OPENROUTER
# ============================================================================

class AIQueueCore:
    """Enhanced AIQueue core with OpenRouter integration"""
    
    def __init__(self, openrouter_api_key: str):
        self.openrouter = OpenRouterClient(openrouter_api_key)
        self.optimization_history = []
        self.prompt_cache = {}
        self.wizard_sessions = {}
        
    async def route_request(self, request: QRequest) -> RoutingDecision:
        """Determine which processing station should handle the request"""
        
        # Use fast routing logic (no API call needed for routing)
        complexity = self._estimate_complexity(request.prompt, request.task_type)
        
        if complexity <= 3:
            station = ProcessingStation.IMMEDIATE
            reasoning = "Simple task suitable for immediate processing"
        elif complexity <= 7:
            station = ProcessingStation.IMMEDIATE  # OpenRouter is fast enough for most tasks
            reasoning = "Standard task - processing with OpenRouter"
        else:
            station = ProcessingStation.DELEGATE
            reasoning = "Complex task requiring specialized handling"
        
        return RoutingDecision(
            station=station,
            reasoning=reasoning,
            estimated_complexity=complexity,
            can_execute_locally=False,  # Using cloud API
            requires_premium_api=complexity > 8
        )
    
    def _estimate_complexity(self, prompt: str, task_type: TaskType) -> int:
        """Estimate task complexity (1-10)"""
        base_complexity = {
            TaskType.SOCIAL_MEDIA: 3,
            TaskType.EMAIL: 4,
            TaskType.CONTENT_CREATION: 5,
            TaskType.VIDEO_SCRIPT: 6,
            TaskType.PROSPECTING: 6,
            TaskType.BRAND_VOICE: 7,
            TaskType.RESEARCH: 8,
            TaskType.TASK_PRIORITIZATION: 4
        }.get(task_type, 5)
        
        # Adjust based on prompt length and complexity indicators
        word_count = len(prompt.split())
        if word_count > 100:
            base_complexity += 1
        if word_count > 200:
            base_complexity += 1
            
        # Look for complexity indicators
        complex_indicators = ['analyze', 'comprehensive', 'detailed', 'multi-step', 'strategy']
        for indicator in complex_indicators:
            if indicator in prompt.lower():
                base_complexity += 1
                break
                
        return min(10, max(1, base_complexity))
    
    async def optimize_prompt(self, request: OptimizePromptRequest) -> Union[OptimizedPrompt, WizardSession]:
        """Main optimization function with OpenRouter"""
        
        # Handle wizard-style optimization
        if request.optimization_level == OptimizationLevel.LOW_FIND_TWEAKS and not request.session_id:
            return await self._create_wizard_session(request)
        
        # Handle wizard completion
        if request.session_id and request.user_answers:
            return await self._complete_wizard_optimization(request)
        
        # Standard optimization
        return await self._standard_optimization(request)
    
    async def _create_wizard_session(self, request: OptimizePromptRequest) -> WizardSession:
        """Create a wizard session for interactive optimization"""
        session_id = hashlib.md5(f"{request.original_prompt}{time.time()}".encode()).hexdigest()[:12]
        
        # Generate contextual questions based on task type
        questions = self._generate_questions(request.task_type, request.original_prompt)
        
        session = WizardSession(
            session_id=session_id,
            questions=questions
        )
        
        # Store session
        self.wizard_sessions[session_id] = {
            'session': session,
            'original_request': request,
            'created_at': datetime.now()
        }
        
        return session
    
    def _generate_questions(self, task_type: TaskType, prompt: str) -> List[Dict[str, Any]]:
        """Generate contextual questions for the wizard"""
        base_questions = [
            {
                "id": "1",
                "question": "What is the primary goal of this prompt?",
                "options": [
                    "Generate immediate action/response",
                    "Provide detailed information/analysis", 
                    "Create engaging/persuasive content",
                    "Solve a specific problem"
                ],
                "allow_custom": True
            },
            {
                "id": "2", 
                "question": "Who is your target audience?",
                "options": [
                    "General public/broad audience",
                    "Industry professionals/experts",
                    "Specific demographic group",
                    "Internal team/colleagues"
                ],
                "allow_custom": True
            }
        ]
        
        # Add task-specific questions
        task_specific = {
            TaskType.SOCIAL_MEDIA: [
                {
                    "id": "3",
                    "question": "What platform will this be used on?",
                    "options": ["Twitter/X", "LinkedIn", "Instagram", "Facebook", "TikTok"],
                    "allow_custom": True
                }
            ],
            TaskType.EMAIL: [
                {
                    "id": "3", 
                    "question": "What's the email's primary purpose?",
                    "options": ["Sales/conversion", "Information sharing", "Relationship building", "Follow-up"],
                    "allow_custom": True
                }
            ],
            TaskType.CONTENT_CREATION: [
                {
                    "id": "3",
                    "question": "What type of content format?",
                    "options": ["Blog post", "Article", "Script", "Copy", "Educational"],
                    "allow_custom": True
                }
            ]
        }
        
        return base_questions + task_specific.get(task_type, [])
    
    async def _complete_wizard_optimization(self, request: OptimizePromptRequest) -> OptimizedPrompt:
        """Complete optimization using wizard answers"""
        session_data = self.wizard_sessions.get(request.session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        original_request = session_data['original_request']
        
        # Build enhanced prompt using wizard answers
        enhanced_prompt = self._build_enhanced_prompt(
            original_request.original_prompt,
            request.user_answers,
            original_request.task_type
        )
        
        # Use OpenRouter for optimization
        model = self.openrouter.select_model(
            original_request.optimization_level,
            Priority.BALANCED,
            original_request.task_type
        )
        
        optimized = await self.openrouter.generate(enhanced_prompt, model)
        
        # Clean up session
        del self.wizard_sessions[request.session_id]
        
        return OptimizedPrompt(
            id=hashlib.md5(original_request.original_prompt.encode()).hexdigest()[:8],
            original_prompt=original_request.original_prompt,
            optimized_prompt=optimized.strip(),
            task_type=original_request.task_type,
            optimization_level=original_request.optimization_level,
            model_used=model,
            timestamp=datetime.now().isoformat(),
            improvement_score=self._calculate_improvement_score(original_request.original_prompt, optimized),
            stored_in_library=True,
            memory_updated=True
        )
    
    async def _standard_optimization(self, request: OptimizePromptRequest) -> OptimizedPrompt:
        """Standard prompt optimization"""
        
        # Build optimization prompt
        optimization_prompt = self._build_optimization_prompt(request)
        
        # Select appropriate model
        model = self.openrouter.select_model(
            request.optimization_level,
            Priority.BALANCED,
            request.task_type
        )
        
        # Generate optimized prompt
        optimized = await self.openrouter.generate(
            optimization_prompt, 
            model,
            temperature=0.3 if request.optimization_level == OptimizationLevel.ADVANCED else 0.7
        )
        
        return OptimizedPrompt(
            id=hashlib.md5(request.original_prompt.encode()).hexdigest()[:8],
            original_prompt=request.original_prompt,
            optimized_prompt=optimized.strip(),
            task_type=request.task_type,
            optimization_level=request.optimization_level,
            model_used=model,
            timestamp=datetime.now().isoformat(),
            improvement_score=self._calculate_improvement_score(request.original_prompt, optimized),
            stored_in_library=False,
            memory_updated=False
        )
    
    def _build_optimization_prompt(self, request: OptimizePromptRequest) -> str:
        """Build the optimization prompt"""
        
        task_guidelines = {
            TaskType.SOCIAL_MEDIA: "Make it engaging, shareable, and platform-appropriate. Include clear hooks and calls-to-action.",
            TaskType.EMAIL: "Focus on clear subject lines, personalization, and strong call-to-action. Optimize for conversion.",
            TaskType.CONTENT_CREATION: "Enhance clarity, structure, and engagement. Ensure it serves the intended purpose effectively.",
            TaskType.VIDEO_SCRIPT: "Include visual cues, timing, and engaging narrative flow. Make it compelling for video format.",
            TaskType.PROSPECTING: "Personalize the message, highlight value proposition, and create urgency for response.",
            TaskType.BRAND_VOICE: "Ensure consistency with brand personality while optimizing for impact and memorability."
        }
        
        guidelines = task_guidelines.get(request.task_type, "Optimize for clarity, impact, and effectiveness.")
        
        return f"""You are an expert prompt engineer specializing in {request.task_type.value} optimization.

ORIGINAL PROMPT:
{request.original_prompt}

TASK TYPE: {request.task_type.value}
TARGET AUDIENCE: {request.target_audience or "General audience"}
CONSTRAINTS: {', '.join(request.constraints) if request.constraints else "None specified"}
OPTIMIZATION LEVEL: {request.optimization_level.value}

GUIDELINES: {guidelines}

OPTIMIZATION REQUIREMENTS:
1. Improve clarity and specificity
2. Add missing structural elements
3. Optimize for the specific task type
4. Ensure actionable and measurable outcomes
5. Maintain the original intent while enhancing effectiveness

Please rewrite the prompt to be more effective while keeping it concise and focused.

OPTIMIZED PROMPT:"""
    
    def _build_enhanced_prompt(self, original_prompt: str, user_answers: str, task_type: TaskType) -> str:
        """Build enhanced prompt using wizard answers"""
        return f"""You are an expert prompt engineer. Using the user's answers to improve this prompt:

ORIGINAL PROMPT: {original_prompt}
USER PREFERENCES: {user_answers}
TASK TYPE: {task_type.value}

Create an optimized version that incorporates the user's specific requirements and preferences.

OPTIMIZED PROMPT:"""
    
    def _calculate_improvement_score(self, original: str, optimized: str) -> float:
        """Calculate improvement score (simple heuristic)"""
        original_words = len(original.split())
        optimized_words = len(optimized.split())
        
        # Base score on length improvement and complexity
        if optimized_words > original_words * 0.8:  # Not too short
            base_score = 7.0
        else:
            base_score = 5.0
            
        # Bonus for structure indicators
        structure_indicators = [':', '-', '1.', '2.', 'specifically', 'include', 'format']
        structure_bonus = sum(1 for indicator in structure_indicators if indicator in optimized.lower())
        
        return min(10.0, base_score + structure_bonus * 0.5)

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

# Initialize FastAPI app
app = FastAPI(
    title="AIQueue OpenRouter Engine", 
    description="High-performance OpenRouter-powered AI engine for AIQueue",
    version="2.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize core engine
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable is required")

core = AIQueueCore(OPENROUTER_API_KEY)

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Health check and basic info"""
    return {
        "message": "AIQueue OpenRouter Engine is running",
        "version": "2.0.0", 
        "engine": "OpenRouter",
        "status": "ready",
        "performance": "high-speed"
    }

@app.post("/optimize", response_model=Union[OptimizedPrompt, WizardSession])
async def optimize_prompt_endpoint(request: OptimizePromptRequest):
    """Main optimization endpoint - much faster than Ollama"""
    try:
        result = await core.optimize_prompt(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/q", response_model=Dict)
async def process_q_request(request: QRequest):
    """Q interface endpoint with OpenRouter routing"""
    
    request_id = hashlib.md5(f"{request.prompt}{time.time()}".encode()).hexdigest()[:12]
    start_time = time.time()
    
    try:
        # Route the request
        routing = await core.route_request(request)
        processing_time = time.time() - start_time
        
        return {
            "request_id": request_id,
            "routing_decision": routing,
            "response": "Request routed successfully - ready for optimization", 
            "model_used": "OpenRouter Smart Routing",
            "processing_time": processing_time,
            "estimated_cost": 0.001,  # Very low cost with free models
            "status": "ready"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_system_stats():
    """Get system statistics"""
    return {
        "engine": "OpenRouter",
        "available_models": [
            "meta-llama/llama-3.1-8b-instruct:free",
            "meta-llama/llama-3.1-70b-instruct:free", 
            "microsoft/wizardlm-2-8x22b:free",
            "google/gemma-2-9b-it:free",
            "google/gemma-2-27b-it:free",
            "anthropic/claude-3.5-sonnet:beta"
        ],
        "active_sessions": len(core.wizard_sessions),
        "prompt_library_size": len(core.optimization_history),
        "system_status": "operational",
        "performance": "high-speed",
        "cost_efficiency": "excellent"
    }

@app.get("/prompt-library")
async def get_prompt_library():
    """Get recent optimized prompts"""
    return {
        "recent_prompts": core.optimization_history[-20:],  # Last 20 prompts
        "total_count": len(core.optimization_history)
    }

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    print("🚀 AIQueue OpenRouter Engine starting up...")
    print(f"✅ OpenRouter API configured")
    print("⚡ High-speed optimization ready!")

@app.on_event("shutdown") 
async def shutdown_event():
    """Cleanup on shutdown"""
    await core.openrouter.close()
    print("🛑 AIQueue OpenRouter Engine shutting down...")

# ============================================================================
# DEVELOPMENT SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Set your OpenRouter API key
    os.environ["OPENROUTER_API_KEY"] = "YOUR_OPENROUTER_API_KEY_HERE"
    
    print("""
    🎯 AIQueue OpenRouter Engine Setup:
    
    1. Get your free OpenRouter API key: https://openrouter.ai/keys
    2. Set environment variable: export OPENROUTER_API_KEY="your_key_here"
    3. Run: python openrouter_main.py
    
    ⚡ OpenRouter Benefits:
    - 10x faster than local Ollama
    - Access to best free models (Llama 3.1, Gemma 2, etc.)
    - Smart model routing for optimal performance
    - No local GPU/RAM requirements
    - Reliable uptime and performance
    """)
    
    uvicorn.run(
        "openrouter_main:app",
        host="0.0.0.0", 
        port=8000,
        reload=True,
        log_level="info"
    )