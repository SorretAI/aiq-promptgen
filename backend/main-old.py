# AIQueue FastAPI Backend - Local Ollama Core Engine
# Transforms your prompt optimizer into a powerful API for the Q interface

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union, Any
from enum import Enum
import asyncio
import ollama
import json
import hashlib
import time
from datetime import datetime
from dataclasses import dataclass, asdict
import os
import logging
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# MODELS & SCHEMAS
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
    IMMEDIATE = "immediate"      # Execute now with local Ollama
    DELEGATE = "delegate"        # Requires external APIs or human input
    BACKBURNER = "backburner"   # Needs research or missing dependencies

class OptimizationLevel(str, Enum):
    BASIC = "basic"             # TinyLlama - ultra fast
    STANDARD = "standard"       # Gemma:2b - balanced
    ADVANCED = "advanced"       # Phi3:mini - high quality

class Priority(str, Enum):
    SPEED = "speed"
    QUALITY = "quality"
    BALANCED = "balanced"

# Request Models
class QRequest(BaseModel):
    """Main Q interface request"""
    prompt: str = Field(..., min_length=1, max_length=10000)
    task_type: Optional[TaskType] = TaskType.CONTENT_CREATION
    priority: Priority = Priority.BALANCED
    user_id: Optional[str] = None
    brand_voice_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = {}

class OptimizePromptRequest(BaseModel):
    """Direct prompt optimization request"""
    original_prompt: str = Field(..., min_length=1, max_length=5000)
    task_type: TaskType = TaskType.CONTENT_CREATION
    target_audience: Optional[str] = ""
    constraints: Optional[List[str]] = []
    optimization_level: OptimizationLevel = OptimizationLevel.STANDARD

class BrandVoiceRequest(BaseModel):
    """Brand voice learning request"""
    content_samples: List[str] = Field(..., min_items=1, max_items=10)
    brand_name: Optional[str] = "Personal Brand"
    tone_keywords: Optional[List[str]] = []
    target_audience: Optional[str] = ""

# Response Models
class RoutingDecision(BaseModel):
    station: ProcessingStation
    reasoning: str
    estimated_complexity: int
    can_execute_locally: bool
    requires_premium_api: bool

class QResponse(BaseModel):
    """Main Q interface response"""
    request_id: str
    routing_decision: RoutingDecision
    response: Optional[str] = None
    optimized_prompt: Optional[str] = None
    model_used: str
    processing_time: float
    estimated_cost: float = 0.0
    next_actions: Optional[List[str]] = []
    status: str  # "completed", "processing", "queued", "error"

class OptimizedPrompt(BaseModel):
    id: str
    original_prompt: str
    optimized_prompt: str
    task_type: TaskType
    optimization_level: OptimizationLevel
    model_used: str
    timestamp: str
    improvement_score: float

class BrandVoice(BaseModel):
    id: str
    brand_name: str
    voice_profile: Dict[str, Any]
    tone_keywords: List[str]
    writing_patterns: List[str]
    sample_count: int
    created_at: str

# ============================================================================
# CORE ENGINE CLASSES
# ============================================================================

@dataclass
class LocalModelConfig:
    name: str
    speed_rating: int  # 1-10
    quality_rating: int  # 1-10
    specialties: List[str]
    context_limit: int
    recommended_for: List[TaskType]

class AIQueueCore:
    """Core engine that orchestrates all local AI operations"""
    
    def __init__(self):
        self.ollama_client = ollama.Client()
        self.local_models = self._initialize_local_models()
        self.brand_voices = {}
        self.processing_queue = {}
        self.optimization_cache = {}
        self._ensure_models_available()
    
    def _initialize_local_models(self) -> Dict[str, LocalModelConfig]:
        """Initialize local Ollama models configuration"""
        return {
            "tinyllama": LocalModelConfig(
                name="tinyllama",
                speed_rating=10,
                quality_rating=4,
                specialties=["routing", "simple_tasks", "quick_decisions"],
                context_limit=2048,
                recommended_for=[TaskType.TASK_PRIORITIZATION]
            ),
            "gemma:2b": LocalModelConfig(
                name="gemma:2b", 
                speed_rating=8,
                quality_rating=7,
                specialties=["content_optimization", "social_media", "general_tasks"],
                context_limit=8192,
                recommended_for=[TaskType.SOCIAL_MEDIA, TaskType.EMAIL, TaskType.CONTENT_CREATION]
            ),
            "phi3:mini": LocalModelConfig(
                name="phi3:mini",
                speed_rating=6,
                quality_rating=9,
                specialties=["reasoning", "analysis", "technical_content", "brand_voice"],
                context_limit=128000,
                recommended_for=[TaskType.PROSPECTING, TaskType.VIDEO_SCRIPT, TaskType.BRAND_VOICE, TaskType.RESEARCH]
            )
        }
    
    def _ensure_models_available(self):
        """Ensure required models are pulled in Ollama"""
        try:
            available_models = [model['name'] for model in self.ollama_client.list()['models']]
            required_models = list(self.local_models.keys())
            
            for model in required_models:
                if model not in available_models:
                    logger.warning(f"Model {model} not found. Please run: ollama pull {model}")
                    
        except Exception as e:
            logger.error(f"Could not check Ollama models: {e}")
    
    async def route_request(self, request: QRequest) -> RoutingDecision:
        """Determine which processing station should handle the request"""
        
        # Use TinyLlama for ultra-fast routing decisions
        routing_prompt = f"""
        Analyze this request and determine the best processing approach:
        
        REQUEST: "{request.prompt}"
        TASK TYPE: {request.task_type}
        PRIORITY: {request.priority}
        
        Classify into one of these stations:
        - IMMEDIATE: Can be handled immediately with local AI (content creation, optimization, simple analysis)
        - DELEGATE: Needs external APIs, human input, or team collaboration
        - BACKBURNER: Requires research, complex analysis, or missing information
        
        Consider:
        - Complexity of the request
        - Available local capabilities
        - Time sensitivity
        - Resource requirements
        
        Respond with just: STATION|REASONING|COMPLEXITY_SCORE(1-10)|LOCAL_CAPABLE(yes/no)|PREMIUM_NEEDED(yes/no)
        """
        
        try:
            response = self.ollama_client.generate(
                model="tinyllama",
                prompt=routing_prompt,
                options={'temperature': 0.1, 'max_tokens': 200}
            )
            
            parts = response['response'].strip().split('|')
            if len(parts) >= 5:
                station = ProcessingStation(parts[0].lower())
                reasoning = parts[1]
                complexity = int(parts[2])
                local_capable = parts[3].lower() == 'yes'
                premium_needed = parts[4].lower() == 'yes'
            else:
                # Fallback parsing
                station = ProcessingStation.IMMEDIATE
                reasoning = "Default routing due to parsing issue"
                complexity = 5
                local_capable = True
                premium_needed = False
                
        except Exception as e:
            logger.error(f"Routing error: {e}")
            # Safe fallback
            station = ProcessingStation.IMMEDIATE
            reasoning = "Fallback to immediate processing"
            complexity = 3
            local_capable = True
            premium_needed = False
        
        return RoutingDecision(
            station=station,
            reasoning=reasoning,
            estimated_complexity=complexity,
            can_execute_locally=local_capable,
            requires_premium_api=premium_needed
        )
    
    def select_local_model(self, task_type: TaskType, complexity: int, priority: Priority) -> str:
        """Select the best local model for the task"""
        
        # Priority-based selection
        if priority == Priority.SPEED:
            if complexity <= 3:
                return "tinyllama"
            else:
                return "gemma:2b"
        elif priority == Priority.QUALITY:
            if complexity >= 7:
                return "phi3:mini"
            else:
                return "gemma:2b"
        else:  # BALANCED
            if task_type in [TaskType.BRAND_VOICE, TaskType.RESEARCH, TaskType.PROSPECTING]:
                return "phi3:mini"
            elif task_type in [TaskType.SOCIAL_MEDIA, TaskType.EMAIL]:
                return "gemma:2b"
            else:
                return "gemma:2b" if complexity <= 6 else "phi3:mini"
    
    async def optimize_prompt(self, request: OptimizePromptRequest) -> OptimizedPrompt:
        """Optimize a prompt using local models"""
        
        # Select model based on optimization level
        model_map = {
            OptimizationLevel.BASIC: "tinyllama",
            OptimizationLevel.STANDARD: "gemma:2b",
            OptimizationLevel.ADVANCED: "phi3:mini"
        }
        model = model_map[request.optimization_level]
        
        # Build optimization prompt
        optimization_prompt = f"""
        You are an expert prompt engineer. Optimize this prompt for {request.task_type.value}:
        
        ORIGINAL: {request.original_prompt}
        TARGET AUDIENCE: {request.target_audience or "general"}
        CONSTRAINTS: {', '.join(request.constraints) if request.constraints else "none"}
        
        Make it:
        1. More specific and actionable
        2. Include clear output format
        3. Add relevant context
        4. Optimize for clarity
        5. Reduce unnecessary words
        
        Return only the optimized prompt, no explanation.
        """
        
        try:
            start_time = time.time()
            response = self.ollama_client.generate(
                model=model,
                prompt=optimization_prompt,
                options={'temperature': 0.3, 'max_tokens': 400}
            )
            processing_time = time.time() - start_time
            
            optimized = response['response'].strip()
            
            # Calculate improvement score (simple heuristic)
            improvement_score = min(10.0, len(optimized.split()) / max(1, len(request.original_prompt.split())) * 5)
            
            return OptimizedPrompt(
                id=hashlib.md5(request.original_prompt.encode()).hexdigest()[:8],
                original_prompt=request.original_prompt,
                optimized_prompt=optimized,
                task_type=request.task_type,
                optimization_level=request.optimization_level,
                model_used=model,
                timestamp=datetime.now().isoformat(),
                improvement_score=improvement_score
            )
            
        except Exception as e:
            logger.error(f"Prompt optimization error: {e}")
            raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")
    
    async def execute_immediate_task(self, request: QRequest, routing: RoutingDecision) -> str:
        """Execute task immediately using local models"""
        
        # Select best local model
        model = self.select_local_model(
            request.task_type, 
            routing.estimated_complexity, 
            request.priority
        )
        
        # Apply brand voice if available
        enhanced_prompt = await self._apply_brand_voice(request.prompt, request.brand_voice_id)
        
        # Execute with selected model
        try:
            response = self.ollama_client.generate(
                model=model,
                prompt=enhanced_prompt,
                options={
                    'temperature': 0.7 if request.task_type in [TaskType.CONTENT_CREATION, TaskType.SOCIAL_MEDIA] else 0.3,
                    'max_tokens': 2000,
                    'top_p': 0.9
                }
            )
            return response['response']
            
        except Exception as e:
            logger.error(f"Task execution error: {e}")
            raise HTTPException(status_code=500, detail=f"Task execution failed: {str(e)}")
    
    async def learn_brand_voice(self, request: BrandVoiceRequest) -> BrandVoice:
        """Learn brand voice from content samples"""
        
        analysis_prompt = f"""
        Analyze these content samples to extract the brand voice profile:
        
        BRAND: {request.brand_name}
        SAMPLES:
        {chr(10).join(f"- {sample}" for sample in request.content_samples)}
        
        Extract:
        1. Tone characteristics (formal/casual, enthusiastic/reserved, etc.)
        2. Common phrases and expressions
        3. Sentence structure patterns
        4. Vocabulary preferences
        5. Writing style elements
        
        Format as JSON:
        {{
            "tone": ["characteristic1", "characteristic2"],
            "common_phrases": ["phrase1", "phrase2"],
            "sentence_patterns": ["pattern1", "pattern2"],
            "vocabulary_style": "description",
            "key_elements": ["element1", "element2"]
        }}
        """
        
        try:
            response = self.ollama_client.generate(
                model="phi3:mini",  # Use best model for brand analysis
                prompt=analysis_prompt,
                options={'temperature': 0.2, 'max_tokens': 1000}
            )
            
            # Parse JSON response
            try:
                voice_profile = json.loads(response['response'])
            except:
                # Fallback parsing
                voice_profile = {
                    "tone": request.tone_keywords or ["professional", "engaging"],
                    "common_phrases": [],
                    "sentence_patterns": ["conversational", "direct"],
                    "vocabulary_style": "accessible",
                    "key_elements": ["clear communication"]
                }
            
            brand_voice = BrandVoice(
                id=hashlib.md5(request.brand_name.encode()).hexdigest()[:8],
                brand_name=request.brand_name,
                voice_profile=voice_profile,
                tone_keywords=request.tone_keywords or [],
                writing_patterns=voice_profile.get("sentence_patterns", []),
                sample_count=len(request.content_samples),
                created_at=datetime.now().isoformat()
            )
            
            # Cache the brand voice
            self.brand_voices[brand_voice.id] = brand_voice
            
            return brand_voice
            
        except Exception as e:
            logger.error(f"Brand voice learning error: {e}")
            raise HTTPException(status_code=500, detail=f"Brand voice learning failed: {str(e)}")
    
    async def _apply_brand_voice(self, prompt: str, brand_voice_id: Optional[str]) -> str:
        """Apply brand voice to prompt if available"""
        if not brand_voice_id or brand_voice_id not in self.brand_voices:
            return prompt
        
        brand_voice = self.brand_voices[brand_voice_id]
        
        enhanced_prompt = f"""
        {prompt}
        
        Apply this brand voice:
        - Tone: {', '.join(brand_voice.voice_profile.get('tone', []))}
        - Style: {brand_voice.voice_profile.get('vocabulary_style', 'professional')}
        - Key elements: {', '.join(brand_voice.voice_profile.get('key_elements', []))}
        
        Maintain the brand's authentic voice while fulfilling the request.
        """
        
        return enhanced_prompt

# ============================================================================
# LIFESPAN CONTEXT MANAGER (MODERN APPROACH)
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern lifespan event handler replacing deprecated startup/shutdown events"""
    # Startup
    logger.info("🚀 AIQueue Local Engine starting up...")
    
    # Initialize core engine
    global core
    core = AIQueueCore()
    logger.info(f"Available models: {list(core.local_models.keys())}")
    
    # Test Ollama connection
    try:
        models = core.ollama_client.list()
        logger.info(f"✅ Ollama connected with {len(models['models'])} models")
    except Exception as e:
        logger.error(f"❌ Ollama connection failed: {e}")
    
    yield  # Application is running
    
    # Shutdown
    logger.info("🛑 AIQueue Local Engine shutting down...")

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="AIQueue Local Engine",
    description="Local Ollama-powered AI engine for the AIQueue system",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global core instance (initialized in lifespan)
core = None

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Health check and basic info"""
    return {
        "message": "AIQueue Local Engine is running",
        "version": "1.0.0",
        "available_models": list(core.local_models.keys()) if core else [],
        "status": "ready"
    }

@app.post("/q", response_model=QResponse)
async def process_q_request(request: QRequest, background_tasks: BackgroundTasks):
    """Main Q interface endpoint - the heart of AIQueue"""
    
    request_id = hashlib.md5(f"{request.prompt}{time.time()}".encode()).hexdigest()[:12]
    start_time = time.time()
    
    try:
        # 1. Route the request
        routing = await core.route_request(request)
        
        # 2. Process based on routing decision
        if routing.station == ProcessingStation.IMMEDIATE:
            # Execute immediately with local models
            response_text = await core.execute_immediate_task(request, routing)
            status = "completed"
            next_actions = []
            
        elif routing.station == ProcessingStation.DELEGATE:
            # Queue for delegation (would integrate with external APIs)
            response_text = f"Task queued for delegation: {routing.reasoning}"
            status = "queued"
            next_actions = ["requires_external_api", "team_collaboration"]
            
        else:  # BACKBURNER
            # Queue for background research
            response_text = f"Task added to backburner: {routing.reasoning}"
            status = "queued" 
            next_actions = ["requires_research", "missing_dependencies"]
        
        processing_time = time.time() - start_time
        
        return QResponse(
            request_id=request_id,
            routing_decision=routing,
            response=response_text,
            model_used=core.select_local_model(request.task_type, routing.estimated_complexity, request.priority),
            processing_time=processing_time,
            estimated_cost=0.0,  # Local processing is free
            next_actions=next_actions,
            status=status
        )
        
    except Exception as e:
        logger.error(f"Q request processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/optimize", response_model=OptimizedPrompt)
async def optimize_prompt(request: OptimizePromptRequest):
    """Optimize prompts using local models"""
    return await core.optimize_prompt(request)

@app.post("/brand-voice/learn", response_model=BrandVoice)
async def learn_brand_voice(request: BrandVoiceRequest):
    """Learn brand voice from content samples"""
    return await core.learn_brand_voice(request)

@app.get("/brand-voice/{voice_id}", response_model=BrandVoice)
async def get_brand_voice(voice_id: str):
    """Get learned brand voice by ID"""
    if voice_id not in core.brand_voices:
        raise HTTPException(status_code=404, detail="Brand voice not found")
    return core.brand_voices[voice_id]

@app.get("/brand-voices", response_model=List[BrandVoice])
async def list_brand_voices():
    """List all learned brand voices"""
    return list(core.brand_voices.values())

@app.get("/models")
async def list_models():
    """List available local models and their capabilities"""
    return {
        "local_models": core.local_models,
        "ollama_status": "connected" if core.ollama_client else "disconnected"
    }

@app.get("/stats")
async def get_system_stats():
    """Get system performance and usage statistics"""
    try:
        ollama_models = core.ollama_client.list()
        return {
            "available_models": [model['name'] for model in ollama_models['models']],
            "brand_voices_count": len(core.brand_voices),
            "processing_queue_size": len(core.processing_queue),
            "cache_size": len(core.optimization_cache),
            "system_status": "operational"
        }
    except Exception as e:
        return {
            "error": str(e),
            "system_status": "error"
        }

# ============================================================================
# DEVELOPMENT SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
    #End of main-old.py v15