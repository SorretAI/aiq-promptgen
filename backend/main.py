# AIQueue FastAPI Backend - Local Ollama Core Engine
# Complete working version with all optimization features

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union, Any
from enum import Enum
import ollama
import json
import hashlib
import time
from datetime import datetime
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

class BrandVoiceRequest(BaseModel):
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
    request_id: str
    routing_decision: RoutingDecision
    response: Optional[str] = None
    optimized_prompt: Optional[str] = None
    model_used: str
    processing_time: float
    estimated_cost: float = 0.0
    next_actions: Optional[List[str]] = []
    status: str

class OptimizationQuestion(BaseModel):
    id: int
    question: str
    options: List[str]
    allow_custom: bool = True

class InteractiveOptimizationResponse(BaseModel):
    session_id: str
    questions: List[OptimizationQuestion]
    instruction: str
    example_format: str

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

class BrandVoice(BaseModel):
    id: str
    brand_name: str
    voice_profile: Dict[str, Any]
    tone_keywords: List[str]
    writing_patterns: List[str]
    sample_count: int
    created_at: str

# ============================================================================
# CORE ENGINE CLASS
# ============================================================================

class AIQueueCore:
    def __init__(self):
        self.ollama_client = ollama.Client()
        self.brand_voices = {}
        self.processing_queue = {}
        self.optimization_cache = {}
        self.interactive_sessions = {}
        self.user_memory_file = "../user_preferences.json"
        self.prompt_library_file = "../prompt_library.json"
        self.user_memory = self._load_user_memory()
        self.prompt_library = self._load_prompt_library()
        self._ensure_models_available()
    
    def _load_user_memory(self) -> Dict:
        try:
            if os.path.exists(self.user_memory_file):
                with open(self.user_memory_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading user memory: {e}")
        
        return {
            "task_preferences": {},
            "common_answers": {},
            "question_patterns": {},
            "last_updated": datetime.now().isoformat()
        }
    
    def _load_prompt_library(self) -> Dict:
        try:
            if os.path.exists(self.prompt_library_file):
                with open(self.prompt_library_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading prompt library: {e}")
        
        return {
            "prompts": [],
            "categories": {},
            "usage_stats": {},
            "last_updated": datetime.now().isoformat()
        }
    
    def _save_user_memory(self):
        try:
            self.user_memory["last_updated"] = datetime.now().isoformat()
            with open(self.user_memory_file, 'w') as f:
                json.dump(self.user_memory, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving user memory: {e}")
    
    def _save_prompt_library(self):
        try:
            self.prompt_library["last_updated"] = datetime.now().isoformat()
            with open(self.prompt_library_file, 'w') as f:
                json.dump(self.prompt_library, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving prompt library: {e}")
    
    def _store_prompt_in_library(self, optimized_prompt: OptimizedPrompt):
        prompt_entry = {
            "id": optimized_prompt.id,
            "original_prompt": optimized_prompt.original_prompt,
            "optimized_prompt": optimized_prompt.optimized_prompt,
            "task_type": optimized_prompt.task_type,
            "optimization_level": optimized_prompt.optimization_level,
            "timestamp": optimized_prompt.timestamp,
            "usage_count": 0,
            "rating": None
        }
        
        self.prompt_library["prompts"].append(prompt_entry)
        
        if optimized_prompt.task_type not in self.prompt_library["categories"]:
            self.prompt_library["categories"][optimized_prompt.task_type] = []
        self.prompt_library["categories"][optimized_prompt.task_type].append(optimized_prompt.id)
        
        self._save_prompt_library()
        return True
    
    def _ensure_models_available(self):
        try:
            available_models = [model['name'] for model in self.ollama_client.list()['models']]
            required_models = ["tinyllama", "gemma:2b", "phi3:mini"]
            
            for model in required_models:
                if model not in available_models:
                    logger.warning(f"Model {model} not found. Please run: ollama pull {model}")
                    
        except Exception as e:
            logger.error(f"Could not check Ollama models: {e}")
    
    async def route_request(self, request: QRequest) -> RoutingDecision:
        routing_prompt = f"""
        Analyze this request and determine the best processing approach:
        
        REQUEST: "{request.prompt}"
        TASK TYPE: {request.task_type}
        PRIORITY: {request.priority}
        
        Classify into exactly one of these stations:
        - IMMEDIATE: Can be handled immediately with local AI
        - DELEGATE: Needs external APIs or human input
        - BACKBURNER: Requires research or missing information
        
        Respond with EXACTLY this format:
        STATION|REASONING|COMPLEXITY_SCORE|LOCAL_CAPABLE|PREMIUM_NEEDED
        
        Where STATION is IMMEDIATE, DELEGATE, or BACKBURNER
        """
        
        try:
            response = self.ollama_client.generate(
                model="tinyllama",
                prompt=routing_prompt,
                options={'temperature': 0.1, 'max_tokens': 200}
            )
            
            parts = response['response'].strip().split('|')
            if len(parts) >= 5:
                station_text = parts[0].strip().upper()
                if station_text == "IMMEDIATE":
                    station = ProcessingStation.IMMEDIATE
                elif station_text == "DELEGATE":
                    station = ProcessingStation.DELEGATE
                elif station_text == "BACKBURNER":
                    station = ProcessingStation.BACKBURNER
                else:
                    station = ProcessingStation.IMMEDIATE
                
                reasoning = parts[1].strip()
                complexity = int(parts[2].strip())
                local_capable = parts[3].strip().lower() == 'yes'
                premium_needed = parts[4].strip().lower() == 'yes'
            else:
                station = ProcessingStation.IMMEDIATE
                reasoning = "Default routing"
                complexity = 5
                local_capable = True
                premium_needed = False
                
        except Exception as e:
            logger.error(f"Routing error: {e}")
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
        if priority == Priority.SPEED:
            return "tinyllama" if complexity <= 3 else "gemma:2b"
        elif priority == Priority.QUALITY:
            return "phi3:mini" if complexity >= 7 else "gemma:2b"
        else:  # BALANCED
            if task_type in [TaskType.BRAND_VOICE, TaskType.RESEARCH, TaskType.PROSPECTING]:
                return "phi3:mini"
            elif task_type in [TaskType.SOCIAL_MEDIA, TaskType.EMAIL]:
                return "gemma:2b"
            else:
                return "gemma:2b" if complexity <= 6 else "phi3:mini"
    
    async def optimize_prompt(self, request: OptimizePromptRequest) -> Union[OptimizedPrompt, InteractiveOptimizationResponse]:
        if request.optimization_level == OptimizationLevel.LOW_REGENERATE:
            return await self._regenerate_prompt_variant(request)
        
        elif request.optimization_level == OptimizationLevel.LOW_FIND_TWEAKS:
            if request.user_answers is None:
                return await self._generate_optimization_questions_response(request)
            else:
                return await self._process_answers_and_optimize(request)
        
        elif request.optimization_level == OptimizationLevel.STANDARD:
            return await self._standard_optimization(request)
        
        elif request.optimization_level == OptimizationLevel.ADVANCED:
            return await self._advanced_optimization(request)
        
        else:
            raise HTTPException(status_code=400, detail="Invalid optimization level")
    
    async def _regenerate_prompt_variant(self, request: OptimizePromptRequest) -> OptimizedPrompt:
        variant_prompt = f"""
        Create a variant of this prompt while keeping the same structure and intent:
        
        ORIGINAL PROMPT: {request.original_prompt}
        TASK TYPE: {request.task_type.value}
        
        Requirements:
        - Keep the same core purpose and structure
        - Change wording and phrasing for variety
        - Maintain the same level of specificity
        - Don't change the fundamental request
        
        Return only the variant prompt, no explanation.
        """
        
        try:
            response = self.ollama_client.generate(
                model="gemma:2b",
                prompt=variant_prompt,
                options={'temperature': 0.8, 'max_tokens': 400}
            )
            
            optimized = response['response'].strip()
            
            result = OptimizedPrompt(
                id=hashlib.md5(f"{request.original_prompt}_variant_{time.time()}".encode()).hexdigest()[:8],
                original_prompt=request.original_prompt,
                optimized_prompt=optimized,
                task_type=request.task_type,
                optimization_level=request.optimization_level,
                model_used="gemma:2b",
                timestamp=datetime.now().isoformat(),
                improvement_score=8.0,
                stored_in_library=True
            )
            
            self._store_prompt_in_library(result)
            return result
            
        except Exception as e:
            logger.error(f"Prompt variant generation error: {e}")
            raise HTTPException(status_code=500, detail=f"Variant generation failed: {str(e)}")
    
    async def _generate_optimization_questions_response(self, request: OptimizePromptRequest) -> InteractiveOptimizationResponse:
        session_id = hashlib.md5(f"{request.original_prompt}_{time.time()}".encode()).hexdigest()[:12]
        
        # Generate questions based on task type
        questions = self._generate_questions_for_task(request.task_type)
        
        self.interactive_sessions[session_id] = {
            "original_request": request,
            "questions": questions,
            "timestamp": datetime.now().isoformat()
        }
        
        return InteractiveOptimizationResponse(
            session_id=session_id,
            questions=questions,
            instruction="Please answer the questions to help me optimize your prompt.",
            example_format="Example: 1.a 2.b,c 3.I want something more casual"
        )
    
    def _generate_questions_for_task(self, task_type: TaskType) -> List[OptimizationQuestion]:
        question_templates = {
            TaskType.SOCIAL_MEDIA: [
                {
                    "question": "What's your target audience?",
                    "options": ["a. general public", "b. business owners", "c. content creators", "d. tech professionals"]
                },
                {
                    "question": "Which platform is this for?",
                    "options": ["a. Twitter/X", "b. LinkedIn", "c. Instagram", "d. Facebook"]
                },
                {
                    "question": "What's your primary goal?",
                    "options": ["a. engagement", "b. brand awareness", "c. lead generation", "d. education"]
                }
            ],
            TaskType.EMAIL: [
                {
                    "question": "What's the email purpose?",
                    "options": ["a. sales outreach", "b. follow-up", "c. newsletter", "d. support"]
                },
                {
                    "question": "Who is the recipient?",
                    "options": ["a. cold prospect", "b. existing client", "c. team member", "d. subscriber"]
                }
            ]
        }
        
        templates = question_templates.get(task_type, [
            {
                "question": "Who is your target audience?",
                "options": ["a. general public", "b. professionals", "c. specific niche", "d. team members"]
            }
        ])
        
        return [OptimizationQuestion(
            id=i + 1,
            question=template["question"],
            options=template["options"],
            allow_custom=True
        ) for i, template in enumerate(templates[:5])]
    
    async def _process_answers_and_optimize(self, request: OptimizePromptRequest) -> OptimizedPrompt:
        if not request.session_id or request.session_id not in self.interactive_sessions:
            raise HTTPException(status_code=400, detail="Invalid session ID")
        
        session = self.interactive_sessions[request.session_id]
        
        optimization_prompt = f"""
        Optimize this prompt based on user feedback:
        
        ORIGINAL PROMPT: {request.original_prompt}
        TASK TYPE: {request.task_type.value}
        USER ANSWERS: {request.user_answers}
        
        Create an improved version that incorporates their specific requirements.
        Return only the optimized prompt.
        """
        
        try:
            response = self.ollama_client.generate(
                model="phi3:mini",
                prompt=optimization_prompt,
                options={'temperature': 0.4, 'max_tokens': 600}
            )
            
            optimized = response['response'].strip()
            
            result = OptimizedPrompt(
                id=hashlib.md5(f"{request.original_prompt}_interactive_{time.time()}".encode()).hexdigest()[:8],
                original_prompt=request.original_prompt,
                optimized_prompt=optimized,
                task_type=request.task_type,
                optimization_level=request.optimization_level,
                model_used="phi3:mini",
                timestamp=datetime.now().isoformat(),
                improvement_score=9.0,
                stored_in_library=True,
                memory_updated=True
            )
            
            self._store_prompt_in_library(result)
            del self.interactive_sessions[request.session_id]
            
            return result
            
        except Exception as e:
            logger.error(f"Interactive optimization error: {e}")
            raise HTTPException(status_code=500, detail=f"Interactive optimization failed: {str(e)}")
    
    async def _standard_optimization(self, request: OptimizePromptRequest) -> OptimizedPrompt:
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
            response = self.ollama_client.generate(
                model="gemma:2b",
                prompt=optimization_prompt,
                options={'temperature': 0.3, 'max_tokens': 400}
            )
            
            optimized = response['response'].strip()
            improvement_score = min(10.0, len(optimized.split()) / max(1, len(request.original_prompt.split())) * 5)
            
            result = OptimizedPrompt(
                id=hashlib.md5(request.original_prompt.encode()).hexdigest()[:8],
                original_prompt=request.original_prompt,
                optimized_prompt=optimized,
                task_type=request.task_type,
                optimization_level=request.optimization_level,
                model_used="gemma:2b",
                timestamp=datetime.now().isoformat(),
                improvement_score=improvement_score,
                stored_in_library=True
            )
            
            self._store_prompt_in_library(result)
            return result
            
        except Exception as e:
            logger.error(f"Standard optimization error: {e}")
            raise HTTPException(status_code=500, detail=f"Standard optimization failed: {str(e)}")
    
    async def _advanced_optimization(self, request: OptimizePromptRequest) -> OptimizedPrompt:
        optimization_prompt = f"""
        You are a world-class prompt engineering expert. Perform advanced optimization:
        
        ORIGINAL PROMPT: {request.original_prompt}
        TASK TYPE: {request.task_type.value}
        TARGET AUDIENCE: {request.target_audience or "general"}
        
        Advanced requirements:
        1. Analyze prompt structure and identify weak points
        2. Add specific examples and success criteria
        3. Include relevant context and background
        4. Optimize for clarity and completeness
        5. Add formatting instructions
        6. Make it actionable and measurable
        
        Return only the fully optimized prompt.
        """
        
        try:
            response = self.ollama_client.generate(
                model="phi3:mini",
                prompt=optimization_prompt,
                options={'temperature': 0.2, 'max_tokens': 800}
            )
            
            optimized = response['response'].strip()
            
            result = OptimizedPrompt(
                id=hashlib.md5(f"{request.original_prompt}_advanced_{time.time()}".encode()).hexdigest()[:8],
                original_prompt=request.original_prompt,
                optimized_prompt=optimized,
                task_type=request.task_type,
                optimization_level=request.optimization_level,
                model_used="phi3:mini",
                timestamp=datetime.now().isoformat(),
                improvement_score=9.5,
                stored_in_library=True
            )
            
            self._store_prompt_in_library(result)
            return result
            
        except Exception as e:
            logger.error(f"Advanced optimization error: {e}")
            raise HTTPException(status_code=500, detail=f"Advanced optimization failed: {str(e)}")
    
    async def execute_immediate_task(self, request: QRequest, routing: RoutingDecision) -> str:
        model = self.select_local_model(request.task_type, routing.estimated_complexity, request.priority)
        
        try:
            response = self.ollama_client.generate(
                model=model,
                prompt=request.prompt,
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

# ============================================================================
# LIFESPAN CONTEXT MANAGER
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 AIQueue Local Engine starting up...")
    
    global core
    core = AIQueueCore()
    logger.info("Available models: tinyllama, gemma:2b, phi3:mini")
    
    try:
        models = core.ollama_client.list()
        logger.info(f"✅ Ollama connected with {len(models['models'])} models")
    except Exception as e:
        logger.error(f"❌ Ollama connection failed: {e}")
    
    yield
    
    logger.info("🛑 AIQueue Local Engine shutting down...")

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="AIQueue Local Engine",
    description="Local Ollama-powered AI engine for the AIQueue system",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

core = None

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {
        "message": "AIQueue Local Engine is running",
        "version": "1.0.0",
        "status": "ready"
    }

@app.post("/q", response_model=QResponse)
async def process_q_request(request: QRequest, background_tasks: BackgroundTasks):
    request_id = hashlib.md5(f"{request.prompt}{time.time()}".encode()).hexdigest()[:12]
    start_time = time.time()
    
    try:
        routing = await core.route_request(request)
        
        if routing.station == ProcessingStation.IMMEDIATE:
            response_text = await core.execute_immediate_task(request, routing)
            status = "completed"
            next_actions = []
        elif routing.station == ProcessingStation.DELEGATE:
            response_text = f"Task queued for delegation: {routing.reasoning}"
            status = "queued"
            next_actions = ["requires_external_api", "team_collaboration"]
        else:
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
            estimated_cost=0.0,
            next_actions=next_actions,
            status=status
        )
        
    except Exception as e:
        logger.error(f"Q request processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/optimize")
async def optimize_prompt(request: OptimizePromptRequest):
    return await core.optimize_prompt(request)

@app.get("/prompt-library")
async def get_prompt_library():
    return {
        "total_prompts": len(core.prompt_library["prompts"]),
        "categories": core.prompt_library["categories"],
        "recent_prompts": core.prompt_library["prompts"][-10:],
        "usage_stats": core.prompt_library["usage_stats"]
    }

@app.get("/user-memory")
async def get_user_memory():
    return {
        "task_preferences": core.user_memory["task_preferences"],
        "common_answers": core.user_memory["common_answers"],
        "memory_stats": {
            "total_tasks": len(core.user_memory["task_preferences"]),
            "last_updated": core.user_memory["last_updated"]
        }
    }

@app.get("/stats")
async def get_system_stats():
    """Get system performance and usage statistics"""
    try:
        # Try direct Ollama API call for more reliable data
        import requests
        ollama_response = requests.get('http://localhost:11434/api/tags', timeout=5)
        
        if ollama_response.status_code == 200:
            ollama_data = ollama_response.json()
            available_models = [model['name'] for model in ollama_data.get('models', [])]
            ollama_error = None
        else:
            available_models = []
            ollama_error = f"Ollama API returned status {ollama_response.status_code}"
            
    except requests.exceptions.ConnectionError:
        available_models = []
        ollama_error = "Cannot connect to Ollama (connection refused)"
    except requests.exceptions.Timeout:
        available_models = []
        ollama_error = "Ollama API timeout"
    except Exception as e:
        available_models = []
        ollama_error = f"Ollama API error: {str(e)}"
        
    # Get other stats safely
    try:
        active_sessions = len(core.interactive_sessions) if core else 0
        prompt_library_size = len(core.prompt_library["prompts"]) if core and core.prompt_library else 0
        user_memory_tasks = len(core.user_memory["task_preferences"]) if core and core.user_memory else 0
    except Exception as e:
        active_sessions = 0
        prompt_library_size = 0
        user_memory_tasks = 0
        
    return {
        "available_models": available_models,
        "active_sessions": active_sessions,
        "prompt_library_size": prompt_library_size,
        "user_memory_tasks": user_memory_tasks,
        "system_status": "operational" if not ollama_error else "degraded",
        "ollama_status": "connected" if not ollama_error else ollama_error
    }
    
@app.get("/models")
async def list_models():
    """List available local models and their capabilities"""
    try:
        ollama_models = core.ollama_client.list()
        models_info = []
        
        if 'models' in ollama_models:
            for model in ollama_models['models']:
                models_info.append({
                    "name": model.get('name', 'unknown'),
                    "size": model.get('size', 0),
                    "modified": model.get('modified_at', ''),
                    "family": model.get('details', {}).get('family', 'unknown'),
                    "parameter_size": model.get('details', {}).get('parameter_size', 'unknown')
                })
        
        return {
            "models": models_info,
            "total_count": len(models_info),
            "ollama_status": "connected"
        }
    except Exception as e:
        return {
            "error": str(e),
            "models": [],
            "ollama_status": "disconnected"
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
    #end of v19