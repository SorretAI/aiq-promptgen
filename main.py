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
    LOW_REGENERATE = "low_regenerate"    # Generate variant of good prompt
    LOW_FIND_TWEAKS = "low_find_tweaks"  # Interactive improvement with questions
    STANDARD = "standard"                # Direct optimization - Gemma:2b
    ADVANCED = "advanced"                # High quality - Phi3:mini

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
    session_id: Optional[str] = None  # For interactive optimization
    user_answers: Optional[str] = None  # Format: "1.a 2.b,c 3.custom_answer"

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

class OptimizationQuestion(BaseModel):
    """Question for interactive optimization"""
    id: int
    question: str
    options: List[str]  # ["a. option1", "b. option2", ...]
    allow_custom: bool = True

class InteractiveOptimizationResponse(BaseModel):
    """Response for interactive optimization - questions phase"""
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
        self.interactive_sessions = {}  # Store active interactive sessions
        self.user_memory_file = "user_preferences.json"
        self.prompt_library_file = "prompt_library.json"
        self.user_memory = self._load_user_memory()
        self.prompt_library = self._load_prompt_library()
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
    
    def _load_user_memory(self) -> Dict:
        """Load user preferences and learning data from local file"""
        try:
            if os.path.exists(self.user_memory_file):
                with open(self.user_memory_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading user memory: {e}")
        
        # Default memory structure
        return {
            "task_preferences": {},  # Preferences by task_type
            "common_answers": {},    # Frequently used answers
            "question_patterns": {}, # Learned question effectiveness
            "last_updated": datetime.now().isoformat()
        }
    
    def _load_prompt_library(self) -> Dict:
        """Load prompt library from local file"""
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
        """Save user memory to local file"""
        try:
            self.user_memory["last_updated"] = datetime.now().isoformat()
            with open(self.user_memory_file, 'w') as f:
                json.dump(self.user_memory, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving user memory: {e}")
    
    def _save_prompt_library(self):
        """Save prompt library to local file"""
        try:
            self.prompt_library["last_updated"] = datetime.now().isoformat()
            with open(self.prompt_library_file, 'w') as f:
                json.dump(self.prompt_library, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving prompt library: {e}")
    
    def _store_prompt_in_library(self, optimized_prompt: OptimizedPrompt):
        """Store optimized prompt in library for future reuse"""
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
        
        # Update categories
        if optimized_prompt.task_type not in self.prompt_library["categories"]:
            self.prompt_library["categories"][optimized_prompt.task_type] = []
        self.prompt_library["categories"][optimized_prompt.task_type].append(optimized_prompt.id)
        
        self._save_prompt_library()
        return True
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
        
        Classify into exactly one of these stations:
        - IMMEDIATE: Can be handled immediately with local AI (content creation, optimization, simple analysis)
        - DELEGATE: Needs external APIs, human input, or team collaboration
        - BACKBURNER: Requires research, complex analysis, or missing information
        
        Consider:
        - Complexity of the request
        - Available local capabilities
        - Time sensitivity
        - Resource requirements
        
        Respond with EXACTLY this format (no extra text):
        STATION|REASONING|COMPLEXITY_SCORE|LOCAL_CAPABLE|PREMIUM_NEEDED
        
        Where:
        - STATION: IMMEDIATE, DELEGATE, or BACKBURNER
        - REASONING: Brief explanation
        - COMPLEXITY_SCORE: Number 1-10
        - LOCAL_CAPABLE: yes or no
        - PREMIUM_NEEDED: yes or no
        """
        
        try:
            response = self.ollama_client.generate(
                model="tinyllama",
                prompt=routing_prompt,
                options={'temperature': 0.1, 'max_tokens': 200}
            )
            
            parts = response['response'].strip().split('|')
            if len(parts) >= 5:
                # Handle case-insensitive station parsing
                station_text = parts[0].strip().upper()
                if station_text == "IMMEDIATE":
                    station = ProcessingStation.IMMEDIATE
                elif station_text == "DELEGATE":
                    station = ProcessingStation.DELEGATE
                elif station_text == "BACKBURNER":
                    station = ProcessingStation.BACKBURNER
                else:
                    station = ProcessingStation.IMMEDIATE  # Safe fallback
                
                reasoning = parts[1].strip()
                complexity = int(parts[2].strip())
                local_capable = parts[3].strip().lower() == 'yes'
                premium_needed = parts[4].strip().lower() == 'yes'
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
    
    async def optimize_prompt(self, request: OptimizePromptRequest) -> Union[OptimizedPrompt, InteractiveOptimizationResponse]:
        """Optimize a prompt using different strategies based on optimization level"""
        
        if request.optimization_level == OptimizationLevel.LOW_REGENERATE:
            return await self._regenerate_prompt_variant(request)
        
        elif request.optimization_level == OptimizationLevel.LOW_FIND_TWEAKS:
            if request.user_answers is None:
                # First call - generate questions
                return await self._generate_optimization_questions_response(request)
            else:
                # Second call - process answers and optimize
                return await self._process_answers_and_optimize(request)
        
        elif request.optimization_level == OptimizationLevel.STANDARD:
            return await self._standard_optimization(request)
        
        elif request.optimization_level == OptimizationLevel.ADVANCED:
            return await self._advanced_optimization(request)
        
        else:
            raise HTTPException(status_code=400, detail="Invalid optimization level")
    
    async def _regenerate_prompt_variant(self, request: OptimizePromptRequest) -> OptimizedPrompt:
        """Generate a variant of the original prompt (LOW_REGENERATE)"""
        
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
                options={'temperature': 0.8, 'max_tokens': 400}  # Higher temp for variety
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
                improvement_score=8.0,  # Variants are generally good
                stored_in_library=True
            )
            
            # Store in library
            self._store_prompt_in_library(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Prompt variant generation error: {e}")
            raise HTTPException(status_code=500, detail=f"Variant generation failed: {str(e)}")
    
    async def _generate_optimization_questions_response(self, request: OptimizePromptRequest) -> InteractiveOptimizationResponse:
        """Generate questions for interactive optimization (LOW_FIND_TWEAKS first call)"""
        
        # Generate session ID
        session_id = hashlib.md5(f"{request.original_prompt}_{time.time()}".encode()).hexdigest()[:12]
        
        # Generate contextual questions
        questions = self._generate_optimization_questions(request.original_prompt, request.task_type)
        
        # Store session for later
        self.interactive_sessions[session_id] = {
            "original_request": request,
            "questions": questions,
            "timestamp": datetime.now().isoformat()
        }
        
        return InteractiveOptimizationResponse(
            session_id=session_id,
            questions=questions,
            instruction="Please answer the questions to help me optimize your prompt. You can combine options or provide custom answers.",
            example_format="Example: 1.a 2.b,c 3.I want something more casual and friendly"
        )
    
    async def _process_answers_and_optimize(self, request: OptimizePromptRequest) -> OptimizedPrompt:
        """Process user answers and generate optimized prompt (LOW_FIND_TWEAKS second call)"""
        
        if not request.session_id or request.session_id not in self.interactive_sessions:
            raise HTTPException(status_code=400, detail="Invalid or expired session ID")
        
        session = self.interactive_sessions[request.session_id]
        questions = session["questions"]
        
        # Parse user answers
        parsed_answers = self._parse_user_answers(request.user_answers, questions)
        
        # Update user memory with answers
        self._update_user_memory(request.task_type, parsed_answers)
        
        # Generate optimization prompt based on answers
        optimization_prompt = self._build_contextual_optimization_prompt(
            request.original_prompt,
            request.task_type,
            parsed_answers,
            questions
        )
        
        try:
            response = self.ollama_client.generate(
                model="phi3:mini",  # Use best model for contextual optimization
                prompt=optimization_prompt,
                options={'temperature': 0.4, 'max_tokens': 600}
            )
            
            optimized = response['response'].strip()
            
            result = OptimizedPrompt(
                id=hashlib.md5(f"{request.original_prompt}_optimized_{time.time()}".encode()).hexdigest()[:8],
                original_prompt=request.original_prompt,
                optimized_prompt=optimized,
                task_type=request.task_type,
                optimization_level=request.optimization_level,
                model_used="phi3:mini",
                timestamp=datetime.now().isoformat(),
                improvement_score=9.0,  # Interactive optimization is usually high quality
                stored_in_library=True,
                memory_updated=True
            )
            
            # Store in library
            self._store_prompt_in_library(result)
            
            # Clean up session
            del self.interactive_sessions[request.session_id]
            
            return result
            
        except Exception as e:
            logger.error(f"Interactive optimization error: {e}")
            raise HTTPException(status_code=500, detail=f"Interactive optimization failed: {str(e)}")
    
    async def _standard_optimization(self, request: OptimizePromptRequest) -> OptimizedPrompt:
        """Standard optimization using current logic"""
        
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
                model="gemma:2b",
                prompt=optimization_prompt,
                options={'temperature': 0.3, 'max_tokens': 400}
            )
            processing_time = time.time() - start_time
            
            optimized = response['response'].strip()
            
            # Calculate improvement score (simple heuristic)
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
            
            # Store in library
            self._store_prompt_in_library(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Standard optimization error: {e}")
            raise HTTPException(status_code=500, detail=f"Standard optimization failed: {str(e)}")
    
    async def _advanced_optimization(self, request: OptimizePromptRequest) -> OptimizedPrompt:
        """Advanced optimization using phi3:mini with detailed analysis"""
        
        # More sophisticated optimization prompt
        optimization_prompt = f"""
        You are a world-class prompt engineering expert. Perform advanced optimization on this prompt:
        
        ORIGINAL PROMPT: {request.original_prompt}
        TASK TYPE: {request.task_type.value}
        TARGET AUDIENCE: {request.target_audience or "general"}
        CONSTRAINTS: {', '.join(request.constraints) if request.constraints else "none"}
        
        Advanced optimization requirements:
        1. Analyze the prompt structure and identify weak points
        2. Add specific examples and clear success criteria
        3. Include relevant context and background information
        4. Optimize for both clarity and completeness
        5. Add formatting instructions for better output
        6. Include edge case handling
        7. Make it more actionable and measurable
        8. Reduce ambiguity while maintaining creativity
        
        Return only the fully optimized prompt with no meta-commentary.
        """
        
        try:
            response = self.ollama_client.generate(
                model="phi3:mini",
                prompt=optimization_prompt,
                options={'temperature': 0.2, 'max_tokens': 800}  # More tokens for detailed optimization
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
                improvement_score=9.5,  # Advanced optimization should be high quality
                stored_in_library=True
            )
            
            # Store in library
            self._store_prompt_in_library(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Advanced optimization error: {e}")
            raise HTTPException(status_code=500, detail=f"Advanced optimization failed: {str(e)}")
    
    def _parse_user_answers(self, user_answers: str, questions: List[OptimizationQuestion]) -> Dict:
        """Parse user answers from format like '1.a 2.b,c 3.custom answer'"""
        
        parsed = {}
        
        try:
            # Split by question numbers
            parts = user_answers.strip().split()
            
            for part in parts:
                if '.' in part:
                    # Check if it starts with a number
                    question_part = part.split('.', 1)
                    if question_part[0].isdigit():
                        q_id = int(question_part[0])
                        answer_part = question_part[1] if len(question_part) > 1 else ""
                        
                        # Handle multiple choice or custom answer
                        if answer_part:
                            # Check if it's option letters (a, b, c) or custom text
                            if all(c in 'abcdefghij,' for c in answer_part.lower()):
                                # Multiple choice answers
                                choices = [c.strip() for c in answer_part.lower().split(',')]
                                parsed[q_id] = {"type": "choices", "value": choices}
                            else:
                                # Custom answer
                                parsed[q_id] = {"type": "custom", "value": answer_part}
                    else:
                        # Continuation of previous custom answer
                        if parsed:
                            last_key = max(parsed.keys())
                            if parsed[last_key]["type"] == "custom":
                                parsed[last_key]["value"] += " " + part
                else:
                    # Continuation of custom answer
                    if parsed:
                        last_key = max(parsed.keys())
                        if parsed[last_key]["type"] == "custom":
                            parsed[last_key]["value"] += " " + part
                            
        except Exception as e:
            logger.error(f"Error parsing user answers: {e}")
            # Fallback - treat entire input as custom answer for question 1
            parsed = {1: {"type": "custom", "value": user_answers}}
        
        return parsed
    
    def _update_user_memory(self, task_type: TaskType, parsed_answers: Dict):
        """Update user memory with new answers"""
        
        if task_type not in self.user_memory["task_preferences"]:
            self.user_memory["task_preferences"][task_type] = {}
        
        task_memory = self.user_memory["task_preferences"][task_type]
        
        for q_id, answer in parsed_answers.items():
            key = f"question_{q_id}"
            
            if key not in task_memory:
                task_memory[key] = []
            
            # Store the answer
            if answer["type"] == "choices":
                task_memory[key].extend(answer["value"])
            else:
                task_memory[key].append(answer["value"])
            
            # Keep only last 10 answers to prevent unlimited growth
            task_memory[key] = task_memory[key][-10:]
        
        self._save_user_memory()
    
    def _build_contextual_optimization_prompt(self, original_prompt: str, task_type: TaskType, parsed_answers: Dict, questions: List[OptimizationQuestion]) -> str:
        """Build optimization prompt based on user answers"""
        
        # Convert answers to readable context
        context_parts = []
        
        for q_id, answer in parsed_answers.items():
            question = next((q for q in questions if q.id == q_id), None)
            if question:
                if answer["type"] == "choices":
                    # Convert letter choices to actual text
                    choice_texts = []
                    for choice_letter in answer["value"]:
                        option_index = ord(choice_letter.lower()) - ord('a')
                        if 0 <= option_index < len(question.options):
                            choice_texts.append(question.options[option_index])
                    
                    context_parts.append(f"{question.question}: {', '.join(choice_texts)}")
                else:
                    context_parts.append(f"{question.question}: {answer['value']}")
        
        context = "\n".join(context_parts)
        
        optimization_prompt = f"""
        You are an expert prompt engineer. Optimize this prompt based on the user's specific requirements:
        
        ORIGINAL PROMPT: {original_prompt}
        TASK TYPE: {task_type.value}
        
        USER REQUIREMENTS:
        {context}
        
        Optimization instructions:
        1. Incorporate all the user's specified requirements into the prompt
        2. Make the prompt more specific and targeted based on their answers
        3. Maintain the core intent while adapting to their preferences
        4. Add relevant context and examples based on their choices
        5. Ensure the output format matches their needs
        6. Make it actionable and clear
        
        Return only the optimized prompt that incorporates all their requirements.
        """
        
        return optimization_prompt3, 'max_tokens': 400}
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

@app.post("/optimize")
async def optimize_prompt(request: OptimizePromptRequest):
    """Optimize prompts using different strategies based on optimization level"""
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

@app.get("/prompt-library")
async def get_prompt_library():
    """Get user's prompt library"""
    return {
        "total_prompts": len(core.prompt_library["prompts"]),
        "categories": core.prompt_library["categories"],
        "recent_prompts": core.prompt_library["prompts"][-10:],  # Last 10 prompts
        "usage_stats": core.prompt_library["usage_stats"]
    }

@app.get("/prompt-library/{task_type}")
async def get_prompts_by_category(task_type: TaskType):
    """Get prompts filtered by task type"""
    category_prompts = [
        prompt for prompt in core.prompt_library["prompts"]
        if prompt["task_type"] == task_type
    ]
    return {
        "task_type": task_type,
        "count": len(category_prompts),
        "prompts": category_prompts
    }

@app.get("/user-memory")
async def get_user_memory():
    """Get user learning memory and preferences"""
    return {
        "task_preferences": core.user_memory["task_preferences"],
        "common_answers": core.user_memory["common_answers"],
        "memory_stats": {
            "total_tasks": len(core.user_memory["task_preferences"]),
            "last_updated": core.user_memory["last_updated"]
        }
    }

@app.delete("/user-memory")
async def reset_user_memory():
    """Reset user memory (for testing or fresh start)"""
    core.user_memory = {
        "task_preferences": {},
        "common_answers": {},
        "question_patterns": {},
        "last_updated": datetime.now().isoformat()
    }
    core._save_user_memory()
    return {"message": "User memory reset successfully"}

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
            "active_sessions": len(core.interactive_sessions),
            "prompt_library_size": len(core.prompt_library["prompts"]),
            "user_memory_tasks": len(core.user_memory["task_preferences"]),
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