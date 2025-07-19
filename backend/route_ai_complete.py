# Complete RouteAI System for SorretAI
# Intelligent model routing with cost optimization

import asyncio
import aiohttp
import ollama
import os
import json
from typing import Dict, List, Optional, Union
from enum import Enum
from dataclasses import dataclass, asdict
import time
from datetime import datetime

class ModelTier(Enum):
    LOCAL = "local"           # Ollama models (FREE)
    FREE_API = "free_api"     # OpenRouter free models  
    PAID_API = "paid_api"     # Premium models for complex tasks

class TaskComplexity(Enum):
    SIMPLE = 1      # Basic content, simple prompts
    MEDIUM = 2      # Standard content creation
    COMPLEX = 3     # Advanced reasoning, long-form content
    CRITICAL = 4    # Business-critical content, video scripts

@dataclass 
class ModelConfig:
    name: str
    tier: ModelTier
    cost_per_1k_tokens: float
    speed_rating: int  # 1-10, 10 being fastest
    quality_rating: int  # 1-10, 10 being highest quality
    context_limit: int
    specialties: List[str]
    api_endpoint: Optional[str] = None

@dataclass
class RouteResult:
    selected_model: str
    estimated_cost: float
    reasoning: str
    tier: ModelTier
    alternatives: List[str]

class SorretRouteAI:
    """
    Intelligent model routing system for SorretAI
    Routes requests to optimal model based on task requirements and cost
    """
    
    def __init__(self, openrouter_key: str = None, together_key: str = None):
        self.models = self._initialize_models()
        self.usage_stats = {}
        self.cost_tracker = 0.0
        self.daily_costs = {}
        self.openrouter_key = openrouter_key or os.getenv("OPENROUTER_API_KEY")
        self.together_key = together_key or os.getenv("TOGETHER_API_KEY")
        
    def _initialize_models(self) -> Dict[str, ModelConfig]:
        """Initialize available models with their configurations"""
        return {
            # LOCAL OLLAMA MODELS (FREE)
            "gemma:2b": ModelConfig(
                name="gemma:2b",
                tier=ModelTier.LOCAL,
                cost_per_1k_tokens=0.0,
                speed_rating=9,
                quality_rating=6,
                context_limit=8192,
                specialties=["prompt_optimization", "basic_content", "social_media"]
            ),
            "phi3:mini": ModelConfig(
                name="phi3:mini", 
                tier=ModelTier.LOCAL,
                cost_per_1k_tokens=0.0,
                speed_rating=7,
                quality_rating=8,
                context_limit=128000,
                specialties=["reasoning", "technical_content", "code", "analysis"]
            ),
            "tinyllama": ModelConfig(
                name="tinyllama",
                tier=ModelTier.LOCAL,
                cost_per_1k_tokens=0.0,
                speed_rating=10,
                quality_rating=4,
                context_limit=2048,
                specialties=["simple_tasks", "quick_responses", "basic_prompts"]
            ),
            
            # FREE API MODELS (OpenRouter)
            "deepseek-r1": ModelConfig(
                name="deepseek/deepseek-r1:free",
                tier=ModelTier.FREE_API,
                cost_per_1k_tokens=0.0,
                speed_rating=6,
                quality_rating=9,
                context_limit=128000,
                specialties=["reasoning", "complex_content", "analysis", "business"],
                api_endpoint="openrouter"
            ),
            "llama-4-maverick": ModelConfig(
                name="meta-llama/llama-4-maverick:free",
                tier=ModelTier.FREE_API,
                cost_per_1k_tokens=0.0,
                speed_rating=7,
                quality_rating=9,
                context_limit=128000,
                specialties=["multimodal", "creative_content", "storytelling", "video_scripts"],
                api_endpoint="openrouter"
            ),
            "qwen-32b": ModelConfig(
                name="qwen/qwq-32b:free",
                tier=ModelTier.FREE_API,
                cost_per_1k_tokens=0.0,
                speed_rating=6,
                quality_rating=8,
                context_limit=32768,
                specialties=["multilingual", "reasoning", "general_tasks"],
                api_endpoint="openrouter"
            ),
            "gemma-3-27b": ModelConfig(
                name="google/gemma-3-27b-it:free",
                tier=ModelTier.FREE_API,
                cost_per_1k_tokens=0.0,
                speed_rating=7,
                quality_rating=8,
                context_limit=8192,
                specialties=["content_creation", "prospecting", "professional"],
                api_endpoint="openrouter"
            ),
            
            # PAID API MODELS (for complex/critical tasks)
            "together-llama-70b": ModelConfig(
                name="meta-llama/Llama-3.1-70B-Instruct-Turbo",
                tier=ModelTier.PAID_API,
                cost_per_1k_tokens=0.9,
                speed_rating=8,
                quality_rating=10,
                context_limit=131072,
                specialties=["premium_content", "business_critical", "complex_reasoning"],
                api_endpoint="together"
            ),
            "together-qwen-72b": ModelConfig(
                name="Qwen/Qwen2.5-72B-Instruct-Turbo",
                tier=ModelTier.PAID_API,
                cost_per_1k_tokens=1.2,
                speed_rating=7,
                quality_rating=10,
                context_limit=32768,
                specialties=["advanced_reasoning", "business_critical", "multilingual"],
                api_endpoint="together"
            )
        }
    
    def route_request(
        self, 
        prompt: str,
        task_type: str,
        complexity: TaskComplexity = TaskComplexity.MEDIUM,
        priority: str = "balanced",  # "speed", "cost", "quality", "balanced"
        max_cost: float = None,
        force_free: bool = False
    ) -> RouteResult:
        """
        Main routing function - selects optimal model for the request
        """
        # 1. Analyze prompt requirements
        requirements = self._analyze_requirements(prompt, task_type, complexity)
        
        # 2. Filter available models
        available_models = self._filter_models(requirements, max_cost, force_free)
        
        # 3. Score and rank models
        ranked_models = self._rank_models(available_models, priority, requirements)
        
        # 4. Select best model
        if not ranked_models:
            selected_model = "gemma:2b"  # Ultimate fallback
            reasoning = "Fallback to local model - no other options available"
        else:
            selected_model = ranked_models[0]
            reasoning = self._generate_reasoning(selected_model, requirements, priority)
        
        # 5. Calculate estimated cost
        estimated_cost = self._calculate_cost(selected_model, requirements)
        
        # 6. Track selection
        self._track_selection(selected_model, requirements, estimated_cost)
        
        return RouteResult(
            selected_model=selected_model,
            estimated_cost=estimated_cost,
            reasoning=reasoning,
            tier=self.models[selected_model].tier,
            alternatives=ranked_models[1:4] if len(ranked_models) > 1 else []
        )
    
    def _analyze_requirements(self, prompt: str, task_type: str, complexity: TaskComplexity) -> Dict:
        """Analyze what the request needs"""
        prompt_tokens = len(prompt.split()) * 1.3  # Rough token estimation
        estimated_response_tokens = self._estimate_response_length(prompt, task_type)
        
        requirements = {
            "prompt_tokens": int(prompt_tokens),
            "estimated_response_tokens": estimated_response_tokens,
            "total_tokens": int(prompt_tokens + estimated_response_tokens),
            "complexity": complexity,
            "task_type": task_type,
            "needs_reasoning": any(word in prompt.lower() for word in 
                                 ["analyze", "explain", "compare", "evaluate", "reason", "think"]),
            "needs_creativity": any(word in prompt.lower() for word in 
                                  ["creative", "story", "imagine", "design", "brainstorm", "original"]),
            "needs_accuracy": any(word in task_type.lower() for word in 
                                 ["prospecting", "business", "sales", "professional", "client"]),
            "is_long_form": estimated_response_tokens > 1000,
            "is_business_critical": complexity in [TaskComplexity.CRITICAL] or 
                                  any(word in task_type.lower() for word in ["business", "sales", "client"])
        }
        
        return requirements
    
    def _estimate_response_length(self, prompt: str, task_type: str) -> int:
        """Estimate how many tokens the response will be"""
        base_estimates = {
            "social_media": 150,
            "email": 250,
            "content_creation": 800,
            "blog_post": 1500,
            "video_script": 600,
            "prospecting": 200,
            "phone_script": 300,
            "analysis": 1000,
            "report": 2000
        }
        
        base = base_estimates.get(task_type, 400)
        
        # Adjust based on prompt length and keywords
        prompt_length = len(prompt.split())
        if prompt_length > 200:
            base *= 2
        elif prompt_length > 100:
            base *= 1.5
        elif prompt_length < 20:
            base *= 0.7
            
        # Adjust for complexity indicators
        if "detailed" in prompt.lower() or "comprehensive" in prompt.lower():
            base *= 1.8
        if "brief" in prompt.lower() or "short" in prompt.lower():
            base *= 0.5
            
        return int(base)
    
    def _filter_models(self, requirements: Dict, max_cost: float, force_free: bool) -> List[str]:
        """Filter models that can handle the requirements"""
        available = []
        
        for model_name, config in self.models.items():
            # Check context limit
            if requirements["total_tokens"] > config.context_limit:
                continue
                
            # Check cost constraint
            if max_cost:
                estimated_cost = (requirements["total_tokens"] / 1000) * config.cost_per_1k_tokens
                if estimated_cost > max_cost:
                    continue
            
            # Force free constraint
            if force_free and config.cost_per_1k_tokens > 0:
                continue
                
            # Check API availability
            if config.tier == ModelTier.FREE_API and not self.openrouter_key:
                continue
            if config.tier == ModelTier.PAID_API and not self.together_key:
                continue
                
            available.append(model_name)
            
        return available
    
    def _rank_models(self, available_models: List[str], priority: str, requirements: Dict) -> List[str]:
        """Rank models based on priority and requirements"""
        if not available_models:
            return []
            
        scored_models = []
        
        for model_name in available_models:
            config = self.models[model_name]
            score = self._calculate_model_score(config, priority, requirements)
            scored_models.append((model_name, score))
        
        # Sort by score (higher is better)
        scored_models.sort(key=lambda x: x[1], reverse=True)
        
        return [model_name for model_name, _ in scored_models]
    
    def _calculate_model_score(self, config: ModelConfig, priority: str, requirements: Dict) -> float:
        """Calculate fitness score for a model"""
        score = 0.0
        
        # Base scores based on priority
        if priority == "speed":
            score += config.speed_rating * 4
            score += config.quality_rating * 1
            score -= config.cost_per_1k_tokens * 5
        elif priority == "cost":
            score -= config.cost_per_1k_tokens * 10
            score += config.speed_rating * 1
            score += config.quality_rating * 2
        elif priority == "quality":
            score += config.quality_rating * 5
            score += config.speed_rating * 1
            score -= config.cost_per_1k_tokens * 2
        else:  # balanced
            score += config.quality_rating * 3
            score += config.speed_rating * 2
            score -= config.cost_per_1k_tokens * 4
        
        # Complexity matching
        complexity = requirements["complexity"]
        if complexity == TaskComplexity.SIMPLE:
            if config.tier == ModelTier.LOCAL:
                score += 8  # Strong preference for local on simple tasks
        elif complexity == TaskComplexity.MEDIUM:
            if config.tier in [ModelTier.LOCAL, ModelTier.FREE_API]:
                score += 5
        elif complexity == TaskComplexity.COMPLEX:
            if config.quality_rating >= 8:
                score += 6
            if config.tier == ModelTier.FREE_API and config.quality_rating >= 8:
                score += 3  # Prefer good free models for complex tasks
        elif complexity == TaskComplexity.CRITICAL:
            if config.quality_rating >= 9:
                score += 8
            if config.tier == ModelTier.PAID_API:
                score += 4  # Prefer paid for critical tasks
                
        # Task type matching
        task_type = requirements["task_type"]
        if task_type in config.specialties:
            score += 5
        if any(specialty in config.specialties for specialty in ["general_tasks", "basic_content"]):
            score += 1
            
        # Special requirements
        if requirements["needs_reasoning"] and "reasoning" in config.specialties:
            score += 4
        if requirements["needs_creativity"] and any(s in config.specialties for s in ["creative_content", "storytelling"]):
            score += 4
        if requirements["needs_accuracy"] and any(s in config.specialties for s in ["business", "professional"]):
            score += 3
        if requirements["is_business_critical"] and "business" in config.specialties:
            score += 5
            
        # Long form content bonus
        if requirements["is_long_form"] and config.context_limit > 32000:
            score += 3
            
        return score
    
    def _generate_reasoning(self, selected_model: str, requirements: Dict, priority: str) -> str:
        """Generate human-readable reasoning for model selection"""
        config = self.models[selected_model]
        reasons = []
        
        # Tier reasoning
        if config.tier == ModelTier.LOCAL:
            reasons.append("Using local model for cost efficiency")
        elif config.tier == ModelTier.FREE_API:
            reasons.append("Using free API model for enhanced capabilities")
        else:
            reasons.append("Using premium model for critical task")
            
        # Complexity reasoning
        complexity = requirements["complexity"]
        if complexity == TaskComplexity.SIMPLE:
            reasons.append("Simple task suitable for basic model")
        elif complexity == TaskComplexity.COMPLEX:
            reasons.append("Complex task requires advanced model")
        elif complexity == TaskComplexity.CRITICAL:
            reasons.append("Critical task requires highest quality model")
            
        # Priority reasoning
        if priority == "speed":
            reasons.append(f"Optimized for speed (rating: {config.speed_rating}/10)")
        elif priority == "quality":
            reasons.append(f"Optimized for quality (rating: {config.quality_rating}/10)")
        elif priority == "cost":
            reasons.append(f"Optimized for cost (${config.cost_per_1k_tokens}/1K tokens)")
            
        return "; ".join(reasons)
    
    def _calculate_cost(self, model_name: str, requirements: Dict) -> float:
        """Calculate estimated cost for the request"""
        config = self.models[model_name]
        total_tokens = requirements["total_tokens"]
        return (total_tokens / 1000) * config.cost_per_1k_tokens
    
    def _track_selection(self, model_name: str, requirements: Dict, cost: float):
        """Track model usage for analytics"""
        today = datetime.now().date().isoformat()
        
        # Track overall usage
        if model_name not in self.usage_stats:
            self.usage_stats[model_name] = {
                "requests": 0,
                "total_tokens": 0,
                "total_cost": 0.0,
                "avg_tokens_per_request": 0
            }
        
        stats = self.usage_stats[model_name]
        stats["requests"] += 1
        stats["total_tokens"] += requirements["total_tokens"]
        stats["total_cost"] += cost
        stats["avg_tokens_per_request"] = stats["total_tokens"] / stats["requests"]
        
        # Track daily costs
        if today not in self.daily_costs:
            self.daily_costs[today] = 0.0
        self.daily_costs[today] += cost
        self.cost_tracker += cost
    
    async def execute_request(self, model_name: str, prompt: str, **kwargs) -> str:
        """Execute the request using the selected model"""
        config = self.models[model_name]
        
        try:
            if config.tier == ModelTier.LOCAL:
                return await self._execute_ollama(model_name, prompt, **kwargs)
            elif config.tier == ModelTier.FREE_API:
                return await self._execute_openrouter(config.name, prompt, **kwargs)
            else:  # PAID_API
                return await self._execute_together(config.name, prompt, **kwargs)
        except Exception as e:
            print(f"Error executing {model_name}: {e}")
            # Fallback to local model
            return await self._execute_ollama("gemma:2b", prompt, **kwargs)
    
    async def _execute_ollama(self, model_name: str, prompt: str, **kwargs) -> str:
        """Execute request using local Ollama"""
        try:
            client = ollama.Client()
            response = client.generate(
                model=model_name,
                prompt=prompt,
                options={
                    'temperature': kwargs.get('temperature', 0.7),
                    'top_p': kwargs.get('top_p', 0.9),
                    'max_tokens': kwargs.get('max_tokens', 2000)
                }
            )
            return response['response']
        except Exception as e:
            raise Exception(f"Ollama execution failed: {e}")
    
    async def _execute_openrouter(self, model_name: str, prompt: str, **kwargs) -> str:
        """Execute request using OpenRouter free models"""
        if not self.openrouter_key:
            raise Exception("OpenRouter API key not configured")
            
        async with aiohttp.ClientSession() as session:
            async with session.post(
                'https://openrouter.ai/api/v1/chat/completions',
                headers={
                    'Authorization': f'Bearer {self.openrouter_key}',
                    'Content-Type': 'application/json'
                },
                json={
                    'model': model_name,
                    'messages': [{'role': 'user', 'content': prompt}],
                    'temperature': kwargs.get('temperature', 0.7),
                    'max_tokens': kwargs.get('max_tokens', 2000)
                }
            ) as response:
                if response.status != 200:
                    raise Exception(f"OpenRouter API error: {response.status}")
                result = await response.json()
                return result['choices'][0]['message']['content']
    
    async def _execute_together(self, model_name: str, prompt: str, **kwargs) -> str:
        """Execute request using Together AI"""
        if not self.together_key:
            raise Exception("Together API key not configured")
            
        async with aiohttp.ClientSession() as session:
            async with session.post(
                'https://api.together.xyz/v1/chat/completions',
                headers={
                    'Authorization': f'Bearer {self.together_key}',
                    'Content-Type': 'application/json'
                },
                json={
                    'model': model_name,
                    'messages': [{'role': 'user', 'content': prompt}],
                    'temperature': kwargs.get('temperature', 0.7),
                    'max_tokens': kwargs.get('max_tokens', 2000)
                }
            ) as response:
                if response.status != 200:
                    raise Exception(f"Together API error: {response.status}")
                result = await response.json()
                return result['choices'][0]['message']['content']
    
    def get_usage_stats(self) -> Dict:
        """Get comprehensive usage statistics"""
        total_requests = sum(stats["requests"] for stats in self.usage_stats.values())
        
        return {
            "total_cost": round(self.cost_tracker, 4),
            "total_requests": total_requests,
            "daily_costs": self.daily_costs,
            "model_usage": self.usage_stats,
            "cost_per_request": round(self.cost_tracker / total_requests, 4) if total_requests > 0 else 0,
            "recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate cost optimization recommendations"""
        recommendations = []
        
        if self.cost_tracker > 50:  # If spending > $50/month
            recommendations.append("💡 Consider using more free tier models for non-critical tasks")
        
        # Analyze model usage patterns
        for model, stats in self.usage_stats.items():
            config = self.models[model]
            if stats["requests"] > 50 and config.tier == ModelTier.PAID_API:
                recommendations.append(f"🔍 High usage of paid model '{model}' - consider free alternatives")
            
            if stats["avg_tokens_per_request"] > 2000 and config.tier == ModelTier.LOCAL:
                recommendations.append(f"📊 Long responses from '{model}' - consider higher context models")
        
        # Daily cost warnings
        today = datetime.now().date().isoformat()
        if today in self.daily_costs and self.daily_costs[today] > 10:
            recommendations.append(f"⚠️ High daily cost: ${self.daily_costs[today]:.2f}")
            
        if not recommendations:
            recommendations.append("✅ Usage patterns look optimized!")
            
        return recommendations
    
    def export_stats(self, filepath: str):
        """Export usage statistics to JSON file"""
        stats = self.get_usage_stats()
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
    
    def reset_stats(self):
        """Reset usage statistics (useful for testing)"""
        self.usage_stats = {}
        self.cost_tracker = 0.0
        self.daily_costs = {}

# CLI Interface for testing
if __name__ == "__main__":
    import argparse
    
    async def main():
        parser = argparse.ArgumentParser(description="SorretAI RouteAI System")
        parser.add_argument("--prompt", required=True, help="Prompt to process")
        parser.add_argument("--task", default="content_creation", help="Task type")
        parser.add_argument("--complexity", default="MEDIUM", choices=["SIMPLE", "MEDIUM", "COMPLEX", "CRITICAL"])
        parser.add_argument("--priority", default="balanced", choices=["speed", "cost", "quality", "balanced"])
        parser.add_argument("--max-cost", type=float, help="Maximum cost per request")
        parser.add_argument("--force-free", action="store_true", help="Only use free models")
        parser.add_argument("--execute", action="store_true", help="Actually execute the request")
        parser.add_argument("--stats", action="store_true", help="Show usage statistics")
        
        args = parser.parse_args()
        
        # Initialize RouteAI
        router = SorretRouteAI()
        
        if args.stats:
            stats = router.get_usage_stats()
            print("\n📊 USAGE STATISTICS:")
            print(f"Total Cost: ${stats['total_cost']}")
            print(f"Total Requests: {stats['total_requests']}")
            print(f"Cost per Request: ${stats['cost_per_request']}")
            print("\n💡 Recommendations:")
            for rec in stats['recommendations']:
                print(f"  {rec}")
            return
        
        # Route the request
        complexity = TaskComplexity[args.complexity]
        result = router.route_request(
            prompt=args.prompt,
            task_type=args.task,
            complexity=complexity,
            priority=args.priority,
            max_cost=args.max_cost,
            force_free=args.force_free
        )
        
        print(f"\n🎯 ROUTING RESULT:")
        print(f"Selected Model: {result.selected_model}")
        print(f"Estimated Cost: ${result.estimated_cost:.4f}")
        print(f"Model Tier: {result.tier.value}")
        print(f"Reasoning: {result.reasoning}")
        
        if result.alternatives:
            print(f"Alternatives: {', '.join(result.alternatives[:3])}")
        
        # Execute if requested
        if args.execute:
            print(f"\n🚀 EXECUTING REQUEST...")
            try:
                response = await router.execute_request(result.selected_model, args.prompt)
                print(f"\n📝 RESPONSE:")
                print(response)
            except Exception as e:
                print(f"❌ Execution failed: {e}")
    
    # Run the CLI
    asyncio.run(main())
