# SorretAI Prompt Optimizer
# Based on Promptify + LangChain + Custom Enhancements
# Integrates with your local Ollama models

import json
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import hashlib
from dataclasses import dataclass, asdict
import ollama
from enum import Enum

class TaskType(Enum):
    SOCIAL_MEDIA = "social_media"
    CONTENT_CREATION = "content_creation"
    PROSPECTING = "prospecting"
    EMAIL = "email"
    VIDEO_SCRIPT = "video_script"
    PHONE_SCRIPT = "phone_script"

class OptimizationLevel(Enum):
    BASIC = "basic"           # TinyLlama
    STANDARD = "standard"     # Gemma:2b
    ADVANCED = "advanced"     # Phi-3-mini

@dataclass
class PromptTemplate:
    """Enhanced prompt template with versioning and optimization tracking"""
    id: str
    original_prompt: str
    optimized_prompt: str
    task_type: TaskType
    optimization_level: OptimizationLevel
    model_used: str
    timestamp: str
    performance_score: Optional[float] = None
    token_savings: Optional[int] = None
    success_metrics: Optional[Dict] = None

class SorretPromptOptimizer:
    """
    Main prompt optimizer class
    Combines best practices from Promptify, LangChain, and custom enhancements
    """
    
    def __init__(self):
        self.ollama_client = ollama.Client()
        self.prompt_cache = {}
        self.optimization_history = []
        self.task_templates = self._load_task_templates()
        
    def _load_task_templates(self) -> Dict:
        """Load predefined templates for different content types"""
        return {
            TaskType.SOCIAL_MEDIA: {
                "structure": "Context + Hook + Value + CTA",
                "tone_keywords": ["engaging", "conversational", "compelling"],
                "constraints": ["character_limits", "hashtag_placement", "mention_guidelines"]
            },
            TaskType.CONTENT_CREATION: {
                "structure": "Introduction + Main Points + Conclusion",
                "tone_keywords": ["informative", "authoritative", "accessible"],
                "constraints": ["SEO_requirements", "keyword_density", "readability"]
            },
            TaskType.PROSPECTING: {
                "structure": "Personalization + Value Prop + Clear Ask",
                "tone_keywords": ["professional", "personalized", "value-focused"],
                "constraints": ["brevity", "specificity", "call_to_action"]
            },
            TaskType.VIDEO_SCRIPT: {
                "structure": "Hook + Problem + Solution + CTA",
                "tone_keywords": ["visual", "narrative", "engaging"],
                "constraints": ["timing", "visual_cues", "pacing"]
            }
        }
    
    def optimize_prompt(
        self, 
        original_prompt: str, 
        task_type: TaskType,
        target_audience: str = "",
        constraints: List[str] = None,
        optimization_level: OptimizationLevel = OptimizationLevel.STANDARD
    ) -> PromptTemplate:
        """
        Main optimization function
        """
        # 1. Analyze the original prompt
        analysis = self._analyze_prompt(original_prompt, task_type)
        
        # 2. Select appropriate Ollama model based on optimization level
        model = self._select_model(optimization_level)
        
        # 3. Build optimization prompt using our templates
        optimization_prompt = self._build_optimization_prompt(
            original_prompt, 
            task_type, 
            target_audience, 
            constraints,
            analysis
        )
        
        # 4. Generate optimized prompt using local Ollama
        optimized_prompt = self._generate_with_ollama(optimization_prompt, model)
        
        # 5. Post-process and validate
        final_prompt = self._post_process_prompt(optimized_prompt, task_type)
        
        # 6. Create prompt template object
        template = PromptTemplate(
            id=self._generate_id(original_prompt),
            original_prompt=original_prompt,
            optimized_prompt=final_prompt,
            task_type=task_type,
            optimization_level=optimization_level,
            model_used=model,
            timestamp=datetime.now().isoformat()
        )
        
        # 7. Cache and track
        self._cache_prompt(template)
        self.optimization_history.append(template)
        
        return template
    
    def _analyze_prompt(self, prompt: str, task_type: TaskType) -> Dict:
        """Analyze prompt structure and identify improvement areas"""
        analysis = {
            "word_count": len(prompt.split()),
            "sentence_count": len(re.split(r'[.!?]+', prompt)),
            "clarity_score": self._calculate_clarity_score(prompt),
            "specificity_score": self._calculate_specificity_score(prompt),
            "task_alignment": self._check_task_alignment(prompt, task_type),
            "missing_elements": self._identify_missing_elements(prompt, task_type)
        }
        return analysis
    
    def _select_model(self, optimization_level: OptimizationLevel) -> str:
        """Select appropriate Ollama model based on optimization level"""
        model_mapping = {
            OptimizationLevel.BASIC: "tinyllama",
            OptimizationLevel.STANDARD: "gemma:2b", 
            OptimizationLevel.ADVANCED: "phi3:mini"
        }
        return model_mapping[optimization_level]
    
    def _build_optimization_prompt(
        self, 
        original_prompt: str, 
        task_type: TaskType,
        target_audience: str,
        constraints: List[str],
        analysis: Dict
    ) -> str:
        """Build the prompt that will be sent to Ollama for optimization"""
        
        task_template = self.task_templates[task_type]
        
        optimization_prompt = f"""
You are an expert prompt engineer specializing in {task_type.value} content.

ORIGINAL PROMPT:
{original_prompt}

TASK TYPE: {task_type.value}
TARGET AUDIENCE: {target_audience or "General audience"}
REQUIRED STRUCTURE: {task_template['structure']}
TONE KEYWORDS: {', '.join(task_template['tone_keywords'])}

CURRENT ANALYSIS:
- Clarity Score: {analysis['clarity_score']}/10
- Specificity Score: {analysis['specificity_score']}/10
- Missing Elements: {', '.join(analysis['missing_elements'])}

CONSTRAINTS:
{chr(10).join(f"- {constraint}" for constraint in (constraints or []))}

OPTIMIZATION GOALS:
1. Improve clarity and specificity
2. Add missing structural elements
3. Optimize for {task_type.value} best practices
4. Reduce token count while maintaining effectiveness
5. Ensure actionable and measurable outcomes

Please rewrite the prompt to be:
- More specific and actionable
- Better structured for the task type
- Optimized for token efficiency
- Clearer in intent and expected output

OPTIMIZED PROMPT:
"""
        return optimization_prompt
    
    def _generate_with_ollama(self, optimization_prompt: str, model: str) -> str:
        """Generate optimized prompt using local Ollama"""
        try:
            response = self.ollama_client.generate(
                model=model,
                prompt=optimization_prompt,
                options={
                    'temperature': 0.3,  # Lower temperature for more consistent optimization
                    'top_p': 0.9,
                    'max_tokens': 500
                }
            )
            return response['response'].strip()
        except Exception as e:
            print(f"Error with Ollama generation: {e}")
            return self._fallback_optimization(optimization_prompt)
    
    def _post_process_prompt(self, optimized_prompt: str, task_type: TaskType) -> str:
        """Clean up and validate the optimized prompt"""
        # Remove any meta-commentary from the model
        lines = optimized_prompt.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith(('OPTIMIZED PROMPT:', 'Here is', 'The optimized')):
                cleaned_lines.append(line)
        
        final_prompt = '\n'.join(cleaned_lines).strip()
        
        # Validate prompt meets minimum requirements
        if len(final_prompt.split()) < 10:
            return self._fallback_optimization(optimized_prompt)
            
        return final_prompt
    
    def _calculate_clarity_score(self, prompt: str) -> float:
        """Calculate clarity score based on various metrics"""
        # Simple heuristic - can be enhanced
        word_count = len(prompt.split())
        sentence_count = len(re.split(r'[.!?]+', prompt))
        
        if sentence_count == 0:
            return 1.0
            
        avg_words_per_sentence = word_count / sentence_count
        
        # Ideal: 10-20 words per sentence
        if 10 <= avg_words_per_sentence <= 20:
            clarity_score = 10.0
        else:
            clarity_score = max(1.0, 10.0 - abs(avg_words_per_sentence - 15) * 0.5)
            
        return min(10.0, clarity_score)
    
    def _calculate_specificity_score(self, prompt: str) -> float:
        """Calculate how specific and actionable the prompt is"""
        specificity_indicators = [
            'specific', 'exactly', 'precisely', 'must include', 'format:',
            'example:', 'structure:', 'tone:', 'audience:', 'goal:', 'outcome:'
        ]
        
        prompt_lower = prompt.lower()
        score = sum(1 for indicator in specificity_indicators if indicator in prompt_lower)
        
        return min(10.0, score * 2)  # Scale to 0-10
    
    def _check_task_alignment(self, prompt: str, task_type: TaskType) -> float:
        """Check how well the prompt aligns with the task type"""
        task_keywords = {
            TaskType.SOCIAL_MEDIA: ['post', 'social', 'engage', 'share', 'hashtag'],
            TaskType.CONTENT_CREATION: ['article', 'blog', 'content', 'write', 'create'],
            TaskType.PROSPECTING: ['prospect', 'lead', 'sales', 'outreach', 'contact'],
            TaskType.VIDEO_SCRIPT: ['script', 'video', 'scene', 'visual', 'narrator']
        }
        
        keywords = task_keywords.get(task_type, [])
        prompt_lower = prompt.lower()
        
        alignment_score = sum(1 for keyword in keywords if keyword in prompt_lower)
        return min(10.0, alignment_score * 2)
    
    def _identify_missing_elements(self, prompt: str, task_type: TaskType) -> List[str]:
        """Identify missing elements based on task type"""
        required_elements = {
            TaskType.SOCIAL_MEDIA: ['target_audience', 'platform', 'call_to_action'],
            TaskType.CONTENT_CREATION: ['topic', 'format', 'tone', 'length'],
            TaskType.PROSPECTING: ['personalization', 'value_proposition', 'next_step'],
            TaskType.VIDEO_SCRIPT: ['hook', 'duration', 'visual_elements']
        }
        
        elements = required_elements.get(task_type, [])
        prompt_lower = prompt.lower()
        
        missing = []
        for element in elements:
            if element.replace('_', ' ') not in prompt_lower and element not in prompt_lower:
                missing.append(element)
                
        return missing
    
    def _generate_id(self, prompt: str) -> str:
        """Generate unique ID for the prompt"""
        return hashlib.md5(prompt.encode()).hexdigest()[:8]
    
    def _cache_prompt(self, template: PromptTemplate):
        """Cache the prompt template"""
        self.prompt_cache[template.id] = template
    
    def _fallback_optimization(self, original: str) -> str:
        """Fallback optimization if Ollama fails"""
        return f"Please {original.lower().strip('.')}. Be specific and actionable in your response."
    
    def get_optimization_history(self) -> List[PromptTemplate]:
        """Get history of all optimizations"""
        return self.optimization_history
    
    def get_prompt_by_id(self, prompt_id: str) -> Optional[PromptTemplate]:
        """Retrieve cached prompt by ID"""
        return self.prompt_cache.get(prompt_id)
    
    def export_prompts(self, filepath: str):
        """Export all prompts to JSON file"""
        export_data = [asdict(template) for template in self.optimization_history]
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)

# Usage Example
if __name__ == "__main__":
    optimizer = SorretPromptOptimizer()
    
    # Example: Optimize a social media prompt
    original = "Write a post about AI"
    
    optimized_template = optimizer.optimize_prompt(
        original_prompt=original,
        task_type=TaskType.SOCIAL_MEDIA,
        target_audience="Tech entrepreneurs",
        constraints=["280 characters", "include hashtags", "engaging hook"],
        optimization_level=OptimizationLevel.STANDARD
    )
    
    print("ORIGINAL:", original)
    print("OPTIMIZED:", optimized_template.optimized_prompt)
    print("MODEL USED:", optimized_template.model_used)
