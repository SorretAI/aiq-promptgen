# Add to prompt_optimizer.py
if __name__ == "__main__":
    import argparse
    
    def main():
        parser = argparse.ArgumentParser(description="SorretAI Prompt Optimizer")
        parser.add_argument("--prompt", required=True, help="Prompt to optimize")
        parser.add_argument("--task", default="content_creation", 
                          choices=["social_media", "content_creation", "prospecting", "video_script"])
        parser.add_argument("--audience", default="", help="Target audience")
        parser.add_argument("--level", default="STANDARD", choices=["BASIC", "STANDARD", "ADVANCED"])
        
        args = parser.parse_args()
        
        optimizer = SorretPromptOptimizer()
        result = optimizer.optimize_prompt(
            original_prompt=args.prompt,
            task_type=TaskType(args.task),
            target_audience=args.audience,
            optimization_level=OptimizationLevel[args.level]
        )
        
        print("ORIGINAL:", result.original_prompt)
        print("OPTIMIZED:", result.optimized_prompt)
        print("MODEL USED:", result.model_used)
    
    main()