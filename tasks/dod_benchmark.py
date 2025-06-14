from inspect_ai import Task, task, eval
from inspect_ai.solver import system_message, generate
from inspect_ai.scorer import model_graded_fact
from inspect_ai.dataset import Sample
import pandas as pd
import os

SYSTEM_MESSAGE = """You are an AI assistant designed to assist with answering Department of Defense policy and operations questions.

## Response instructions
Please answer questions factually and provide a clear answer alongside relevant reasoning. If you're not sure, please say so — do not try to make up an answer.
"""

def load_dod_benchmark():
    df = pd.read_csv("../datasets/dod_benchmark.csv")
    print(f"Loading {len(df)} samples from dod_benchmark.csv")
    
    samples = []
    for _, row in df.iterrows():
        samples.append(
            Sample(
                input=row["question"],
                target=row["answer"],
                metadata={
                    "directorate": row.get("directorate", "unknown"),
                    "question_type": row.get("question_type", "unknown")
                }
            )
        )
    
    return samples

@task
def dodbench_eval():
    return Task(
        dataset=load_dod_benchmark(),
        solver=[
            system_message(SYSTEM_MESSAGE),
            generate(max_tokens=500),  # Reasonable limit for DoD policy answers
        ],
        scorer=model_graded_fact(model="openai/gpt-4o-mini")
    )

if __name__ == "__main__":
    print("Starting evaluation...")
    
    # Run the full evaluation
    results = eval(
        dodbench_eval(),
        model=os.environ.get("INSPECT_EVAL_MODEL", "anthropic/claude-3-7-sonnet-20250219"),
        log_dir="./eval_logs",
        # Remove limit to run on all 60 samples
    )
    
    # Print results
    print(f"\n{'='*50}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*50}")
    print(f"Total Samples: {results.total}")
    print(f"Completed: {results.completed}")
    print(f"Failed: {results.total - results.completed}")
    print(f"Mean Score: {results.mean:.2%}")
    print(f"{'='*50}")