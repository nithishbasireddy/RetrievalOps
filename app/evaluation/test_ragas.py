from ragas import EvaluationDataset

sample = EvaluationDataset.from_list(
    [
        {
            "user_input": "Who owns intellectual property created during employment?",
            "response": "The employer owns intellectual property created during employment.",
            "reference": "The employer owns intellectual property, inventions, designs, source code, software, and other work products created by the employee during the course of employment."
        }
    ]
)

print("Ragas installed successfully")
print(sample)