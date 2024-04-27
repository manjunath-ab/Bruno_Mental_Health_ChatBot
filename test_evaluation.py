from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.metrics import BiasMetric, AnswerRelevancyMetric
from deepeval.metrics import HallucinationMetric
from pathlib import Path
import json
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ToxicityMetric
import os

def load_environment_variables():
    dotenv_path = Path('/Users/abhis/.env')
    load_dotenv(dotenv_path=dotenv_path)

load_environment_variables()
for filename in os.listdir('C:/Users/abhis/Desktop/AI4MentalHealth/pipelines/chat_evaluation'):
    file_path = os.path.join('C:/Users/abhis/Desktop/AI4MentalHealth/pipelines/chat_evaluation', filename)
    if os.path.isfile(file_path):
        # Process each file
        with open(file_path, 'r') as file:
            data = json.load(file)
            test_cases = []
            human_inputs = data["human"]
            ai_outputs = data["ai"]

            for i in range(len(human_inputs)):
                    human_input = human_inputs[i]
                    ai_output = ai_outputs[i]
                    retrieval_context = None
                    context = ["A chatbot trying to be a friend and understanding the user's mental state and providing relief"]

                    test_case = LLMTestCase(
                        input=human_input,
                        actual_output=ai_output,
                        expected_output= None,  # Assuming the AI output is the expected output
                        context=context,
                    )
                    test_cases.append(test_case)

answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5, model="gpt-3.5-turbo-0125")
hallucination_metric = HallucinationMetric(threshold=0.5, model="gpt-3.5-turbo-0125")
bias_metric = BiasMetric(threshold=0.5, model="gpt-3.5-turbo-0125")
toxicity_metric = ToxicityMetric(threshold=0.5, model="gpt-3.5-turbo-0125")

# Evaluate the test cases
results = evaluate(
    test_cases, [answer_relevancy_metric, hallucination_metric, bias_metric, toxicity_metric]
)