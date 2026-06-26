# Ensure your dependencies are installed with:
# pip install openai pandas weave

# Find your OpenAI API key at: https://platform.openai.com/api-keys
# Ensure that your OpenAI API key is available at:
# os.environ['OPENAI_API_KEY'] = "<your_openai_api_key>"

import asyncio
import os
import re
from textwrap import dedent

import openai
import weave


class JsonModel(weave.Model):
    prompt: weave.Prompt = weave.StringPrompt(
        dedent("""
You are an assistant that answers questions about JSON data provided by the user. The JSON data represents structured information of various kinds, and may be deeply nested. In the first user message, you will receive the JSON data under a label called 'context', and a question under a label called 'question'. Your job is to answer the question with as much accuracy and brevity as possible. Give only the answer with no preamble. You must output the answer in XML format, between <answer> and </answer> tags.
""")
    )
    model: str = "gpt-4.1-nano"
    _client: openai.OpenAI

    def __init__(self):
        super().__init__()
        self._client = openai.OpenAI()

    @weave.op
    def predict(self, context: str, question: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.prompt.format()},
                {
                    "role": "user",
                    "content": f"Context: {context}\nQuestion: {question}",
                },
            ],
        )
        assert response.choices[0].message.content is not None
        return response.choices[0].message.content


@weave.op
def correct_answer_format(answer: str, output: str) -> dict[str, bool]:
    parsed_output = re.search(r"<answer>(.*?)</answer>", output, re.DOTALL)
    if parsed_output is None:
        return {"correct_answer": False, "correct_format": False}
    return {"correct_answer": parsed_output.group(1) == answer, "correct_format": True}


if __name__ == "__main__":
    if not os.environ.get('OPENAI_API_KEY'):
        print("OPENAI_API_KEY is not set - make sure to export it in your environment or assign it in this script")
        exit(1)

    # Find your wandb API key at: https://wandb.ai/authorize
    weave.init("joy-bagchi-cloudpulse-innovations/intro-example")

    jsonqa = weave.Dataset.from_uri(
        "weave:///wandb/json-qa/object/json-qa:v3"
    ).to_pandas()

    model = JsonModel()

    eval = weave.Evaluation(
        name="json-qa-eval",
        dataset=weave.Dataset.from_pandas(jsonqa),
        scorers=[correct_answer_format],
    )

    asyncio.run(eval.evaluate(model))