# Ensure your dependencies are installed with:
# pip install openai weave

# Find your OpenAI API key at: https://platform.openai.com/api-keys
# Ensure that your OpenAI API key is available at:
# os.environ['OPENAI_API_KEY'] = "<your_openai_api_key>"

import os
import weave
from openai import OpenAI

# Find your wandb API key at: https://wandb.ai/authorize
weave.init('joy-bagchi-cloudpulse-innovations/intro-example')  # 🐝


@weave.op  # 🐝 Decorator to track requests
def create_completion(message: str) -> str:
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": message}
        ],
    )
    return response.choices[0].message.content


message = "What is Singular Value Decomposition?"
create_completion(message)
