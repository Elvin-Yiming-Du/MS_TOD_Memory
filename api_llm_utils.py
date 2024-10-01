import os
from openai import OpenAI
import json
os.environ["OPENAI_API_KEY"] = "sk-v3vzSqMLo0TxJf77440c430e75B04a90A6D02fCb0506B0D4"
os.environ["OPENAI_BASE_URL"] = "https://xiaoai.plus/v1"
client = OpenAI(
  api_key=os.environ.get("OPENAI_API_KEY"),
  base_url=os.environ.get("OPENAI_BASE_URL"),
)

def gpt4o_generate(user_prompt, system_prompt="你是个评测助手，请根据用户要求返回分数0或者1。",
                   model_name = "gpt-4o-2024-05-13",
                   temperature = 0.9,
                   max_length = 4000):
    TRY_NUM = 5
    success = False

    for attempt in range(TRY_NUM):
        try:
            completion = get4_generate(
                model=model_name,
                temperature=temperature,
                max_tokens=max_length,
                top_p=0.95,
                messages=[{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
                ])
        except Exception as e:
            print(f"{e}")
        else:
            success = True
            break
        if success:
            result = completion.model_dump_json()
        else:
            result = ''
        response = json.loads(result)["choices"][0]["message"]["content"]
    return response



def get4mini_generate(user_prompt, system_prompt="你是个评测助手，请根据用户要求返回分数0或者1。",
                   model_name = "gpt-4o-mini",
                   temperature = 0.9,
                   max_length = 3000):
    TRY_NUM = 5
    success = False

    for attempt in range(TRY_NUM):
        try:
            completion = get4_generate(
                model=model_name,
                temperature=temperature,
                max_tokens=max_length,
                top_p=0.95,
                messages=[{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
                ])
        except Exception as e:
            print(f"{e}")
        else:
            success = True
            break
        if success:
            result = completion.model_dump_json()
        else:
            result = ''
        response = json.loads(result)["choices"][0]["message"]["content"]
    return response


if __name__ == '__main__':
    input_message = [{"role": "user", "content": "Hi"}]
    gpt4_generate(input_message)

