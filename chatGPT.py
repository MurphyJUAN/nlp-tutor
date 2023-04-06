# %%
import json
import numpy as np
from sklearn.model_selection import train_test_split
import requests
import time
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# %%
# Create 
# 用自己 OpenAI 的 account 創造 api_key 並置於此處
api_key = ''
# 放置 instruction prompt
system_prompt = "I will provide you with a presentation by a manager in an earning call conference. I would like you to play the role of several financial analysts participating in the Q&A section of the earning call conference. These analysts possess strong analytical skills that can delve deep into a company's financial data, business model, and market trends, as well as industry and market knowledge and a forward-thinking mindset. Based on the content of the presentation that I will provide you, please ask some professional questions that financial analysts would typically ask in the real world. For example: The 50,000 new users added for the CRM last quarter -- can give us a sense how much of that was hosted, how much of that was on premise?  What's going on there in the mix?"
# %%
result = []
# %%
# 放置 main prompt
user_prompt = ""
# %%
response = requests.post(
        'https://api.openai.com/v1/chat/completions',
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        },
        json = {
            'model': 'gpt-3.5-turbo', # 一定要用chat可以用的模型
            'messages' : [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        })
json = response.json()
# %%
print(response)