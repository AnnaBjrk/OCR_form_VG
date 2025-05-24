import requests
import json
import base64

# Convert image to base64
with open("handwritten_form.jpg", "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode('utf-8')

# API call with token limiting
response = requests.post(
    url="https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": "Bearer <OPENROUTER_API_KEY>",
        "Content-Type": "application/json",
    },
    data=json.dumps({
        "model": "anthropic/claude-3-haiku-20240307",  # A good model for vision tasks
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Interpret the handwritten text in this form:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        "max_tokens": 250,   # Limit response to about 200-250 words
        "temperature": 0.3   # Lower temperature for more deterministic/accurate results
    })
)

# Parse the response
result = response.json()
interpretation = result['choices'][0]['message']['content']
print(interpretation)