import torch
from transformers import pipeline

# Load the model
model_id = "/root/machinelearning/llama3.2/llama3.2-3B-Instruct/full-model"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Initialize the conversation with a system prompt
conversation_history = [
    {"role": "system", "content": "You are Hongyuan (Steven) Liu and you will answer some questions about yourself."}
]


# Loop to simulate multi-turn conversation
while True:
    # User input
    user_input = input("User: ")
    
    # Add user message to conversation history
    conversation_history.append({"role": "user", "content": user_input})
    
    # Generate response from the model
    outputs = pipe(conversation_history, max_new_tokens=1000, pad_token_id=50256)
    
    # Decode the generated tokens
    generated_text = outputs[0]['generated_text'][-1]['content'].strip()
    
    # Add the assistant's response to the conversation history
    conversation_history.append({"role": "assistant", "content": generated_text})
    
    # Print the assistant's response
    print(f"Assistant: {generated_text}\n")
