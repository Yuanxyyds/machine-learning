from unsloth import FastLanguageModel
from config import Config
from transformers import TextStreamer

# Load the model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/root/machinelearning/llama3.2/llama3.2-3B-Instruct/model",
    max_seq_length=Config.MAX_SEQ_LENGTH,
    load_in_4bit=False,
)
FastLanguageModel.for_inference(model)
text_streamer = TextStreamer(tokenizer)
base_info = {"role": "system", "content": "You are Hongyuan (Steven) Liu, and you should answer questions about yourself"}


# Function to get model response
def get_response(input_message):
    messages = {"role": "user", "content": input_message}
    inputs = tokenizer.apply_chat_template(
        [ messages],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    outputs = model.generate(
        input_ids=inputs, streamer=text_streamer, max_new_tokens=1000, use_cache=True
    )



# Main loop for continuous conversation
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Conversation ended.")
        break

    # Get the model's response
    print("Assistant: ", end="")
    get_response(user_input)
