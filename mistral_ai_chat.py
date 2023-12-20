from huggingface_hub import InferenceClient  # Importing the InferenceClient class from huggingface_hub
import random  # Importing the random module for generating random numbers
import time  # Importing the time module for measuring time
import api_key  # Importing the 'api_key' module to access the API key

# Setting the API URL for model inference
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"  # FASTER BUT LESS STABLE
# API_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1" #QUITE SLOWER BUT MEMORY EFFICIENT AND MORE STABLE

# # Replace YOUR_API_KEY_HERE with the obtained API key from Hugging Face
headers = {"Authorization": F"Bearer {api_key.API_KEY}"}
# Setting up the authorization header with the provided API key

# Function to format prompt
def format_prompt(message, custom_instructions=None):
    # Defining a function to format the prompt with optional custom instructions
    prompt = ""
    if custom_instructions:
        prompt += f"[INST] {custom_instructions} [/INST]"  # Adding custom instructions to the prompt

    prompt += f"[INST] {message} [/INST]"  # Adding the main message to the prompt
    return prompt
# Function to generate response based on user input
def generate(prompt, temperature=0.9, max_new_tokens=512, top_p=0.95, repetition_penalty=1.0):

    # Setting up keyword arguments for text generation
    temperature = float(temperature)  # Convert temperature to a float for precise control
    if temperature < 1e-2:
        temperature = 1e-2  # Ensure a minimum temperature to avoid division by zero issues

    top_p = float(top_p)  # Convert top_p to a float for precise control

    # Creating a dictionary of keyword arguments for text generation
    generate_kwargs = dict(
        temperature=temperature,  # Set temperature for controlling randomness in text generation
        max_new_tokens=max_new_tokens,  # Set the maximum number of new tokens to generate
        top_p=top_p,  # Set top_p for nucleus sampling to control diversity in generated text
        repetition_penalty=repetition_penalty,  # Set repetition penalty to discourage repeated phrases
        do_sample=True,  # Enable sampling to introduce randomness in text generation
        seed=random.randint(0, 10**7),  # Set a random seed for reproducibility
    )


    custom_instructions = "Demonstrate a personable and witty nature during the conversation, showcasing humor and wit. Act As like you are a person in the Conversation"
    formatted_prompt = format_prompt(prompt, custom_instructions)
    # Formatting the user prompt with custom instructions

    client = InferenceClient(API_URL, headers=headers)
    # Creating an instance of InferenceClient for model inference
    response = client.text_generation(formatted_prompt, **generate_kwargs)
    # Generating a text response using the model

    return response

print("MODEL HAS BEEN STARTED..............")
while True:
    # Get user input
    user_prompt = input("You: ")

    start = time.time()
    # Generate a response based on user input
    generated_text = generate(user_prompt)
    end = time.time()
    print("Bot:", generated_text)
    print(f"TIME TAKEN TO RESPOND: {end-start}")
    # Printing the generated response and the time taken to respond
    