# Import necessary libraries
from gradio_client import Client  # Import the Gradio Client for making predictions
import time  # Import the time module for performance measurement

# Initialize the Gradio Client with the provided API endpoint
gradio_client = Client("https://osanseviero-mistral-super-fast.hf.space/")

# Main loop for continuous user interaction
while True:
    # Get user input for generating a chat response
    user_query = input("ENTER YOUR MESSAGE\n")

    # Record the start time for performance measurement
    start_time = time.time()

    # Make a prediction using the Gradio Client for chat generation
    chat_response = gradio_client.predict(
        user_query , # Default System Command / Custom Instruction
        0.8,          # Control the randomness of the response
        1024,          # Limit the length of the generated response
        0.9,                # Set a probability threshold for word selection
        1.1,   # Penalize repeated phrases in the response
        api_name="/chat"
    )

    # Record the end time for performance measurement
    end_time = time.time()

    # Display the generated chat response and time taken to respond
    print(f"CHAT RESPONSE: {chat_response[:-4]}\nTIME TAKEN TO RESPOND: {round(end_time - start_time, 2)} sec")

