import openai ,os
from dotenv import load_dotenv

load_dotenv('.env')

# Set your OpenAI API key
openai.api_key = os.environ['api_key']

paragraph = ""

# Your paragraph
# paragraph = """
# OpenAI is an AI research and deployment company. Our mission is to ensure that artificial general intelligence benefits all of humanity. OpenAI will build safe and beneficial AGI or help others achieve this outcome.
# """
def extract_text_from_srt(file_path):
    global paragraph
    paragraph = ""
    with open(os.path.join(os.getcwd(),'audio-text') +f"/{file_path}.srt", 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            # Skip timecodes and empty lines
            if not line.isdigit() and '-->' not in line and line:
                paragraph += line + " "
    return paragraph.strip()

def chat_(path_,question):
    global paragraph
    # Example usage
    srt_file_path = path_
    paragraph = extract_text_from_srt(srt_file_path)

    # Define the question you want to ask
    # question = "What is the context here?"

    # Construct the messages for the ChatCompletion
    messages = [
        # {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Here is a paragraph:\n\n{paragraph}\n\nQuestion: {question}\nAnswer:"}
    ]

    # Make a request to OpenAI API
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",  # or "gpt-4" if available
        messages=messages,
        max_tokens=100,  # Adjust as needed
        temperature=0.5
    )

    # Get the answer from the response
    answer = response.choices[0].message.content.strip()

    # Print the answer
    # print(f"Question: {question}\nAnswer: {answer}")
    return f"Question: {question}\nAnswer: {answer}"
