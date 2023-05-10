from flask import Flask, render_template, request
from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI
import os

app = Flask(__name__)

# Retrieve your OpenAI API key from the environment variable
openai_api_key = os.getenv("secretopenai")

def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 1024
    max_chunk_overlap = 20
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=num_outputs, api_key=openai_api_key))

    documents = SimpleDirectoryReader(directory_path).load_data()

    index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    index.save_to_disk('index.json')

    return index

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_text = request.form['user_input']
        index = GPTSimpleVectorIndex.load_from_disk('index.json')
        response = index.query(input_text, response_mode="compact")
        return render_template('index.html', response=response.response,user_input=input_text)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    index = construct_index("docs")
    app.run(debug=True)
