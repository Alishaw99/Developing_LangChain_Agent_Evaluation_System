from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()
model = init_chat_model(model="gpt-6-luna")

# response = model.invoke("tell me something about the sun in two sentences?")

# print(response)