from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    temperature=0.7,
    google_api_key = api_key
)

while True:
    prompt = input("\nYou: ")

    if prompt.lower() == "exit":
        print("Goodbye!")
        break

    response = llm.invoke(prompt)

    print("\nAI:", response.content)

template = "Give me 3 career skills that are in high demand in {year}."
prompt_template = PromptTemplate.from_template(template)

parser = StrOutputParser()
chain = prompt_template | llm | parser
response = chain.invoke({"year": "2026"})
print("\n Career Skills in 2026:\n", response)
