import os
import bs4
from dotenv import load_dotenv
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# ✅ Load API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# ✅ Input URLs
urls = [
    "https://extension.umn.edu/plant-diseases/apple-scab",
    "https://extension.umn.edu/plant-diseases/black-rot-apple",
]

# ✅ Step 1: Load and split content
def load_and_split(urls):
    all_chunks = []
    for url in urls:
        loader = WebBaseLoader(
            web_paths=(url,),
            bs_kwargs={"parse_only": bs4.SoupStrainer(name=["p", "h1", "h2", "h3"])}
        )
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        all_chunks.extend(chunks)
    return all_chunks

# ✅ Step 2: Create or load Chroma vectorstore
VECTORSTORE_PATH = "chroma_store"
def build_vectorstore(docs):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    if os.path.exists(VECTORSTORE_PATH):
        print("📂 Loading existing vectorstore...")
        return Chroma(persist_directory=VECTORSTORE_PATH, embedding_function=embeddings)
    else:
        print("🆕 Creating new vectorstore...")
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=VECTORSTORE_PATH
        )
        vectorstore.persist()
        return vectorstore

# ✅ Step 3: Create RAG chain
def build_rag_chain(vectorstore):
    llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0.3)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
    )

# ✅ Step 4: Run it
docs = load_and_split(urls)
vectorstore = build_vectorstore(docs)
rag_chain = build_rag_chain(vectorstore)

print("✅ RAG chatbot ready. Type your question or 'exit' to quit.\n")
while True:
    query = input("👨‍🌾 You: ")
    if query.lower() in ["exit", "quit"]:
        print("👋 Goodbye!")
        break
    try:
        response = rag_chain({"question": query})
        print("🤖 Bot:", response["answer"])
    except Exception as e:
        print("❌ Error:", str(e))
