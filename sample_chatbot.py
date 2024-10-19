
import os
from llama_index.core import PromptTemplate, SimpleDirectoryReader, VectorStoreIndex, Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.ollama import OllamaEmbedding
from langchain_community.llms import Ollama
import streamlit as st

# Set the embedding model
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# Define the path for saving/loading the index
persist_dir = "./saved_index"

# Define the path where you saved "Documents"
file_dir = '/Users/kaspiper/Documents/GitHub/osu_eorgs_LLM/'


def create_or_load_index(force_rebuild=False):
    # Check if the index already exists
    if os.path.exists(persist_dir) and not force_rebuild:
        print("Loading existing index...")
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)
    else:
        print("Creating new index...")
        # Load documents from both directories
        smart_columbus_loader = SimpleDirectoryReader(
            input_dir=file_dir+'Documents/smart-columbus',
            required_exts=[".pdf"],
            recursive=True
        )
        ohio_tech_news_loader = SimpleDirectoryReader(
            input_dir=file_dir+'Documents/ohio-tech-news',
            required_exts=[".md"],
            recursive=True
        )
        # Combine documents from both loaders
        documents = smart_columbus_loader.load_data() + ohio_tech_news_loader.load_data()

        # Create the index
        index = VectorStoreIndex.from_documents(documents)

        # Save the index
        index.storage_context.persist(persist_dir=persist_dir)
        print(f"Index saved to {persist_dir}")

    return index

# Set up the language model
llm = Ollama(model="llama3.2")

# Define the QA prompt template
qa_prompt_tmpl_str = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information above I want you to think step by step to answer the query in a crisp manner, incase case you don't know the answer say 'I don't know!'.\n"
    "Query: {query_str}\n"
    "Answer: "
)
qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

# Create or load the index
index = create_or_load_index()

# Create the query engine
query_engine = index.as_query_engine(similarity_top_k=4, llm=llm)

# Optional to speed things up
# os.environ["OPENAI_API_KEY"] = 'YOUR_API_KEY_HERE'
# query_engine = index.as_query_engine(streaming=True, similarity_top_k=4)

query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt_tmpl})

# Streamlit app
st.set_page_config(page_title="Columbus Chatbot", page_icon="🤖")

# Placeholder for logo
st.image("https://images.cvent.com/e95086fd534542bfbecbef975cc27526/pix/91275ab5fb4e4680822acfcef4a5019a!_!eb8f5813464ff4cb52a7ac51a76f161c.jpg?f=webp")
st.image("https://engineering.osu.edu/themes/custom/osu_kinetic/images/logo.png")
st.title("Columbus Chatbot")

# User input
user_question = st.text_input("Ask a question about Columbus:")

if user_question:
    with st.spinner("Thinking..."):
        response = query_engine.query(user_question)
        st.write(str(response))

st.sidebar.markdown("This chatbot provides information about Columbus using Smart Columbus and OhioX's Ohio Tech News, built during Construct I/O.")


