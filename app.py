import os
import traceback

import pandas as pd
import streamlit as st
import networkx as nx
from dotenv import load_dotenv
from src.embeddings import embedding_factory
from src.llms import llm_factory
from src.loaders import *
from src.graph_processor.processor import (
    create_rag,
    insert_document,
    query_rag
)
from src.graph_processor.utils import graphml_to_json

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="GraphRAG Explorer",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if "rag_model" not in st.session_state:
    st.session_state["rag_model"] = None
if "full_content" not in st.session_state:
    st.session_state["full_content"] = None
if "chart" not in st.session_state:
    st.session_state["chart"] = False
if "graph" not in st.session_state:
    st.session_state["graph"] = None

# Title and description
st.title("GraphRAG Explorer")
st.markdown("Explore and visualize Graph-based Retrieval-Augmented Generation. More detail: [here](https://github.com/KKenny0/GraphRAGxplorer)")

# File upload section
file_sec_1, file_sec_2 = st.columns([1.5, 2])

file_sec_1.header("Upload Your Data")
uploaded_file = file_sec_1.file_uploader(
    "Choose a file (DOCX, TXT, or PDF)",
    type=['docx', 'txt', 'pdf'],
    help="Upload your document or dataset for analysis"
)

# Data preview section
if uploaded_file is not None:
    file_sec_2.subheader("Data Preview")

    # Get file extension
    file_extension = uploaded_file.name.split('.')[-1].lower()
    try:
        loader = loader_factory.get_loader(uploaded_file.name)
        data = loader.load_data(uploaded_file)
        st.session_state["full_content"] = "\n".join(data)
        file_sec_2.text_area(
            "Text Content Preview",
            st.session_state["full_content"][:1000] + ("..." if len(st.session_state["full_content"]) > 1000 else ""),
            height=200
        )

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

st.divider()

# Sidebar for configurations
with st.sidebar:
    st.header("Configuration")
    
    # 1. Embedding Service Configuration
    st.subheader("1. Embedding Service")
    embedding_service = st.selectbox(
        "Select Embedding Service",
        ["OpenAI", "Sentence Transformers", "Ollama"]
    )
    
    if embedding_service == "OpenAI":
        st.session_state["openai_api_base"] = st.text_input("Enter API Base")
        st.session_state["openai_api_key"] = st.text_input("Enter API Key")
        embedding_model = st.text_input("Enter Model")
        if st.session_state["openai_api_base"]:
            embedding_factory.create(
                "openai",
                api_base=st.session_state["openai_api_base"],
                api_key=st.session_state["openai_api_key"],
                model_name=embedding_model,
            )

    elif embedding_service == "Sentence Transformers":
        embedding_model = st.selectbox(
            "Select Model",
            ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]
        )
        embedding_factory.create(
            "sentence-transformer",
            model_name=embedding_model
        )
    else:
        st.session_state["ollama_host"] = st.text_input("Enter server host", value="http://localhost:11434")
        embedding_model = st.selectbox(
            "Select Model",
            ["nomic-embed-text", "mxbai-embed-large", "snowflake-arctic-embed", "bge-m3", "bge-large"]
        )
        embedding_factory.create(
            "ollama",
            model_name=embedding_model,
            host=st.session_state["ollama_host"]
        )
    
    # 2. LLM Service Configuration
    st.subheader("2. LLM Service")
    llm_type = st.selectbox(
        "Select LLM Service",
        ["OpenAI", "Ollama"]
    )
    
    if llm_type == "OpenAI":
        if not st.session_state["openai_api_key"]:
            st.session_state["openai_api_key"] = st.text_input("Enter OpenAI API key")
        if not st.session_state["openai_api_base"]:
            st.session_state["openai_api_base"] = st.text_input("Enter OpenAI API base URL")
        llm_model = st.text_input("Enter OpenAI model", value="gpt-3.5-turbo")
        llm_factory.register_llm(
            llm_type="openai",
            model=llm_model,
            api_key=st.session_state["openai_api_key"],
            base_url=st.session_state["openai_api_base"]
        )

    else:
        if "ollama_host" not in st.session_state:
            st.session_state["ollama_host"] = st.text_input("Enter Ollama host", value="http://localhost:11434")
        llm_model = st.text_input("Enter Ollama model", value="qwen2.5:14b-instruct-q4_K_M")
        llm_factory.register_llm(
            llm_type="ollama",
            model=llm_model,
            host=st.session_state["ollama_host"]
        )

    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    llm_factory.update_config(temperature=temperature)

    # 3. GraphRAG Method Selection
    st.subheader("3. GraphRAG Method")
    rag_type = st.selectbox(
        "Select Graph Method",
        ["GraphRAG", "LightRAG"]
    )

    if st.button("Build Graph DB"):
        st.session_state["graph"] = None
        with st.spinner("Building Vector DB"):
            try:
                # Create RAG model
                st.session_state["rag_model"] = create_rag(rag_type=rag_type)

                # Insert document if content is available
                if st.session_state["full_content"]:
                    insert_document(st.session_state["rag_model"], st.session_state["full_content"])
                    st.success("Graph DB built successfully!")
                else:
                    st.warning("Please upload a document first.")

            except Exception as e:
                st.error(f"Error building Graph DB: {str(e)}. Details will be toasted.")
                st.toast(f"{traceback.format_exc()}", icon="⚠️")

            st.session_state["chart"] = True

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Graph Data Preview")

    graph_data_file = list(filter(lambda x: ".graphml" in x, os.listdir("graphrag_cache")))
    # Placeholder for graph visualization
    if st.session_state["chart"] and graph_data_file:
        st.session_state["graph"], graph_data = graphml_to_json("graphrag_cache/{}".format(graph_data_file[0]))

        node_df = pd.DataFrame(graph_data.get("nodes", []))
        edge_df = pd.DataFrame(graph_data.get("links", []))

        if len(node_df):
            del node_df["source_id"]
            if "clusters" in node_df.columns:
                del node_df["clusters"]
            st.dataframe(node_df)

        if len(edge_df):
            del edge_df["source_id"]
            del edge_df["order"]
            st.dataframe(edge_df)

with col2:
    # Statistics
    if st.session_state["rag_model"] is not None and st.session_state["graph"]:
        st.write("Node Statistics:")
        st.metric("Total Nodes", len(st.session_state["graph"].nodes()))
        st.metric("Total Edges", len(st.session_state["graph"].edges()))

        st.write("Graph Properties:")
        st.json({
            "Average Degree": sum(dict(st.session_state["graph"].degree()).values()) / len(st.session_state["graph"]),
            "Density": nx.density(st.session_state["graph"]),
            "Is Connected": nx.is_connected(st.session_state["graph"])
        })

    st.subheader("Query and Results")
    
    # Query input
    query = st.text_input("Enter your query")
    if st.button("Submit Query") and query:
        if st.session_state["rag_model"] is not None:
            with st.spinner("Processing query..."):
                try:
                    _, answer = query_rag(
                        st.session_state["rag_model"],
                        query,
                        rag_type=rag_type
                    )
                    st.write("Answer:", answer)
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
        else:
            st.warning("Please build the Graph DB first.")

# Footer
st.markdown("---")
st.markdown("GraphRAG Explorer - Powered by Streamlit")
