import os
import string
import random
import requests
import pandas as pd
import fitz
import json
from openai import OpenAI
from pinecone import Pinecone
from typing import Optional, List, Dict, Any, Generator
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import streamlit as st

load_dotenv()


OPENAI_KEY = os.getenv("OPENAI_KEY")
PINE_KEY = os.getenv('PINE_KEY')
pc = Pinecone(api_key=PINE_KEY)
client = OpenAI(api_key=OPENAI_KEY)

def get_embeddings(text: str, model: str) :
    """Generates embeddings for a given text using the OpenAI client."""
    text = text.replace("\n", " ")
    return client.embeddings.create(input=text, model=model).data[0].embedding

def generate_ids(number: int, size: int) -> List[str]:
    ids = []
    for _ in range(number):
        res = ''.join(random.choices(string.ascii_letters, k=size))
        while res in ids:
            res = ''.join(random.choices(string.ascii_letters, k=size))
        ids.append(res)
    return ids

def load_chunks(split_text: List[str], model: str) -> pd.DataFrame:
    df = pd.DataFrame(columns=['id', 'values', 'metadata'])
    ids = generate_ids(len(split_text), 7)
    for i, chunk in enumerate(split_text):
        df.loc[i] = [ids[i], get_embeddings(chunk, model=model), {'text': chunk}]
    return df

def convert_data(chunk: pd.DataFrame) -> List[Dict[str, Any]]:
    return chunk.to_dict('records')

def load_chunker(seq: pd.DataFrame, size: int) -> Generator[pd.DataFrame, None, None]:
    for pos in range(0, len(seq), size):
        yield seq.iloc[pos:pos + size]


def embed_and_upload_to_pinecone(
    file_bytes: bytes,
    file_name: str,
    chunk_size: int = 800,
    chunk_overlap: int = 200,
    embedding_model: str = "text-embedding-3-small"
) -> Dict[str, Any]:

    raw_text = ""
    if file_name.lower().endswith('.pdf'):
        try:
            # Read directly from bytes
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                raw_text = "".join(page.get_text() for page in doc)
        except Exception as e:
            return {"status": "error", "message": f"Failed to read PDF bytes: {e}"}
    else:
        return {"status": "error", "message": "Unsupported file type. Only PDF is supported via upload."}

    if not raw_text:
        return {"status": "error", "message": "Extracted text is empty. Nothing to process."}

    my_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    chunks = my_splitter.split_text(raw_text)
    if not chunks:
        return {"status": "error", "message": "Text splitting resulted in zero chunks."}

    try:
        my_df = load_chunks(chunks, model=embedding_model)
    except Exception as e:
        return {"status": "error", "message": f"Failed to generate embeddings: {e}"}

    try:
        target_index = pc.Index('sage')
    except Exception as e:
        return {"status": "error", "message": f"Failed to connect to Pinecone index '{'sage'}': {e}"}

    total_upserted = 0
    batch_size = 100
    try:
        for load_chunk in load_chunker(my_df, batch_size):
            vectors = convert_data(load_chunk)
            target_index.upsert(vectors)
            total_upserted += len(vectors)
    except Exception as e:
        return {"status": "error", "message": f"Failed during Pinecone upsert: {e}"}

    return {
        "status": "success",
        "total_chunks_processed": len(chunks),
        "total_vectors_upserted": total_upserted,
        "index_name": 'sage',
        "processed_file": file_name
    }

def get_context(query: str, embed_model: str = 'text-embedding-3-small', k: int = 5) -> Dict[str, Any]:
    try:
        index = pc.Index('sage')
        query_embeddings = get_embeddings(query, model=embed_model)
        pinecone_response = index.query(
            vector=query_embeddings, 
            top_k=k, 
            include_metadata=True
        )
        contexts = [item['metadata']['text'] for item in pinecone_response['matches']]
        
        if not contexts:
            return {"status": "success", "message": "Query successful, but no matching contexts were found."}
            
        return {"status": "success", "contexts_found": contexts}
    except Exception as e:
        return {"status": "error", "message": f"Failed to retrieve context: {e}"}


tools = [
    {
        "type": "function",
        "function": {
            "name": "embed_and_upload_to_pinecone",
            "description": "Processes an uploaded PDF file, chunks it, creates embeddings, and upserts to Pinecone. The file must be uploaded in the UI first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_name": {"type": "string", "description": "The name of the file that has been uploaded by the user and is ready for processing."},
                    "chunk_size": {"type": "integer", "default": 800},
                    "chunk_overlap": {"type": "integer", "default": 200},
                    "embedding_model": {"type": "string", "default": "text-embedding-3-small"}
                },
                "required": ["file_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_context",
            "description": "Retrieves relevant text contexts from the Pinecone index based on a user's search query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query to find relevant context for."},
                    "k": {"type": "integer", "default": 5},
                    "embed_model": {"type": "string", "default": "text-embedding-3-small"}
                },
                "required": ["query"]
            }
        }
    }
]

available_tools = {
    "embed_and_upload_to_pinecone": embed_and_upload_to_pinecone,
    "get_context": get_context,
}

def main():

    st.set_page_config(page_title="SAGE Assistant")
    st.title("S.A.G.E. Assistant")

    with st.sidebar:
        st.header("File Upload")
        uploaded_file = st.file_uploader("Upload a PDF to process", type="pdf")
        
        if uploaded_file is not None:
            if st.session_state.get("uploaded_file_name") != uploaded_file.name:
                st.session_state.uploaded_file_bytes = uploaded_file.read()
                st.session_state.uploaded_file_name = uploaded_file.name
                st.success(f"File '{uploaded_file.name}' loaded. Ask the assistant to process it.")
        
        if "uploaded_file_name" in st.session_state:
             st.info(f"File in memory: {st.session_state.uploaded_file_name}")

    if "messages" not in st.session_state:
        st.session_state.base_system_prompt = "You are a helpful assistant. You have two tools: one to upload documents to a Pinecone index, and one to retrieve context from it to answer questions."
        st.session_state.messages = [
            {"role": "system", "content": st.session_state.base_system_prompt}
        ]

    for message in st.session_state.messages:
        role = message.get("role")
        if role == "system":
            continue
        
            
        with st.chat_message(role):
            if role == "user":
                st.write(message.get("content"))
            
            elif role == "assistant":
                content = message.get("content")
                tool_calls = message.get("tool_calls")
                
                if content:
                    st.write(content)
                    
                if tool_calls:
                    st.write("Calling tools... _hang on!_")
                    for tool_call in tool_calls:
                        func_name = tool_call.get("function", {}).get("name")
                        func_args = tool_call.get("function", {}).get("arguments")
                        if func_name:
                             with st.expander(f"Tool Call: `{func_name}`"):
                                 try:
                                     st.json(func_args)
                                 except:
                                     st.write(func_args)
            
            elif role == "tool":
                tool_name = message.get("name")
                tool_content = message.get("content")
                with st.expander(f"Tool Result: `{tool_name}`"):
                    try:
                        st.json(tool_content)
                    except:
                        st.write(tool_content)

    if user_prompt := st.chat_input("What would you like to do?"):
        
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.write(user_prompt)

        try:
            messages_for_api = [msg for msg in st.session_state.messages]
            current_system_prompt = st.session_state.base_system_prompt
            
            if "uploaded_file_name" in st.session_state:
                file_context = (
                    f" \n\nIMPORTANT CONTEXT: A file named '{st.session_state.uploaded_file_name}' "
                    "has been uploaded by the user and is in memory. "
                    "Make the tool call 'embed_and_upload_to_pinecone' using this file name."
                )
                current_system_prompt += file_context

            messages_for_api[0] = {"role": "system", "content": current_system_prompt}

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages_for_api, 
                tools=tools,
                tool_choice="auto",
            )
            
            response_message = response.choices[0].message
            st.session_state.messages.append(response_message.to_dict())
            tool_calls = response_message.tool_calls

            if tool_calls:
                with st.chat_message("assistant"):
                    st.write("Calling tools...")
                    for tool_call in tool_calls:
                         with st.expander(f"Tool Call: `{tool_call.function.name}`"):
                             st.json(tool_call.function.arguments)
                
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_to_call = available_tools.get(function_name)
                    
                    if not function_to_call:
                        function_response = json.dumps({"status": "error", "message": f"Unknown function '{function_name}'"})
                    else:
                        try:
                            function_args = json.loads(tool_call.function.arguments)
                            function_response_data = {}

                            if function_name == "embed_and_upload_to_pinecone":
                                if "uploaded_file_bytes" in st.session_state:
                                    
                                    function_response_data = function_to_call(
                                        file_bytes=st.session_state.uploaded_file_bytes,
                                        file_name=st.session_state.uploaded_file_name,
                                        chunk_size=function_args.get("chunk_size", 800),
                                        chunk_overlap=function_args.get("chunk_overlap", 200),
                                        embedding_model=function_args.get("embedding_model", "text-embedding-3-small")
                                    )
                                    
                                    
                                else:
                                    function_response_data = {"status": "error", "message": "No file found in memory. Please upload a file using the sidebar first."}
                            else:
                                function_response_data = function_to_call(**function_args)
                            
                            function_response = json.dumps(function_response_data)

                        except Exception as e:
                            function_response = json.dumps({"status": "error", "message": str(e)})

                    tool_message = {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response,
                    }
                    st.session_state.messages.append(tool_message)

                    with st.chat_message("tool"):
                        with st.expander(f"Tool Result: `{function_name}`"):
                            try:
                                st.json(function_response)
                            except:
                                st.write(function_response)

                final_response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=st.session_state.messages,
                )
                final_answer = final_response.choices[0].message.content
                st.session_state.messages.append({"role": "assistant", "content": final_answer})
                with st.chat_message("assistant"):
                    st.write(final_answer)

            else:
                assistant_response = response_message.content
                with st.chat_message("assistant"):
                    st.write(assistant_response)

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
        if "uploaded_file_bytes" in st.session_state:
            del st.session_state.uploaded_file_bytes
        if "uploaded_file_name" in st.session_state:
            del st.session_state.uploaded_file_name

if __name__ == "__main__":
    main()