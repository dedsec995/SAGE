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
    url: Optional[str] = None,
    text: Optional[str] = None,
    chunk_size: int = 800,
    chunk_overlap: int = 200,
    embedding_model: str = "text-embedding-3-small"
) -> Dict[str, Any]:
    """
    Processes text from a URL (PDF/TXT) or raw string, chunks it,
    creates embeddings, and upserts to a Pinecone index.
    """
    index = pc.Index('sage')
    raw_text = ""

    if url:
        try:
            response = requests.get(url)
            response.raise_for_status()
            if url.lower().endswith('.pdf'):
                with fitz.open(stream=response.content, filetype="pdf") as doc:
                    raw_text = "".join(page.get_text() for page in doc)
            elif url.lower().endswith('.txt'):
                raw_text = response.text
            else:
                return {"status": "error", "message": "Unsupported file type. URL must end in .pdf or .txt"}
        except requests.exceptions.RequestException as e:
            return {"status": "error", "message": f"Failed to download or access URL: {e}"}
    elif text:
        raw_text = text
    else:
        return {"status": "error", "message": "No input provided. You must specify either 'url' or 'text'."}

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
        "index_name": 'sage'
    }

def get_context(query: str, embed_model: str = 'text-embedding-3-small', k: int = 5) -> Dict[str, Any]:
    """
    Retrieves relevant text contexts from the Pinecone index
    based on a user's search query.
    """
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
            "description": "Processes text from a URL (PDF/TXT) or raw string, chunks it, creates embeddings, and upserts to Pinecone. One of 'url' or 'text' must be provided.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL of the PDF or TXT file to process. If provided, the 'text' parameter is ignored."},
                    "text": {"type": "string", "description": "A string of raw text to process. This is used only if the 'url' parameter is not provided."},
                    "chunk_size": {"type": "integer", "default": 800},
                    "chunk_overlap": {"type": "integer", "default": 200},
                    "embedding_model": {"type": "string", "default": "text-embedding-3-small"}
                },
                # "required": ["url"]
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
    """
    Main loop to run the chat-with-tools.
    """
    print("Starting chat... (type 'quit' to exit)")
    messages = [
        {"role": "system", "content": "You are a helpful assistant. You have two tools: one to upload documents to a Pinecone index, and one to retrieve context from it to answer questions."}
    ]

    while True:
        try:
            # Get user input
            user_prompt = input("You: ")
            if user_prompt.lower() == 'quit':
                print("Ending chat. Goodbye!")
                break
            
            messages.append({"role": "user", "content": user_prompt})

            # --- First API Call: Get model response or tool call ---
            response = client.chat.completions.create(
                model="gpt-4o",  # Or your preferred model
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )
            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls

            # --- Check if the model wants to call a tool ---
            if tool_calls:
                # Append the assistant's request to the message history
                messages.append(response_message)
                
                # --- Execute all tool calls ---
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_to_call = available_tools.get(function_name)
                    
                    if not function_to_call:
                        print(f"Error: Model tried to call unknown function '{function_name}'")
                        continue
                        
                    try:
                        # Parse the JSON arguments
                        function_args = json.loads(tool_call.function.arguments)
                        
                        print(f"--- Calling Tool: {function_name}({function_args}) ---")
                        
                        # Call the corresponding Python function
                        function_response = function_to_call(**function_args)
                        
                        print(f"--- Tool Response: {function_response} ---")
                        
                        # Append the tool's output to the message history
                        messages.append(
                            {
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "name": function_name,
                                "content": json.dumps(function_response),  # Convert response to JSON string
                            }
                        )
                    except Exception as e:
                        print(f"Error executing tool {function_name}: {e}")
                        messages.append(
                            {
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "name": function_name,
                                "content": json.dumps({"status": "error", "message": str(e)}),
                            }
                        )

                final_response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                )
                final_answer = final_response.choices[0].message.content
                print(f"Assistant: {final_answer}")
                messages.append({"role": "assistant", "content": final_answer})

            else:
                # --- No tool call, just a direct answer ---
                assistant_response = response_message.content
                print(f"Assistant: {assistant_response}")
                messages.append({"role": "assistant", "content": assistant_response})

        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            break

if __name__ == "__main__":
    main()