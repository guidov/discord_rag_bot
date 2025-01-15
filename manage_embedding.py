from llama_index import SimpleDirectoryReader, VectorStoreIndex, load_index_from_storage, ServiceContext
from llama_index.storage.storage_context import StorageContext
from llama_index.llm_predictor import LLMPredictor
from llama_index.llms import CustomLLM, LLMMetadata
from llama_index.embeddings import OpenAIEmbedding
from openai import OpenAI
from typing import Optional, List, Mapping, Any
from dotenv import load_dotenv
import logging
import sys
import os

load_dotenv()

# Configure DeepSeek LLM
DEEPSEEK_API_BASE = "https://api.deepseek.com"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

class DeepSeekLLM(CustomLLM):
    model_name: str = "deepseek-chat"
    _context_window: int = 32768
    _max_tokens: int = 4096
    _client: OpenAI

    def __init__(self):
        super().__init__()
        object.__setattr__(self, '_client', OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_API_BASE,
            default_headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
        ))

    @property
    def metadata(self) -> LLMMetadata:
        """Get the metadata for the LLM."""
        from llama_index.llms.base import LLMMetadata
        return LLMMetadata(
            context_window=self._context_window,
            num_output=self._max_tokens,
            model_name=self.model_name
        )

    def _get_context_window(self) -> int:
        """Get the context window for the LLM."""
        return self._context_window

    def _get_max_tokens(self) -> int:
        """Get the maximum tokens for the LLM."""
        return self._max_tokens

    def complete(self, prompt: str, **kwargs: Any) -> Any:
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant that answers questions based on the provided context."},
            {"role": "user", "content": prompt}
        ]
        response = self._client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            stream=False
        )
        from llama_index.llms import CompletionResponse
        return CompletionResponse(text=response.choices[0].message.content)

    async def acomplete(self, prompt: str, **kwargs: Any) -> Any:
        from llama_index.llms import CompletionResponse
        response = await self.complete(prompt, **kwargs)
        return CompletionResponse(text=response)

    async def stream_complete(self, prompt: str, **kwargs: Any) -> Any:
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant that answers questions based on the provided context."},
            {"role": "user", "content": prompt}
        ]
        response = self._client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            stream=True
        )
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

if not DEEPSEEK_API_KEY:
    raise ValueError("DeepSeek API key not found in environment variables")

deepseek_client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_API_BASE,
    default_headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
)

llm = DeepSeekLLM()

# Use OpenAI's embedding model
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found in environment variables")

embed_model = OpenAIEmbedding(
    api_key=OPENAI_API_KEY,
    model="text-embedding-ada-002",
    api_base="https://api.openai.com/v1"
)

# enable INFO level logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


async def load_index(directory_path : str = r'data'):
    documents = SimpleDirectoryReader(directory_path, filename_as_id=True).load_data()
    print(f"loaded documents with {len(documents)} pages")
    
    # Create service context with our custom LLM
    llm_predictor = LLMPredictor(llm=llm)
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor,
        embed_model=embed_model
    )
    
    try:
        # Rebuild storage context
        storage_context = StorageContext.from_defaults(persist_dir="./storage")
        # Try to load the index from storage
        index = load_index_from_storage(
            storage_context,
            service_context=service_context
        )
        logging.info("Index loaded from storage.")
    except FileNotFoundError:
        logging.info("Index not found. Creating a new one...")
        index = VectorStoreIndex.from_documents(
            documents,
            service_context=service_context,
            storage_context=StorageContext.from_defaults()
        )
        # Persist index to disk
        index.storage_context.persist()
        logging.info("New index created and persisted to storage.")

    return index

async def chat_with_deepseek(messages):
    """Direct chat with DeepSeek using the new client"""
    response = deepseek_client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        stream=False
    )
    return response.choices[0].message.content

async def update_index(directory_path : str = r'data'):
    try:
        documents = SimpleDirectoryReader(directory_path, filename_as_id=True).load_data()
    except FileNotFoundError:
        logging.error("Invalid document directory path.")
        return None
    try:
        # Rebuild storage context
        storage_context = StorageContext.from_defaults(persist_dir="./storage")
        # Try to load the index from storage
        index = load_index_from_storage(storage_context)
        logging.info("Existing index loaded from storage.")
        refreshed_docs = index.refresh_ref_docs(documents, update_kwargs={"delete_kwargs": {"delete_from_docstore": True}})
        # index.update_ref_doc()
        print(refreshed_docs)
        print('Number of newly inserted/refreshed docs: ', sum(refreshed_docs))

        index.storage_context.persist()
        logging.info("Index refreshed and persisted to storage.")
        return refreshed_docs

        
    except FileNotFoundError:
    # Run refresh_ref_docs function to check for document updates
        logging.error("Index is not created yet.")
        return None

