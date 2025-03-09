import os
import logging
import time
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

class TextProcessor:
    """
    Text processing model that handles text-based queries and generates responses.
    
    This class integrates with language models (LLMs) and implements RAG for enhanced responses.
    """
    
    def __init__(self):
        """Initialize the text processor with the configured language model."""
        self.model_name = os.getenv("TEXT_MODEL", "gpt-4")
        logger.info(f"Initializing TextProcessor with model: {self.model_name}")
        
        # Initialize the language model based on configuration
        self._initialize_model()
        
        # Initialize memory system for RAG
        from app.models.memory import MemorySystem
        self.memory_system = MemorySystem()
    
    def _initialize_model(self):
        """Initialize the language model based on the configuration."""
        if self.model_name.startswith("gpt"):
            self._initialize_openai_model()
        elif self.model_name.startswith("llama"):
            self._initialize_llama_model()
        elif self.model_name.startswith("mistral"):
            self._initialize_mistral_model()
        else:
            logger.warning(f"Unknown model: {self.model_name}, falling back to OpenAI")
            self.model_name = "gpt-4"
            self._initialize_openai_model()
    
    def _initialize_openai_model(self):
        """Initialize OpenAI model."""
        try:
            import openai
            from langchain_openai import ChatOpenAI
            
            openai.api_key = os.getenv("OPENAI_API_KEY")
            self.model = ChatOpenAI(
                model_name=self.model_name,
                temperature=0.7,
                max_tokens=1000
            )
            logger.info(f"Initialized OpenAI model: {self.model_name}")
        except ImportError:
            logger.error("Failed to import OpenAI libraries")
            raise
        except Exception as e:
            logger.error(f"Error initializing OpenAI model: {str(e)}")
            raise
    
    def _initialize_llama_model(self):
        """Initialize LLaMA model."""
        try:
            from langchain.llms import HuggingFacePipeline
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            
            model_id = "meta-llama/Llama-2-7b-chat-hf"
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_8bit=True
            )
            
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=1000,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.15
            )
            
            self.model = HuggingFacePipeline(pipeline=pipe)
            logger.info(f"Initialized LLaMA model: {model_id}")
        except ImportError:
            logger.error("Failed to import HuggingFace libraries")
            raise
        except Exception as e:
            logger.error(f"Error initializing LLaMA model: {str(e)}")
            raise
    
    def _initialize_mistral_model(self):
        """Initialize Mistral model."""
        try:
            from langchain.llms import HuggingFacePipeline
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            
            model_id = "mistralai/Mistral-7B-Instruct-v0.1"
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_8bit=True
            )
            
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=1000,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.15
            )
            
            self.model = HuggingFacePipeline(pipeline=pipe)
            logger.info(f"Initialized Mistral model: {model_id}")
        except ImportError:
            logger.error("Failed to import HuggingFace libraries")
            raise
        except Exception as e:
            logger.error(f"Error initializing Mistral model: {str(e)}")
            raise
    
    def process(
        self, 
        text: str, 
        context: Optional[List[Dict[str, str]]] = None,
        use_rag: bool = True,
        user_id: Optional[str] = None
    ) -> Tuple[str, Optional[List[Dict[str, Any]]], float]:
        """
        Process text input and generate a response.
        
        Args:
            text: The input text to process
            context: Previous conversation context
            use_rag: Whether to use Retrieval-Augmented Generation
            user_id: User ID for personalization
            
        Returns:
            Tuple containing:
            - response: The generated text response
            - sources: Sources used for RAG (if applicable)
            - confidence: Confidence score of the response
        """
        logger.info(f"Processing text: {text[:50]}...")
        start_time = time.time()
        
        # Retrieve relevant information if RAG is enabled
        sources = None
        if use_rag:
            try:
                # Search for relevant information in memory
                search_results, _ = self.memory_system.search(
                    query=text,
                    user_id=user_id,
                    limit=3,
                    min_score=0.7
                )
                
                if search_results:
                    sources = search_results
                    logger.info(f"Found {len(sources)} relevant sources for RAG")
            except Exception as e:
                logger.warning(f"Error retrieving RAG sources: {str(e)}")
        
        # Prepare the prompt with context and sources
        prompt = self._prepare_prompt(text, context, sources)
        
        # Generate response using the language model
        try:
            if self.model_name.startswith("gpt"):
                response = self._generate_openai_response(prompt)
            else:
                response = self._generate_hf_response(prompt)
            
            # Calculate confidence (simplified)
            confidence = 0.85  # In a real implementation, this would be model-specific
            
            # Store the interaction in memory for future RAG
            if use_rag:
                try:
                    self.memory_system.store(
                        content=f"User: {text}\nAssistant: {response}",
                        metadata={
                            "type": "conversation",
                            "user_id": user_id,
                            "timestamp": time.time()
                        },
                        user_id=user_id
                    )
                except Exception as e:
                    logger.warning(f"Error storing interaction in memory: {str(e)}")
            
            logger.info(f"Generated response in {time.time() - start_time:.2f}s")
            return response, sources, confidence
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
    
    def _prepare_prompt(
        self, 
        text: str, 
        context: Optional[List[Dict[str, str]]] = None,
        sources: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Prepare the prompt for the language model.
        
        Args:
            text: The input text
            context: Previous conversation context
            sources: Retrieved sources for RAG
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = []
        
        # Add system instructions
        prompt_parts.append(
            "You are IntelAssist, an intelligent AI assistant that provides helpful, accurate, "
            "and thoughtful responses to user queries. Be concise but thorough."
        )
        
        # Add retrieved sources if available
        if sources:
            prompt_parts.append("\nRelevant information:")
            for i, source in enumerate(sources, 1):
                prompt_parts.append(f"{i}. {source['content']}")
        
        # Add conversation context if available
        if context:
            prompt_parts.append("\nConversation history:")
            for message in context:
                role = message.get("role", "user")
                content = message.get("content", "")
                prompt_parts.append(f"{role.capitalize()}: {content}")
        
        # Add the current query
        prompt_parts.append(f"\nUser: {text}")
        prompt_parts.append("Assistant:")
        
        return "\n".join(prompt_parts)
    
    def _generate_openai_response(self, prompt: str) -> str:
        """Generate a response using OpenAI models."""
        from langchain.schema import HumanMessage, SystemMessage
        
        # Extract system instructions and user query
        parts = prompt.split("\nUser: ")
        system_content = parts[0]
        user_content = parts[1].split("\nAssistant:")[0] if len(parts) > 1 else ""
        
        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=user_content)
        ]
        
        response = self.model.invoke(messages)
        return response.content
    
    def _generate_hf_response(self, prompt: str) -> str:
        """Generate a response using HuggingFace models."""
        response = self.model.invoke(prompt)
        
        # Extract the assistant's response
        if "Assistant:" in response:
            return response.split("Assistant:")[1].strip()
        return response.strip()
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        use_rag: bool = True,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Handle multi-turn conversations.
        
        Args:
            messages: List of conversation messages
            use_rag: Whether to use RAG
            user_id: User ID for personalization
            
        Returns:
            Dictionary with the response and metadata
        """
        logger.info(f"Processing chat with {len(messages)} messages")
        
        # Extract the latest user message
        latest_user_message = None
        for message in reversed(messages):
            if message.get("role") == "user":
                latest_user_message = message.get("content")
                break
        
        if not latest_user_message:
            return {"error": "No user message found"}
        
        # Process the latest message with context
        response, sources, confidence = self.process(
            text=latest_user_message,
            context=messages[:-1] if len(messages) > 1 else None,
            use_rag=use_rag,
            user_id=user_id
        )
        
        return {
            "response": response,
            "sources": sources,
            "confidence": confidence,
            "model": self.model_name
        }
    
    def summarize(self, text: str, max_length: int = 100) -> str:
        """
        Summarize a long text into a concise summary.
        
        Args:
            text: The text to summarize
            max_length: Maximum length of the summary
            
        Returns:
            Summarized text
        """
        logger.info(f"Summarizing text of length {len(text)}")
        
        prompt = (
            f"Please summarize the following text in no more than {max_length} words:\n\n"
            f"{text}\n\nSummary:"
        )
        
        try:
            if self.model_name.startswith("gpt"):
                summary = self._generate_openai_response(prompt)
            else:
                summary = self._generate_hf_response(prompt)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error summarizing text: {str(e)}")
            raise 