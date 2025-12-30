"""
utils/ollama_client.py

Ollama client wrapper with connection testing and error handling.
"""

from typing import Dict, Optional
import time

try:
    from ollama import Client, ResponseError
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    Client = None
    ResponseError = Exception

from config.settings import OLLAMA


class OllamaConnectionError(Exception):
    """Raised when Ollama connection fails"""
    pass


class OllamaClient:
    """
    Wrapper for Ollama client with built-in error handling and retry logic.
    """
    
    def __init__(
        self, 
        model: str = None,
        host: str = None,
        timeout: int = None,
        max_retries: int = None
    ):
        """
        Initialize Ollama client.
        
        Args:
            model: Model name (default from config)
            host: Ollama server URL (default from config)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        
        Raises:
            OllamaConnectionError: If Ollama is not available
        """
        if not OLLAMA_AVAILABLE:
            raise OllamaConnectionError(
                "Ollama package not installed. Install with: pip install ollama"
            )
        
        self.model = model or OLLAMA.MODEL
        self.host = host or OLLAMA.URL
        self.timeout = timeout or OLLAMA.TIMEOUT
        self.max_retries = max_retries or OLLAMA.MAX_RETRIES
        
        self.client = Client(host=self.host)
        self._is_connected = False
    
    def test_connection(self, verbose: bool = True) -> bool:
        """
        Test connection to Ollama server.
        
        Args:
            verbose: Print connection status
        
        Returns:
            True if connected, False otherwise
        """
        try:
            models = self.client.list()
            available_models = [m['model'] for m in models.get('models', [])]
            
            if verbose:
                print(f"✅ Connected to Ollama at {self.host}")
                print(f"   Available models: {len(available_models)}")
                
                if self.model in available_models:
                    print(f"   ✅ Model '{self.model}' is available")
                else:
                    print(f"   ⚠️  Model '{self.model}' not found")
                    print(f"   Available: {available_models[:3]}...")
                    if available_models:
                        self.model = available_models[0]
                        print(f"   Switching to: {self.model}")
            
            self._is_connected = True
            return True
            
        except Exception as e:
            if verbose:
                print(f"❌ Error connecting to Ollama at {self.host}")
                print(f"   Error: {str(e)}")
                print(f"   Make sure Ollama is running: ollama serve")
            
            self._is_connected = False
            return False
    
    def generate(
        self,
        prompt: str,
        system: str = None,
        temperature: float = None,
        format: str = None,
        options: Dict = None
    ) -> Dict:
        """
        Generate text from prompt with retry logic.
        
        Args:
            prompt: Input prompt
            system: System message
            temperature: Sampling temperature
            format: Output format ('json', etc.)
            options: Additional options
        
        Returns:
            Response dictionary
        
        Raises:
            OllamaConnectionError: If generation fails after retries
        """
        if not self._is_connected:
            if not self.test_connection(verbose=False):
                raise OllamaConnectionError(
                    f"Not connected to Ollama at {self.host}. "
                    f"Start Ollama with: ollama serve"
                )
        
        # Prepare options
        if options is None:
            options = {}
        
        if temperature is not None:
            options['temperature'] = temperature
        
        # Build request kwargs
        request_kwargs = {
            'model': self.model,
            'prompt': prompt,
            'options': options
        }
        
        if system:
            request_kwargs['system'] = system
        
        if format:
            request_kwargs['format'] = format
        
        # Retry loop
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.generate(**request_kwargs)
                return response
                
            except ResponseError as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"⚠️  Attempt {attempt + 1} failed, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    break
            
            except Exception as e:
                last_error = e
                break
        
        # All retries failed
        raise OllamaConnectionError(
            f"Failed to generate after {self.max_retries} attempts. "
            f"Last error: {str(last_error)}"
        )
    
    def is_connected(self) -> bool:
        """Check if client is connected"""
        return self._is_connected
    
    def get_model(self) -> str:
        """Get current model name"""
        return self.model
    
    def set_model(self, model: str):
        """Change model"""
        self.model = model


def test_ollama_connection(host: str = None, model: str = None) -> bool:
    """
    Quick function to test Ollama connection.
    
    Args:
        host: Ollama server URL
        model: Model name
    
    Returns:
        True if connected, False otherwise
    """
    try:
        client = OllamaClient(host=host, model=model)
        return client.test_connection(verbose=True)
    except OllamaConnectionError as e:
        print(f"❌ {str(e)}")
        return False


if __name__ == "__main__":
    print("\n" + "="*80)
    print("🔧 TESTING OLLAMA CONNECTION")
    print("="*80 + "\n")
    
    success = test_ollama_connection()
    
    if success:
        print("\n✅ Ollama is ready to use!")
    else:
        print("\n❌ Ollama connection failed")
        print("\nTroubleshooting:")
        print("1. Check if Ollama is installed: ollama --version")
        print("2. Start Ollama server: ollama serve")
        print("3. Pull a model: ollama pull gemma3:4b")