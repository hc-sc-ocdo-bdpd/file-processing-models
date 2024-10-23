import pytest
from file_processing_models.gguf_model_processor import GgufModelProcessor
from file_processing_models.errors import (
    ModelLoadingError,
    InferenceError,
    EmbeddingGenerationError,
)

def test_model_loading(gguf_processor):
    assert gguf_processor.model is not None, "Model should be loaded successfully."

def test_infer(gguf_processor):
    prompt = "Hello, how are you?"
    result = gguf_processor.infer(prompt)
    assert isinstance(result, str), "Inference result should be a string."
    assert len(result) > 0, "Inference result should not be empty."

def test_create_embeddings(gguf_processor):
    text = "This is a test sentence."
    embedding = gguf_processor.create_embeddings(text)
    assert isinstance(embedding, list), "Embedding should be a list."
    assert len(embedding) > 0, "Embedding list should not be empty."

def test_invalid_model_loading(invalid_model_path):
    with pytest.raises(ModelLoadingError):
        GgufModelProcessor(invalid_model_path)

@pytest.mark.parametrize("prompt", [
    "Tell me a story.",
    "",
    None,
])
def test_inference_errors(gguf_processor, prompt):
    if prompt is None:
        with pytest.raises(InferenceError):
            gguf_processor.infer(prompt)
    else:
        result = gguf_processor.infer(prompt)
        assert isinstance(result, str), "Inference result should be a string."

@pytest.mark.parametrize("text", [
    "Sample text for embedding.",
    "",
    None,
])
def test_embedding_generation_errors(gguf_processor, text):
    if text is None:
        with pytest.raises(EmbeddingGenerationError):
            gguf_processor.create_embeddings(text)
    else:
        embedding = gguf_processor.create_embeddings(text)
        assert isinstance(embedding, list), "Embedding should be a list."

def test_infer_without_model(gguf_processor):
    # Simulate the model not being loaded
    gguf_processor.model = None
    with pytest.raises(ModelLoadingError):
        gguf_processor.infer("Test prompt")

def test_create_embeddings_without_model(gguf_processor):
    # Simulate the model not being loaded
    gguf_processor.model = None
    with pytest.raises(ModelLoadingError):
        gguf_processor.create_embeddings("Test text")