from file_processing.errors import FileProcessingError

class FileProcessingModelsError(FileProcessingError):
    """Base exception class for file-processing-models errors."""
    pass

class ModelLoadingError(FileProcessingModelsError):
    """Exception raised when a model fails to load."""
    pass

class InferenceError(FileProcessingModelsError):
    """Exception raised during inference failures."""
    pass

class EmbeddingGenerationError(FileProcessingModelsError):
    """Exception raised when embedding generation fails."""
    pass