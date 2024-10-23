import pytest
from file_processing_models.gguf_model_processor import GgufModelProcessor
from file_processing_models.errors import ModelLoadingError

@pytest.fixture(scope='session')
def valid_model_path(tmp_path_factory):
    # Provide a path to a valid GGUF model for testing.
    # For the purposes of this example, we'll assume it's located at 'tests/models/valid_model.gguf'.
    # In practice, you'd provide an actual model file or mock as needed.
    return 'tests/models/valid_model.gguf'

@pytest.fixture(scope='session')
def invalid_model_path(tmp_path_factory):
    # Provide a path to an invalid GGUF model (or non-existent file).
    return 'tests/models/invalid_model.gguf'

@pytest.fixture(scope='session')
def gguf_processor(valid_model_path):
    # Initialize and return a GgufModelProcessor instance.
    try:
        processor = GgufModelProcessor(valid_model_path)
        return processor
    except ModelLoadingError as e:
        pytest.fail(f"Failed to load valid model for testing: {e}")
