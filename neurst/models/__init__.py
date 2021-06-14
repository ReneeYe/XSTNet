import importlib
import os

from neurst.models.model import BaseModel
from neurst.utils.registry import setup_registry

build_model, register_model = setup_registry(BaseModel.REGISTRY_NAME, base_class=BaseModel, create_fn="new",
                                             verbose_creation=True)

models_dir = os.path.dirname(__file__)
for file in os.listdir(models_dir):
    path = os.path.join(models_dir, file)
    if not file.startswith('_') and not file.startswith('.') and file.endswith('.py'):
        model_name = file[:file.find('.py')] if file.endswith('.py') else file
        module = importlib.import_module('neurst.models.' + model_name)
