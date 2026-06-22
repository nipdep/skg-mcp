import abc
import json
from typing import Type, TypeVar, Generic, Callable, Any, Dict 
from pydantic import BaseModel, TypeAdapter, ValidationError

T = TypeVar("T") 

class BaseLLMGenerator(abc.ABC, Generic[T]):

    def __init__(self, model_name: str, api_key: str = None):
        self.model_name = model_name
        self.api_key = api_key

        self._setup_client()

    @abc.abstractmethod
    def _setup_client(self):
        pass
    
    @abc.abstractmethod
    def text_generate(self, prompt: str) -> str:
        raise NotImplementedError("Subclasses must implement this method")

    @abc.abstractmethod
    def structured_text_generate(self, prompt: str, expected_type: Type[T]) -> T:
        raise NotImplementedError("Subclasses must implement this method")

    def _parse_structured_output(self, json_string: str, expected_type: Type[T]) -> T:
        start_obj = json_string.find('{')
        start_arr = json_string.find('[')

        if start_obj == -1 and start_arr == -1:
            raise ValueError("Could not find a valid JSON object or array start.")
        elif start_obj == -1:
            start_index = start_arr
        elif start_arr == -1:
            start_index = start_obj
        else:
            start_index = min(start_obj, start_arr)

        end_obj = json_string.rfind('}')
        end_arr = json_string.rfind(']')
        end_index = max(end_obj, end_arr)
        
        if end_index == -1 or end_index < start_index:
            raise ValueError("Could not find a valid JSON object or array end.")

        json_substring = json_string[start_index : end_index + 1]
        try:
            adapter = TypeAdapter(expected_type)
            parsed_data = adapter.validate_json(json_substring)
            # print(f"Parsed structured output: {repr(parsed_data)}")
            return parsed_data
        
        except ValidationError as e:
            print(f"Error validating JSON: {e}")
            raise e
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON string: {e}")
            raise e
        
    
