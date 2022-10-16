from typing import List, Tuple
from pydantic import BaseModel

class ApiRequest(BaseModel):
    molform: List[int]
    spectra: List[Tuple[int, str, int]]
