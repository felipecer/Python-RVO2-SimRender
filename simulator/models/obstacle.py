from typing import Tuple, List, Optional, Literal, Union
from pydantic import BaseModel

class Obstacle(BaseModel):
    polygon: List[Tuple[float, float]]