from typing import Any, Dict, Tuple
import yaml
from pydantic import BaseModel, Field

class ColorScheme(BaseModel):
    agent_color: Tuple[int, int, int] = Field(..., description="Color for agents")
    obstacle_color: Tuple[int, int, int] = Field(..., description="Color for obstacles")
    background_color: Tuple[int, int, int] = Field(..., description="Background color")
    goal_color: Tuple[int, int, int] = Field(..., description="Color for goals")
    velocity_color: Tuple[int, int, int] = Field(..., description="Color for current velocity arrow")
    pref_velocity_color: Tuple[int, int, int] = Field(..., description="Color for preferred velocity arrow")
    detection_radius_color: Tuple[int, int, int] = Field(..., description="Color for detection radius")
    distance_line_color: Tuple[int, int, int] = Field(..., description="Color for distance marker to goal")
    additional_info: Dict[str, Any] = Field(default_factory=dict)

    def get_agent_color(self, behaviour: str) -> Tuple[int, int, int]:
        """
        Returns the color associated with a behavior. If no color is specified
        for that behavior, returns the default color `agent_color`.
        """
        return self.additional_info.get(behaviour, {}).get("agent_color", self.agent_color)

class ColorSchemeConfig(BaseModel):
    schemes: Dict[str, ColorScheme]

def load_color_schemes(file_path: str) -> ColorSchemeConfig:
    """Loads the YAML file of color schemes and converts it to ColorSchemeConfig."""
    with open(file_path, 'r') as file:
        color_schemes = yaml.safe_load(file)

    # The YAML should now have a dictionary under 'color_schemes'
    return ColorSchemeConfig(schemes=color_schemes['color_schemes'])
