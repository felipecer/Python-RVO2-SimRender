import yaml
from pydantic import BaseModel, Field
from typing import Any, Dict, Tuple, List


class ColorScheme(BaseModel):
    agent_color: Tuple[int, int,
                       int] = Field(..., description="Color para los agentes")
    obstacle_color: Tuple[int, int,
                          int] = Field(..., description="Color para los obstáculos")
    background_color: Tuple[int, int,
                            int] = Field(..., description="Color de fondo")
    goal_color: Tuple[int, int,
                      int] = Field(..., description="Color para las metas")
    velocity_color: Tuple[int, int, int] = Field(
        ..., description="Color de la flecha de velocidad actual")
    pref_velocity_color: Tuple[int, int, int] = Field(
        ..., description="Color de la flecha de velocidad preferida")
    detection_radius_color: Tuple[int, int, int] = Field(
        ..., description="Color para el radio de detección")
    distance_line_color: Tuple[int, int, int] = Field(
        ..., description="Color para el marcador de distancia a la meta")
    additional_info: Dict[str, Any] = Field(default_factory=dict)

    def get_agent_color(self, behaviour: str) -> Tuple[int, int, int]:
        """
        Retorna el color asociado a un comportamiento. Si no se especifica un color
        para ese comportamiento, retorna el color por defecto `agent_color`.
        """
        return self.additional_info.get(behaviour, {}).get("agent_color", self.agent_color)


class ColorSchemeConfig(BaseModel):
    schemes: Dict[str, ColorScheme]


def load_color_schemes(file_path: str) -> ColorSchemeConfig:
    """Carga el archivo YAML de esquemas de color y lo convierte en ColorSchemeConfig."""
    with open(file_path, 'r') as file:
        color_schemes = yaml.safe_load(file)

    # El YAML ahora debe tener un diccionario bajo 'color_schemes'
    return ColorSchemeConfig(schemes=color_schemes['color_schemes'])
