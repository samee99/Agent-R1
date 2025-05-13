from typing import Dict, List, Any
from agent_r1.tool.base import BaseTool

from rcwa import Material, Layer, LayerStack, Source, Solver
import numpy as np

class RCWATool(BaseTool):
    name = "rcwa"
    description = "Simulate the transmission spectrum of a 4-layer optical stack using RCWA."
    parameters = {
        "type": "object",
        "properties": {
            "layers": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of four material names in order from incident to transmission"
            }
        },
        "required": ["layers"]
    }

    def __init__(self):
        super().__init__()
        # ✅ Use np.linspace to guarantee exactly 57 points (matching dataset)
        self.wavelengths = np.linspace(0.25, 0.80, 57)  # microns
        self.source = Source(wavelength=0.25)
        self.reflection_layer = Layer(n=1.0)  # Air
        self.substrate = Material(name='Si')  # transmission side
        self.transmission_layer = Layer(material=self.substrate)

    def execute(self, args: Dict) -> Dict[str, Any]:
        try:
            layer_names = args["layers"]
            if len(layer_names) != 4:
                return {"content": "Exactly 4 layers required", "success": False}

            layers = [Layer(material=Material(name=m), thickness=0.1) for m in layer_names]
            stack = LayerStack(*layers, incident_layer=self.reflection_layer, transmission_layer=self.transmission_layer)
            solver = Solver(stack, self.source, (1, 1))
            result = solver.solve(wavelength=self.wavelengths)

            spectrum = np.array(result["TTot"]).tolist()
            return {
                "content": {
                    "spectrum": spectrum
                },
                "success": True
            }

        except Exception as e:
            return {"content": str(e), "success": False}

    def batch_execute(self, args_list: List[Dict]) -> List[Dict[str, Any]]:
        return [self.execute(args) for args in args_list]
