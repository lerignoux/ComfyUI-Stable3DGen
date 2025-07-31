from .stable_3d import Stable3DGenerate3D, Stable3DLoadModels


NODE_CLASS_MAPPINGS = {
    "Stable3DGenerate3D": Stable3DGenerate3D,
    "Stable3DLoadModels": Stable3DLoadModels
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Stable3DGenerate3D": "Stable-3D Generate 3D",
    "Stable3DLoadModels": "Stable-3D Load Models"
}

__all__ = [NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS]
