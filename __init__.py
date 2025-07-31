from .stable_3d import Stable3DGenerate3D


NODE_CLASS_MAPPINGS = {
    "Stable3DGenerate3D": Stable3DGenerate3D
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Stable3DGenerate3D": "Stable-3D Generate 3D"
}

__all__ = [NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS]
