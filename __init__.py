from .stable_3d import Stable3DGenerate3D, Stable3DPreprocessMesh


NODE_CLASS_MAPPINGS = {
    "Stable3DGenerate3D": Stable3DGenerate3D,
    "Stable3DPreprocessMesh": Stable3DPreprocessMesh
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Stable3DGenerate3D": "Stable-3D Generate 3D",
    "Stable3DPreprocessMesh": "Stable-3D Preprocess Mesh"
}

__all__ = [NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS]
