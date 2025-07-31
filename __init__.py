from .stable_3d import Stable3DGenerate3D, Stable3DLoadModels, Stable3DPreprocessImage


NODE_CLASS_MAPPINGS = {
    "Stable3DGenerate3D": Stable3DGenerate3D,
    "Stable3DLoadModels": Stable3DLoadModels,
    "Stable3DPreprocessImage": Stable3DPreprocessImage
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Stable3DGenerate3D": "Stable-3D Generate 3D",
    "Stable3DLoadModels": "Stable-3D Load Models",
    "Stable3DPreprocessImage": "Stable-3D Preprocess Image"
}

__all__ = [NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS]
