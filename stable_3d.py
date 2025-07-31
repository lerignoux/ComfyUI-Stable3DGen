import datetime
import io
import json
import logging
import os
import numpy
import sys
import torch
import trimesh
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import folder_paths

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Stable3DGen'))

from hi3dgen.pipelines import Hi3DGenPipeline

log = logging.getLogger(__name__)


MAX_SEED = numpy.iinfo(numpy.int32).max
WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights')
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# Initialize normal predictor
"""
predictor_model = os.path.join(torch.hub.get_dir(), 'hugoycj_StableNormal_main')
log.info(f"Loading torch predictor model: {predictor_model}")
try:
    normal_predictor = torch.hub.load(
        predictor_model,
        "StableNormal_turbo",
        yoso_version='yoso-normal-v1-8-1',
        source='local',
        local_cache_dir='./weights',
        pretrained=True
    )
except Exception as e:
    new_model = "hugoycj/StableNormal"
    log.info(f"Failed loading local torch {predictor_model} downloading {new_model}, {e}")
    normal_predictor = torch.hub.load(
        "hugoycj/StableNormal",
        "StableNormal_turbo",
        trust_repo=True,
        yoso_version='yoso-normal-v1-8-1',
        local_cache_dir='./weights'
    )
"""
# Loads model to ~/.cache/torch/hub/
normal_predictor = torch.hub.load("Stable-X/StableNormal", "StableNormal_turbo", trust_repo=True)


def cache_weights(weights_dir: str) -> dict:
    """
    Load weights locally if missing.
    Needs to be adapted to match ComfyUI Models storage
    """
    import os
    from huggingface_hub import snapshot_download

    os.makedirs(weights_dir, exist_ok=True)
    model_ids = [
        "Stable-X/trellis-normal-v0-1",
        "Stable-X/yoso-normal-v1-8-1",
        "ZhengPeng7/BiRefNet",
    ]
    cached_paths = {}
    for model_id in model_ids:
        log.info(f"Caching weights for: {model_id}")
        local_path = os.path.join(weights_dir, model_id.split("/")[-1])
        if os.path.exists(local_path):
            log.info(f"Already cached at: {local_path}")
            cached_paths[model_id] = local_path
            continue
        log.info(f"Downloading and caching model: {model_id}")
        local_path = snapshot_download(repo_id=model_id, local_dir=os.path.join(weights_dir, model_id.split("/")[-1]), force_download=False)
        cached_paths[model_id] = local_path
        log.info(f"Cached at: {local_path}")

    # torch.hub.load('facebookresearch/dinov2', name, pretrained=True)

    return cached_paths


cache_weights(WEIGHTS_DIR)

class Stable3DGenerate3D:
    """
    A node to generate a Stable3D asset
    """
    def __init__(self):
        self.output_directory = folder_paths.get_output_directory()
        self.temp_directory = folder_paths.get_temp_directory()
        self.compress_level = 4
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "seed": (
                    "INT",
                    {
                        "tooltip": "The generation seed"
                    }
                ),
                "ss_guidance_strength": (
                    "INT",
                    {
                        "default": 3,
                        "tooltip": "the titles for each slide."
                    }
                ),
                "ss_sampling_steps": (
                    "INT",
                    {
                        "default": 50,
                        "step": 1,
                        "tooltip": ""
                    }
                ),
                "slat_guidance_strength": (
                    "INT",
                    {
                        "default": 3,
                        "tooltip": ""
                    }
                ),
                "slat_sampling_steps": (
                    "INT",
                    {
                        "default": 6,
                        "tooltip": ""
                    }
                )
            },
        }

    CATEGORY = "stable_3d_gen"
    DESCRIPTION = "Generates a Stable 3D Gen mesh from an imput image"
    FUNCTION = "generate_3d"
    INPUT_IS_LIST = False
    OUTPUT_NODE = True
    RETURN_NAMES = ("mesh_file_path",)
    RETURN_TYPES = ("STRING",)

    def save_3d_asset(self, generated_mesh, filename=None):
        """
        Save the 3d asset file to user output directory
        """
        output_id = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
        if filename is None:
            filename = f"{output_id}_mesh.glb"
        if '.glb' not in filename:
            filename = f"{filename}.glb"
        mesh_path = os.path.join(self.output_directory, filename)

        trimesh_mesh = generated_mesh.to_trimesh(transform_pose=True)
        trimesh_mesh.export(mesh_path)

        return mesh_path

    def generate_3d(
        self,
        image,
        seed=-1,
        ss_guidance_strength=3,
        ss_sampling_steps=50,
        slat_guidance_strength=3,
        slat_sampling_steps=6
    ):
        if image is None:
            return None, None, None

        if seed == -1:
            seed = numpy.random.randint(0, MAX_SEED)

        hi3dgen_pipeline = Hi3DGenPipeline.from_pretrained("custom_nodes/ComfyUI-Stable3DGen/weights/trellis-normal-v0-1")
        hi3dgen_pipeline.cuda()

        image = torch.rand(1, 512, 512, 3) # Example tensor
        numpy_image = image.squeeze(0).cpu().numpy()
        numpy_image_scaled = numpy.clip(numpy_image * 255, 0, 255).astype(numpy.uint8)
        pil_image = Image.fromarray(numpy_image_scaled)

        # FIXME We should properly handle batch here.
        image = hi3dgen_pipeline.preprocess_image(pil_image, resolution=512)
        normal_image = normal_predictor(pil_image, resolution=512, match_input_resolution=True, data_type='object')

        outputs = hi3dgen_pipeline.run(
            normal_image,
            seed=seed,
            formats=["mesh",],
            preprocess_image=False,
            sparse_structure_sampler_params={
                "steps": ss_sampling_steps,
                "cfg_strength": ss_guidance_strength,
            },
            slat_sampler_params={
                "steps": slat_sampling_steps,
                "cfg_strength": slat_guidance_strength,
            },
        )
        generated_mesh = outputs['mesh'][0]
        saved_file = self.save_3d_asset(generated_mesh)

        return saved_file

        @classmethod
        def IS_CHANGED(s, images, seed, ss_guidance_strength, ss_sampling_steps, slat_guidance_strength, slat_sampling_steps):
            # FIXME We should properly handle re-generation depending on the input parameters.
            return float("NaN")
