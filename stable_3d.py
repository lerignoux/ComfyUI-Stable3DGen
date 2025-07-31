import datetime
import io
import json
import logging
import os
import numpy
import sys
import torch
import trimesh
from huggingface_hub import snapshot_download
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import folder_paths

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Stable3DGen'))
from hi3dgen.pipelines import Hi3DGenPipeline

log = logging.getLogger(__name__)


MAX_SEED = numpy.iinfo(numpy.int32).max

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

class Stable3DLoadModels:
    """
    A node to load the models necessary for Stable3D
    Node will download the models from huggingface or torch if missing.
    """
    def __init__(self):
        self.weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights')
        os.makedirs(self.weights_dir, exist_ok=True)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trellis_model": (
                    "STRING",
                    {
                        "tooltip": "The trellis model to use",
                        "default": "Stable-X/trellis-normal-v0-1"
                    }
                ),
                "normal_model": (
                    "STRING",
                    {
                        "tooltip": "The normal generation model",
                        "default": "Stable-X/yoso-normal-v1-8-1"
                    }
                ),
                "birefnet_model": (
                    "STRING",
                    {
                        "default": "ZhengPeng7/BiRefNet",
                        "tooltip": "the Background removal model."
                    }
                )
            },
        }

    CATEGORY = "stable_3d_gen"
    DESCRIPTION = "Load the models necessary for Stable3D Gen"
    FUNCTION = "load_models"
    INPUT_IS_LIST = False
    OUTPUT_NODE = False
    RETURN_NAMES = ("Trellis model", "Normal predictor", "Birefnet model")
    RETURN_TYPES = ("TRELLIS_MODEL", "STABLE3D_NORMAL", "STABLE3D_BIREFNET")


def load_models(self, trellis_model, normal_model, birefnet_model):
    """
    Load weights locally if missing.
    Needs to be adapted to match ComfyUI Models storage
    """

    model_ids = [trellis_model, normal_model, birefnet_model]
    cached_paths = {}
    loaded_models = []
    for model_id in model_ids:
        log.info(f"Caching weights for: {model_id}")
        local_path = os.path.join(self.weights_dir, model_id.split("/")[-1])
        if os.path.exists(local_path):
            log.info(f"Already cached at: {local_path}")
            cached_paths[model_id] = local_path
            loaded_models.append(local_path)
            continue
        log.info(f"Downloading and caching model: {model_id}")
        local_path = snapshot_download(repo_id=model_id, local_dir=os.path.join(weights_dir, model_id.split("/")[-1]), force_download=False)
        cached_paths[model_id] = local_path
        log.info(f"Cached at: {local_path}")

    # Loads model to ~/.cache/torch/hub/
    normal_predictor = torch.hub.load("Stable-X/StableNormal", "StableNormal_turbo", trust_repo=True)

    # torch.hub.load('facebookresearch/dinov2', name, pretrained=True)

    return (loaded_models[0], normal_predictor, loaded_models[2])


class Stable3DPreprocessImage:
    """
    A node to Preprocess an input image into a normal representation for 3d generation
    """
    def __init__(self):
        self.temp_directory = folder_paths.get_temp_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "background_removal_model": (
                    "STABLE3D_BIREFNET",
                    {"tooltip": "The model to remove background (BiRefNet)."}
                ),
                "normal_predictor": (
                    "STABLE3D_NORMAL",
                    {"tooltip": "The normal predictor model to generate the image normal."}
                ),
                "image": ("IMAGE",)
            },
        }

    CATEGORY = "stable_3d_gen"
    DESCRIPTION = "Preprocess an input image into normal suitable for Stable 3D Generation."
    FUNCTION = "preprocess_image"
    INPUT_IS_LIST = False
    OUTPUT_NODE = True
    RETURN_NAMES = ("normal_image",)
    RETURN_TYPES = ("IMAGE",)

    def save_normal_image(self, normal_image):
        output_id = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
        filename = f"{output_id}_normal.png"
        path = os.path.join(self.temp_directory, filename)

        normal_image.save(path)

    def preprocess_image(self, background_removal_model, normal_predictor, image):

        hi3dgen_pipeline = Hi3DGenPipeline.from_pretrained("custom_nodes/ComfyUI-Stable3DGen/weights/trellis-normal-v0-1")
        hi3dgen_pipeline.cuda()

        image = torch.rand(1, 512, 512, 3) # Example tensor
        numpy_image = image.squeeze(0).cpu().numpy()
        numpy_image_scaled = numpy.clip(numpy_image * 255, 0, 255).astype(numpy.uint8)
        pil_image = Image.fromarray(numpy_image_scaled)

        # FIXME We should properly handle batch here.
        image = hi3dgen_pipeline.preprocess_image(pil_image, resolution=512)
        normal_image = normal_predictor(pil_image, resolution=512, match_input_resolution=True, data_type='object')

        self.save_normal_image(normal_image)

        return normal_image


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
                "trellis_model": ("TRELLIS_MODEL", ),
                "normal_image": ("IMAGE",),
                "seed": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": MAX_SEED,
                        "tooltip": "The generation seed, use -1 for a random seed."
                    }
                ),
                "ss_guidance_strength": (
                    "INT",
                    {
                        "default": 3,
                        "tooltip": "The ss guidance strength"
                    }
                ),
                "ss_sampling_steps": (
                    "INT",
                    {
                        "default": 50,
                        "step": 1,
                        "tooltip": "The ss sampling steps. increasing the steps increase generation time"
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
                        "tooltip": "The slat sampling steps. increasing the steps increase generation time"
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
        if filename is None:
            output_id = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
            filename = f"{output_id}_mesh.glb"
        if '.glb' not in filename:
            filename = f"{filename}.glb"
        mesh_path = os.path.join(self.output_directory, filename)

        trimesh_mesh = generated_mesh.to_trimesh(transform_pose=True)
        trimesh_mesh.export(mesh_path)

        return mesh_path

    def generate_3d(
        self,
        trellis_model,
        normal_image,
        seed=-1,
        ss_guidance_strength=3,
        ss_sampling_steps=50,
        slat_guidance_strength=3,
        slat_sampling_steps=6
    ):
        if normal_image is None:
            return None, None, None

        if seed == -1:
            seed = numpy.random.randint(0, MAX_SEED)

        hi3dgen_pipeline = Hi3DGenPipeline.from_pretrained("custom_nodes/ComfyUI-Stable3DGen/weights/trellis-normal-v0-1")
        hi3dgen_pipeline.cuda()

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
