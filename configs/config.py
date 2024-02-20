from types import SimpleNamespace
import yaml
import dataclasses


class Config:
    def __init__(self, yaml_path):
        with open(yaml_path, "r") as yamlfile:
            self.yaml_config = yaml.load(yamlfile, Loader=yaml.FullLoader)
            self.config = self._to_obj(self.yaml_config)

    def __getattr__(self, item):
        return self.config.__getattribute_

    def __getattr__(self, item):
        return self.config.__getattribute__(item)

    def _to_obj(self, d):
        if isinstance(d, list):
            d = [self._to_obj(x) for x in d]

        if isinstance(d, dict):
            obj = SimpleNamespace()
            for k in d:
                obj.__dict__[k] = self._to_obj(d[k])
            return obj
        else:
            return d

    def _to_dict(self, obj):
        if isinstance(obj, list):
            return [self._to_dict(x) for x in obj]
        elif hasattr(obj, "__dict__"):
            attrs = obj.__dict__
            if len(attrs) == 0:
                return obj
            else:
                return {k: self._to_dict(obj.__getattribute__(k)) for k in attrs}
        else:
            return obj

    def _update_yaml(self):
        self.yaml_config = self._to_dict(self.config)

    def write_yaml(self, yaml_path):
        with open(yaml_path, 'w') as yamlfile:
            self._update_yaml()
            yaml.dump(self.yaml_config, yamlfile)
            yamlfile.close()

    def __str__(self, indent=1):
        self._update_yaml()
        return yaml.dump(self.yaml_config, allow_unicode=True, default_flow_style=False)


class FeaturesConfig(Config):
    @dataclasses.dataclass
    class DataClass:
        dataset: str
        datasets_path: str
        debug_path: str
        result_path: str
        debug: bool
        save_object_images: bool
        num_query_images: int
        num_db_images: int

    @dataclasses.dataclass
    class FeaturesClass:
        interpolation_mode: str
        clip_embedding_size: int
        clip_input_size: int
        clip_model: str
        clip_model_weights: str
        fastsam_channels: int
        fastsam_input_size: int
        fastsam_model_weights: str
        fastsam_features: list
        use_clip_features: bool
        use_fastsam_features: bool
        flip: bool
        use_masks_fastsam: bool
        use_masks_clip: bool
        use_context_clip: bool
        use_context_fastsam: bool
        use_gt_masks: bool
        context_clip: str
        context_fastsam: str
        context_crop_factor_small: float
        context_crop_factor_medium: float
        small_box_area: int
        medium_box_area: int
        zoom: str
        zoom_factor: float

    data: DataClass
    features: FeaturesClass


class HashingConfig(Config):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config.train.config_path = kwargs["yaml_path"]

    @dataclasses.dataclass
    class ModelClass:
        hash_size: int
        fastsam_model_weights: str
        fastsam_channels: int
        fastsam_output_size: int
        fastsam_input_size: int
        clip_model: str
        clip_model_weights: str
        clip_input_size: int
        clip_embedding_size: int
        hash_method: str
        mask_features_method: str
        fc_layer: bool
        fc_size: int

    @dataclasses.dataclass
    class FeaturesClass:
        context_clip: str
        context_crop_factor_small: int
        context_crop_factor_medium: float
        small_box_area: int
        medium_box_area: int

    @dataclasses.dataclass
    class DataClass:
        val_datasets_path: str
        label_path: str
        images_path: str

    @dataclasses.dataclass
    class TrainClass:
        masks_per_image: int
        num_workers: int
        num_epochs: int
        batch_factor: int
        batch_size: int
        log_steps: int
        ckpt_steps: int
        top_k: int
        limit_batches: int
        weight_decay: int
        lr: float
        optimizer: str

    @dataclasses.dataclass
    class ValClass:
        num_workers: int
        masks_per_image: int
        num_images: int
        batch_size: int
        steps: int
        retrieval_queries: int
        retrieval_db: int
        temp_image_dir: str
        save_images: bool

    @dataclasses.dataclass
    class VisClass:
        num_queries: int
        num_retrievals: int

    model: ModelClass
    features: FeaturesClass
    data: DataClass
    train: TrainClass
    val: ValClass
    vis: VisClass


# if __name__ == "__main__":
#     # Read and write config
#     # cfg = HashingConfig("256bit_masked.yaml")
#     # cfg.write_yaml("test.yaml")
#
#     # Generate config class
#     #from base import Config
#
#     #config = Config.from_yaml("test.yaml")
