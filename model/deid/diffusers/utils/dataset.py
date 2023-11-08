from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        size=512,
        center_crop=False,
        instance_mask_root=None,
        class_mask_root=None,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if instance_mask_root is not None:
            self.instance_mask_root = Path(instance_mask_root)
            self.instance_mask_root.mkdir(parents=True, exist_ok=True)
            self.instance_masks_path = list(self.instance_mask_root.iterdir())
            self.num_instance_masks = len(self.instance_masks_path)
        else:
            self.class_mask_root = None

        if class_mask_root is not None:
            self.class_mask_root = Path(class_mask_root)
            self.class_mask_root.mkdir(parents=True, exist_ok=True)
            self.class_masks_path = list(self.class_mask_root.iterdir())
            self.num_class_masks = len(self.class_masks_path)
        else:
            self.class_mask_root = None

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        self.image_transforms_resize_and_crop = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            ]
        )

        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        instance_image = self.image_transforms_resize_and_crop(instance_image)
        example["PIL_images"] = instance_image
        example["instance_images"] = self.image_transforms(instance_image)

        if self.instance_mask_root:
            instance_mask = Image.open(self.instance_masks_path[index % self.num_instance_masks])
            if not instance_mask.mode == "RGB":
                instance_mask = instance_mask.convert("RGB")
            instance_mask = self.image_transforms_resize_and_crop(instance_mask)
            example["PIL_masks"] = instance_mask

        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            class_image = self.image_transforms_resize_and_crop(class_image)
            example["class_images"] = self.image_transforms(class_image)
            example["class_PIL_images"] = class_image

            if self.class_mask_root:
                class_mask = Image.open(self.class_masks_path[index % self.num_class_masks])
                if not class_mask.mode == "RGB":
                    class_mask = class_mask.convert("RGB")
                class_mask = self.image_transforms_resize_and_crop(class_mask)
                example["class_PIL_masks"] = class_mask

            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids

        return example


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example