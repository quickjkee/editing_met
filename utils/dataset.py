import requests

from PIL import Image
from tqdm import tqdm
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torch.utils.data import Dataset
from io import BytesIO
from transformers import BertTokenizer
from torchvision.transforms import v2

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

# 3rd party utils
############################################################################
rnd_trans = v2.Compose([
     v2.RandomResizedCrop(size=(224, 224), antialias=True),
     v2.RandomHorizontalFlip(p=0.5),
     v2.RandomRotation(degrees=(-90, 90)),
     v2.RandomVerticalFlip(p=0.5),
     ])

def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def url_to_img(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img


def open_img(path):
    img = Image.open(path)
    return img


def init_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token': '[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens': ['[ENC]']})
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
    return tokenizer
############################################################################



class Dataset(Dataset):

    # -----------------------------------------------------
    def __init__(self, df, tokenizer=None, local_path=None, preprocess=None, is_train=False):
        if preprocess is None:
            preprocess = _transform(224)
        if local_path is None:
            print('Dataset will be loaded from urls')
        else:
            print(f'Dataset downloaded locally in {local_path}')
        if tokenizer is None:
            tokenizer = init_tokenizer()

        self.local_path = local_path
        self.is_train = is_train

        self.preprocess = preprocess
        self.df = df
        self.tokenizer = tokenizer
        self.data = self.make_data()
    # -----------------------------------------------------


    # -----------------------------------------------------
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.df)
    # -----------------------------------------------------


    # -----------------------------------------------------
    def make_data(self):
        list_of_dicts = self.df.to_dict('records')

        # Make images from urls
        bar = tqdm(range(len(list_of_dicts)), desc=f'making dataset: ')
        for j, item in enumerate(list_of_dicts):

            # img_source
            img_source_url = item['image_source']
            if self.local_path is not None:
                splitted_path = img_source_url.split('yandex-research')
                path = f'{self.local_path}/{splitted_path[-1]}'
                pil_image = open_img(path)
            else:
                pil_image = url_to_img(img_source_url)
            image_1 = self.preprocess(pil_image)
            list_of_dicts[j]['image_source'] = image_1

            # img1
            img_1_url = item['image_1']
            if self.local_path is not None:
                splitted_path = img_source_url.split('yandex-research')
                path = f'{self.local_path}/{splitted_path[-1]}'
                pil_image = open_img(path)
            else:
                pil_image = url_to_img(img_1_url)
            image_1 = self.preprocess(pil_image)
            list_of_dicts[j]['image_1'] = image_1

            # img2
            img_2_url = item['image_2']
            if self.local_path is not None:
                splitted_path = img_source_url.split('yandex-research')
                path = f'{self.local_path}/{splitted_path[-1]}'
                pil_image = open_img(path)
            else:
                pil_image = url_to_img(img_2_url)
            image_2 = self.preprocess(pil_image)
            list_of_dicts[j]['image_2'] = image_2

            # txt tgt and src
            target_prompt = list_of_dicts[j]['target_prompt']
            source_prompt = list_of_dicts[j]['source_prompt']
            text_input_source = self.tokenizer(source_prompt,
                                        padding='max_length', truncation=True, max_length=35, return_tensors="pt")
            text_input_target = self.tokenizer(target_prompt,
                                        padding='max_length', truncation=True, max_length=35, return_tensors="pt")
            list_of_dicts[j]['text_ids_source'] = text_input_source.input_ids
            list_of_dicts[j]['text_mask_source'] = text_input_source.attention_mask
            list_of_dicts[j]['text_ids_target'] = text_input_target.input_ids
            list_of_dicts[j]['text_mask_target'] = text_input_target.attention_mask

            bar.update(1)

        return list_of_dicts
    # -----------------------------------------------------