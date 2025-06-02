import random
import json
import albumentations as A
import numpy as np
from transformers import BertTokenizer
from transformers import ViTFeatureExtractor
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import Image
import torch.utils.data as data
import os
import re
import torch


class FieldParser:
    def __init__(
            self,
            args
    ):
        super().__init__()
        self.args = args
        self.BASE_DIR = args.base_dir
        self.tokenizer = BertTokenizer.from_pretrained(
            self.args.bert_dir, use_fast=True, cache_dir=self.args.bert_cache_dir
        )

        self.vit_feature_extractor = ViTFeatureExtractor.from_pretrained(
            self.args.vit_dir, cache_dir=self.args.vit_cache_dir,
            size=(self.args.image_width, self.args.image_height), do_normalize=True,
            image_mean=IMAGENET_DEFAULT_MEAN, image_std=IMAGENET_DEFAULT_STD
        )

        self.image_argument_funcs = [
            A.RandomBrightnessContrast(p=0.1),
            A.ToGray(p=0.1),
            A.ColorJitter(p=0.1),
            A.RandomResizedCrop(180, 180, p=0.1),
            A.HorizontalFlip(p=0.1),
            A.ImageCompression(quality_lower=50, quality_upper=80, p=0.1),
            A.GaussNoise(p=0.1),
            A.ToSepia(p=0.1),
            A.FancyPCA(p=0.1),
            A.RGBShift(p=0.1),
            A.Sharpen(p=0.1),
            A.CoarseDropout(p=0.1)
        ]

        self.image_argument_funcs2 = [
            A.RandomBrightnessContrast(p=1),
            A.ToGray(p=1),
            A.ColorJitter(p=1),
            A.RandomResizedCrop(180, 180, p=1),
            A.HorizontalFlip(p=1),
            A.ImageCompression(quality_lower=50, quality_upper=80, p=1),
            A.GaussNoise(p=1),
            A.ToSepia(p=1),
            A.FancyPCA(p=1),
            A.RGBShift(p=1),
            A.Sharpen(p=1),
            A.CoarseDropout(p=0.1)
        ]

        self.ans2idx = json.load(open('/root/VQA_Main/Modified_MedVQA-main/data/ans2label_create.json','r'))
        self.ans2idx_closed = json.load(open('/root/VQA_Main/Modified_MedVQA-main/data/ans2label_closeded.json','r'))
        self.ans2idx_opended = json.load(open('/root/VQA_Main/Modified_MedVQA-main/data/ans2label_opened.json','r'))

        self.manual_map = { 'none': '0',
              'zero': '0',
              'one': '1',
              'two': '2',
              'three': '3',
              'four': '4',
              'five': '5',
              'six': '6',
              'seven': '7',
              'eight': '8',
               'nine': '9',
              'ten': '10'}
        self.articles = ['a', 'an', 'the']
        self.period_strip = re.compile("(?!<=\d)(\.)(?!\d)")
        self.comma_strip = re.compile("(\d)(\,)(\d)")
        self.punct = [';', r"/", '[', ']', '"', '{', '}',
                        '(', ')', '=', '+', '\\', '_', '-',
                        '>', '<', '@', '`', ',', '?', '!']
        self.contractions = {
            "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve":
            "could've", "couldnt": "couldn't", "couldn'tve": "couldn't've",
            "couldnt've": "couldn't've", "didnt": "didn't", "doesnt":
            "doesn't", "dont": "don't", "hadnt": "hadn't", "hadnt've":
            "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent":
            "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve":
            "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll",
            "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", "Im":
            "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've":
            "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's",
            "maam": "ma'am", "mightnt": "mightn't", "mightnt've":
            "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
            "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't",
            "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
            "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat":
            "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve":
            "she'd've", "she's": "she's", "shouldve": "should've", "shouldnt":
            "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve":
            "shouldn't've", "somebody'd": "somebodyd", "somebodyd've":
            "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll":
            "somebody'll", "somebodys": "somebody's", "someoned": "someone'd",
            "someoned've": "someone'd've", "someone'dve": "someone'd've",
            "someonell": "someone'll", "someones": "someone's", "somethingd":
            "something'd", "somethingd've": "something'd've", "something'dve":
            "something'd've", "somethingll": "something'll", "thats":
            "that's", "thered": "there'd", "thered've": "there'd've",
            "there'dve": "there'd've", "therere": "there're", "theres":
            "there's", "theyd": "they'd", "theyd've": "they'd've", "they'dve":
            "they'd've", "theyll": "they'll", "theyre": "they're", "theyve":
            "they've", "twas": "'twas", "wasnt": "wasn't", "wed've":
            "we'd've", "we'dve": "we'd've", "weve": "we've", "werent":
            "weren't", "whatll": "what'll", "whatre": "what're", "whats":
            "what's", "whatve": "what've", "whens": "when's", "whered":
            "where'd", "wheres": "where's", "whereve": "where've", "whod":
            "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl":
            "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll",
            "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve":
            "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
            "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll":
            "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've",
            "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd":
            "you'd", "youd've": "you'd've", "you'dve": "you'd've", "youll":
            "you'll", "youre": "you're", "youve": "you've"
        }
    def _parse_image(self, img):
        outputs = self.vit_feature_extractor(img.transpose([2, 0, 1]))  # (3, width, height)
        return outputs["pixel_values"][0]

    def _do_argument(self, image, always_apply=False):
        if not always_apply:
            img_argument = random.choice(self.image_argument_funcs)
        else:
            img_argument = random.choice(self.image_argument_funcs2)
        try:
            transformed = img_argument(image=image)  # (width, height, 3)
            transformed_image = transformed["image"]  # (width, height, 3)
        except Exception as e:
            transformed_image = image
        return transformed_image


    def _parse_question(self, text1, max_length=64):

        encoded_inputs = self.tokenizer(text1, max_length=max_length, padding='max_length', truncation=True)
        input_ids = np.array(encoded_inputs['input_ids'])
        mask = np.array(encoded_inputs['attention_mask'])
        token_type_ids = np.array(encoded_inputs["token_type_ids"])

        to_return = {
            "question_input_ids": input_ids,
            "question_mask": mask,
            "question_token_type_ids": token_type_ids
        }
        return to_return
    

    def _parse_answer(self, text1, max_length=64):
        encoded_inputs = self.tokenizer(text1, max_length=max_length, padding='max_length', truncation=True)
        input_ids = np.array(encoded_inputs['input_ids'])
        input_ids[input_ids == 101] = 0
        input_ids[input_ids == 102] = 0
        mask = np.array(encoded_inputs['attention_mask'])
        token_type_ids = np.array(encoded_inputs["token_type_ids"])
        target_id = np.append(input_ids[1:], 0)
        to_return = {
            "answer_input_ids": input_ids,
            "answer_mask": mask,
            "answer_token_type_ids": token_type_ids,
            "target_ids": target_id
        }
        return to_return


    def _clean_report(self, report):
        report = str(report)
        report_cleaner = lambda t: t.strip().lower()
        return report_cleaner(report)


    # for generation task 
    def parse(self, features, training=False):
        to_return = {'id': str(features['qid'])}
        question = self._clean_report(features.get('question', ''))
        answer = self._clean_report(features.get('answer', ''))
        
        question_feat = self._parse_question(question, max_length=self.args.bert_question_max_length)
        to_return.update(question_feat)
        answer_feat= self._parse_answer(answer, max_length=self.args.bert_answer_max_length)
        to_return.update(answer_feat)

        # chest x-ray images
        try:
            with Image.open(os.path.join(self.BASE_DIR, features['image_name'])) as pil:
                array = np.array(pil, dtype=np.uint8)
                if array.shape[-1] != 3 or len(array.shape) != 3:
                    array = np.array(pil.convert("RGB"), dtype=np.uint8)
                features['image'] = array
                to_return["image_mask"] = 0
        except Exception as e:
            print('can not find image')
            features['image'] = np.zeros((self.args.image_width, self.args.image_height, 3), dtype=np.uint8)
            to_return["image_mask"] = 1

        if training:
            to_return["pixel_values"] = self._parse_image(self._do_argument(features['image']))
        else:
            to_return["pixel_values"] = self._parse_image(features['image'])
    
        return to_return


    def parse_YN(self, features, training=False):
        to_return = {'id': str(features['qid'])}
        question = self._clean_report(features.get('question', ''))

        question_feat = self._parse_question(question, max_length=self.args.bert_question_max_length)
        to_return.update(question_feat)

        answer = self._clean_report(features.get('answer', ''))
        if answer == 'yes':
            to_return['target_ids'] = torch.tensor([1.0, 0.0])
        else:
            to_return['target_ids'] = torch.tensor([0.0, 1.0])
        # chest x-ray images
        try:
            with Image.open(os.path.join(self.BASE_DIR, features['image_name'])) as pil:
                array = np.array(pil, dtype=np.uint8)
                if array.shape[-1] != 3 or len(array.shape) != 3:
                    array = np.array(pil.convert("RGB"), dtype=np.uint8)
                features['image'] = array
                to_return["image_mask"] = 0
        except Exception as e:
            print('can not find image')
            features['image'] = np.zeros((self.args.image_width, self.args.image_height, 3), dtype=np.uint8)
            to_return["image_mask"] = 1

        # if training:
        #     to_return["pixel_values"] = self._parse_image(self._do_argument(features['image']))
        # else:
        to_return["pixel_values"] = self._parse_image(features['image'])

        return to_return

    def parse_closed(self, features, training=False):
        to_return = {'id': str(features['qid'])}
        question = self._clean_report(features.get('question', ''))

        question_feat = self._parse_question(question, max_length=self.args.bert_question_max_length)
        to_return.update(question_feat)
        # print("=================features is : ================",features)
        closed_answer = features.get('answer', '')
        # if closed_answer is not None:
        answer_idx = torch.tensor(self.ans2idx_closed[str(closed_answer)])
        # print(answer_idx)
        to_return['target_ids'] = torch.nn.functional.one_hot(answer_idx, 57)

        # chest x-ray images
        try:
            with Image.open(os.path.join(self.BASE_DIR, features['image_name'])) as pil:
                array = np.array(pil, dtype=np.uint8)
                if array.shape[-1] != 3 or len(array.shape) != 3:
                    array = np.array(pil.convert("RGB"), dtype=np.uint8)
                features['image'] = array
                to_return["image_mask"] = 0
        except Exception as e:
            print('can not find image')
            features['image'] = np.zeros((self.args.image_width, self.args.image_height, 3), dtype=np.uint8)
            to_return["image_mask"] = 1

        # if training:
        #     to_return["pixel_values"] = self._parse_image(self._do_argument(features['image']))
        # else:
        to_return["pixel_values"] = self._parse_image(features['image'])

        return to_return

    def parse_opened(self, features, training=False):       
        to_return = {'id': str(features['qid'])}
        question = self._clean_report(features.get('question', ''))
        answer_type = self._clean_report(features.get('answer_type', ''))
        question_feat = self._parse_question(question, max_length=self.args.bert_question_max_length)
        to_return.update(question_feat)
        to_return["answer_type"] = answer_type
        opened_answer = features.get('answer', '')
        answer_idx = torch.tensor(self.ans2idx_opended[str(opened_answer)])
        # print(answer_idx)
        to_return['target_ids'] = torch.nn.functional.one_hot(answer_idx, 424)

        # chest x-ray images
        try:
            with Image.open(os.path.join(self.BASE_DIR, features['image_name'])) as pil:
                array = np.array(pil, dtype=np.uint8)
                if array.shape[-1] != 3 or len(array.shape) != 3:
                    array = np.array(pil.convert("RGB"), dtype=np.uint8)
                features['image'] = array
                to_return["image_mask"] = 0
        except Exception as e:
            print('can not find image')
            features['image'] = np.zeros((self.args.image_width, self.args.image_height, 3), dtype=np.uint8)
            to_return["image_mask"] = 1

        # if training:
        #     to_return["pixel_values"] = self._parse_image(self._do_argument(features['image']))
        # else:
        to_return["pixel_values"] = self._parse_image(features['image'])

        return to_return
    def parse_all(self, features, training=False):
        to_return = {'id': str(features['qid'])}
        question = self._clean_report(features.get('question', ''))
        
        
        question_feat = self._parse_question(question, max_length=self.args.bert_question_max_length)
        to_return.update(question_feat)

        cur_answer = self.preprocess_answer(features.get('answer', ''))
        # if str(features.get('answer', '')) in self.ans2idx:
        answer_idx = torch.tensor(self.ans2idx[str(cur_answer)])
        to_return['target_ids'] = torch.nn.functional.one_hot(answer_idx, 458)
            
        to_return['answer_type'] = features['answer_type']
        to_return['image_name'] = features['image_name']
        try:
            with Image.open(os.path.join(self.BASE_DIR, features['image_name'])) as pil:
                array = np.array(pil, dtype=np.uint8)
                if array.shape[-1] != 3 or len(array.shape) != 3:
                    array = np.array(pil.convert("RGB"), dtype=np.uint8)
                features['image'] = array
                to_return["image_mask"] = 0
        except Exception as e:
            print('can not find image')
            features['image'] = np.zeros((self.args.image_width, self.args.image_height, 3), dtype=np.uint8)
            to_return["image_mask"] = 1

        # if training:
        #     to_return["pixel_values"] = self._parse_image(self._do_argument(features['image']))
        # else:
        to_return["pixel_values"] = self._parse_image(features['image'])

        return to_return

    def transform_with_parse(self, inputs, training=True):
        # cur_answer = self.preprocess_answer(inputs.get('answer', ''))
        # if str(cur_answer) in self.ans2idx:
        #     
        return self.parse_all(inputs, training)
        # return self.parse_YN(inputs, training)
        # return self.parse_closed(inputs, training)
        # return self.parse_opened(inputs, training)
    
    def preprocess_answer(self, answer):
        answer = str(answer)
        answer = self.process_digit_article(self.process_punctuation(answer))
        answer = answer.replace(',', '').replace('x ray', 'xray')
        return answer
    
    def process_digit_article(self, inText):
        outText = []
        tempText = inText.lower().split()
        for word in tempText:
            word = self.manual_map.setdefault(word, word)
            if word not in self.articles:
                outText.append(word)
            else:
                pass
        for wordId, word in enumerate(outText):
            if word in self.contractions:
                outText[wordId] = self.contractions[word]
        outText = ' '.join(outText)
        return outText    
    
    def process_punctuation(self, inText):
        outText = inText
        for p in self.punct:
            if (p + ' ' in inText or ' ' + p in inText) \
            or (re.search(self.comma_strip, inText) != None):
                outText = outText.replace(p, '')
            else:
                outText = outText.replace(p, ' ')
        outText = self.period_strip.sub("", outText, re.UNICODE)
        return outText    
    
    
# for mimic_cxr dataset
class ParseDataset(data.Dataset):
    def __init__(self, args, split='train'):
        self.args = args
        self.meta = json.load(open(args.annotation, 'r'))
        self.meta = self.meta[split]
        # self.meta = [i for i in self.meta if "OPEN" in i['answer_type']]
        # self.meta = [i for i in self.meta if "CLOSED" in i['answer_type']]
        # self.meta = [i for i in self.meta if "CLOSED" in i['answer_type']]
        # self.meta = [i for i in self.meta if i['answer'].lower() in ['yes', 'no']]
        self.parser = FieldParser(args)
        self.training = True if split == 'train' else False

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        return self.parser.transform_with_parse(self.meta[index], self.training)


# create datasets for mimic_cxr
def create_datasets(args):
    train_dataset = ParseDataset(args, 'train')
    dev_dataset = ParseDataset(args, 'test')
    test_dataset = ParseDataset(args, 'test')
    return train_dataset, dev_dataset, test_dataset

    
if __name__ == '__main__':
    data = json.load(open('/root/VQA_Main/Modified_MedVQA-main/data/VQA_RAD Dataset Public.json', 'r'))