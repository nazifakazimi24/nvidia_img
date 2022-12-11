import requests
import torch
from PIL import Image

from transformers import VisionEncoderDecoderModel, GPT2TokenizerFast, ViTFeatureExtractor, CLIPProcessor, CLIPModel, \
    DetrFeatureExtractor, DetrForObjectDetection


def vit_gpt2_model(image):
    """
    Deploys a ViT model with a GPT2 tokenizer for producing natural language image captions.
    Prints the results to the console.
    :param image: a .jpg image
    :return: nothing
    """

    # initializes a model, a tokenizer and a feature extractor
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    pixel_values = feature_extractor(image, return_tensors="pt").pixel_values

    # uses the generated ids from the image to create a natural language caption
    generated_ids = model.generate(pixel_values)
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(generated_text)


def clip_vit_model(image, keywords: list):
    """
    Deploys a CLIP model, using a ViT transformer as an image encoder, to match given keywords to an image with a
    certain probability.
    :param image: a .jpg image
    :param keywords: a list of keywords you wish to test the image against
    :return: nothing
    """

    # initialises CLIP model from the pretrained model by OpenAI
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    inputs = processor(text=keywords, images=image, return_tensors="pt", padding=True)

    # gets output of the model, specifically the scores for each prompt and calculates their probability in percent
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

    # prints the probabilities in a readable manner
    for tag, index in zip(keywords, range(len(keywords))):
        print(f"Probability of tag '{tag}': {round(probs[0][index].item(), 4)}")


def detr(image):
    """
    Deploys a DETR model with a ResNet-50 backbone for extracting features from an image
    :param image: a .jpg image
    :return: nothing
    """

    # initializes the feature extractor and the transformer model
    feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

    # reads the inputs for the extractor
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    target_sizes = torch.tensor([image.size[::-1]])
    results = feature_extractor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]

    # unzips the tensor so that the results can be printed in a readable fashion
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        # let's only keep detections with score > 0.9
        if score > 0.9:
            print(
                f"Detected {model.config.id2label[label.item()]} with confidence "
                f"{round(score.item(), 3)} at location {box}"
            )


if __name__ == "__main__":
    url_cats = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image_cats = Image.open("images/cats.jpeg")

    tags = ["a photo of a dog", "a photo of a cat", "a photo of a human"]

    print("___OUTPUT VIT-GPT2 MODEL:___")
    vit_gpt2_model(image_cats)
    print("___OUTPUT CLIP MODEL:___")
    clip_vit_model(image_cats, tags)
    print("___OUTPUT DETR MODEL:___")
    detr(image_cats)

