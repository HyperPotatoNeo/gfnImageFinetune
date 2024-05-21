import argparse
import os
import glob
from PIL import Image
import torch
from transformers import CLIPTokenizer, CLIPModel, CLIPProcessor
import ImageReward as imagereward
from diffusers import UNet2DConditionModel
from pipeline_stable_diffusion_extended import StableDiffusionPipelineExtended
from scheduling_ddim_extended import DDIMSchedulerExtended
import dpok_utils as utils

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation script.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="runwayml/stable-diffusion-v1-5", help="Path to pretrained model.")
    parser.add_argument("--revision", type=str, default=None, help="Revision of pretrained model.")
    parser.add_argument("--task", type=str, default="four_roses", help="Evaluation task.")
    parser.add_argument("--model", type=str, default="dpok", help="Model type.")
    return parser.parse_args()

def calculate_reward(pipe, args, reward_tokenizer, tokenizer, weight_dtype, reward_clip_model, image_reward, imgs, prompts, test_flag=False):
    image_pil = imgs if test_flag else pipe.numpy_to_pil(imgs)[0]
    blip_reward, _ = utils.image_reward_get_reward(image_reward, image_pil, prompts, weight_dtype)
    inputs = reward_tokenizer(prompts, max_length=tokenizer.model_max_length, padding="do_not_pad", truncation=True)
    padded_tokens = reward_tokenizer.pad({"input_ids": inputs.input_ids}, padding=True, return_tensors="pt")
    txt_emb = reward_clip_model.get_text_features(input_ids=padded_tokens.input_ids.to("cuda").unsqueeze(0))
    return blip_reward.cpu().squeeze(0).squeeze(0), txt_emb.squeeze(0).to("cuda")

def main():
    args = parse_args()
    weight_dtype = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision)
    reward_clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    reward_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    reward_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    image_reward = imagereward.load("ImageReward-v1.0")
    image_reward.requires_grad_(False).to(device, dtype=weight_dtype)
    reward_clip_model.requires_grad_(False).to(device)

    pipe = StableDiffusionPipelineExtended.from_pretrained(args.pretrained_model_name_or_path, torch_dtype=weight_dtype).to(device)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision).to(device)
    pipe.scheduler = DDIMSchedulerExtended.from_config(pipe.scheduler.config)
    unet.requires_grad_(False)
    unet.eval()

    single_prompt = {
        "cat_dog": 'A cat and a dog',
        "green_rabbit": 'A green colored rabbit',
        "four_roses": 'Four roses',
        "cat_on_mars": 'A cat on mars'
    }.get(args.task, 'Four roses')

    image_paths = glob.glob(f"test_output/{args.task}/{args.model}/*.png")
    images = [Image.open(x).convert('RGB') for x in image_paths]

    reward_list = [calculate_reward(pipe, args, reward_tokenizer, tokenizer, weight_dtype, reward_clip_model, image_reward, img, single_prompt, test_flag=True)[0] for img in images]
    reward_list = torch.stack(reward_list).detach().cpu()
    print(f"Average reward: {reward_list.mean()} ± {reward_list.std()}")

if __name__ == "__main__":
    main()
