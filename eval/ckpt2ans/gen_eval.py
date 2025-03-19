import argparse
import json
import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor

# Uncomment the following line to enable deterministic behavior
from pytorch_lightning import seed_everything

# specify the path to the model
model_path = "deepseek-ai/Janus-1.3B"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

@torch.inference_mode()
def generate(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    temperature: float = 1,
    n_samples: int = 16,
    cfg_weight: float = 5,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
):
    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids)

    tokens = torch.zeros((n_samples*2, len(input_ids)), dtype=torch.int).cuda()
    for i in range(n_samples*2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id

    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)

    generated_tokens = torch.zeros((n_samples, image_token_num_per_image), dtype=torch.int).cuda()

    for i in range(image_token_num_per_image):
        outputs = mmgpt.language_model.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)
        hidden_states = outputs.last_hidden_state
        
        logits = mmgpt.gen_head(hidden_states[:, -1, :])
        
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]
        
        logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
        probs = torch.softmax(logits / temperature, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)

        next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)

    dec = mmgpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[n_samples, 8, img_size//patch_size, img_size//patch_size])
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

    dec = np.clip((dec + 1) / 2 * 255, 0, 255)

    visual_img = np.zeros((n_samples, img_size, img_size, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec

    return visual_img

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "metadata_file",
        type=str,
        help="JSONL file containing lines of metadata for each prompt"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature for sampling",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=4,
        help="number of parallel generations",
    )
    parser.add_argument(
        "--cfg_weight",
        type=float,
        default=5.0,
        help="classifier-free guidance weight",
    )
    parser.add_argument(
        "--image_token_num_per_image",
        type=int,
        default=576,
        help="number of image tokens per image",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=384,
        help="image size",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=16,
        help="patch size",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    opt = parser.parse_args()
    return opt

def main(opt):
    # Load prompts
    with open(opt.metadata_file) as fp:
        metadatas = [json.loads(line) for line in fp]

    for index, metadata in enumerate(metadatas):
        # Uncomment the following line to enable deterministic behavior
        seed_everything(opt.seed)

        outpath = os.path.join(opt.outdir, f"{index:0>5}")
        os.makedirs(outpath, exist_ok=True)

        prompt = metadata['prompt']
        print(f"Prompt ({index: >3}/{len(metadatas)}): '{prompt}'")

        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
            json.dump(metadata, fp)

        visual_img = generate(
            vl_gpt,
            vl_chat_processor,
            prompt,
            temperature=opt.temperature,
            n_samples=opt.n_samples,
            cfg_weight=opt.cfg_weight,
            image_token_num_per_image=opt.image_token_num_per_image,
            img_size=opt.img_size,
            patch_size=opt.patch_size,
        )

        for i in range(opt.n_samples):
            save_path = os.path.join(sample_path, f"000{i}.png")
            Image.fromarray(visual_img[i]).save(save_path)

    print("Done.")

if __name__ == "__main__":
    opt = parse_args()
    main(opt)