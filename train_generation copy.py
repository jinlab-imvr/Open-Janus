import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from torch.optim import AdamW
from PIL import Image
from torchvision import transforms

def preprocess_image(image_path, vl_gpt, vl_chat_processor, device, img_size=384, patch_size=16):
    """
    Preprocess image for generation.
    Args:
        image_path: str, path to the image
        vl_gpt: MultiModalityCausalLM, the model
        vl_chat_processor: VLChatProcessor, the processor
        device: torch.device, the device
        img_size: int, the image size
        patch_size: int, the patch size

    Reutrn:
        image_tokens: torch.Tensor, the image tokens
    """
    
    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)  #  (1, 3, img_size, img_size)
    image_tensor = image_tensor.to(torch.bfloat16)
    
    quan, loss, info = vl_gpt.gen_vision_model.encode(image_tensor)
    gt_tokens = info[-1].unsqueeze(0)
    
    image_tokens = gt_tokens.view(1, (img_size // patch_size) ** 2).to(torch.long)
    
    
    return image_tokens

def train_step(model, optimizer, text_prompts, target_images, device, temperature=1.0, cfg_weight=5.0, gradient_accumulation_steps=4):
    """
    Single training step (optimize memory usage, accumulate loss to avoid storing 576 logits).

    Args:
        model: MultiModalityCausalLM, the model
        optimizer: torch.optim, the optimizer
        text_prompts: List[str], the text prompts
        target_images: torch.Tensor, the target images
        device: torch.device, the device
        temperature: float, the temperature
        cfg_weight: float, the CFG weight
        gradient_accumulation_steps: int, the gradient accumulation steps

    Returns:
        float, the loss
    """
    
    input_ids = tokenizer(
        text_prompts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).input_ids.to(device)
    
    input_ids_cfg = torch.zeros((input_ids.shape[0] * 2, input_ids.shape[1]), dtype=torch.int, device=device)
    
    for i in range(input_ids.shape[0] * 2):
        input_ids_cfg[i, :] = input_ids[i // 2]
        if i % 2 != 0:
            input_ids_cfg[i, 1:-1] = vl_chat_processor.pad_id
    
    inputs_embeds = model.language_model.get_input_embeddings()(input_ids_cfg).to(dtype=torch.bfloat16)
    
    batch_size = input_ids.shape[0]
    # Manually set vocab_size to match logits.shape[-1]
    vocab_size = 16384  
    # print(f"🚀 vocab_size (manually set): {vocab_size}")
    
    past_key_values = None  
    loss_total = 0  
    
    for i in range(576):

        outputs = model.language_model.model(
            inputs_embeds=inputs_embeds,
            use_cache=False,  
        )
        hidden_states = outputs.last_hidden_state
        
        logits = model.gen_head(hidden_states[:, -1, :])
        # CFG Logits
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]
        logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
        
        probs = F.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
        img_embeds = model.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)
        
        # Accumulate the loss to save memory
        loss_total += F.cross_entropy(logits, target_images[:, i], reduction="sum")
    
    torch.cuda.empty_cache()
    
    # Normalize loss
    loss = loss_total / (batch_size * 576)
    loss.backward()
    
    if gradient_accumulation_steps > 1:
        for step in range(gradient_accumulation_steps):
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
    else:
        optimizer.step()
        optimizer.zero_grad()
    
    return loss.item()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_path = "deepseek-ai/Janus-1.3B"
    global vl_chat_processor
    vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
    global tokenizer
    tokenizer = vl_chat_processor.tokenizer
    
    model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True
    ).to(device, dtype=torch.bfloat16)
    
    optimizer = AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.01)
    
    batch_size = 1
    # Your Prompt and Image Path
    dummy_prompts = ["A meme shows decoupling vision encoder is better than single vision encoder." for _ in range(batch_size)]
    image_path = "/mnt/data1/fangzheng/Janus/images/doge.png"  
    image_tokens = preprocess_image(image_path, model, vl_chat_processor, device)
    
    loss = train_step(model, optimizer, dummy_prompts, image_tokens, device)
    for i in range(10):
        loss = train_step(model, optimizer, dummy_prompts, image_tokens, device)

        print(f"✅ Training Success, Loss: {loss:.4f}")
    
if __name__ == "__main__":
    main()

