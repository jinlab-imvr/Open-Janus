import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM
from accelerate import Accelerator
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images


accelerator = Accelerator(mixed_precision="bf16")  
device = accelerator.device


model_path = "deepseek-ai/Janus-1.3B"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
).to(device)

vl_gpt.train()  
for name, param in vl_gpt.named_parameters():
    if "gen_embed" in name:  # 
        print(f"Parameter: {name}, Shape: {param.shape}, Requires Grad: {param.requires_grad}")
        # freeze gen_embed parameters
        param.requires_grad = False
        # check if the parameters are frozen
        print(f"Parameter: {name}, Shape: {param.shape}, Requires Grad: {param.requires_grad}")


lr = 1e-4 
optimizer = optim.AdamW(vl_gpt.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
criterion = nn.CrossEntropyLoss(ignore_index=-100)  
gradient_clip = 1.0


vl_gpt, optimizer = accelerator.prepare(vl_gpt, optimizer)


def train_step(model, optimizer, criterion):
    model.train()
    optimizer.zero_grad()


    conversation = [
        {
            "role": "User",
            "content": "<image_placeholder>\nConvert the formula into latex code.",
            "images": ["images/equation.png"],
        },
        {"role": "Assistant", "content": ""},
    ]


    pil_images = load_pil_images(conversation)


    prepare_inputs = vl_chat_processor(
        conversations=conversation, images=pil_images, force_batchify=True
    ).to(device) 


    model = model.module if hasattr(model, "module") else model  
    model = model.to(torch.bfloat16)

    inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)


    with accelerator.autocast():  
        outputs = model.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
        )
        logits = outputs.logits  # (batch_size, seq_len, vocab_size)


    labels = prepare_inputs.input_ids.clone().detach()
    labels[labels == tokenizer.pad_token_id] = -100  
    loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

    accelerator.backward(loss)  
    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)  
    optimizer.step()
    optimizer.zero_grad()

    return loss.item()

loss = train_step(vl_gpt, optimizer, criterion)
print(f"Training loss: {loss}")

for i in range(10):
    loss = train_step(vl_gpt, optimizer, criterion)
    print(f"Training loss: {loss}")