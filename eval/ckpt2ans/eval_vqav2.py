import argparse
import os
import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images

def eval_model(args):
    # Load model and processor
    model_path = args.model_path
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True
    )
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

    # Load questions
    with open(args.question_file, "r") as f:
        questions = [json.loads(line) for line in f]

    # Prepare output file
    answers_file = args.answers_file
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    with open(answers_file, "w") as ans_file:
        for i in tqdm(range(0, len(questions), args.batch_size)):
            batch_questions = questions[i:i + args.batch_size]
            conversations = [
                {
                    "role": "User",
                    "content": f"<image_placeholder>\n{question['text']}",
                    "images": [os.path.join(args.image_folder, question['image'])],
                }
                for question in batch_questions
            ]

            # Load images and prepare inputs
            pil_images = load_pil_images(conversations)
            prepare_inputs = vl_chat_processor(
                conversations=conversations, images=pil_images, force_batchify=True
            ).to(vl_gpt.device)
            # Run image encoder to get the image embeddings
            inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

            # Run the model to get the response
            outputs = vl_gpt.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=512,
                do_sample=False,
                use_cache=True,
            )

            answers = tokenizer.batch_decode(outputs.cpu(), skip_special_tokens=True)

            # Write the answers to the file
            for question, answer in zip(batch_questions, answers):
                ans_file.write(json.dumps({
                    "question_id": question["question_id"],
                    "prompt": question["text"],
                    "text": answer,
                    "image": question["image"]
                }) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="deepseek-ai/Janus-1.3B")
    parser.add_argument("--image-folder", type=str, default="/mnt/data1/fangzheng/Janus/data/test2015")
    parser.add_argument("--question-file", type=str, default="/mnt/data1/fangzheng/Janus/data/json/vqav2/llava_vqav2_mscoco_test-dev2015.jsonl")
    parser.add_argument("--answers-file", type=str, default="/mnt/data1/fangzheng/Janus/outputs/answer.jsonl")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)