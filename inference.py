import torch
import json
from transformers import AutoTokenizer
from model.mini_gemini_gemma import MiniGeminiGemmaForCausalLM
from tqdm import tqdm


def prepare_image_embeddings(image_tokens, model):
    """
    image_tokens: Tensor [B, 3, 8, 256]  --> 3 images per sample
    return: Tensor [B, 3*8*256, hidden_dim]
    """
    B, N, C, T = image_tokens.shape  # [B, 3, 8, 256]
    image_tokens = image_tokens.view(B * N, C, T).to(torch.long).to(model.device)  # [B*3, 8, 256]

    with torch.no_grad():
        image_embed = model.image_token_embedding(image_tokens)  # [B*3, 256, dim]
    image_embed = image_embed.view(B, N, C * T, -1)  # [B, 3, 2048, dim]
    image_embed = image_embed.view(B, -1, image_embed.size(-1))  # [B, 3*2048, dim]
    return image_embed


def build_prompts_and_masks(prompts, tokenizer, model):
    """
    tokenizer text, get text embedding and masks
    return:
        input_ids: [B, L]
        text_embeds: [B, L, D]
        attn_mask: [B, L]
    """
    encoded = tokenizer(prompts, return_tensors='pt', padding=True).to(model.device)
    input_ids = encoded['input_ids']
    attn_mask = encoded['attention_mask']

    with torch.no_grad():
        text_embeds = model.language_model.embed_tokens(input_ids)  # [B, L, D]

    return input_ids, text_embeds, attn_mask


def main():
    # ==== Load input ====
    image_token_path = 'vq_codes.pt'          # [B, 3, 8, 256]
    waypoint_path = 'waypoints.json'          # [[(x,y), (x,y), (x,y)]]
    model_path = 'your_model_path'
    output_file = 'output_predictions.txt'

    # ==== Load model ====
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.add_special_tokens({'additional_special_tokens': ['<boi>', '<eoi>', '<bos>', '<eos>']})

    model = MiniGeminiGemmaForCausalLM.from_pretrained(
        model_path,
        attn_implementation='flash_attention_2',
        torch_dtype=torch.bfloat16,
        device_map='auto'
    )
    model.eval()

    image_tokens = torch.load(image_token_path)  # [B, 3, 8, 256]
    with open(waypoint_path, 'r') as f:
        waypoints = json.load(f)

    # ==== Prepare inputs ====
    prompts = []
    for i in range(len(waypoints)):
        part = []
        for j in range(3):
            xy = waypoints[i][j]
            part.append(f"<boi> <eoi>[{xy[0]}, {xy[1]}]")
        text = (
            "<bos>human:We already know three frames and their waypoints:\n"
            + ",\n".join(part)
            + "\nnow predict the next frames, its waypoint is\n"
            + "gpt:"
        )
        prompts.append(text)

    input_ids, text_embeds, text_mask = build_prompts_and_masks(prompts, tokenizer, model)
    image_embeds = prepare_image_embeddings(image_tokens, model)  # [B, N, D]
    image_mask = torch.ones(image_embeds.shape[:-1], dtype=torch.long, device=model.device)

    # ==== Concat multimodal ====
    input_embeds = torch.cat([text_embeds, image_embeds], dim=1)
    attn_mask = torch.cat([text_mask, image_mask], dim=1)

    # ==== Inference ====
    outputs = model.generate(
        inputs_embeds=input_embeds,
        attention_mask=attn_mask,
        max_new_tokens=128,
        do_sample=True,
        top_p=0.95,
        top_k=50,
        temperature=0.9,
        eos_token_id=tokenizer.eos_token_id,
    )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    with open(output_file, 'w') as f:
        for r in decoded:
            f.write(r + '\n')
    print(f"Saved predictions to {output_file}")


if __name__ == "__main__":
    main()
