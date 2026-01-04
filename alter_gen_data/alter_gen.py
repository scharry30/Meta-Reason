import __init__
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer



def generate_hf(
    model, tokenizer, input_ids, max_new_tokens=10,
    temperature=1.0, top_k=50, top_p=0.95
):

    generated_ids_hf = model.generate(
        input_ids=input_ids,
        attention_mask=(input_ids != tokenizer.pad_token_id).long(),  # Ensure mask aligns with pad tokens
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=(temperature > 0 and (top_k > 0 or top_p < 1.0)),  
        use_cache=True,
        return_dict_in_generate=False
    )
    return generated_ids_hf

def generate_with_partial_kv(
    model, tokenizer, input_ids, past_key_values=None, max_new_tokens=10,
    temperature=1.0, top_k=50, top_p=0.95
):
    device = input_ids.device
    
    # input_ids
    if input_ids.numel() == 0 or input_ids.shape[1] == 0:
        raise ValueError("input_ids cannot be empty")
    
    seq_len = input_ids.shape[1]

    # Step 1:  KV Cache
    if past_key_values is None:
        #  KV Cache
        if seq_len > 1:
            with torch.no_grad():
                outputs = model(input_ids=input_ids[:, :-1], use_cache=True, return_dict=True)
                past_key_values = outputs.past_key_values
    else:
        #  token
        cached_len = past_key_values[0][0].shape[2]  # KV Cache  token 
        if cached_len < seq_len - 1:
            new_input_ids = input_ids[:, cached_len:-1] 
            if new_input_ids.shape[1] > 0:  # token 
                with torch.no_grad():
                    outputs = model(input_ids=new_input_ids, past_key_values=past_key_values, use_cache=True, return_dict=True)
                    past_key_values = outputs.past_key_values

    # Step 2: `do_sample`
    do_sample = temperature > 0 and (top_k > 0 or top_p < 1.0)
    
    # Step 3: new token
    try:
        output = model.generate(
            input_ids=input_ids,  # last token
            attention_mask= (input_ids != tokenizer.pad_token_id).long(),  # Ensure mask aligns with pad tokens torch.ones_like(input_ids).long(), #
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            use_cache=True,
            return_dict_in_generate=True,
            past_key_values=past_key_values,
            pad_token_id=tokenizer.eos_token_id
        )
    except Exception as e:
        print(f"Error in model.generate: {e}")
        print(f"past_key_values type: {type(past_key_values)}")
        if past_key_values is not None:
            print(f"past_key_values length: {len(past_key_values)}")
            print(f"first layer shape: {past_key_values[0][0].shape if len(past_key_values) > 0 else 'N/A'}")
    # Step 4:  token ID
    generated_ids = output.sequences
    past_key_values = output.past_key_values

    return generated_ids, past_key_values

if __name__ == '__main__':

    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id 
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")


    prompt = "Once upon a time"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")


    split_token_count = 5  
    added_tokens_count = 3  
    final_token_count = 10  
    temperature = 0
    top_k = 50
    top_p = 0.95


    generated_ids_hf1 = model.generate(
        input_ids=input_ids,
        max_new_tokens=split_token_count,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=(temperature > 0 and (top_k > 0 or top_p < 1.0)),
        use_cache=True,
        return_dict_in_generate=False
    )


    generated_ids_kv1, kv_cache = generate_with_partial_kv(
        model, tokenizer, input_ids, None, max_new_tokens=split_token_count,
        temperature=temperature, top_k=top_k, top_p=top_p
    )


    first_match = torch.equal(generated_ids_kv1, generated_ids_hf1)


    added_tokens = tokenizer([''], return_tensors="pt").input_ids.to("cuda")

    generated_ids_kv1 = torch.cat([generated_ids_kv1, added_tokens], dim=-1)


    generated_ids_hf2 = model.generate(
        input_ids=generated_ids_kv1,
        max_new_tokens=final_token_count,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=(temperature > 0 and (top_k > 0 or top_p < 1.0)),
        use_cache=True,
        return_dict_in_generate=False
    )

    generated_ids_kv2, kv_cache = generate_with_partial_kv(
        model, tokenizer, generated_ids_kv1, kv_cache, max_new_tokens=final_token_count,
        temperature=temperature, top_k=top_k, top_p=top_p
    )



    final_match = torch.equal(generated_ids_kv2, generated_ids_hf2)


    generated_text_kv = tokenizer.decode(generated_ids_kv2[0], skip_special_tokens=True)
    generated_text_hf = tokenizer.decode(generated_ids_hf2[0], skip_special_tokens=True)
