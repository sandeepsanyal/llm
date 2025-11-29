from transformers import AutoModel, AutoTokenizer
import torch
import os
import ollama

# Optional: pin GPU device if you have multiple
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

OCR_MODEL = "deepseek-ai/DeepSeek-OCR"

SUMMARY_MODEL = "gemma3:1b"

# trust_remote_code is required for custom model logic
tokenizer = AutoTokenizer.from_pretrained(OCR_MODEL, trust_remote_code=True)
model = AutoModel.from_pretrained(
    OCR_MODEL,
    # _attn_implementation="flash_attention_2",  # use if flash-attn is installed & supported
    trust_remote_code=True,
    use_safetensors=True
)

if torch.cuda.is_available():
    device = torch.device("cuda")
    dtype = torch.bfloat16
# elif torch.backends.mps.is_available():
#     device = torch.device("mps")
#     dtype = torch.bfloat16
else:
    device = torch.device("cpu")
    dtype = torch.float32

model = model.eval().to(device).to(dtype)


prompt = "<image>\n<|grounding|>Convert the document to markdown."
image_file = "/Users/wrngnfreeman/GitHub/llm/artifacts/diabetes-infographics-layout-vector-12130742.jpg"  # replace with your image
output_path = r"/Users/wrngnfreeman/GitHub/llm/temp_stuff"  # directory for saved results

# Monkey-patch .cuda() to .to(device)
old_cuda = torch.Tensor.cuda
def to_device(self, *args, **kwargs):
    return self.to(device)
torch.Tensor.cuda = to_device

# Monkey-patch autocast("cuda") to autocast(device.type)
old_autocast = torch.autocast
def autocast_patch(device_type, *args, **kwargs):
    if device_type == "cuda":
        device_type = device.type
    return old_autocast(device_type, *args, **kwargs)
torch.autocast = autocast_patch

# Now call infer safely
res = model.infer(
    tokenizer,
    prompt=prompt,
    image_file=image_file,
    output_path=output_path,
    base_size=1024,
    image_size=640,
    crop_mode=True,
    save_results=True,
    test_compress=True,
    eval_mode=True
)

print("Extracted Text: \n{res}\n\n")


### Summarize Extract
prompt2 = f"{res}\n\nWhere it is talking about Mixture of Experts Architecture?"
# OCR processing
response = ollama.chat(
    model=OCR_MODEL,
    messages=[
        {
            "role": "user",
            "content": prompt2
        }
    ]
)

# Output the result
print(response['message']['content'])
