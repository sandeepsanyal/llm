import ollama

OCR_MODEL = "deepseek-ocr:latest"
SUMMARY_MODEL = "llama3.2:3b"

image_file = "/Users/wrngnfreeman/GitHub/llm/artifacts/2021_diabetes_awareness_larg.jpg"  # replace with your image
output_path = r"/Users/wrngnfreeman/GitHub/llm/temp_stuff"  # directory for saved results


prompt = "<image>\n<|grounding|>Convert the document to markdown."
response = ollama.generate(
    model=OCR_MODEL,
    prompt=prompt,
    images=[image_file],
    options={
        "output_path": output_path,
        "base_size": 1024,
        "image_size": 640,
        "crop_mode": True,
        "save_results": True,
        "test_compress": True,
        "eval_mode": True
    }
)

res = response['response']


print("Extracted Text: \n{}\n\n".format(res))


### Summarize Extract
prompt2 = f"{res}\n\nWhere is it talking about Mixture of Experts Architecture?"
# OCR processing
response = ollama.chat(
    model=SUMMARY_MODEL,
    messages=[
        {
            "role": "user",
            "content": prompt2
        }
    ]
)


print("Summary: \n")
# Output the result
print(response['message']['content'])
