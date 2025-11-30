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
        "eval_mode": True
    }
)

res = response['response']


print("Extracted Text: \n{}\n\n".format(res))


### Summarize Extract
prompt2 = f"{res}\n\nConvert this into a well structured Markdown."
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
