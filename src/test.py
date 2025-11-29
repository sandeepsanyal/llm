import ollama

model=r'gemma3:1b'

# to print response as a stream
stream = ollama.chat(
    model=model,
    messages=[
        {
            'role': 'user',
            'content': 'Hello, I am Sandeep. Introduce yourself please.'
        }
    ],
    stream=True
)
for chunk in stream:
    print(
        chunk['message']['content'],
        end='',
        flush=True
    )



# to store response
response = ollama.chat(
    model=model,
    messages=[
        {
            'role': 'user',
            'content': 'Hello, I am Sandeep. Introduce yourself please.'
        }
    ]
)
resp = response['message']['content']
print(resp)




def add_two_numbers(a: int, b: int) -> int:
    return a + b

result = add_two_numbers(5, 3)

message = f"The result of adding 5 and 3 is {result}."

response = ollama.chat(
    model=model,
    messages=[{"role": "user", "content": message}]
)
resp = response['message']['content']
print(resp)
