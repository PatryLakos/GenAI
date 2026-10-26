import os
import sys
import time
from dotenv import load_dotenv
from groq import Groq
from groq import APIError
import ollama

sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()
client = Groq(api_key=os.getenv('GROQ_API_KEY'))

MODEL = 'openai/gpt-oss-120b'
MODELE_LOKALNE = ['llama3.1', 'gemma3:1b']

PROMPT = ('Jesteś ekspertem od sztucznej inteligencji. '
          'Wyjaśnij w 3-4 zdaniach, czym jest transformer '
          'w kontekście modeli językowych. Użyj prostego języka, '
          'zrozumiałego dla studenta pierwszego roku informatyki.')

print(f'=== {MODEL} ===')
try:
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{'role': 'user', 'content': PROMPT}],
        temperature=0.7,
        max_tokens=1000,
    )
    t = time.perf_counter() - start
    print(response.choices[0].message.content)
    print(f'Czas odpowiedzi: {t:.2f} s')
except APIError as e:
    print(f'[Blad Groq API] {e}')

for model in MODELE_LOKALNE:
    print(f'=== {model} ===')
    try:
        for przebieg in (1, 2):
            start = time.perf_counter()
            local = ollama.chat(model=model,
                                messages=[{'role': 'user', 'content': PROMPT}])
            t = time.perf_counter() - start
            if przebieg == 2:
                print(local['message']['content'])
            print(f'Przebieg {przebieg}, czas odpowiedzi: {t:.2f} s')
    except Exception as e:
        print(f'[Blad Ollama] {e}')
        print(f'Upewnij sie, ze serwer Ollama dziala i model {model} '
              f'jest pobrany (ollama pull {model}).')
