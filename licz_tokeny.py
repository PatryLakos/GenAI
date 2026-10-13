import os
import sys

from dotenv import load_dotenv
from groq import APIError, Groq

# poprawne wyświetlanie polskich znaków w konsoli Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()
client = Groq(api_key=os.getenv('GROQ_API_KEY'))

MODEL = 'openai/gpt-oss-120b'
# ceny płatnej warstwy API w USD za 1 mln tokenów (stan na 2026 r.)
CENA_WE = 0.15
CENA_WY = 0.60

PROMPT = (
    'Jesteś ekspertem od sztucznej inteligencji. '
    'Wyjaśnij w 3-4 zdaniach, czym jest tokenizacja '
    'w kontekście modeli językowych. Użyj prostego języka, '
    'zrozumiałego dla studenta pierwszego roku informatyki.'
)

try:
    # wysłanie zadania i odczyt statystyk z pola usage
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{'role': 'user', 'content': PROMPT}],
        temperature=0.7,
        max_tokens=1000,
    )
    print(response.choices[0].message.content)
    tok_we = response.usage.prompt_tokens
    tok_wy = response.usage.completion_tokens
    print(f'Tokeny wejściowe: {tok_we} | tokeny wyjściowe: {tok_wy}')

    # szacunkowy koszt pojedynczego zapytania
    koszt = (tok_we * CENA_WE + tok_wy * CENA_WY) / 1_000_000
    print(f'Szacunkowy koszt zapytania: {koszt:.6f} USD')
except APIError as e:
    # np. błąd 429 (przekroczony limit zapytań) lub problem z kluczem/siecią
    print(f'[Błąd Groq API] {e}')


# porownanie tokenizacji: polski vs angielski 
zdania = { 
    'PL': 'Sztuczna inteligencja zmienia współczesną informatykę.', 
    'EN': 'Artificial intelligence is changing modern computing.', 
} 
for jezyk, tekst in zdania.items(): 
    r = client.chat.completions.create(model=MODEL, 
        messages=[{'role': 'user', 'content': tekst}], max_tokens=1) 
    print(f'{jezyk}: {r.usage.prompt_tokens} tokenów | {len(tekst)} znaków') 


# tokeny w modelu lokalnym (Ollama) 
import ollama 
local = ollama.chat(model='llama3.1', 
                    messages=[{'role': 'user', 'content': PROMPT}]) 
print('Tokeny wejściowe (Ollama):', local['prompt_eval_count']) 
print('Tokeny wyjściowe (Ollama):', local['eval_count']) 