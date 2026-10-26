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
MODEL_LOKALNY = 'llama3.1'

ROLA = ('Jesteś doświadczonym inżynierem DevOps, który przygotowuje '
        'pliki konfiguracyjne gotowe do użycia w środowisku produkcyjnym.')

PROMPTY = {
    'Iteracja 1': 'Napisz Dockerfile dla aplikacji w Pythonie.',
    'Iteracja 2': (
        'Napisz Dockerfile dla aplikacji FastAPI w Pythonie 3.12. '
        'Zależności są w pliku requirements.txt, aplikacja startuje '
        'poleceniem uvicorn main:app na porcie 8000. '
        'Użyj lekkiego obrazu bazowego.'
    ),
    'Iteracja 3': (
        'Zadanie: napisz Dockerfile dla aplikacji FastAPI.\n'
        'Kontekst: Python 3.12, zależności w requirements.txt, '
        'aplikacja startuje poleceniem uvicorn main:app na porcie 8000, '
        'obraz będzie wdrażany w klastrze Kubernetes.\n'
        'Wymagania:\n'
        '- obraz bazowy python:3.12-slim;\n'
        '- build wieloetapowy;\n'
        '- uruchamianie jako użytkownik nieuprzywilejowany;\n'
        '- poprawne wykorzystanie cache warstw przy instalacji zależności;\n'
        '- HEALTHCHECK;\n'
        '- zmienne PYTHONUNBUFFERED i PYTHONDONTWRITEBYTECODE.\n'
        'Format: sam Dockerfile w jednym bloku kodu, po nim lista '
        'maksymalnie 5 punktów z uzasadnieniem kluczowych decyzji.'
    ),
}

for opis, prompt in PROMPTY.items():
    print(f'\n\n{"#" * 64}\n### {opis}\n{"#" * 64}')
    print(prompt)
    print('-' * 64)

    print(f'--- {MODEL} ---')
    try:
        start = time.perf_counter()
        r = client.chat.completions.create(
            model=MODEL,
            messages=[{'role': 'system', 'content': ROLA},
                      {'role': 'user', 'content': prompt}],
            temperature=0.7,
            max_tokens=2000,
        )
        print(r.choices[0].message.content)
        print(f'Czas odpowiedzi: {time.perf_counter() - start:.2f} s')
    except APIError as e:
        print(f'[Blad Groq API] {e}')

    print(f'--- {MODEL_LOKALNY} ---')
    try:
        start = time.perf_counter()
        local = ollama.chat(model=MODEL_LOKALNY,
                            messages=[{'role': 'system', 'content': ROLA},
                                      {'role': 'user', 'content': prompt}])
        print(local['message']['content'])
        print(f'Czas odpowiedzi: {time.perf_counter() - start:.2f} s')
    except Exception as e:
        print(f'[Blad Ollama] {e}')
