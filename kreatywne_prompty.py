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
UZYJ_OLLAMA = False
MODEL_LOKALNY = 'llama3.1'

PROMPTY = {
    'A - analogia': (
        'Wyjaśnij, czym jest rekurencja, używając analogii '
        'do rosyjskich matrioszek. Zakończ jednym zdaniem, '
        'wskazując, co w tej analogii odpowiada warunkowi '
        'stopu. Maksymalnie 120 słów.'
    ),
    'B - storytelling': (
        'Opowiedz krótką historię (do 150 słów) o studencie '
        'informatyki, który podczas rodzinnego obiadu tłumaczy '
        'dziadkowi, czym jest rekurencja. Historia ma zawierać '
        'poprawne merytorycznie wyjaśnienie pojęcia.'
    ),
    'C - zmiana perspektywy': (
        'Jesteś stosem wywołań programu. Opisz w pierwszej '
        'osobie, co dzieje się z tobą podczas rekurencyjnego '
        'obliczania silni z liczby 4. Maksymalnie 120 słów.'
    ),
}


def uruchom_groq(prompt):
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{'role': 'user', 'content': prompt}],
        temperature=0.9,
        max_tokens=400,
        reasoning_effort='low',
    )
    return r.choices[0].message.content


def uruchom_ollama(prompt):
    r = ollama.chat(
        model=MODEL_LOKALNY,
        messages=[{'role': 'user', 'content': prompt}],
    )
    return r['message']['content']


for opis, prompt in PROMPTY.items():
    print(f'\n\n{"#" * 64}\n### WARIANT {opis}\n{"#" * 64}')
    print(prompt)
    print('-' * 64)
    try:
        start = time.perf_counter()
        if UZYJ_OLLAMA:
            odpowiedz = uruchom_ollama(prompt)
        else:
            odpowiedz = uruchom_groq(prompt)
        print(odpowiedz)
        print(f'Czas odpowiedzi: {time.perf_counter() - start:.2f} s')
    except APIError as e:
        print(f'[Blad Groq API] {e}')
    except Exception as e:
        print(f'[Blad] {e}')
