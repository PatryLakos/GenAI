import os
import sys
import time

from dotenv import load_dotenv
from groq import APIError, Groq
import ollama

# poprawne wyświetlanie polskich znaków w konsoli Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()
client = Groq(api_key=os.getenv('GROQ_API_KEY'))

# testujemy dwa modele: zewnętrzny (API) i lokalny (Ollama)
MODEL = 'openai/gpt-oss-120b'
MODEL_LOKALNY = 'llama3.1'

# Krok 1 - prompt bez struktury
PROMPT_BEZ_STRUKTURY = 'Napisz wprowadzenie do raportu o wykrywaniu ataków sieciowych.'

# Kroki 2-3 - dekompozycja i prompt ze strukturą (rola + zadanie + kontekst + format)
ROLA = (
    'Jesteś doświadczonym inżynierem cyberbezpieczeństwa '
    'i autorem dokumentacji technicznej.'
)
PROMPT_ZE_STRUKTURA = (
    'Zadanie: napisz wprowadzenie do raportu technicznego '
    'z projektu studenckiego. '
    'Kontekst: projekt dotyczy zastosowania uczenia '
    'maszynowego do wykrywania ataków w sieci komputerowej '
    '(system klasy IDS); porównywane są klasyczne metody '
    'sygnaturowe z klasyfikatorami uczonymi na zbiorze '
    'CIC-IDS2017; odbiorcą jest nauczyciel przedmiotu '
    'Projekt zespołowy na kierunku informatyka. '
    'Format: około 250 słów, trzy akapity (motywacja '
    'i znaczenie problemu; krótki przegląd podejść; cel '
    'i zakres projektu); styl formalny, bez wypunktowań.'
)


def uruchom(opis, prompt, rola=None):
    print(f'\n\n{"#" * 64}\n###  ODPOWIEDZ -> {opis}\n{"#" * 64}')

    # model zewnętrzny: Groq (API)
    print(f'\n--- [{opis}] model zewnętrzny: {MODEL} ---')
    try:
        wiadomosci_api = [{'role': 'user', 'content': prompt}]
        if rola:
            wiadomosci_api.insert(0, {'role': 'system', 'content': rola})
        start = time.perf_counter()
        response = client.chat.completions.create(
            model=MODEL,
            messages=wiadomosci_api,
            temperature=0.7,
            max_tokens=1000,
        )
        print(response.choices[0].message.content)
        print(f'Czas odpowiedzi: {time.perf_counter() - start:.2f} s')
    except APIError as e:
        print(f'[Błąd Groq API] {e}')

    # model lokalny: Ollama (rola trafia do wiadomości systemowej)
    print(f'\n--- [{opis}] model lokalny: {MODEL_LOKALNY} ---')
    try:
        wiadomosci = []
        if rola:
            wiadomosci.append({'role': 'system', 'content': rola})
        wiadomosci.append({'role': 'user', 'content': prompt})
        start = time.perf_counter()
        local = ollama.chat(model=MODEL_LOKALNY, messages=wiadomosci)
        print(local['message']['content'])
        print(f'Czas odpowiedzi: {time.perf_counter() - start:.2f} s')
    except Exception as e:
        print(f'[Błąd Ollama] {e}')
        print(
            f'Upewnij się, że serwer Ollama działa i model {MODEL_LOKALNY} '
            f'jest pobrany (ollama pull {MODEL_LOKALNY}).'
        )


# Krok 1 - bez struktury (bez roli systemowej)
uruchom('KROK 1: prompt bez struktury', PROMPT_BEZ_STRUKTURY)

# Krok 3 - ze strukturą; rola jako instrukcja systemowa
uruchom('KROK 3: prompt ze strukturą', PROMPT_ZE_STRUKTURA, rola=ROLA)