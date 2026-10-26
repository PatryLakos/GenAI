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

ROLA = ('Jesteś doświadczonym specjalistą ds. komunikacji w dziale IT '
        'uczelni i autorem komunikatów dla społeczności akademickiej.')

ZADANIE = ('Zadanie: napisz komunikat o planowanej przerwie technicznej '
           'uczelnianej platformy e-learningowej.')

KONTEKST = ('Kontekst: przerwa odbędzie się w sobotę 14 listopada 2026 '
            'od godziny 22:00 do 02:00 w niedzielę, czyli potrwa 4 godziny; '
            'w tym czasie platforma będzie całkowicie niedostępna; '
            'przyczyną jest aktualizacja systemu i migracja bazy danych; '
            'odbiorcą komunikatu jest: ')

FORMAT = ('Format: komunikat w języku polskim, ton uprzejmy i rzeczowy, '
          'maksymalnie 120 słów, bez wypunktowań, z tytułem w pierwszej '
          'linii. Poziom szczegółowości technicznej: ')

ODBIORCY = {
    'studenci': ('studenci',
                 'minimalny, bez żargonu; podaj tylko termin, czas trwania '
                 'i radę, aby nie zostawiać pracy na platformie na ostatnią '
                 'chwilę.'),
    'wykladowcy': ('wykładowcy',
                   'średni; podaj termin, czas trwania, wpływ na testy '
                   'i zadania z terminem w tym okresie oraz zalecenie '
                   'przesunięcia terminów.'),
    'administratorzy': ('administratorzy IT',
                        'wysoki; podaj termin, czas trwania, zakres prac '
                        '(aktualizacja systemu, migracja bazy danych), '
                        'konieczność wykonania kopii zapasowej przed '
                        'rozpoczęciem oraz wyłączenie monitoringu na czas '
                        'przerwy.'),
}

for nazwa, (odbiorca, szczegolowosc) in ODBIORCY.items():
    prompt = f'{ZADANIE}\n{KONTEKST}{odbiorca}.\n{FORMAT}{szczegolowosc}'
    print(f'\n\n{"#" * 64}\n### {nazwa.upper()}\n{"#" * 64}')
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
            max_tokens=1000,
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
