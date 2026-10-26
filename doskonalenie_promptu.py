import os
import sys
import time
from dotenv import load_dotenv
from groq import Groq
from groq import APIError

sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()
client = Groq(api_key=os.getenv('GROQ_API_KEY'))

MODEL = 'openai/gpt-oss-120b'

PROMPTY = {
    'Iteracja 1': 'Napisz funkcję Python do walidacji adresu e-mail.',
    'Iteracja 2': (
        'Napisz funkcję Python walidującą studencki adres e-mail. '
        'Adres jest poprawny wyłącznie w domenie pollub.edu.pl. '
        'Funkcja ma używać wyrażeń regularnych i zwracać True lub False. '
        'Dodaj testy jednostkowe dla przypadków: adres poprawny, '
        'zła domena, brak znaku @, pusty łańcuch znaków.'
    ),
    'Iteracja 3': (
        'Jesteś doświadczonym programistą Python w dziale IT uczelni. '
        'Napisz funkcję validate_student_email(email: str) -> bool '
        'z użyciem modułu re. Wymagania:\n'
        '– adres jest poprawny wyłącznie w domenie pollub.edu.pl;\n'
        '– część lokalna to imie.nazwisko (małe litery łacińskie, '
        'bez polskich znaków) albo sześciocyfrowy numer albumu;\n'
        '– dodaj docstring z trzema przykładami użycia;\n'
        '– przygotuj co najmniej 8 testów jednostkowych w pytest, '
        'obejmujących m.in.: poprawny adres z nazwiskiem, poprawny adres '
        'z numerem albumu, złą domenę, brak znaku @, wielkie litery, '
        'polskie znaki, pusty łańcuch, dwie kropki w części lokalnej;\n'
        '– kod ma być zgodny z PEP 8 i zawierać komentarz wyjaśniający '
        'wyrażenie regularne.'
    ),
}


def uruchom(prompt):
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{'role': 'user', 'content': prompt}],
        temperature=0.7,
        max_tokens=2000,
    )
    return r.choices[0].message.content


for opis, prompt in PROMPTY.items():
    print(f'\n\n{"#" * 64}\n### {opis}\n{"#" * 64}')
    print(prompt)
    print('-' * 64)
    try:
        start = time.perf_counter()
        print(uruchom(prompt))
        print(f'Czas odpowiedzi: {time.perf_counter() - start:.2f} s')
    except APIError as e:
        print(f'[Blad Groq API] {e}')
