import os
from dotenv import load_dotenv
from groq import Groq, APIError, APIStatusError, APIConnectionError

# wczytanie klucza z pliku .env do zmiennej srodowiskowej
load_dotenv()

api_key = os.getenv('GROQ_API_KEY')
if not api_key:
    raise SystemExit('Nie znaleziono klucza. Sprawdz, czy plik .env znajduje sie w tym katalogu i czy zawiera GROQ_API_KEY.')

client = Groq(api_key=api_key)

MODEL = 'openai/gpt-oss-120b'

messages = [
    {'role': 'system', 'content': 'Odpowiadaj zwiezle, w jezyku polskim.'},
]

print('Wpisz prompt i zatwierdz Enterem. Aby zakonczyc, wpisz "koniec".\n')

while True:
    try:
        prompt = input('Ty: ').strip()
    except (EOFError, KeyboardInterrupt):
        print('\nPrzerwano. Do zobaczenia!')
        break

    if prompt.lower() == 'koniec':
        print('Do zobaczenia!')
        break

    if not prompt:
        continue

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages + [{'role': 'user', 'content': prompt}],
            temperature=0.7,
            max_tokens=1000,
        )
        print(f'\nModel: {response.choices[0].message.content}\n')
    except APIConnectionError:
        print('\n[Blad polaczenia] Nie udalo sie polaczyc z serwerem. Sprawdz internet i sprobuj ponownie.\n')
    except APIStatusError as e:
        print(f'\n[Blad API {e.status_code}] {e.message}')
        print('Sprobuj ponownie lub wpisz "koniec", aby wyjsc.\n')
    except APIError as e:
        print(f'\n[Blad API] {e.message}\n')
    except Exception as e:
        print(f'\n[Nieoczekiwany blad] {e}\n')