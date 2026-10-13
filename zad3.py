import os
from dotenv import load_dotenv
from groq import Groq, APIError

load_dotenv()

api_key = os.getenv('GROQ_API_KEY')
if not api_key:
    raise SystemExit('Nie znaleziono klucza. Sprawdz plik .env (GROQ_API_KEY).')

client = Groq(api_key=api_key)

MODEL = 'openai/gpt-oss-120b'
QUESTION = 'Czym jest rekurencja w programowaniu?'

# dwie persony zdefiniowane przez wiadomosc systemowa
personas = {
    'Lakoniczny ekspert techniczny': (
        'Jestes doswiadczonym inzynierem oprogramowania. Odpowiadaj po polsku, '
        'maksymalnie zwiezle i precyzyjnie, uzywaj terminologii technicznej, '
        'bez wstepow i bez przykladow, chyba ze sa niezbedne.'
    ),
    'Nauczyciel prostego jezyka': (
        'Jestes cierpliwym nauczycielem informatyki. Odpowiadaj po polsku, '
        'prostym jezykiem, tak jakbys tlumaczyl osobie bez doswiadczenia. '
        'Uzyj analogii z zycia codziennego i krotkiego przykladu.'
    ),
}

results = []

for name, instruction in personas.items():
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {'role': 'system', 'content': instruction},
                {'role': 'user', 'content': QUESTION},
            ],
            temperature=0.7,
            max_tokens=1000,
        )
    except APIError as e:
        print(f'[Blad API] {e.message}')
        continue

    usage = response.usage
    results.append((name, usage.completion_tokens))

    print(f'===== {name} =====')
    print(response.choices[0].message.content)
    print(f'\n[tokeny] prompt: {usage.prompt_tokens} | '
          f'odpowiedz: {usage.completion_tokens} | '
          f'razem: {usage.total_tokens}\n')

# podsumowanie
print('===== PODSUMOWANIE (tokeny odpowiedzi) =====')
for name, tokens in results:
    print(f'{name}: {tokens}')