# Lab 10: Dataset Frankenstein — budujemy dane do ML

## Czy wiesz, że...

W 2001 roku Tim Menzies opublikował jeden z pierwszych publicznie dostępnych datasetów do predykcji defektów (NASA MDP). Od tamtego czasu badacze odkryli, że dane były pełne błędów — duplikaty, brakujące wartości, niespójne etykiety. Okazuje się, że budowanie dobrego datasetu to najtrudniejsza część machine learningu. Kto by pomyślał.

## Kontekst

Przez ostatnie laby zbieraliście metryki kodu na różne sposoby: LOC, złożoność cyklomatyczną, metryki OO, code churn, ownership. Na labach 8-9 zobaczyliście jak platforma (SonarQube) robi to automatycznie. Teraz czas na pytanie: **czy te metryki mogą predykować bugi?**

Żeby odpowiedzieć, potrzebujemy datasetu — tabeli, gdzie każdy wiersz to plik, kolumny to metryki, a ostatnia kolumna mówi czy plik jest "buggy" czy "clean". Dziś taki dataset zbudujemy od zera, łącząc dane z poprzednich labów.

## Cel laboratorium

Po tym laboratorium będziesz potrafić:
- zbudować dataset do predykcji defektów z danych z repozytorium,
- łączyć metryki produktowe (LOC, CC) z procesowymi (churn, authors),
- etykietować pliki jako buggy/clean na podstawie historii commitów,
- eksplorować dane w Jupyter Notebook (rozkłady, korelacje, balans klas).

## Wymagania wstępne

- Python 3.9+ z bibliotekami: `pandas`, `matplotlib`, `seaborn`, `radon`
- Jupyter Notebook (lokalnie lub z Dockera z lab03)
- Sklonowany projekt open-source z co najmniej kilkuletnią historią

## Zadania

### Zadanie 1: Jupyter setup (30 min)

**Krok 1:** Uruchomcie Jupyter Notebook:

```bash
# Opcja A: lokalnie
pip install jupyter pandas matplotlib seaborn radon
jupyter notebook

# Opcja B: z Dockera z lab03
docker compose -f /sciezka/do/lab03/docker-compose.yml up jupyter
# Jupyter będzie dostępny na http://localhost:8888
```

**Krok 2:** Otwórzcie szablon `dataset_builder.ipynb` z tego repozytorium.

Szablon zawiera strukturę notebooka z komentarzami i miejscami do uzupełnienia (`# TODO`). Nie musicie zaczynać od zera.

**Krok 3:** Upewnijcie się, że macie sklonowany projekt OSS:

```bash
git clone https://github.com/psf/requests.git /tmp/requests
```

### Zadanie 2: Budowanie datasetu (75 min)

Pracujcie w notebooku `dataset_builder.ipynb`. Cel: zbudować CSV z metrykami per plik.

**Schemat datasetu:**

| Kolumna | Opis | Źródło |
|---------|------|--------|
| `filename` | ścieżka do pliku .py | system plików |
| `loc` | liczba linii kodu | podobnie jak lab04 |
| `avg_cc` | średnia złożoność cyklomatyczna | radon (lab05) |
| `max_cc` | maksymalna CC w pliku | radon (lab05) |
| `num_functions` | liczba funkcji/metod | radon |
| `churn` | total adds + deletes | git log --numstat (lab07) |
| `num_commits` | ile razy plik był zmieniany | git log (lab07) |
| `num_authors` | ilu autorów dotknęło pliku | git log (lab07) |
| `age_days` | dni od pierwszego commitu dotykającego pliku | git log |
| `is_buggy` | 1 = buggy, 0 = clean | heurystyka (patrz niżej) |

**Heurystyka etykietowania:**

Plik jest "buggy" jeśli był zmieniany w commicie, którego wiadomość zawiera słowa kluczowe sugerujące naprawę błędu:
- `fix`, `bug`, `error`, `fault`, `defect`, `patch`, `repair`, `crash`, `issue`

```python
BUG_KEYWORDS = ["fix", "bug", "error", "fault", "defect", "patch", "repair", "crash", "issue"]

def is_bug_commit(message: str) -> bool:
    msg_lower = message.lower()
    return any(keyword in msg_lower for keyword in BUG_KEYWORDS)
```

To uproszczona heurystyka (w badaniach naukowych używa się bardziej zaawansowanych metod, np. linkowania commitów z issue trackerami), ale na potrzeby tego laba jest wystarczająca.

**Krok po kroku:**

1. Zbierzcie listę wszystkich plików `.py` w projekcie (pominijcie testy, setup.py, conftest.py)
2. Dla każdego pliku obliczcie metryki produktowe (LOC, CC) używając `radon`
3. Dla każdego pliku obliczcie metryki procesowe (churn, authors, age) z `git log --numstat`
4. Dla każdego pliku sprawdźcie czy jest "buggy" (heurystyka powyżej)
5. Złączcie wszystko w jeden DataFrame i zapiszcie jako `dataset.csv`

**Krok 4:** Eksploracja danych w notebooku:

1. **Rozkłady:** histogram każdej cechy (LOC, CC, churn, itd.)
2. **Balans klas:** ile plików buggy vs clean? (% każdej klasy)
3. **Korelacje:** macierz korelacji między cechami (seaborn heatmap)
4. **Porównanie:** boxploty cech dla buggy vs clean (czy buggy pliki mają wyższy churn?)

**Krok 5:** Odpowiedzcie na pytania (w notebooku jako komórki Markdown):

1. Ile plików .py znaleźliście? Ile z nich jest buggy, a ile clean?
2. Czy dataset jest zbalansowany? (Jaki jest stosunek buggy:clean?)
3. Które cechy najbardziej różnią się między buggy a clean? (Patrz boxploty)
4. Czy widzicie korelacje między cechami? Które cechy są ze sobą skorelowane?
5. Czy heurystyka etykietowania jest idealna? Jakie są jej wady?

### Zadanie 3: Feature engineering (30 min) — dla ambitnych

Rozszerzcie dataset o dodatkowe cechy:

1. **Metryki OO** (z lab06): WMC, DIT, CBO dla plików z klasami
2. **Code ownership** (z lab07): owner_pct — jaki % commitów ma główny owner
3. **Wiek ostatniej modyfikacji**: dni od ostatniego commitu dotykającego pliku
4. **Stosunek komentarzy**: % linii będących komentarzami

Sprawdźcie czy nowe cechy poprawiają separację buggy/clean na boxplotach.

## Co oddajecie

W swoim branchu `lab10_nazwisko1_nazwisko2`:

1. **`dataset_builder.ipynb`** — wypełniony notebook z kodem i odpowiedziami
2. **`dataset.csv`** — wygenerowany dataset
3. *(opcjonalnie)* rozszerzony dataset z dodatkowymi cechami

## Kryteria oceny

- Notebook się uruchamia od początku do końca bez błędów
- Dataset zawiera co najmniej 6 cech (LOC, avg_cc, max_cc, churn, num_authors, age_days)
- Etykietowanie buggy/clean działa na podstawie heurystyki
- Eksploracja danych: histogramy, balans klas, macierz korelacji, boxploty
- Odpowiedzi na pytania w komórkach Markdown

## FAQ

**P: Mój projekt ma za mało plików .py (mniej niż 20).**
O: Wybierz większy projekt (flask, django, httpx) lub dodaj pliki testowe do analizy.

**P: Dataset jest bardzo niezbalansowany (np. 90% clean, 10% buggy).**
O: To normalne! W realnych projektach większość plików nie jest "buggy". Zanotujcie to jako obserwację. Na lab11 nauczymy się z tym radzić.

**P: Radon nie potrafi sparsować niektórych plików.**
O: Opakujcie wywołanie radona w try/except i pomiń pliki, które powodują błędy. Zanotujcie ile plików pominęliście.

**P: Heurystyka etykietowania oznacza za dużo plików jako buggy.**
O: Słowo "fix" pojawia się w wielu commitach, które nie dotyczą bugów (np. "fix typo in README"). To znana wada tej heurystyki. Zanotujcie to w odpowiedziach.

**P: Jak policzyć "age" pliku z git log?**
O: `git log --format="%ad" --date=short --diff-filter=A -- sciezka/do/pliku.py` pokaże datę pierwszego commitu, w którym plik się pojawił. Odejmijcie od dzisiejszej daty.

## Przydatne linki

- [pandas documentation](https://pandas.pydata.org/docs/)
- [seaborn heatmap](https://seaborn.pydata.org/generated/seaborn.heatmap.html)
- [radon documentation](https://radon.readthedocs.io/)
- [PROMISE repository](http://promise.site.uottawa.ca/SERepository/)
- [The Art and Science of Analyzing Software Data (book)](https://www.elsevier.com/books/the-art-and-science-of-analyzing-software-data/bird/978-0-12-411519-4)

---
*"Zbieramy części z różnych miejsc i ożywiamy potwora... to znaczy dataset. Najgorsza część? Potwór żyje, ale ma brakujące wartości."* — dr Frankenstein (wersja data science)
