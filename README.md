# transcriber_and_video_editor

Lokalny zestaw skryptow do pracy z polskim komentarzem w filmach:

- transkrypcja audio/wideo do `.txt`, `.srt` i `.json`
- transkrypcja po krotkich take'ach wykrytych z pauz
- wykrywanie fragmentow mowy i usuwanie ciszy
- renderowanie dynamicznej wersji filmu w 1080p

## Foldery

```text
wejscie/  - tutaj wrzucasz surowe filmy lub audio
wyniki/   - tutaj trafiaja gotowe filmy, napisy i listy ciec
```

Surowe filmy i wyniki sa ignorowane przez Git, bo moga miec wiele gigabajtow.
Na GitHub trafia kod, instrukcje i pusta struktura folderow.

## Instalacja

```powershell
python -m pip install -r requirements.txt
```

Pierwsze uruchomienie danego modelu Whisper moze pobrac model z internetu.

## Transkrypcja podstawowa

```powershell
python transkrybuj.py "wejscie\film.mp4" -o "wyniki" --model medium --json --word-timestamps --bez-kontekstu
```

## Transkrypcja po take'ach

Tego uzywamy, gdy Whisper skleja powtorki albo traktuje kilka prob jako jeden tekst.

```powershell
python transkrybuj_takeami.py "wejscie\film.mp4" -o "wyniki" --model medium --split-pause 0.15 --padding 0.08 --window 0.04 --word-timestamps
```

## Montaz dynamiczny

Wersja dynamiczna ma miec maksymalnie okolo `0.5 s` ciszy bez gadania.
Przy podobnych powtorkach zostawiamy ostatni take.

Render 1080p z wybranych fragmentow:

```powershell
python edytuj_przedzialy.py "wejscie\film.mp4" "wyniki\film_DYNAMIC_1080p.mp4" --width 1920 --height 1080 --keep START-KONIEC
```

Napisy dopasowane do wersji po cieciach:

```powershell
python generuj_srt_po_cieciach.py "wyniki\film.json" "wyniki\film_DYNAMIC_1080p.srt" --keep START-KONIEC
```

## Praca z wieloma filmami

Jesli w folderze `wejscie` jest wiele duzych filmow, przetwarzamy je zawsze po kolei.
Nie uruchamiamy kilku transkrypcji lub renderow rownolegle, bo moze to zapchac CPU,
RAM, dysk albo mocno spowolnic komputer.

Bezpieczny tryb pracy:

1. Wrzuc filmy do `wejscie`.
2. Przetwarzaj tylko nowe pliki, ktore nie maja jeszcze wyniku w `wyniki`.
3. Rob jeden film naraz.
4. Po kazdym filmie sprawdz wynik i dopiero wtedy przejdz do kolejnego.

## Najwazniejsza zasada montazu

Nie ufamy samej transkrypcji przy bardzo powtarzanych frazach.

Whisper moze skleic kilka podobnych prob w jedna linijke. Dlatego podejrzane
fragmenty dzielimy po pauzach na krotkie take'i i dla powtarzanych blokow
zostawiamy ostatnia wersje.
