# Workflow montazu dynamicznego

Cel: montaz talking-head / komentarza tak, zeby zostawic najlepsze take'i,
usunac przejezyczenia, powtorki i przerwy dluzsze niz okolo 0.5 s.

## Zasada krytyczna

Nie ufamy samej transkrypcji przy bardzo powtarzanych frazach.

Whisper potrafi skleic kilka podobnych powtorek w jedna linijke SRT. Dlatego
kazdy blok, w ktorym mowca powtarza podobne slowa kilka razy, trzeba dodatkowo
sprawdzic jako krotkie take'i audio.

Regula montazowa:

- jesli kilka take'ow ma bardzo podobne slownictwo, zostawiamy ostatnia wersje
- jesli transkrypcja pokazuje jedna linijke, ale audio ma kilka osobnych take'ow,
  decyzje podejmujemy na podstawie take'ow audio, nie na podstawie samego SRT
- maksymalna przerwa bez mowienia w wersji dynamicznej: okolo 0.5 s
- wynik domyslny: 1080p

## Standardowy przebieg

Jesli w `wejscie` znajduje sie kilka filmow, zawsze przetwarzamy je po kolei.
Nie odpalamy kilku transkrypcji ani renderow rownolegle. To jest domyslna
zasada nawet wtedy, gdy polecenie brzmi tylko "przetworz nowe filmy".

1. Transkrypcja:

```powershell
python transkrybuj.py "wejscie\film.mp4" -o "wyniki" --model medium --json --word-timestamps --bez-kontekstu
```

2. Wykrycie dynamicznych fragmentow mowy:

```powershell
python wykryj_przedzialy_dynamiczne.py "wejscie\film.mp4" --allow START-KONIEC --output "wyniki\film_intervals.txt" --max-silence 0.50 --padding 0.18
```

3. Dla podejrzanych powtorek wykrycie bardzo krotkich take'ow:

```powershell
python wykryj_przedzialy_dynamiczne.py "wejscie\film.mp4" --allow START-KONIEC --output "wyniki\film_takei.txt" --max-silence 0.15 --padding 0.08 --window 0.04
```

4. Jesli w `film_takei.txt` widac kilka take'ow tej samej frazy, w finalnej
   liscie zostaje tylko ostatni take.

5. Render 1080p:

```powershell
python edytuj_przedzialy.py "wejscie\film.mp4" "wyniki\film_DYNAMIC_1080p.mp4" --width 1920 --height 1080 --keep START-KONIEC
```

6. SRT dopasowane do wersji po cieciach:

```powershell
python generuj_srt_po_cieciach.py "wyniki\film.json" "wyniki\film_DYNAMIC_1080p.srt" --keep START-KONIEC
```

## Co sprawdzac recznie po renderze

- czy pierwsze zdanie nie ma ukrytych powtorek sklejonych przez Whispera
- czy przy podobnych zdaniach zostala ostatnia wersja
- czy nie ma przerw dluzszych niz okolo 0.5 s
- czy wynik ma 1920x1080 i audio 48 kHz
