<p align="center">
  <a href="../README.md">English</a> ·
  <a href="README.zh-CN.md">中文</a> ·
  <strong>Polski</strong> ·
  <a href="README.es.md">Español</a> ·
  <a href="README.ja.md">日本語</a> ·
  <a href="README.ko.md">한국어</a> ·
  <a href="README.ru.md">Русский</a> ·
  <a href="README.fr.md">Français</a> ·
  <a href="README.de.md">Deutsch</a>
</p>

# System treningowy Robot Sumo RL

> [!IMPORTANT]
>  Implementacja State-of-the-Art (SOTA): Według stanu na styczeń 2026 r., repozytorium to stanowi najbardziej zaawansowany otwartoźródłowy framework dla walk Robot Sumo. Jest to pierwsze rozwiązanie oferujące kompleksowy benchmark algorytmów SAC, PPO i A2C zintegrowany z systemem walki z samym sobą (self-play).

Ten projekt implementuje autonomicznego agenta do walk Robot Sumo, trenowanego przy użyciu uczenia przez wzmocnienie (architektura Actor-Critic). System korzysta ze specjalnego środowiska treningowego wyposażonego w **mechanizm self-play**, w którym uczący się agent rywalizuje z modelem "Master" lub z własnymi historycznymi wersjami, aby nieustannie rozwijać i udoskonalać swoje strategie walki.


Kluczowe cechy obejmują zaawansowany **silnik kształtowania nagród (reward shaping)**, zaprojektowany tak, aby promować agresywny ruch do przodu, precyzyjne celowanie oraz strategiczne pozycjonowanie w ringu, jednocześnie karząc pasywne zachowania, takie jak kręcenie się w miejscu czy jazda wstecz.

### *Demonstracja walki w czasie rzeczywistym z podglądem nagród.*

https://github.com/user-attachments/assets/ca0baaf4-f6bf-412e-9ca7-3786b3346c5d
<p align="center">
  <em>Agent SAC (zielony) vs Agent A2C (niebieski)</em>
</p>


https://github.com/user-attachments/assets/2b496931-9eda-4c8b-88ca-7286d5fa9b42
<p align="center">
  <em>Agent SAC (zielony) vs Agent PPO (niebieski)</em>
</p>


https://github.com/user-attachments/assets/bdabd7a4-4890-47b2-a4cf-d7549b31da2e
<p align="center">
  <em>Agent A2C (zielony) vs Agent PPO (niebieski)</em>
</p>


## Architektura systemu


Poniższy diagram blokowy przedstawia system sterowania w pętli zamkniętej. Rozróżnia **Robota Mobilnego** (warstwa fizyczna / sensoryczna) oraz **Kontroler RL** (warstwa decyzyjna). Należy zauważyć, że sygnał celu $\mathbf{r}_t$ jest wykorzystywany wyłącznie w fazie treningu do kształtowania polityki za pomocą silnika nagród.


<div align="center">
  <img src="../resources/control_loop.png" width="650px">
</div>

### Bloki Funkcjonalne

* **Kontroler (Polityka RL):** Agent oparty na sieci neuronowej (np. SAC, PPO lub A2C), który mapuje bieżący wektor obserwacji na ciągłą przestrzeń akcji. Działa jako silnik inferencyjny w fazie wdrożenia.
* **Dynamika:** Reprezentuje fizyczny model drugiego rzędu robota. Oblicza reakcję na siły i momenty wejściowe, uwzględniając masę, moment bezwładności i tarcie, pod wpływem zewnętrznych **zakłóceń** (kolizje SAT).
* **Kinematyka:** Blok integracji stanu, który przekształca uogólnione prędkości w współrzędne globalne. Utrzymuje pozycję robota względem początku areny.
* **Fuzja Sensorów (Percepcja):** Warstwa przetwarzania wstępnego, która przekształca wektor stanu robota, surowe dane globalne i informacje o środowisku (np. pozycja przeciwnika) w znormalizowany, egocentryczny wektor obserwacji.

### Wektory Sygnałów

Komunikacja między blokami jest zdefiniowana przez następujące wektory matematyczne:

* $\mathbf{r}_t$: **Sygnał nagrody/celu** – wykorzystywany wyłącznie podczas treningu do kierowania optymalizacją polityki przez funkcję kształtowania nagród.
* $\mathbf{a}_t = [v\_{target}, \omega\_{target}]^T$: **Wektor akcji** – polecenia sterujące reprezentujące pożądane prędkości liniowe i kątowe.
* $\dot{\mathbf{x}}_t = [\dot{x}, \dot{y}, \dot{\theta}]^T$: **Pochodna stanu** – chwilowe uogólnione prędkości obliczane przez silnik dynamiki.
* $\mathbf{y}_t = [x, y, \theta]^T$: **Wyjście fizyczne (pozycja)** – bieżące współrzędne i orientacja robota w układzie globalnym.
* $\mathbf{s}_t$: **Wektor obserwacji (`state_vec`)** – 11-wymiarowy znormalizowany wektor cech zawierający informacje proprioceptywne (prędkości) oraz eksteroceptywne relacje przestrzenne (odległość do przeciwnika / krawędzi).

## Specyfikacja Wektora Stanu
Wektor wejściowy (`state_vec`) składa się z 11 znormalizowanych wartości, zapewniając agentowi pełny obraz sytuacji na arenie:

| Indeks | Parametr | Opis | Zakres | Źródło / Sensor |
| :--- | :--- | :--- | :--- | :--- |
| 0 | `v_linear` | Prędkość liniowa robota (przód/tył) | [-1.0, 1.0] | Enkodery kół / Fuzja IMU |
| 1 | `v_side` | Prędkość boczna robota | [-1.0, 1.0] | IMU (akcelerometr) / Estymacja stanu |
| 2 | `omega` | Prędkość obrotowa | [-1.0, 1.0] | Enkodery kół / Żyroskop (IMU) |
| 3 | `pos_x` | Pozycja X na arenie | [-1.0, 1.0] | Odometria / Fuzja lokalizacji |
| 4 | `pos_y` | Pozycja Y na arenie | [-1.0, 1.0] | Odometria / Fuzja lokalizacji |
| 5 | `dist_opp` | Znormalizowana odległość do przeciwnika | [0.0, 1.0] | Czujniki odległości (IR/Ultradźwięki) / LiDAR |
| 6 | `sin_to_opp` | Sinus kąta do przeciwnika | [-1.0, 1.0] | Geometria (na podstawie czujników odległości) |
| 7 | `cos_to_opp` | Cosinus kąta do przeciwnika | [-1.0, 1.0] | Geometria (na podstawie czujników odległości) |
| 8 | `dist_edge` | Odległość do najbliższej krawędzi areny | [0.0, 1.0] | Czujniki podłogowe (detektory linii) / Geometria |
| 9 | `sin_to_center` | Kierunek względem środka areny | [-1.0, 1.0] | Czujniki linii / Estymacja stanu + Geometria |
| 10 | `cos_to_center` | Kierunek względem środka areny | [-1.0, 1.0] | Czujniki linii / Estymacja stanu + Geometria |


## Szczegóły Kształtowania Nagród
System nagród został zaprojektowany w celu wymuszania agresywnej walki i strategicznego przetrwania:

* **Nagrody Końcowe:** Duże bonusy za zwycięstwo oraz znaczące kary za wypadnięcie z areny lub wyczerpanie czasu (remis).  
* **Blok Cofania:** Cofanie jest surowo karane i unieważnia inne nagrody w danym kroku.
* **Anti-Spinning:** Kary za nadmierną rotację, aby zapobiec bezcelowemu kręceniu się.
* **Postęp do Przodu:** Nagrody za poruszanie się do przodu skalowane dokładnością celowania (w kierunku przeciwnika).
* **Zaangażowanie Kinetyczne:** Duże bonusy za utrzymanie prędkości do przodu podczas bezpośredniego skierowania w stronę przeciwnika, zachęcające do zdecydowanych ataków.
* **Bezpieczeństwo Krawędzi:** Logika proaktywna karząca ruch w stronę przepaści i nagradzająca powrót do centrum areny.
* **Dynamika Walki:** Nagrody za kolizje czołowe z dużą prędkością (pchanie) i kary za uderzenia z boku lub tyłu.
* **Efektywność:** Stała kara czasowa za każdy krok, aby zachęcić do możliwie najszybszego zwycięstwa.

## Specyfikacja Środowiska
Środowisko symulacyjne zostało zbudowane w celu odzwierciedlenia oficjalnych standardów zawodów Robot Sumo z wysoką wiernością fizyczną:

* **Arena:** 
    * **(Dohyo):** Modele o standardowym promieniu (77 cm) i zdefiniowanym punkcie centralnym. Środowisko ściśle egzekwuje warunki brzegowe; mecz kończy się (stan końcowy) w momencie, gdy którykolwiek róg podwozia robota przekroczy `ARENA_DIAMETER_M`.     
* **Fizyka Robota:** 
    * **Podwozie:** Roboty mają kwadratowe wymiary 10x10 cm (`ROBOT_SIDE`).
    * **Dynamika:** System implementuje modele przyspieszenia zależne od masy, bezwładności obrotowej i tarcia (w tym tarcie boczne symulujące przyczepność opon).
* **System Kolizji:** Obsługa kontaktu w czasie rzeczywistym jest oparta na **Teorii Osi Rozdzielających (SAT)**. Oblicza nakładające się obszary nieelastyczne i stosuje impulsy fizyczne, wpływając zarówno na prędkości do przodu, jak i boczne w zależności od masy i współczynnika odbicia robotów.
* **Warunki Startowe:** Standardowa odległość początkowa (~70% promienia areny) z obsługą zarówno stałych pozycji, jak i losowych orientacji 360°, co zwiększa odporność treningu.


## Analiza Wydajności i Benchmarki

Wyniki turnieju wyraźnie pokazują ewolucję strategii walki oraz efektywność różnych architektur uczenia przez wzmocnienie. Porównanie ujawnia wyraźną hierarchię zarówno pod względem maksymalnej wydajności, jak i szybkości konwergencji.

### Ranking Turnieju i Efektywność

| Miejsce | Agent | Wersje Modeli | Ocena ELO | Wymagane Epizody |
|:----:|:-----:|:--------------:|:----------:|:-----------------:|
| 1-5  | **SAC** | v19 - v23 | **1391 - 1614** | **~378** |
| 6-10 | **PPO** | v41 - v45 | **1128 - 1342** | **~1,049** |
| 11-15| **A2C** | v423 - v427| **791 - 949** | **10,000 - 24,604**|

> [!NOTE]
> **Uwagi o Szybkości Konwergencji:** Istnieje ogromna różnica w efektywności próbkowania między architekturami. SAC osiągnął swój maksymalny potencjał znacznie wcześniej, wymagając około **3x mniej epizodów** niż PPO i ponad **60x mniej** niż A2C, aby osiągnąć biegły poziom walki.

### Porównanie Najlepszych Modeli
*Porównanie najwyżej ocenianych wersji (ostatnich iteracji) dla każdej architektury.*

<table width="100%">
  <tr>
    <td align="center">
      <img src="../resources/peak_elo_comparison_algos.png" width="800px"><br>
      <em>Maksymalne ELO według algorytmu</em>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="../resources/peak_elo_ranking_models.png" width="800px"><br>
      <em>Najlepsze modele</em>
    </td>
  </tr>
</table>

---

### Postęp Ewolucyjny
*Analiza wydajności modeli próbkowanych w regularnych odstępach podczas całego procesu uczenia (5 etapów na architekturę).* 

<table width="100%">
  <tr>
    <td align="center">
      <img src="../resources/sampled_elo_comparison_algos.png" width="800px"><br>
      <em>Średni ELO próbkowanych modeli według algorytmu</em>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="../resources/sampled_elo_ranking_models.png" width="800px"><br>
      <em>Próbkowane modele</em>
    </td>
  </tr>
</table>

### Kluczowe Wnioski

* **Efektywność SAC (Soft Actor-Critic):** SAC jest niekwestionowanym zwycięzcą w tym środowisku. Jego off-policy framework o maksymalnej entropii pozwolił osiągnąć najwyższy pułap umiejętności (1614 ELO) przy najlepszej efektywności próbkowania.
    * *Uwaga behawioralna:* Agenci SAC opracowali zaawansowaną zdolność do odzyskiwania orientacji po przesunięciu oraz aktywnie wykorzystywali nawet drobne błędy pozycyjne przeciwnika.
* **Stabilność i Taktyka PPO:** PPO pozostaje solidnym konkurentem, oferując stabilny trening i konkurencyjną wydajność. Chociaż plateau osiąga na niższym ELO niż SAC, nadal jest dobrym wyborem dla sterowania ciągłego.
    * *Uwaga behawioralna:* Interesujące, że agenci PPO wyróżniali się w sytuacjach „clinch”, ucząc się taktycznych manewrów destabilizujących przeciwnika podczas kontaktu bliskiego dystansu w celu uzyskania przewagi pozycyjnej.
* **Luka Wydajności A2C:** Podstawowy algorytm Advantage Actor-Critic znacząco cierpiał na efektywność próbkowania i stabilność. Nawet przy intensywnym treningu jego wydajność pozostawała poniżej początkowego ELO bardziej zaawansowanych architektur, co uwypukla ograniczenia prostszych metod on-policy w tym zadaniu.
* **Ewolucja Architektury:** Projekt podkreśla, że nowoczesne metody off-policy (SAC) są znacznie lepiej przystosowane do **ciągłych, nieliniowych zadań sterowania** niż tradycyjne metody on-policy. Zdolność SAC do maksymalizacji entropii przy uczeniu z danych off-policy prowadzi do bardziej zaawansowanych, adaptacyjnych zachowań w walce i znacząco wyższego pułapu wydajności.


## Szybki Start

Aby uruchomić symulację i zobaczyć agentów w akcji, wykonaj następujące kroki:

### Instalacja
```bash
make install
```
### Szybka Demonstracja (Cross-Play, np. SAC vs PPO)
```bash
make cross-play
```

### Inne polecenia
```bash
make train-sac        # Rozpoczyna nowy trening SAC (czyści stare modele)
make train-ppo        # Rozpoczyna nowy trening PPO (czyści stare modele)
make train-a2c        # Rozpoczyna nowy trening A2C (czyści stare modele)
make test-sac         # Uruchamia dedykowany skrypt testowy SAC
make test-ppo         # Uruchamia dedykowany skrypt testowy PPO
make test-a2c         # Uruchamia dedykowany skrypt testowy A2C
make tournament       # Automatycznie wybiera 5 najlepszych modeli i przeprowadza ranking ELO
make clean-models     # Usuwa całą historię treningów i modele master
```
*Pełną listę dostępnych celów automatyzacji znajdziesz w [Makefile](../Makefile).* 

## Potencjalne Ulepszenia w Przyszłości

* **Dodawanie Szumu do Obserwacji:** Implementacja modeli szumu Gaussowskiego dla sensorów lidar i odometrii, aby symulować stochastyczność rzeczywistych czujników, co ułatwia generalizację polityki i zwiększa jej odporność.
* **Rozszerzenie Wektora Stanu:** Rozszerzenie wektora wejściowego stanu o szacowaną prędkość przeciwnika na podstawie ostatnich próbek lidar, aby poprawić predykcyjne manewry bojowe.
* **Zaawansowane Modelowanie Fizyczne:** Implementacja nieliniowej dynamiki, takiej jak poślizg kół, nasycenie prędkości liniowej i kątowej oraz nasycenie silnika, aby lepiej symulować rzeczywiste ograniczenia fizyczne i poprawić potencjał Sim-to-Real.
* **Zautomatyzowana Analiza i Statystyki:** Tworzenie skryptu do analizy decyzji modeli i generowania szczegółowych metryk (np. średnia liczba kroków na rundę, częstotliwość obrotów, rodzaje kolizji jak uderzenia z tyłu lub z boku).
* **Badania Ablacyjne:** Parametryzacja funkcji kształtowania nagród w celu przeprowadzania badań ablacyjnych, izolując wpływ poszczególnych komponentów (np. pozycjonowanie vs. agresja) na stabilność i konwergencję SAC i PPO.
* **Środowiska Ewaluacyjne i Testy Regresyjne:** Opracowanie zestawu stałych scenariuszy taktycznych (np. wyzwania odzyskiwania pozycji przy krawędzi, specyficzne orientacje startowe) jako zestawu testów regresyjnych, aby upewnić się, że nowe wersje modeli nie tracą fundamentalnych umiejętności przy jednoczesnej optymalizacji dla wyższego ELO.

## Cytowanie

Jeżeli to repozytorium pomogło Ci w trakcie badań, możesz je zacytować:

**Styl APA**
> Brzustowicz, S. (2026). Robot-Sumo-RL: Uczenie przez wzmocnienie dla robotów sumo z wykorzystaniem algorytmów SAC, PPO, A2C (Wersja 1.0.0) [Kod źródłowy]. https://github.com/sebastianbrzustowicz/Robot-Sumo-RL

**BibTeX**
```bibtex
@software{brzustowicz_robot_sumo_rl_2026,
  author = {Sebastian Brzustowicz},
  title = {Robot-Sumo-RL: Uczenie przez wzmocnienie dla robotów sumo z wykorzystaniem algorytmów SAC, PPO, A2C},
  url = {https://github.com/sebastianbrzustowicz/Robot-Sumo-RL},
  version = {1.0.0},
  year = {2026}
}
```
> [!TIP]
> Możesz również użyć przycisku **"Cytuj to repozytorium"** w pasku bocznym, aby automatycznie skopiować te cytowania lub pobrać surowy plik metadanych.

## Licencja

Licencja Robot-Sumo-RL Source-Available (zakaz użycia AI).  
Pełne warunki i ograniczenia znajdziesz w pliku [LICENSE](../LICENSE).

## Autor

Sebastian Brzustowicz &lt;Se.Brzustowicz@gmail.com&gt;