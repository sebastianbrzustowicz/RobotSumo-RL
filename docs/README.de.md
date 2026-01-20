<p align="center">
  <a href="../README.md">English</a> ·
  <a href="README.zh-CN.md">中文</a> ·
  <a href="README.pl.md">Polski</a> ·
  <a href="README.es.md">Español</a> ·
  <a href="README.ja.md">日本語</a> ·
  <a href="README.ko.md">한국어</a> ·
  <a href="README.ru.md">Русский</a> ·
  <a href="README.fr.md">Français</a> ·
  <strong>Deutsch</strong>
</p>

# RobotSumo RL Trainingssystem

Dieses Projekt implementiert einen autonomen RobotSumo-Kampfagenten, der mithilfe von **Reinforcement Learning (Actor-Critic-Architektur)** trainiert wird. Das System verwendet eine spezialisierte Trainingsumgebung mit einem **Self-Play-Mechanismus**, bei dem der lernende Agent gegen ein "Master"-Modell oder seine eigenen historischen Versionen antritt, um kontinuierlich seine Kampftaktiken zu entwickeln und zu verfeinern.  

Wesentliche Merkmale umfassen eine ausgeklügelte **Reward-Shaping-Engine**, die aggressives Vorwärtsbewegen, präzises Zielen und strategisches Positionieren im Ring fördert, während passive Verhaltensweisen wie Drehen auf der Stelle oder Rückwärtsfahren bestraft werden.

### *Echtzeit-Kampfdemonstration mit Live-Belohnungsverfolgung.*

https://github.com/user-attachments/assets/ca0baaf4-f6bf-412e-9ca7-3786b3346c5d
<p align="center">
  <em>SAC-Agent (Grün) vs A2C-Agent (Blau)</em>
</p>

https://github.com/user-attachments/assets/2b496931-9eda-4c8b-88ca-7286d5fa9b42
<p align="center">
  <em>SAC-Agent (Grün) vs PPO-Agent (Blau)</em>
</p>

https://github.com/user-attachments/assets/bdabd7a4-4890-47b2-a4cf-d7549b31da2e
<p align="center">
  <em>A2C-Agent (Grün) vs PPO-Agent (Blau)</em>
</p>


## Systemarchitektur

Das folgende Blockdiagramm zeigt das Regelungssystem im geschlossenen Kreis. Es unterscheidet zwischen dem **mobilen Roboter** (physikalische/Sensor-Ebene) und dem **RL-Controller** (Entscheidungsebene). Beachten Sie, dass das Zielsiganl $\mathbf{r}_t$ nur während der Trainingsphase verwendet wird, um die Policy über die Reward-Engine zu formen.

<div align="center">
  <img src="../resources/control_loop.png" width="650px">
</div>

### Funktionale Blöcke

* **Controller (RL-Policy):** Ein auf neuronalen Netzen basierender Agent (z. B. SAC, PPO oder A2C), der den aktuellen Beobachtungsvektor auf einen kontinuierlichen Aktionsraum abbildet. Er fungiert während der Einsatzphase als Inferenzmaschine.
* **Dynamik:** Stellt das physikalische Modell zweiter Ordnung des Roboters dar. Es berechnet die Reaktion auf Eingabekräfte und -momente unter Berücksichtigung von Masse, Trägheitsmoment und Reibung, beeinflusst durch externe **Störungen** (SAT-Kollisionen).
* **Kinematik:** Ein Zustandsintegrationsblock, der generalisierte Geschwindigkeiten in globale Koordinaten transformiert. Er hält die Pose des Roboters relativ zum Ursprung der Arena aufrecht.
* **Sensorfusion (Wahrnehmung):** Eine Vorverarbeitungsschicht, die den Roboterzustandsvektor, rohe globale Zustandsdaten und Umgebungsinformationen (z. B. Position des Gegners) in einen normalisierten, egozentrischen Beobachtungsvektor umwandelt.

### Signalvektoren

Die Kommunikation zwischen den Blöcken wird durch die folgenden mathematischen Vektoren definiert:

* $\mathbf{r}_t$: **Belohnungs-/Zielsignal** – wird ausschließlich während des Trainings verwendet, um die Policy-Optimierung über die Reward-Shaping-Funktion zu steuern.
* $\mathbf{a}_t = [v\_{target}, \omega\_{target}]^T$: **Aktionsvektor** – Steuerbefehle, die die gewünschten linearen und Winkelgeschwindigkeiten repräsentieren.
* $\dot{\mathbf{x}}_t = [\dot{x}, \dot{y}, \dot{\theta}]^T$: **Zustandsableitung** – die momentanen generalisierten Geschwindigkeiten, berechnet durch die Dynamik-Engine.
* $\mathbf{y}_t = [x, y, \theta]^T$: **Physikalische Ausgabe (Pose)** – die aktuellen Koordinaten und Orientierung des Roboters im globalen Rahmen.
* $\mathbf{s}_t$: **Beobachtungsvektor (`state_vec`)** – ein 11-dimensionaler normalisierter Merkmalsvektor, der propriozeptive Hinweise (Geschwindigkeit) und exterozeptive räumliche Beziehungen (Abstand zum Gegner/Rand) enthält.

## Spezifikation des Zustandsvektors
Der Eingabe-Zustandsvektor (`state_vec`) besteht aus 11 normalisierten Werten und bietet dem Agenten eine umfassende Übersicht über die Situation in der Arena:

| Index | Parameter | Beschreibung | Bereich | Quelle / Sensor |
| :--- | :--- | :--- | :--- | :--- |
| 0 | `v_linear` | Lineargeschwindigkeit des Roboters (vorwärts/rückwärts) | [-1.0, 1.0] | Raddrehgeber / IMU-Fusion |
| 1 | `v_side` | Laterale Geschwindigkeit des Roboters | [-1.0, 1.0] | IMU (Beschleunigungssensor) / Zustandsabschätzung |
| 2 | `omega` | Rotationsgeschwindigkeit | [-1.0, 1.0] | Raddrehgeber / Gyroskop (IMU) |
| 3 | `pos_x` | X-Position in der Arena | [-1.0, 1.0] | Odometriedaten / Lokalisierungsfusion |
| 4 | `pos_y` | Y-Position in der Arena | [-1.0, 1.0] | Odometriedaten / Lokalisierungsfusion |
| 5 | `dist_opp` | Normalisierte Entfernung zum Gegner | [0.0, 1.0] | Abstandssensoren (IR/Ultraschall) / LiDAR |
| 6 | `sin_to_opp` | Sinus des Winkels zum Gegner | [-1.0, 1.0] | Geometrie (basierend auf Abstandssensoren) |
| 7 | `cos_to_opp` | Kosinus des Winkels zum Gegner | [-1.0, 1.0] | Geometrie (basierend auf Abstandssensoren) |
| 8 | `dist_edge` | Entfernung zum nächsten Arena-Rand | [0.0, 1.0] | Bodensensoren (Linien-Detektoren) / Geometrie |
| 9 | `sin_to_center` | Richtung relativ zum Arenazentrum | [-1.0, 1.0] | Liniensensoren / Zustandsabschätzung + Geometrie |
| 10 | `cos_to_center` | Richtung relativ zum Arenazentrum | [-1.0, 1.0] | Liniensensoren / Zustandsabschätzung + Geometrie |


## Details zum Reward Shaping
Das Belohnungssystem ist darauf ausgelegt, aggressives Kämpfen und strategisches Überleben zu fördern:

* **Terminal Rewards:** Große Boni für Siege und erhebliche Strafen für Herausfallen oder Zeitüberschreitung (Unentschieden).  
* **Backward Block:** Rückwärtsfahren wird strikt bestraft und hebt andere Belohnungen für diesen Schritt auf.
* **Anti-Spinning:** Strafen für übermäßige Rotation, um zielloses Drehen zu verhindern.
* **Forward Progress:** Belohnungen für Vorwärtsbewegung werden anhand der Zielgenauigkeit (Ausrichtung auf den Gegner) skaliert.
* **Kinetische Interaktion:** Hochwertige Boni für Beibehaltung der Vorwärtsgeschwindigkeit bei direkter Ausrichtung auf den Gegner, um entschlossene Angriffe zu fördern.
* **Rand-Sicherheit:** Proaktive Logik, die Bewegungen Richtung Abgrund bestraft und die Rückkehr zum Arena-Zentrum belohnt.
* **Kampfdynamik:** Belohnungen für Kopf-an-Kopf-Kollisionen mit hoher Geschwindigkeit (Schubsen) und Strafen für Treffer von der Seite oder von hinten.
* **Effizienz:** Eine konstante Zeitstrafe pro Schritt, um den schnellstmöglichen Sieg zu fördern.

## Umgebungs-Spezifikation
Die Simulationsumgebung wurde entwickelt, um den offiziellen RobotSumo-Wettkampfstandards mit hoher physikalischer Genauigkeit zu entsprechen:

* **Arena:** 
    * **(Dohyo):** Modelliert mit einem Standardradius von 77 cm und einem definierten Mittelpunkt. Die Umgebung erzwingt strikt die Randbedingungen; ein Match endet (Terminalzustand), sobald eine Ecke des Roboterchassis den `ARENA_DIAMETER_M` überschreitet.     
* **Robotik-Physik:** 
    * **Chassis:** Roboter halten die quadratischen Abmessungen von 10x10 cm (`ROBOT_SIDE`) ein.
    * **Dynamik:** Das System implementiert massenbasierte Beschleunigung, Rotationsinertien und Reibungsmodelle (einschließlich lateraler Reibung zur Simulation des Reifen-Grips).
* **Kollisionssystem:** Die Echtzeit-Kontaktbehandlung basiert auf dem **Separating Axis Theorem (SAT)**. Es berechnet nicht-elastische Überlappungen und wendet physikalische Impulse an, die sowohl die Vorwärts- als auch die Lateralgschwindigkeit unter Berücksichtigung der Masse und Rückpralleigenschaften der Roboter beeinflussen.
* **Startbedingungen:** Standardstartabstand (~70% des Arenaradius) mit Unterstützung für feste Startpositionen und zufällige 360°-Orientierungen, um die Trainingsrobustheit zu erhöhen.


## Leistungsanalyse & Benchmarks

Die Turnierergebnisse zeigen deutlich die Entwicklung der Kampfstrategien und die Effizienz der verschiedenen Reinforcement-Learning-Architekturen. Der Vergleich zeigt eine klare Hierarchie sowohl in der Spitzenleistung als auch in der Geschwindigkeit der Konvergenz.

### Turnier-Rangliste & Effizienz

| Rang | Agent | Modellversionen | ELO-Wert | Benötigte Episoden |
|:----:|:-----:|:--------------:|:----------:|:-----------------:|
| 1-5  | **SAC** | v19 - v23 | **1391 - 1614** | **~378** |
| 6-10 | **PPO** | v41 - v45 | **1128 - 1342** | **~1,049** |
| 11-15| **A2C** | v423 - v427| **791 - 949** | **10,000 - 24,604**|

> [!NOTE]
> **Hinweis zur Konvergenzrate:** Es gibt eine enorme Diskrepanz in der Proben-Effizienz zwischen den Architekturen. SAC erreichte sein Spitzenpotenzial deutlich früher, benötigte etwa **3x weniger Episoden** als PPO und über **60x weniger** als A2C, um ein leistungsfähiges Kampf-Level zu erreichen.

### Vergleich der Top-Modelle
*Vergleich der leistungsstärksten Versionen (Enditerationen) jeder Architektur.*

<table width="100%">
  <tr>
    <td align="center">
      <img src="../resources/peak_elo_comparison_algos.png" width="800px"><br>
      <em>Maximales ELO nach Algorithmus</em>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="../resources/peak_elo_ranking_models.png" width="800px"><br>
      <em>Top-Modelle</em>
    </td>
  </tr>
</table>

---

### Evolutionärer Fortschritt
*Analyse der Modellleistung, die in regelmäßigen Abständen über den gesamten Lernprozess hinweg (5 Stufen pro Architektur) ausgewertet wurde.*

<table width="100%">
  <tr>
    <td align="center">
      <img src="../resources/sampled_elo_comparison_algos.png" width="800px"><br>
      <em>Durchschnittliches ELO der ausgewählten Modelle nach Algorithmus</em>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="../resources/sampled_elo_ranking_models.png" width="800px"><br>
      <em>Ausgewählte Modelle</em>
    </td>
  </tr>
</table>

### Wichtige Erkenntnisse

* **SAC (Soft Actor-Critic) Effizienz:** SAC ist der unbestrittene Sieger in dieser Umgebung. Sein Off-Policy-Maximum-Entropy-Framework ermöglichte es, das höchste Fähigkeitsniveau (1614 ELO) mit der besten Proben-Effizienz zu erreichen.  
    * *Verhaltenshinweis:* SAC-Agenten entwickelten die Fähigkeit, ihre Orientierung nach Verschiebungen wiederherzustellen und selbst kleinere Positionsfehler des Gegners aktiv auszunutzen.
* **PPO Stabilität & Taktik:** PPO bleibt ein verlässlicher Kandidat und bietet stabiles Training sowie wettbewerbsfähige Leistung. Auch wenn es ein niedrigeres ELO als SAC erreicht, bleibt es eine robuste Wahl für kontinuierliche Steuerung.  
    * *Verhaltenshinweis:* Interessanterweise zeigten PPO-Agenten besondere Stärke in "Clinchen"-Situationen, indem sie taktische Manöver erlernten, um den Gegner bei engem Kontakt aus dem Gleichgewicht zu bringen und sich einen Positionsvorteil zu verschaffen.
* **A2C Leistungslücke:** Der grundlegende Advantage Actor-Critic-Algorithmus hatte erhebliche Schwierigkeiten mit Proben-Effizienz und Stabilität. Selbst bei umfangreichem Training blieb seine Leistung unter dem Start-ELO der fortschrittlicheren Architekturen, was die Grenzen einfacherer On-Policy-Methoden in dieser Aufgabe verdeutlicht.
* **Architekturentwicklung:** Das Projekt zeigt, dass moderne Off-Policy-Methoden (SAC) deutlich besser für **kontinuierliche, nichtlineare Steuerungsaufgaben** geeignet sind als traditionelle On-Policy-Methoden. Die Fähigkeit von SAC, Entropie zu maximieren und gleichzeitig aus Off-Policy-Daten zu lernen, führt zu komplexeren, adaptiven Kampfverhalten und einer deutlich höheren Leistungsgrenze.


## Einfacher Start

Um die Simulation auszuführen und die Agenten in Aktion zu sehen, gehen Sie wie folgt vor:

### Installation
```bash
make install
```
### Schnell-Demo (Cross-Play, z.B. SAC vs PPO)
```bash
make cross-play
```

### Weitere Befehle
```bash
make train-sac        # Startet neues SAC-Training (alte Modelle werden gelöscht)
make train-ppo        # Startet neues PPO-Training (alte Modelle werden gelöscht)
make train-a2c        # Startet neues A2C-Training (alte Modelle werden gelöscht)
make test-sac         # Führt das spezielle SAC-Testskript aus
make test-ppo         # Führt das spezielle PPO-Testskript aus
make test-a2c         # Führt das spezielle A2C-Testskript aus
make tournament       # Wählt automatisch die Top-5 trainierten Modelle aus & erstellt ELO-Ranking
make clean-models     # Löscht die gesamte Trainingshistorie und Master-Modelle
```
*Eine vollständige Liste aller verfügbaren Automatisierungsziele finden Sie im [Makefile](../Makefile).*


## Mögliche zukünftige Verbesserungen

* **Rausch-Injektion für Sensoren:** Implementierung von Gaußschen Rauschmodellen für Lidar- und Odometriersensoren, um die stochastische Natur realer Sensoren zu simulieren und so die Generalisierung und Robustheit der Policy zu verbessern.  
* **Erweiterung des Zustandsvektors:** Erweiterung des Eingabestatusvektors um geschätzte Gegnergeschwindigkeiten basierend auf aktuellen Lidar-Daten, um prädiktive Kampfmanöver zu verbessern.  
* **Erweiterte Physiksimulation:** Implementierung nichtlinearer Dynamiken wie Radschlupf, Sättigung der Linear- und Winkelgeschwindigkeit sowie Motorsättigung, um physikalische Realitätsbedingungen besser zu simulieren und das Sim-to-Real-Potenzial zu steigern.  
* **Automatisierte Analysen & Statistiken:** Erstellung eines Skripts zur Analyse von Modellentscheidungen und Generierung detaillierter Metriken (z. B. durchschnittliche Schritte pro Runde, Drehfrequenz, spezifische Kollisionsarten wie Heck- oder Seitenaufprall).  
* **Ablationsstudien:** Parametrisierung der Reward-Shaping-Funktion, um Ablationsstudien durchzuführen und zu isolieren, wie einzelne Komponenten (z. B. Positionierung vs. Aggression) zur Stabilität und Konvergenz von SAC und PPO beitragen.  
* **Evaluierungs-Framework & Regressionstests:** Entwicklung einer Reihe fester taktischer Szenarien (z. B. Randwiederherstellungs-Herausforderungen, spezifische Startorientierungen), die als Regressionstest dienen, um sicherzustellen, dass neue Modellversionen grundlegende Fähigkeiten beibehalten und gleichzeitig höhere ELO-Werte erreichen.

## Zitation

Wenn dieses Repository Ihre Forschung unterstützt hat, können Sie es wie folgt zitieren:

**APA-Stil**
> Brzustowicz, S. (2026). RobotSumo-RL: Reinforcement Learning für Sumo-Roboter unter Verwendung der SAC-, PPO- und A2C-Algorithmen (Version 1.0.0) [Quellcode]. https://github.com/sebastianbrzustowicz/RobotSumo-RL

**BibTeX**
```bibtex
@software{brzustowicz_robotsumo_rl_2026,
  author = {Sebastian Brzustowicz},
  title = {RobotSumo-RL: Reinforcement Learning für Sumo-Roboter unter Verwendung der SAC-, PPO- und A2C-Algorithmen},
  url = {https://github.com/sebastianbrzustowicz/RobotSumo-RL},
  version = {1.0.0},
  year = {2026}
}
```
> [!TIP]
> Sie können auch die **„Dieses Repository zitieren“**-Schaltfläche in der Seitenleiste verwenden, um die Zitation automatisch zu kopieren oder die Rohmetadaten herunterzuladen.

## Lizenz

RobotSumo-RL Source-Available License (Keine KI-Nutzung).
Siehe die [LICENSE](../LICENSE) für vollständige Bedingungen und Einschränkungen.

## Autor

Sebastian Brzustowicz &lt;Se.Brzustowicz@gmail.com&gt;
