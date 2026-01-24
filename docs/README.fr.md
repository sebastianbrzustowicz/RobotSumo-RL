<p align="center">
  <a href="../README.md">English</a> ·
  <a href="README.zh-CN.md">中文</a> ·
  <a href="README.pl.md">Polski</a> ·
  <a href="README.es.md">Español</a> ·
  <a href="README.ja.md">日本語</a> ·
  <a href="README.ko.md">한국어</a> ·
  <a href="README.ru.md">Русский</a> ·
  <strong>Français</strong> ·
  <a href="README.de.md">Deutsch</a>
</p>

# Système d'Entraînement Robot Sumo RL

> [!IMPORTANT]
>  Implémentation State-of-the-Art (SOTA) : En date de janvier 2026, ce dépôt représente le framework open-source le plus avancé pour le combat de Robot Sumo, étant le premier à fournir un benchmark complet des algorithmes SAC, PPO et A2C intégrés à un mécanisme d'auto-apprentissage (self-play) compétitif.

Ce projet implémente un agent de combat autonome Robot Sumo entraîné via l'apprentissage par renforcement (architecture Actor-Critic). Le système utilise un environnement d'entraînement spécialisé avec un **mécanisme de self-play**, où l'agent en apprentissage affronte un modèle « Master » ou ses propres versions historiques pour faire évoluer et affiner continuellement ses stratégies de combat.

Les fonctionnalités clés incluent un **moteur de reward shaping** sophistiqué, conçu pour favoriser les mouvements agressifs vers l'avant, une visée précise et un positionnement stratégique sur le ring, tout en pénalisant les comportements passifs tels que le spinning ou la marche arrière.

### *Démonstration de combat en temps réel avec suivi des récompenses.*

https://github.com/user-attachments/assets/ca0baaf4-f6bf-412e-9ca7-3786b3346c5d
<p align="center">
  <em>Agent SAC (Vert) vs Agent A2C (Bleu)</em>
</p>

https://github.com/user-attachments/assets/2b496931-9eda-4c8b-88ca-7286d5fa9b42
<p align="center">
  <em>Agent SAC (Vert) vs Agent PPO (Bleu)</em>
</p>

https://github.com/user-attachments/assets/bdabd7a4-4890-47b2-a4cf-d7549b31da2e
<p align="center">
  <em>Agent A2C (Vert) vs Agent PPO (Bleu)</em>
</p>


## Architecture du Système

Le diagramme ci-dessous illustre le système de contrôle en boucle fermée. Il distingue le **Robot Mobile** (couche physique/capteurs) et le **Contrôleur RL** (couche décisionnelle). Notez que le signal d'objectif $\mathbf{r}_t$ est utilisé uniquement pendant la phase d'entraînement pour façonner la politique via le moteur de récompense.

<div align="center">
  <img src="../resources/control_loop.png" width="650px">
</div>

### Blocs Fonctionnels

* **Contrôleur (Politique RL) :** Un agent basé sur un réseau neuronal (par ex., SAC, PPO ou A2C) qui mappe le vecteur d'observation actuel vers un espace d'actions continues. Il fonctionne comme moteur d'inférence lors de la phase de déploiement.
* **Dynamique :** Représente le modèle physique du robot du second ordre. Il calcule la réponse aux forces et couples appliqués, en tenant compte de la masse, du moment d'inertie et des frottements, influencé par des **Perturbations** externes (collisions SAT).
* **Cinématique :** Un bloc d'intégration d'état qui transforme les vitesses généralisées en coordonnées globales. Il maintient la position du robot par rapport à l'origine de l'arène.
* **Fusion de Capteurs (Perception) :** Une couche de prétraitement qui transforme le vecteur d'état du robot, les données brutes de l'état global et les informations de l'environnement (ex. : position de l’adversaire) en un vecteur d'observation normalisé et centré sur le robot.

### Vecteurs de Signal

La communication entre les blocs est définie par les vecteurs mathématiques suivants :

* $\mathbf{r}_t$ : **Signal de Récompense/Objectif** – utilisé exclusivement pendant l'entraînement pour guider l'optimisation de la politique via la fonction de reward shaping.
* $\mathbf{a}_t = [v\_{target}, \omega\_{target}]^T$ : **Vecteur d’Action** – commandes de contrôle représentant les vitesses linéaires et angulaires souhaitées.
* $\dot{\mathbf{x}}_t = [\dot{x}, \dot{y}, \dot{\theta}]^T$ : **Dérivée de l’État** – vitesses généralisées instantanées calculées par le moteur dynamique.
* $\mathbf{y}_t = [x, y, \theta]^T$ : **Sortie Physique (Pose)** – coordonnées et orientation actuelles du robot dans le repère global.
* $\mathbf{s}_t$ : **Vecteur d’Observation (`state_vec`)** – vecteur de caractéristiques normalisées à 11 dimensions, contenant des indices proprioceptifs (vitesse) et des relations spatiales exteroceptives (distance à l’adversaire/bords).

## Spécification du Vecteur d’État
Le vecteur d’état d’entrée (`state_vec`) se compose de 11 valeurs normalisées, fournissant à l’agent une vue complète de la situation dans l’arène :

| Index | Paramètre | Description | Plage | Source / Capteur |
| :--- | :--- | :--- | :--- | :--- |
| 0 | `v_linear` | Vitesse linéaire du robot (avant/arrière) | [-1.0, 1.0] | Encodeurs de roues / Fusion IMU |
| 1 | `v_side` | Vitesse latérale du robot | [-1.0, 1.0] | IMU (Accéléromètre) / Estimation d’état |
| 2 | `omega` | Vitesse de rotation | [-1.0, 1.0] | Encodeurs de roues / Gyroscope (IMU) |
| 3 | `pos_x` | Position X dans l’arène | [-1.0, 1.0] | Odometry / Fusion de localisation |
| 4 | `pos_y` | Position Y dans l’arène | [-1.0, 1.0] | Odometry / Fusion de localisation |
| 5 | `dist_opp` | Distance normalisée par rapport à l’adversaire | [0.0, 1.0] | Capteurs de distance (IR/Ultrason) / LiDAR |
| 6 | `sin_to_opp` | Sinus de l’angle vers l’adversaire | [-1.0, 1.0] | Géométrie (basée sur les capteurs de distance) |
| 7 | `cos_to_opp` | Cosinus de l’angle vers l’adversaire | [-1.0, 1.0] | Géométrie (basée sur les capteurs de distance) |
| 8 | `dist_edge` | Distance jusqu’au bord le plus proche de l’arène | [0.0, 1.0] | Capteurs au sol (détecteurs de ligne) / Géométrie |
| 9 | `sin_to_center` | Direction par rapport au centre de l’arène | [-1.0, 1.0] | Capteurs de ligne / Estimation d’état + Géométrie |
| 10 | `cos_to_center` | Direction par rapport au centre de l’arène | [-1.0, 1.0] | Capteurs de ligne / Estimation d’état + Géométrie |

## Détails du Reward Shaping
Le système de récompense est conçu pour encourager le combat agressif et la survie stratégique :

* **Récompenses Terminales :** Gros bonus pour la victoire et pénalités importantes pour être sorti ou en cas de match nul.   
* **Blocage du Recul :** La marche arrière est strictement pénalisée et annule les autres récompenses pour cette étape.
* **Anti-rotation :** Pénalités pour rotation excessive afin d’éviter les spins inutiles.
* **Progrès en Avant :** Les récompenses pour avancer sont modulées par la précision du ciblage (orientation vers l’adversaire).
* **Engagement Cinétique :** Gros bonus pour maintenir la vitesse avant tout en faisant face directement à l’adversaire, encourageant les attaques décisives.
* **Sécurité aux Bords :** Logique proactive qui pénalise le mouvement vers l’abîme et récompense le retour vers le centre de l’arène.
* **Dynamique de Combat :** Récompenses pour collisions frontales à haute vitesse (poussée) et pénalités pour les coups reçus sur le côté ou l’arrière.
* **Efficacité :** Pénalité de temps constante par étape pour encourager la victoire la plus rapide possible.

## Spécification de l’Environnement
L’environnement de simulation est conçu pour refléter les standards officiels de la compétition Robot Sumo avec une grande fidélité physique :

* **Arène :** 
    * **(Dohyo) :** Modélisée avec un rayon standard (77 cm) et un point central défini. L’environnement applique strictement les limites ; un match se termine (État Terminal) dès qu’un coin du châssis d’un robot dépasse le `ARENA_DIAMETER_M`.     
* **Physique du Robot :** 
    * **Châssis :** Les robots respectent les dimensions carrées de 10x10 cm (`ROBOT_SIDE`).
    * **Dynamique :** Le système implémente des modèles d’accélération basés sur la masse, d’inertie rotationnelle et de friction (y compris la friction latérale pour simuler l’adhérence des roues).
* **Système de Collision :** La gestion des contacts en temps réel est assurée par le **Théorème de l’Axe Séparateur (SAT)**. Il calcule les chevauchements non élastiques et applique des impulsions physiques, affectant les vitesses avant et latérales en fonction de la masse et de la restitution des robots.
* **Conditions de Départ :** Distance de départ standard (~70 % du rayon de l’arène) avec support pour des positions fixes et des orientations aléatoires à 360° afin d’améliorer la robustesse de l’entraînement.

## Analyse des Performances & Référentiels

Les résultats du tournoi démontrent clairement l’évolution des stratégies de combat et l’efficacité des différentes architectures d’Apprentissage par Renforcement. La comparaison montre une hiérarchie nette tant sur les performances maximales que sur la vitesse de convergence.

### Classement du Tournoi & Efficacité

| Rang | Agent | Versions du Modèle | Classement ELO | Épisodes Requis |
|:----:|:-----:|:-----------------:|:--------------:|:---------------:|
| 1-5  | **SAC** | v19 - v23 | **1391 - 1614** | **~378** |
| 6-10 | **PPO** | v41 - v45 | **1128 - 1342** | **~1,049** |
| 11-15| **A2C** | v423 - v427| **791 - 949** | **10,000 - 24,604**|

> [!NOTE]
> **Remarque sur le Taux de Convergence :** Il existe une disparité massive dans l’efficacité d’échantillonnage entre les architectures. SAC a atteint son potentiel maximal beaucoup plus tôt, nécessitant environ **3x moins d’épisodes** que PPO et plus de **60x moins** que A2C pour converger vers un niveau de combat performant.

### Comparaison des Meilleurs Modèles
*Comparaison des versions les plus performantes (itérations finales) pour chaque architecture.*

<table width="100%">
  <tr>
    <td align="center">
      <img src="../resources/peak_elo_comparison_algos.png" width="800px"><br>
      <em>ELO maximal par algorithme</em>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="../resources/peak_elo_ranking_models.png" width="800px"><br>
      <em>Modèles au sommet</em>
    </td>
  </tr>
</table>

---

### Progression Évolutive
*Analyse des performances des modèles échantillonnés à intervalles réguliers tout au long du processus d’apprentissage (5 étapes par architecture).*

<table width="100%">
  <tr>
    <td align="center">
      <img src="../resources/sampled_elo_comparison_algos.png" width="800px"><br>
      <em>ELO moyen des modèles échantillonnés par algorithme</em>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="../resources/sampled_elo_ranking_models.png" width="800px"><br>
      <em>Modèles échantillonnés</em>
    </td>
  </tr>
</table>

### Points Clés

* **Efficacité du SAC (Soft Actor-Critic) :** Le SAC est le vainqueur incontesté dans cet environnement. Son cadre hors-politique à entropie maximale lui a permis d’atteindre le plafond de compétence le plus élevé (1614 ELO) avec la meilleure efficacité d’échantillonnage. 
    * *Remarque comportementale :* Les agents SAC ont développé une capacité sophistiquée à retrouver leur orientation lorsqu’ils sont déplacés et exploitent activement même de petites erreurs de positionnement de l’adversaire.
* **Stabilité et Tactiques du PPO :** Le PPO reste un concurrent fiable, offrant un entraînement stable et des performances compétitives. Bien qu’il atteigne un plateau à un ELO inférieur à celui du SAC, il demeure un choix robuste pour le contrôle continu.
    * *Remarque comportementale :* Fait intéressant, les agents PPO ont excellé dans les situations de "clinch", apprenant des manœuvres tactiques pour déséquilibrer l’adversaire lors de contacts rapprochés afin de gagner un avantage de positionnement.
* **Écart de Performance de l’A2C :** L’algorithme de base Advantage Actor-Critic a rencontré de grandes difficultés en termes d’efficacité d’échantillonnage et de stabilité. Même après un entraînement intensif, ses performances restaient inférieures à l’ELO de départ des architectures plus avancées, soulignant les limites des méthodes simples on-policy pour cette tâche.
* **Évolution de l’Architecture :** Le projet montre que les méthodes modernes hors-politique (SAC) sont beaucoup mieux adaptées aux **tâches de contrôle continu non linéaires** que les méthodes traditionnelles on-policy. La capacité du SAC à maximiser l’entropie tout en apprenant à partir de données hors-politique conduit à des comportements de combat plus sophistiqués et adaptatifs et à un plafond de performance significativement plus élevé.


## Démarrage Simple

Pour lancer la simulation et voir les agents en action, suivez ces étapes :

### Installation
```bash
make install
```
### Démo Rapide (Cross-Play, par ex. SAC vs PPO)
```bash
make cross-play
```

### Démo Rapide (Cross-Play, par ex. SAC vs PPO)
```bash
make train-sac        # Démarre un nouvel entraînement SAC (supprime les anciens modèles)
make train-ppo        # Démarre un nouvel entraînement PPO (supprime les anciens modèles)
make train-a2c        # Démarre un nouvel entraînement A2C (supprime les anciens modèles)
make test-sac         # Exécute le script de test dédié SAC
make test-ppo         # Exécute le script de test dédié PPO
make test-a2c         # Exécute le script de test dédié A2C
make tournament       # Sélectionne automatiquement les 5 meilleurs modèles entraînés et lance le classement ELO
make clean-models     # Supprime tout l’historique d’entraînement et les modèles principaux
```
*Pour la liste complète des cibles d’automatisation disponibles, veuillez consulter le [Makefile](../Makefile).*


## Améliorations Potentielles Futures

* **Injection de Bruit dans les Observations** : Implémentation de modèles de bruit gaussien pour les capteurs lidar et d’odométrie afin de simuler la stochasticité des capteurs réels, favorisant une meilleure généralisation et robustesse de la politique.
* **Extension du Vecteur d’État** : Expansion du vecteur d’état d’entrée avec la vitesse estimée de l’adversaire à partir des échantillons récents du lidar pour améliorer les manœuvres de combat prédictives.
* **Modélisation Avancée de la Physique** : Implémentation de dynamiques non linéaires telles que le glissement des roues, la saturation des vitesses linéaires-angulaires et la saturation des moteurs pour mieux simuler les contraintes physiques réelles et améliorer le potentiel Sim-to-Real.
* **Analytique et Statistiques Automatisées** : Création d’un script pour analyser les décisions du modèle et générer des métriques détaillées (par ex., nombre moyen d’étapes par round, fréquence de rotation et types spécifiques de collisions comme l’arrière ou latérales).
* **Études d’Ablation** : Paramétrisation de la fonction de shaping des récompenses pour réaliser des études d’ablation, isolant comment les composants individuels (par ex., positionnement vs agressivité) contribuent à la stabilité et à la convergence des modèles SAC et PPO.
* **Environnements d’Évaluation & Tests de Régression** : Développement d’un ensemble de scénarios tactiques fixes (par ex., défis de récupération en bordure, orientations de départ spécifiques) servant de suite de tests de régression, garantissant que les nouvelles versions du modèle ne perdent pas les compétences fondamentales tout en optimisant pour un ELO plus élevé.

## Citation

Si ce dépôt vous a été utile dans vos recherches, vous pouvez le citer :

**Style APA**
> Brzustowicz, S. (2026). Robot-Sumo-RL : Apprentissage par renforcement pour robots sumo utilisant les algorithmes SAC, PPO, A2C (Version 1.0.0) [Code source]. https://github.com/sebastianbrzustowicz/Robot-Sumo-RL

**BibTeX**
```bibtex
@software{brzustowicz_robot_sumo_rl_2026,
  author = {Sebastian Brzustowicz},
  title = {Robot-Sumo-RL: Apprentissage par renforcement pour robots sumo utilisant les algorithmes SAC, PPO, A2C},
  url = {https://github.com/sebastianbrzustowicz/Robot-Sumo-RL},
  version = {1.0.0},
  year = {2026}
}
```
> [!TIP]
> Vous pouvez également utiliser le bouton **"Citer ce dépôt"** dans la barre latérale pour copier automatiquement ces citations ou télécharger le fichier de métadonnées brut.

## Licence

Licence Source-Disponible Robot-Sumo-RL (Pas d’utilisation AI).
Voir le fichier [LICENSE](../LICENSE) pour les termes et restrictions complets.

## Auteur

Sebastian Brzustowicz &lt;Se.Brzustowicz@gmail.com&gt;
