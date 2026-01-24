<p align="center">
  <a href="../README.md">English</a> ·
  <a href="README.zh-CN.md">中文</a> ·
  <a href="README.pl.md">Polski</a> ·
  <strong>Español</strong> ·
  <a href="README.ja.md">日本語</a> ·
  <a href="README.ko.md">한국어</a> ·
  <a href="README.ru.md">Русский</a> ·
  <a href="README.fr.md">Français</a> ·
  <a href="README.de.md">Deutsch</a>
</p>

# Sistema de Entrenamiento Robot Sumo RL

> [!IMPORTANT]
>  Implementación State-of-the-Art (SOTA): A partir de enero de 2026, este repositorio representa el marco de código abierto más avanzado para el combate de Robot Sumo, siendo el primero en proporcionar un benchmark exhaustivo de los algoritmos SAC, PPO y A2C integrados con un mecanismo competitivo de self-play.

Este proyecto implementa un agente autónomo de combate Robot Sumo entrenado mediante Aprendizaje por Refuerzo (arquitectura Actor-Critic). El sistema utiliza un entorno de entrenamiento especializado que incorpora un **mecanismo de auto-juego (self-play)**, en el que el agente en aprendizaje compite contra un modelo "Maestro" o contra sus propias versiones históricas para evolucionar y refinar continuamente sus estrategias de combate.

Las características clave incluyen un sofisticado **motor de conformación de recompensas (reward shaping)** diseñado para promover un movimiento agresivo hacia adelante, una puntería precisa y un posicionamiento estratégico en el ring, mientras penaliza comportamientos pasivos como girar sin propósito o conducir hacia atrás.

### *Demostración de combate en tiempo real con seguimiento de recompensas en vivo.*

https://github.com/user-attachments/assets/ca0baaf4-f6bf-412e-9ca7-3786b3346c5d
<p align="center">
  <em>Agente SAC (Verde) vs Agente A2C (Azul)</em>
</p>

https://github.com/user-attachments/assets/2b496931-9eda-4c8b-88ca-7286d5fa9b42
<p align="center">
  <em>Agente SAC (Verde) vs Agente PPO (Azul)</em>
</p>

https://github.com/user-attachments/assets/bdabd7a4-4890-47b2-a4cf-d7549b31da2e
<p align="center">
  <em>Agente A2C (Verde) vs Agente PPO (Azul)</em>
</p>


## Arquitectura del Sistema

El siguiente diagrama de bloques ilustra el sistema de control en lazo cerrado. Distingue entre el **Robot Móvil** (capa física/de sensado) y el **Controlador RL** (capa de decisión). Obsérvese que la señal objetivo $\mathbf{r}_t$ se utiliza únicamente durante la fase de entrenamiento para dar forma a la política a través del motor de recompensas.

<div align="center">
  <img src="../resources/control_loop.png" width="650px">
</div>

### Bloques Funcionales

* **Controlador (Política RL):** Un agente basado en redes neuronales (p. ej., SAC, PPO o A2C) que mapea el vector de observación actual a un espacio de acciones continuas. Opera como motor de inferencia durante la fase de despliegue.
* **Dinámica:** Representa el modelo físico de segundo orden del robot. Calcula la respuesta a fuerzas y torques de entrada, teniendo en cuenta la masa, el momento de inercia y la fricción, influenciado por **Perturbaciones** externas (colisiones SAT).
* **Cinemática:** Un bloque de integración de estados que transforma velocidades generalizadas en coordenadas globales. Mantiene la pose del robot relativa al origen de la arena.
* **Fusión de Sensores (Percepción):** Una capa de preprocesamiento que transforma el vector de estado del robot, los datos de estado global sin procesar y la información del entorno (p. ej., posición del oponente) en un vector de observación normalizado y egocéntrico.

### Vectores de Señal

La comunicación entre bloques se define mediante los siguientes vectores matemáticos:

* $\mathbf{r}_t$: **Señal de recompensa/objetivo** – utilizada exclusivamente durante el entrenamiento para guiar la optimización de la política mediante la función de conformación de recompensas.
* $\mathbf{a}_t = [v\_{target}, \omega\_{target}]^T$: **Vector de acción** – comandos de control que representan las velocidades lineal y angular deseadas.
* $\dot{\mathbf{x}}_t = [\dot{x}, \dot{y}, \dot{\theta}]^T$: **Derivada de estado** – las velocidades generalizadas instantáneas calculadas por el motor de dinámica.
* $\mathbf{y}_t = [x, y, \theta]^T$: **Salida física (Pose)** – las coordenadas actuales y la orientación del robot en el marco global.
* $\mathbf{s}_t$: **Vector de observación (`state_vec`)** – un vector de características normalizadas de 11 dimensiones que contiene señales propioceptivas (velocidad) y relaciones espaciales exteroceptivas (distancia al oponente/bordes).

## Especificación del Vector de Estado
El vector de estado de entrada (`state_vec`) consta de 11 valores normalizados, que proporcionan al agente una visión completa de la situación en la arena:

| Índice | Parámetro | Descripción | Rango | Fuente / Sensor |
| :--- | :--- | :--- | :--- | :--- |
| 0 | `v_linear` | Velocidad lineal del robot (adelante/atrás) | [-1.0, 1.0] | Encoders de ruedas / Fusión IMU |
| 1 | `v_side` | Velocidad lateral del robot | [-1.0, 1.0] | IMU (Acelerómetro) / Estimación de estado |
| 2 | `omega` | Velocidad de rotación | [-1.0, 1.0] | Encoders de ruedas / Giroscopio (IMU) |
| 3 | `pos_x` | Posición X en la arena | [-1.0, 1.0] | Odómetro / Fusión de localización |
| 4 | `pos_y` | Posición Y en la arena | [-1.0, 1.0] | Odómetro / Fusión de localización |
| 5 | `dist_opp` | Distancia normalizada al oponente | [0.0, 1.0] | Sensores de distancia (IR/Ultrasonido) / LiDAR |
| 6 | `sin_to_opp` | Seno del ángulo hacia el oponente | [-1.0, 1.0] | Geometría (basada en sensores de distancia) |
| 7 | `cos_to_opp` | Coseno del ángulo hacia el oponente | [-1.0, 1.0] | Geometría (basada en sensores de distancia) |
| 8 | `dist_edge` | Distancia al borde más cercano de la arena | [0.0, 1.0] | Sensores de suelo (detectores de línea) / Geometría |
| 9 | `sin_to_center` | Dirección relativa al centro de la arena | [-1.0, 1.0] | Sensores de línea / Estimación de estado + Geometría |
| 10 | `cos_to_center` | Dirección relativa al centro de la arena | [-1.0, 1.0] | Sensores de línea / Estimación de estado + Geometría |


## Detalles de la Conformación de Recompensas
El sistema de recompensas está diseñado para imponer combate agresivo y supervivencia estratégica:

* **Recompensas Terminales:** Grandes bonificaciones por ganar y penalizaciones significativas por caer fuera o agotar el tiempo (empate).  
* **Bloqueo de Retroceso:** La conducción en reversa se penaliza estrictamente y anula otras recompensas en ese paso.
* **Anti-Giro:** Penalizaciones por rotación excesiva para evitar giros sin objetivo.
* **Progreso hacia Adelante:** Las recompensas por avanzar se escalan según la precisión de apuntado (estar de frente al oponente).
* **Compromiso Cinético:** Bonificaciones de alta magnitud por mantener velocidad hacia adelante mientras se encara directamente al oponente, fomentando ataques decisivos.
* **Seguridad en el Borde:** Lógica proactiva que penaliza el movimiento hacia el abismo y recompensa el retorno al centro de la arena.
* **Dinámica de Combate:** Recompensas por colisiones frontales de alta velocidad (empuje) y penalizaciones por ser golpeado desde el costado o la retaguardia.
* **Eficiencia:** Una penalización temporal constante por paso para incentivar la victoria lo más rápida posible.

## Especificación del Entorno
El entorno de simulación está construido para reflejar los estándares oficiales de las competiciones Robot Sumo con alta fidelidad física:

* **Arena:**  
    * **(Dohyo):** Modelada con un radio estándar (77 cm) y un punto central definido. El entorno aplica estrictamente las condiciones de borde; un combate termina (Estado Terminal) tan pronto como cualquier esquina del chasis del robot supera `ARENA_DIAMETER_M`.  
* **Física del Robot:**  
    * **Chasis:** Los robots cumplen con las dimensiones cuadradas de 10x10 cm (`ROBOT_SIDE`).  
    * **Dinámica:** El sistema implementa modelos de aceleración basados en masa, inercia rotacional y fricción (incluida la fricción lateral para simular el agarre de los neumáticos).  
* **Sistema de Colisiones:** La gestión de contactos en tiempo real está impulsada por el **Teorema del Eje Separador (SAT)**. Calcula solapamientos no elásticos y aplica impulsos físicos, afectando tanto a las velocidades longitudinales como laterales según la masa y la restitución de los robots.  
* **Condiciones de Inicio:** Incluye una distancia inicial estándar (~70% del radio de la arena) con soporte tanto para posiciones fijas como para orientaciones aleatorias de 360 grados, con el fin de aumentar la robustez del entrenamiento.


## Análisis de Rendimiento y Benchmarks

Los resultados del torneo demuestran claramente la evolución de las estrategias de combate y la eficiencia de las distintas arquitecturas de Aprendizaje por Refuerzo. La comparación muestra una jerarquía clara tanto en el rendimiento máximo como en la velocidad de convergencia.

### Clasificación del Torneo y Eficiencia

| Rango | Agente | Versiones del Modelo | Rating ELO | Episodios Requeridos |
|:----:|:-----:|:-------------------:|:----------:|:-------------------:|
| 1-5  | **SAC** | v19 - v23 | **1391 - 1614** | **~378** |
| 6-10 | **PPO** | v41 - v45 | **1128 - 1342** | **~1,049** |
| 11-15| **A2C** | v423 - v427| **791 - 949** | **10,000 - 24,604**|

> [!NOTE]
> **Nota sobre la Tasa de Convergencia:** Existe una enorme disparidad en la eficiencia muestral entre arquitecturas. SAC alcanzó su máximo potencial significativamente antes, requiriendo aproximadamente **3 veces menos episodios** que PPO y más de **60 veces menos** que A2C para converger a un nivel de combate competente.

### Comparación de los Mejores Modelos
*Comparación de las versiones con mayor rendimiento (iteraciones finales) para cada arquitectura.*

<table width="100%">
  <tr>
    <td align="center">
      <img src="../resources/peak_elo_comparison_algos.png" width="800px"><br>
      <em>ELO máximo por algoritmo</em>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="../resources/peak_elo_ranking_models.png" width="800px"><br>
      <em>Modelos en su pico de rendimiento</em>
    </td>
  </tr>
</table>

---

### Progreso Evolutivo
*Análisis del rendimiento de los modelos muestreados a intervalos regulares a lo largo de todo el proceso de aprendizaje (5 etapas por arquitectura).*

<table width="100%">
  <tr>
    <td align="center">
      <img src="../resources/sampled_elo_comparison_algos.png" width="800px"><br>
      <em>ELO promedio de los modelos muestreados por algoritmo</em>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="../resources/sampled_elo_ranking_models.png" width="800px"><br>
      <em>Modelos muestreados</em>
    </td>
  </tr>
</table>

### Conclusiones Clave

* **Eficiencia de SAC (Soft Actor-Critic):** SAC es el ganador indiscutible en este entorno. Su marco off-policy de máxima entropía le permitió alcanzar el techo de habilidad más alto (1614 ELO) con la mejor eficiencia muestral.  
    * *Nota de Comportamiento:* Los agentes SAC desarrollaron una capacidad sofisticada para recuperar su orientación cuando son desplazados y para explotar activamente incluso pequeños errores de posicionamiento del oponente.
* **Estabilidad y Tácticas de PPO:** PPO sigue siendo un contendiente fiable, ofreciendo entrenamiento estable y rendimiento competitivo. Aunque se estanca en un ELO inferior al de SAC, sigue siendo una opción robusta para control continuo.  
    * *Nota de Comportamiento:* Curiosamente, los agentes PPO destacaron en situaciones de “clinch”, aprendiendo maniobras tácticas para desestabilizar al oponente durante el contacto a corta distancia y ganar ventaja de posicionamiento.
* **Brecha de Rendimiento de A2C:** El algoritmo básico Advantage Actor-Critic tuvo serias dificultades con la eficiencia muestral y la estabilidad. Incluso con un entrenamiento extensivo, su rendimiento se mantuvo por debajo del ELO inicial de las arquitecturas más avanzadas, lo que resalta las limitaciones de los métodos on-policy más simples en esta tarea.
* **Evolución de las Arquitecturas:** El proyecto destaca que los métodos modernos off-policy (SAC) están mucho mejor adaptados a **tareas de control continuo y no lineal** que los métodos tradicionales on-policy. La capacidad de SAC para maximizar la entropía mientras aprende de datos off-policy conduce a comportamientos de combate más sofisticados y adaptativos, y a un techo de rendimiento significativamente más alto.


## Inicio Rápido

Para ejecutar la simulación y ver a los agentes en acción, sigue estos pasos:

### Instalación
```bash
make install
```
### Demostración Rápida (Cross-Play, por ejemplo SAC vs PPO)
```bash
make cross-play
```

### Demostración Rápida (Cross-Play, por ejemplo SAC vs PPO)
```bash
make train-sac        # Inicia un entrenamiento nuevo de SAC (elimina modelos antiguos)
make train-ppo        # Inicia un entrenamiento nuevo de PPO (elimina modelos antiguos)
make train-a2c        # Inicia un entrenamiento nuevo de A2C (elimina modelos antiguos)
make test-sac         # Ejecuta el script de prueba dedicado para SAC
make test-ppo         # Ejecuta el script de prueba dedicado para PPO
make test-a2c         # Ejecuta el script de prueba dedicado para A2C
make tournament       # Selecciona automáticamente los 5 mejores modelos entrenados y ejecuta el ranking ELO
make clean-models     # Elimina todo el historial de entrenamiento y los modelos principales

```
*Para una lista completa de los objetivos de automatización disponibles, consulta el [Makefile](../Makefile).*


## Posibles Mejoras Futuras

* **Inyección de Ruido en la Observación**: Implementar modelos de ruido gaussiano para los sensores lidar y de odometría para simular la estocasticidad de sensores del mundo real, facilitando una mejor generalización de la política y mayor robustez.
* **Expansión del Estado de Entrada**: Ampliar el vector de estado de entrada con la velocidad estimada del oponente basada en muestras recientes de sensores lidar para mejorar las maniobras predictivas de combate.
* **Modelado Avanzado de Física**: Implementar dinámicas no lineales como deslizamiento de ruedas, saturación de velocidad lineal-angular y saturación de motores para simular mejor las restricciones físicas reales y mejorar el potencial Sim-to-Real.
* **Análisis y Estadísticas Automatizadas**: Crear un script para analizar las decisiones del modelo y generar métricas detalladas (por ejemplo, pasos promedio por ronda, frecuencia de giros y tipos específicos de colisión como impactos traseros o laterales).
* **Estudios de Ablación**: Parametrizar la función de reward shaping para realizar estudios de ablación, aislando cómo contribuyen los componentes individuales (por ejemplo, posicionamiento vs. agresión) a la estabilidad y convergencia de SAC y PPO.
* **Entornos de Evaluación y Pruebas de Regresión**: Desarrollar un conjunto de escenarios tácticos fijos (por ejemplo, desafíos de recuperación de borde, orientaciones de inicio específicas) para servir como suite de pruebas de regresión, asegurando que nuevas versiones del modelo no pierdan habilidades fundamentales mientras optimizan para un mayor ELO.

## Cita

Si este repositorio te ha ayudado durante tu investigación, siéntete libre de citarlo:

**Estilo APA**
> Brzustowicz, S. (2026). Robot-Sumo-RL: Reinforcement Learning for sumo robots using SAC, PPO, A2C algorithms (Versión 1.0.0) [Código fuente]. https://github.com/sebastianbrzustowicz/Robot-Sumo-RL

**BibTeX**
```bibtex
@software{brzustowicz_robot_sumo_rl_2026,
  author = {Sebastian Brzustowicz},
  title = {Robot-Sumo-RL: Reinforcement Learning for sumo robots using SAC, PPO, A2C algorithms},
  url = {https://github.com/sebastianbrzustowicz/Robot-Sumo-RL},
  version = {1.0.0},
  year = {2026}
}
```
> [!TIP]
> También puedes usar el botón **"Citar este repositorio"** en la barra lateral para copiar automáticamente estas citas o descargar el archivo de metadatos.

## Licencia

Licencia Source-Available de Robot-Sumo-RL (Sin uso de IA).
Consulta el archivo [LICENSE](../LICENSE) para los términos y restricciones completos.

## Autor

Sebastian Brzustowicz &lt;Se.Brzustowicz@gmail.com&gt;
