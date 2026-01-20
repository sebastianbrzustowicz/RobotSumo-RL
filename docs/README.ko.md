<p align="center">
  <a href="../README.md">English</a> ·
  <a href="README.zh-CN.md">中文</a> ·
  <a href="README.pl.md">Polski</a> ·
  <a href="README.es.md">Español</a> ·
  <a href="README.ja.md">日本語</a> ·
  <strong>한국어</strong> ·
  <a href="README.ru.md">Русский</a> ·
  <a href="README.fr.md">Français</a> ·
  <a href="README.de.md">Deutsch</a>
</p>

# RobotSumo RL 훈련 시스템

이 프로젝트는 강화 학습(Actor-Critic 아키텍처)을 사용하여 학습된 자율 RobotSumo 전투 에이전트를 구현합니다. 시스템은 **셀프 플레이 메커니즘**이 적용된 특수 훈련 환경을 사용하며, 학습 에이전트는 "마스터" 모델이나 자신의 이전 버전과 경쟁하면서 전투 전략을 지속적으로 발전시키고 개선합니다.

주요 기능으로는 공격적인 전진, 정확한 조준, 전략적 링 포지셔닝을 장려하면서 회전이나 후진과 같은 수동적 행동을 벌점하는 정교한 **보상 셰이핑 엔진**이 포함되어 있습니다.

### *실시간 전투 시연 및 보상 추적.*

https://github.com/user-attachments/assets/ca0baaf4-f6bf-412e-9ca7-3786b3346c5d
<p align="center">
  <em>SAC 에이전트 (초록) vs A2C 에이전트 (파랑)</em>
</p>

https://github.com/user-attachments/assets/2b496931-9eda-4c8b-88ca-7286d5fa9b42
<p align="center">
  <em>SAC 에이전트 (초록) vs PPO 에이전트 (파랑)</em>
</p>

https://github.com/user-attachments/assets/bdabd7a4-4890-47b2-a4cf-d7549b31da2e
<p align="center">
  <em>A2C 에이전트 (초록) vs PPO 에이전트 (파랑)</em>
</p>


## 시스템 아키텍처

아래 블록 다이어그램은 폐루프(closed-loop) 제어 시스템을 보여줍니다. 여기서는 **모바일 로봇**(물리/센싱 레이어)과 **RL 컨트롤러**(의사결정 레이어)를 구분합니다. 목표 신호 $\mathbf{r}_t$는 정책을 보상 엔진을 통해 최적화할 때 **훈련 단계에서만** 사용됩니다.

<div align="center">
  <img src="../resources/control_loop.png" width="650px">
</div>

### 기능 블록

* **컨트롤러 (RL 정책):** 신경망 기반 에이전트(SAC, PPO, A2C 등)로, 현재 관측 벡터를 연속적인 행동 공간으로 매핑합니다. 배포 단계에서는 추론 엔진으로 동작합니다.
* **동역학(Dynamics):** 로봇의 2차 물리 모델을 나타냅니다. 질량, 관성 모멘트, 마찰 등을 고려하여 입력 힘과 토크에 대한 반응을 계산하며, 외부 **외란(Disturbances, SAT 충돌)**의 영향을 받습니다.
* **운동학(Kinematics):** 일반화 속도를 전역 좌표계로 변환하는 상태 통합 블록입니다. 아레나 기준 좌표에서 로봇의 자세를 유지합니다.
* **센서 융합(Sensor Fusion, 인지):** 로봇 상태 벡터, 원시 글로벌 상태 데이터, 환경 정보(예: 상대 위치)를 정규화되고 자기 중심적(egocentric)인 관측 벡터로 변환하는 전처리 레이어입니다.

### 신호 벡터

블록 간 통신은 다음 수학적 벡터로 정의됩니다:

* $\mathbf{r}_t$: **보상/목표 신호** – 보상 셰이핑 함수를 통해 정책 최적화를 안내하기 위해 훈련 시에만 사용됩니다.
* $\mathbf{a}_t = [v\_{target}, \omega\_{target}]^T$: **행동 벡터** – 원하는 선속도와 각속도를 나타내는 제어 명령입니다.
* $\dot{\mathbf{x}}_t = [\dot{x}, \dot{y}, \dot{\theta}]^T$: **상태 미분 벡터** – 동역학 엔진에 의해 계산된 순간 일반화 속도입니다.
* $\mathbf{y}_t = [x, y, \theta]^T$: **물리적 출력 (Pose)** – 글로벌 좌표계에서 로봇의 현재 위치와 방향입니다.
* $\mathbf{s}_t$: **관측 벡터(`state_vec`)** – 11차원 정규화 특징 벡터로, 고유 감각(proprioceptive) 신호(속도)와 외부 감각(exteroceptive) 공간 정보(상대/모서리까지 거리)를 포함합니다.

## 상태 벡터 사양

입력 상태 벡터(`state_vec`)는 11개의 정규화된 값으로 구성되어 있으며, 에이전트가 아레나 내 상황을 종합적으로 이해할 수 있도록 합니다:

| 인덱스 | 파라미터 | 설명 | 범위 | 소스 / 센서 |
| :--- | :--- | :--- | :--- | :--- |
| 0 | `v_linear` | 로봇의 선형 속도 (앞/뒤) | [-1.0, 1.0] | 휠 엔코더 / IMU 융합 |
| 1 | `v_side` | 로봇의 측면 속도 | [-1.0, 1.0] | IMU (가속도계) / 상태 추정 |
| 2 | `omega` | 회전 속도 | [-1.0, 1.0] | 휠 엔코더 / 자이로스코프 (IMU) |
| 3 | `pos_x` | 아레나 X 위치 | [-1.0, 1.0] | 오도메트리 / 위치 추정 융합 |
| 4 | `pos_y` | 아레나 Y 위치 | [-1.0, 1.0] | 오도메트리 / 위치 추정 융합 |
| 5 | `dist_opp` | 상대까지 정규화된 거리 | [0.0, 1.0] | 거리 센서 (IR/초음파) / LiDAR |
| 6 | `sin_to_opp` | 상대 방향 각도의 사인 값 | [-1.0, 1.0] | 거리 센서 기반 기하 계산 |
| 7 | `cos_to_opp` | 상대 방향 각도의 코사인 값 | [-1.0, 1.0] | 거리 센서 기반 기하 계산 |
| 8 | `dist_edge` | 가장 가까운 아레나 모서리까지 거리 | [0.0, 1.0] | 바닥 센서 (라인 감지기) / 기하 계산 |
| 9 | `sin_to_center` | 아레나 중심 기준 방향 | [-1.0, 1.0] | 라인 센서 / 상태 추정 + 기하 계산 |
| 10 | `cos_to_center` | 아레나 중심 기준 방향 | [-1.0, 1.0] | 라인 센서 / 상태 추정 + 기하 계산 |

## 보상 셰이핑 상세

보상 시스템은 공격적 전투와 전략적 생존을 장려하도록 설계되었습니다:

* **종료 보상(Terminal Rewards):** 승리 시 큰 보너스, 탈락 또는 시간 초과(무승부) 시 큰 페널티.
* **후진 제한(Backward Block):** 후진 주행은 엄격히 페널티 적용되며, 해당 스텝의 다른 보상을 무효화합니다.
* **회전 방지(Anti-Spinning):** 무의미한 회전을 방지하기 위한 과도한 회전 페널티.
* **전진 보상(Forward Progress):** 상대를 향하는 정확도에 따라 전진 시 보상 증가.
* **운동적 참여(Kinetic Engagement):** 상대를 직접 바라보며 전진 속도를 유지할 경우 높은 보상, 결정적 공격 장려.
* **모서리 안전(Edge Safety):** 절벽 방향 이동 페널티, 아레나 중심 복귀 보상.
* **전투 역학(Combat Dynamics):** 고속 정면 충돌 시 보상, 측면/후방 충격 시 페널티.
* **효율성(Efficiency):** 스텝당 일정 시간 페널티를 부과하여 가능한 한 빠른 승리를 장려.

## 환경 사양 (Environment Specification)

시뮬레이션 환경은 공식 RobotSumo 대회 표준을 반영하도록 높은 물리적 정확도로 구축되었습니다:

* **아레나(Arena):** 
    * **도효(Dohyo):** 표준 반지름(77cm)과 정의된 중심점을 모델링. 환경은 경계 조건을 엄격히 적용하며, 로봇 섀시의 어떤 코너라도 `ARENA_DIAMETER_M`를 초과하면 경기 종료(종료 상태, Terminal State).
* **로봇 물리(Robot Physics):** 
    * **섀시(Chassis):** 로봇은 10x10 cm 정사각형(`ROBOT_SIDE`) 규격 준수.
    * **동역학(Dynamics):** 질량 기반 가속, 회전 관성, 마찰 모델 구현 (타이어 그립 시뮬레이션을 위한 측면 마찰 포함).
* **충돌 시스템(Collision System):** 실시간 접촉 처리는 **분리축 정리(Separating Axis Theorem, SAT)** 기반. 비탄성 중첩 계산 후 물리적 충격을 적용하여, 로봇 질량과 반발계수에 따라 전진 및 측면 속도에 영향을 줌.
* **시작 조건(Start Conditions):** 표준 시작 거리(아레나 반지름 약 70%) 제공, 고정 위치와 무작위 360도 방향을 모두 지원하여 학습의 강건성 향상.

## 성능 분석 및 벤치마크 (Performance Analysis & Benchmarks)

토너먼트 결과는 전투 전략의 진화와 서로 다른 강화학습 구조의 효율성을 명확히 보여줍니다. 비교 결과, 최대 성능과 수렴 속도 모두에서 뚜렷한 계층 구조가 확인됩니다.

### 토너먼트 리더보드 & 효율성

| 순위 | 에이전트 | 모델 버전 | ELO 점수 | 필요 에피소드 수 |
|:----:|:-----:|:--------------:|:----------:|:-----------------:|
| 1-5  | **SAC** | v19 - v23 | **1391 - 1614** | **~378** |
| 6-10 | **PPO** | v41 - v45 | **1128 - 1342** | **~1,049** |
| 11-15| **A2C** | v423 - v427| **791 - 949** | **10,000 - 24,604** |

> [!NOTE]
> **수렴 속도 참고:** 아키텍처별 샘플 효율성에 큰 차이가 있습니다. SAC는 최고 성능에 훨씬 빨리 도달했으며, PPO보다 약 **3배 적은 에피소드**, A2C보다 **60배 이상 적은 에피소드**로 숙련된 전투 수준에 수렴했습니다.


### 최상위 모델 비교 (Top Models Comparison)
*각 아키텍처별로 가장 성능이 높은 버전(최종 반복)을 비교합니다.*

<table width="100%">
  <tr>
    <td align="center">
      <img src="../resources/peak_elo_comparison_algos.png" width="800px"><br>
      <em>알고리즘별 최고 ELO</em>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="../resources/peak_elo_ranking_models.png" width="800px"><br>
      <em>최고 모델</em>
    </td>
  </tr>
</table>

---

### 진화적 진행 상황 (Evolutionary Progress)
*학습 전체 과정에서 정기적으로 샘플링한 모델 성능 분석 (아키텍처당 5단계).*

<table width="100%">
  <tr>
    <td align="center">
      <img src="../resources/sampled_elo_comparison_algos.png" width="800px"><br>
      <em>알고리즘별 샘플링 모델의 평균 ELO</em>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="../resources/sampled_elo_ranking_models.png" width="800px"><br>
      <em>샘플링된 모델</em>
    </td>
  </tr>
</table>

### 주요 시사점 (Key Takeaways)

* **SAC (Soft Actor-Critic) 효율성:** SAC는 이 환경에서 명백한 승자입니다. 오프폴리시 최대 엔트로피 프레임워크를 통해 최고 기술 한계(1614 ELO)에 도달하며, 샘플 효율성도 가장 높습니다.  
    * *행동적 주석:* SAC 에이전트는 위치가 변경되었을 때 방향을 회복하는 정교한 능력을 개발했으며, 상대의 작은 위치 실수까지도 적극적으로 활용했습니다.
* **PPO 안정성 및 전략:** PPO는 안정적인 학습과 경쟁력 있는 성능을 제공하는 신뢰할 수 있는 후보입니다. SAC보다 낮은 ELO에서 성능이 정체되지만, 연속 제어(continuous control)에서는 여전히 강력한 선택입니다.  
    * *행동적 주석:* 흥미롭게도 PPO 에이전트는 근접 접촉 상황("클린치")에서 뛰어난 성능을 보였으며, 상대를 불균형하게 만드는 전술적 동작을 학습하여 위치 우위를 확보했습니다.
* **A2C 성능 격차:** 기본 Advantage Actor-Critic 알고리즘은 샘플 효율성과 안정성에서 상당한 어려움을 겪었습니다. 충분한 학습에도 불구하고, 성능은 더 발전된 아키텍처의 초기 ELO보다 낮았으며, 단순한 온폴리시 방법의 한계를 보여줍니다.
* **아키텍처 진화:** 본 프로젝트는 현대 오프폴리시 방법(SAC)이 **연속적이고 비선형적인 제어 작업**에 기존 온폴리시 방법보다 훨씬 적합하다는 것을 강조합니다. SAC는 오프폴리시 데이터를 학습하면서 엔트로피를 극대화할 수 있어, 더 정교하고 적응적인 전투 행동을 가능하게 하고, 성능 한계도 크게 향상됩니다.


## 간단 시작 (Simple Start)

시뮬레이션을 실행하고 에이전트의 동작을 확인하려면 다음 단계를 따르세요:

### 설치 (Installation)
```bash
make install
```
### 빠른 데모 (크로스 플레이, 예: SAC vs PPO)
```bash
make cross-play
```

### 기타 명령어 (Other commands)
```bash
make train-sac        # 새 SAC 학습 시작 (기존 모델 삭제)
make train-ppo        # 새 PPO 학습 시작 (기존 모델 삭제)
make train-a2c        # 새 A2C 학습 시작 (기존 모델 삭제)
make test-sac         # SAC 전용 테스트 스크립트 실행
make test-ppo         # PPO 전용 테스트 스크립트 실행
make test-a2c         # A2C 전용 테스트 스크립트 실행
make tournament       # 상위 5개 학습 모델 자동 선택 후 ELO 순위 계산
make clean-models     # 모든 학습 기록과 마스터 모델 삭제
```
*사용 가능한 모든 자동화 타겟 목록은 [Makefile](../Makefile)을 참조하세요.*


## 향후 개선 가능성 (Future Potential Improvements)

* **관측 노이즈 주입 (Observation Noise Injection)**: 라이다 및 오도메트리 센서에 가우시안 노이즈 모델을 적용하여 실제 센서의 불확실성을 시뮬레이션하고, 정책의 일반화 능력과 강인성을 향상시킵니다.
* **입력 상태 확장 (Input State Expansion)**: 최근 라이다 센서 샘플을 기반으로 상대 속도를 추정하여 입력 상태 벡터에 추가함으로써 예측적 전투 동작을 개선합니다.
* **고급 물리 모델링 (Advanced Physics Modeling)**: 바퀴 미끄러짐, 선형-각속도 포화, 모터 포화 등 비선형 역학을 구현하여 현실 세계의 물리적 제약을 보다 정확하게 시뮬레이션하고, Sim-to-Real 가능성을 향상시킵니다.
* **자동 분석 및 통계 (Automated Analytics & Statistics)**: 모델 결정 과정을 분석하고 자세한 지표(예: 라운드당 평균 스텝, 회전 빈도, 후방/측면 충돌 유형 등)를 생성하는 스크립트를 작성합니다.
* **제거 연구 (Ablation Studies)**: 보상 셰이핑 함수를 파라미터화하여 제거 연구를 수행하고, 각 요소(예: 위치 vs 공격성)가 SAC 및 PPO의 안정성과 수렴에 어떻게 기여하는지 분리합니다.
* **평가 환경 및 회귀 테스트 (Evaluation Harnesses & Regression Testing)**: 고정된 전술 시나리오(예: 가장자리 회복 도전, 특정 시작 방향)를 구축하여 회귀 테스트 환경으로 활용, 새 모델 버전이 기본 기술을 잃지 않으면서 ELO 최적화를 보장합니다.

## 인용 (Citation)

연구에 이 저장소가 도움이 되었다면 다음과 같이 인용할 수 있습니다:

**APA 스타일**
> Brzustowicz, S. (2026). RobotSumo-RL: Reinforcement Learning for sumo robots using SAC, PPO, A2C algorithms (Version 1.0.0) [Source code]. https://github.com/sebastianbrzustowicz/RobotSumo-RL

**BibTeX**
```bibtex
@software{brzustowicz_robotsumo_rl_2026,
  author = {Sebastian Brzustowicz},
  title = {RobotSumo-RL: Reinforcement Learning for sumo robots using SAC, PPO, A2C algorithms},
  url = {https://github.com/sebastianbrzustowicz/RobotSumo-RL},
  version = {1.0.0},
  year = {2026}
}
```
> [!TIP]
> 사이드바의 **"Cite this repository"** 버튼을 사용하면 자동으로 인용문을 복사하거나 메타데이터 파일을 다운로드할 수 있습니다.

## 라이선스 (License)

RobotSumo-RL 소스 사용 가능 라이선스 (AI 사용 불가).
전체 조건과 제한 사항은 [LICENSE](../LICENSE) 파일을 참조하세요.

## 작성자 (Author)

Sebastian Brzustowicz &lt;Se.Brzustowicz@gmail.com&gt;
