---
layout: libdoc/page
title: Projects
order: 4
category: Main
---

# 진행 중인 & 과거 프로젝트

## 현재 진행 중인 프로젝트

### SCRC 연구실 프로젝트

준비 중입니다.

---

## Tech Stack

### Languages
<p>
  <img src="https://skillicons.dev/icons?i=cpp,c,java,php,python,bash,rust" />
</p>

### Backend / Framework
<p>
  <img src="https://skillicons.dev/icons?i=spring" />
</p>

### Systems / Runtime
<p>
  <img src="https://skillicons.dev/icons?i=linux,cmake" />
</p>

### Database
`Oracle` / `Tibero` / `MySQL`

### Dev / Infra
<p>
  <img src="https://skillicons.dev/icons?i=git,docker" />
</p>

---

## Side Projects

### llmrc
Rust 중심으로 구성한 로컬 LLM 실행 런타임 프로젝트입니다.  
양자화된 GGUF 모델을 로드하고 C++ 애플리케이션과 연동하여 추론을 제공하는 구조를 구현하고 있습니다.

#### 주요 내용
- Rust 기반 모델 로딩 및 런타임 관리  
- GGUF 포맷 처리  
- FFI를 통한 C++ 연동  
- HTTP 기반 추론 인터페이스  
- macOS / Linux 빌드 및 실행 환경 구성

#### 목표
- 로컬 환경에서 재현 가능한 LLM 실행 구조 확립  
- 언어 간 책임 분리를 통한 확장 가능한 아키텍처 구성  
- 서비스로 연결 가능한 런타임 기반 확보

#### 상태
기본 실행 파이프라인을 구현했으며, 안정화와 기능 확장을 진행 중입니다.

**Repo:**  
`https://github.com/Azabell1993/llmrc`  


---  

### QT_Kernel_OS
Qt/C++ 기반으로 운영체제 구조와 동작 원리를 이해하기 위한 CLI 학습 프로젝트입니다.

#### 내용
- 콘솔 인터페이스 설계  
- 프로세스 / 메모리 / 파일 시스템 개념 구현  
- 커널 구조 이해를 위한 실험 환경 제작

**Demo:**  
`https://azabell1993.github.io/QT_Kernel_OS/files.html`  

---

### ml-engine
C++과 LibTorch를 사용하여 로컬 모델을 로딩하고 API 형태로 제공하는 추론 환경을 구성하는 프로젝트입니다.

#### 내용
- 모델 로딩 및 추론 파이프라인 구현  
- 경량 REST 서버 구성  
- GGUF 및 llama.cpp 연동 실험
  
**Repo:**  
`https://github.com/Azabell1993/ml-engine`    
  
---

## Interests

- On-device AI  
- Local LLM Runtime  
- Inference Architecture  
- System Performance  
- Language Interoperability (Rust ↔ C++)
