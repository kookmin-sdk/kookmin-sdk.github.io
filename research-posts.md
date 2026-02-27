---
layout: libdoc/page
title: Research Posts
order: 3
category: Main
---

# 연구 포스팅

통신 네트워크와 엣지 AI 관련 다양한 주제에 대한 상세한 기술 논문 및 연구 노트를 여기서 찾을 수 있습니다.

{% for post in site.posts %}
- [{{ post.title }}]({{ post.url }}) - *{{ post.date | date: "%Y년 %m월 %d일" }}*
  
  {{ post.excerpt | strip_html | truncatewords: 20 }}
{% endfor %}

---

## 카테고리

### 주요 포스팅
우리 연구를 이해하기 위한 포괄적인 개요와 중요한 개념 프레임워크를 제공하는 논문들입니다.

### 기술 심화 분석
특정 기술, 알고리즘, 구현 세부사항에 대한 심층적인 기술 탐구입니다.

### 프로젝트 리포트
진행 중인 연구 프로젝트의 업데이트 및 내용 정리입니다.

---

**새로운 연구 포스팅에 대한 최신 소식을 받으시려면 정기적으로 방문해주세요!**
