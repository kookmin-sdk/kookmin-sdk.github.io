---
layout: libdoc/page
title: Research Posts
order: 3
category: Main
---

# Research Posts

Here you can find detailed technical articles and research notes on various topics related to our work in communication networks and edge AI.

{% for post in site.posts %}
- [{{ post.title }}]({{ post.url }}) - *{{ post.date | date: "%B %d, %Y" }}*
  
  {{ post.excerpt | strip_html | truncatewords: 20 }}
{% endfor %}

---

## Categories

### Featured Posts
Articles that provide comprehensive overviews and important conceptual frameworks for understanding our research.

### Technical Deep-Dives
In-depth technical explorations of specific technologies, algorithms, and implementation details.

### Project Reports
Updates and documentation from our ongoing research projects.

---

**Subscribe to stay updated on new research posts!**
