---
layout: page
title: Notes
permalink: /notes
---

<div id='blog'>
{% assign all_notes = site.notes | concat: site.coding | sort: 'date' | reverse %}
{% assign current_year = '' %}

{% for note in all_notes %}
  {% capture year %}{{ note.date | date: '%Y' }}{% endcapture %}
  {% if year != current_year %}
    {% assign current_year = year %}
    <p class='year'>{{ year }}</p>
  {% endif %}
  <p class='post-title'><a href='{{ note.url }}'>{{ note.title }}</a></p>
  <p class='post-date'>{{ note.date | date: '%d %B %Y' }}</p>
  {% if note.blurb and note.blurb != '' %}
    <p class='post-subtitle'>{{ note.blurb }}</p>
  {% else %}
    <p class='post-subtitle'></p>
  {% endif %}
{% endfor %}
</div>
