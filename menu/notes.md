---
layout: default
title: Notes
permalink: /notes
---

<h1>Notes</h1>

{% assign all_notes = site.notes | concat: site.coding | sort: 'date' | reverse %}
{% assign current_year = '' %}

<ul class="notes-list">
{% for note in all_notes %}
  {% capture year %}{{ note.date | date: '%Y' }}{% endcapture %}
  {% if year != current_year %}
    {% assign current_year = year %}
    <li class="notes-year">{{ year }}</li>
  {% endif %}
  <li class="notes-entry">
    <span class="notes-date">{{ note.date | date: "%b %-d" }}</span>
    <a href="{{ note.url }}">{{ note.title }}</a>
  </li>
{% endfor %}
</ul>
