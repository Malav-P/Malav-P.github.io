---
layout: page
title: 
permalink: /technical-notes
---

Here I will present notes from my learning throughout my undergrad and PhD.

<ul class="posts-container">
  {% for post in site.posts %}
    {% unless post.img == "essay" %}
      <div style="margin-top:10%;">
      {% unless post.next %}
        <h2>{{ post.date | date: '%Y' }}</h2>
      {% else %}
        {% capture year %}{{ post.date | date: '%Y' }}{% endcapture %}
        {% capture nyear %}{{ post.next.date | date: '%Y' }}{% endcapture %}
        {% if year != nyear %}
          <h2>{{ post.date | date: '%Y' }}</h2>
        {% endif %}
      {% endunless %}
    </div>
      <li itemscope>
        <div style="margin-top:10%;">
          <a href="{{ site.github.url }}{{ post.url }}" style="text-decoration:none;">{{ post.title }}</a>
          <span class="post-date"> {{ post.date | date: "%B %-d" }}</span>
          <p class="post-date">{{ post.blurb }}</p>
        </div>
      </li>
    {% endunless %}
  {% endfor %}
</ul>
