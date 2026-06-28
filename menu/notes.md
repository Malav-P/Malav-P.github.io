---
layout: page
title: Notes
permalink: /notes
---

<style>
.tab-bar {
  display: flex;
  gap: 24px;
  border-bottom: 1px solid rgba(0,0,0,0.12);
  margin-bottom: 24px;
}
.tab-bar button {
  background: none;
  border: none;
  border-bottom: 2px solid transparent;
  padding: 8px 0;
  margin-bottom: -1px;
  font-size: 13px;
  font-family: inherit;
  color: rgba(0,0,0,0.4);
  cursor: pointer;
  font-weight: normal;
  letter-spacing: 0.03em;
}
.tab-bar button.active {
  color: rgba(0,0,0,0.85);
  font-weight: bold;
  border-bottom: 2px solid rgba(0,0,0,0.85);
}
</style>

<div class="tab-bar">
  <button class="active" onclick="filterPosts('all', this)">All</button>
  <button onclick="filterPosts('notes', this)">Notes</button>
  <button onclick="filterPosts('coding', this)">Coding</button>
</div>

<div id='blog'>
{% assign all_posts = site.notes | concat: site.coding | sort: 'date' | reverse %}
{% assign current_year = '' %}

{% for post in all_posts %}
  {% capture year %}{{ post.date | date: '%Y' }}{% endcapture %}
  {% capture collection %}{{ post.collection }}{% endcapture %}
  {% if year != current_year %}
    {% assign current_year = year %}
    <p class='year' data-collection='{{ collection }}'>{{ year }}</p>
  {% endif %}
  <div class='post-entry' data-collection='{{ collection }}'>
    <p class='post-title'><a href='{{ post.url }}'>{{ post.title }}</a></p>
    <p class='post-date'>{{ post.date | date: '%d %B %Y' }}</p>
    {% if post.blurb and post.blurb != '' %}
      <p class='post-subtitle'>{{ post.blurb }}</p>
    {% else %}
      <p class='post-subtitle'></p>
    {% endif %}
  </div>
{% endfor %}
</div>

<script>
function filterPosts(collection, btn) {
  document.querySelectorAll('.tab-bar button').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');

  const entries = document.querySelectorAll('.post-entry');
  entries.forEach(el => {
    el.style.display = (collection === 'all' || el.dataset.collection === collection) ? '' : 'none';
  });

  // Show/hide year labels: only show a year label if at least one visible entry follows it
  const children = Array.from(document.getElementById('blog').children);
  children.forEach((el, i) => {
    if (!el.classList.contains('year')) return;
    // Find the next year label or end
    let hasVisible = false;
    for (let j = i + 1; j < children.length; j++) {
      if (children[j].classList.contains('year')) break;
      if (children[j].classList.contains('post-entry') && children[j].style.display !== 'none') {
        hasVisible = true;
        break;
      }
    }
    el.style.display = hasVisible ? '' : 'none';
  });
}
</script>
