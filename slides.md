---
# You can also start simply with 'default'
theme: seriph
addons:
  - "@twitwi/slidev-addon-ultracharger"
addonsConfig:
  ultracharger:
    inlineSvg:
      markersWorkaround: false
    disable:
      - metaFooter
      - tocFooter

background: /mountain.jpg

# some information about your slides (markdown enabled)
info: |
  ## Slidev Starter Template
  Presentation slides for developers.

  Learn more at [Sli.dev](https://sli.dev)
# apply unocss classes to the current slide
class: text-center
# https://sli.dev/features/drawing
drawings:
  persist: false
# slide transition: https://sli.dev/guide/animations.html#slide-transitions
# transition: slide-down
# enable MDC Syntax: https://sli.dev/features/mdc
mdc: true

title: Deep Learning
hideInToc: true
subtitle: Large Language Models
date: 21/04/2025
venue: HSE
author: Alexey Boldyrev
---

# <span style="font-size:28.0pt" v-html="$slidev.configs.title?.replaceAll(' ', '<br/>')"></span>
# <span style="font-size:32.0pt" v-html="$slidev.configs.subtitle?.replaceAll(' ', '<br/>')"></span>
# <span style="font-size:18.0pt" v-html="$slidev.configs.author?.replaceAll(' ', '<br/>')"></span>
<div>
<br>
<span style="color:#b3b3b3ff; font-size: 11px; float: right;">Image credit: ‘Glacier du Rhone au haut du Valais’<br> by Claude Niquet after Jean Séraphin Désiré Besson<br>
<a href="https://wellcomecollection.org/works/e3y95vtv">https://wellcomecollection.org/works/e3y95vtv</a>
</span>
</div>

<style>
  :deep(footer) { padding-bottom: 3em !important; }
</style>

---
src: ./slides/0_introduction.md
---

---
src: ./slides/0_end.md
---
