# Why we need Language Models?

#### Many NLP tasks require natural language output: 
* **Machine translation**: return text in the target language
* **Speech recognition**: return a transcript of what was said
* **Natural language generation**: return text in natural language
* **Spell checking**: return the corrected spelling of the input

<br>

### Language models define **probability distributions** over strings or sentences (natural language)<br> ➔ We can use a language model **to evaluate possible strings** so that we can choose the best (i.e. most likely) one:<br> if $P_{LM}(A) > P_{LM}(B)$, return $A$ rather than $B$

---

# What is Large Language Model (LLM)?

* Scaled up versions of Transformer architecture, e.g. millions/billions of parameters
* Typically pretrained on massive amounts of “general” textual data (e.g. web corpus)
* Training objective is typically “next token prediction”: $P(W_{t+1} | W_t, W_{t-1}, ..., W_1)$
* Emergent abilities as they scale up (e.g. chain-of-thought reasoning)
* Heavy computational cost (time, money, GPUs)
* Larger general ones: “plug-and-play” with few or zero-shot learning
  * Train once, then adapt to other tasks without needing to retrain
  * E.g. in-context learning and prompting

---

# How "Large" are Large Language Models?

#### Today we are mainly dealing with two types of models:

* Medium-sized models:
  * [BERT](https://arxiv.org/abs/1810.04805)/[RoBERTa](https://arxiv.org/abs/1907.11692) models (110M or 355M **parameters**)
  * [GPT-1](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) (117M)
  * [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (1.5B)
  * [T5 models](https://arxiv.org/abs/1910.10683) (220M, 770M, 3B)
* “Very” large language (foundation) models: > 100B
  * [GPT-3](https://arxiv.org/abs/2005.14165) (**175B**, 13B, 6.7B, 2.7B, 1.3B, 760M, 350M, 125M)
  * GPT-3.5 (~175B)
  * [GPT-4](https://arxiv.org/pdf/2303.08774) (~1.7T), GPT-4o (?), OpenAI o1 (?), GPT-4.5 (?)
  * DeepSeek [V3](https://arxiv.org/abs/2412.19437) (671B), [R1](https://arxiv.org/abs/2501.12948) (671B)

---

# Main properties of LLMs

<div class="grid grid-cols-[2fr_2fr] gap-12">
<div>
<br>

* Scalable network architecture
  * **Transformer** vs. CNN/RNN

* Scalable target
  * Autoregressive LM vs. masked LM

* Scalable data
  * Plain text everywhere vs. labeled data
</div>
<div>
    <figure>
    <img src="/NIPS-2017-attention-is-all-you-need-Paper.svg" style="width: 285px !important;">
      <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute"><br>Image source:
      <a href="https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf">arXiv:1706.03762 Fig. 1</a> ("Attention Is All You Need")
    </figcaption>
  </figure>
</div>
</div>

---

# Natural Language Processing

* Natural language processing (NLP) studies interactions between computers and humans using natural languages
* *Self-supervised learning* has been extensively used to **pretrain** text representations

<br>
<center>
  <figure>
    <img src="/nlp-map-pretrain.svg" style="width: 520px !important;">
      <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute"><br>Image source:
      <a href="https://d2l.ai/chapter_natural-language-processing-pretraining/index.html">d2l.ai fig. 15.1</a>
    </figcaption>
  </figure>
</center>

---

# Why Pretraining?

#### Pre-training allows you to:
* Build a rich representation of the language
* Initialize weights for a strong NLP model
* Build a probability distribution of the language from which tokens can be sampled

<br>

#### Pretraining for different types of architectures:
* **Encoder**: Get bidirectional contexts - can predict
  * Examples: BERT and its variants (RoBERTa)
* **Decoder**: Great for generation - can't predict
  * Examples: GPT, LaMDa
* **Encoder-Decoder**: All the good things from the encoder and decoder
  * Examples: Transformer, T5

---

# Embedding Models

* Context-independent:
  * [**Word2Vec**](https://d2l.ai/chapter_natural-language-processing-pretraining/word2vec.html)
    * Prediction method
    * Vectors are used to represent words
    * Two variants:
      * Skip-gram: a word can be used to generate its surrounding words in a text sequence
      * Continuous bag of words (CBOW): a center word is generated based on its surrounding context words

  * [**GloVe**](https://d2l.ai/chapter_natural-language-processing-pretraining/glove.html) (**Glo**bal **Ve**ctors)
    * Information comes from global corpus statistics
    * Vectors are learned by gradient descent

---

# Embedding Models

* Context-sensitive:
  * [**TagLM**](https://arxiv.org/abs/1705.00108) (language-model-augmented sequence tagger)
  * [**CoVe**](https://arxiv.org/abs/1708.00107) (Context Vectors)
    * Encode not only individual words but words along with their context
    * CoVe train Neural Machine Translation (LSTMs and Attention) system and use its encoder
    * Bidirectional encoder: knows both left and right contexts
  * [**ELMo**](https://arxiv.org/abs/1802.05365) (Embeddings from Language Models)
    * Three layers:
      * L0 (embeddings) - output of the character-level CNN
      * L1 - concatenated representations from layer 1 of both forward and backward LSTMs
      * L2 - concatenated representations from layer 2 of both forward and backward LSTMs
    * The weighted sum of representations from all layers, is used to represent a word

---

# Transfer Through Word Embeddings

<br>
<br>
<br>

<center>
  <figure>
    <img src="/embs_what_to_do_with_data-min.png" style="width: 760px !important;">
      <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute"><br>Image source:
      <a href="https://lena-voita.github.io/nlp_course/transfer_learning.html">https://lena-voita.github.io/nlp_course/transfer_learning.html</a>
    </figcaption>
  </figure>
</center>

---

# GPT: From Task-Specific to Task-Agnostic

* ELMo improved NLP tasks but still required a task-specific architecture
* Designing a new architecture for every task is impractical
* [**GPT**](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) (Generative Pre-Training) is **task-agnostic model** for context-sensitive text representations
  * Built on a Transformer decoder architecture
  * Pretrains on large text data to learn representations
  * For downstream tasks, adds a simple linear output layer
* Main differences from ELMo:
  * ELMo keeps its pretrained parameters frozen
  * GPT fine-tunes all model parameters during supervised learning
* GPT achieved state-of-the-art results on many tasks: natural language inference, question answering, sentence similarity, and classification
  * Minimal model changes needed for new tasks

<!--
Limitation  
- Because GPT is autoregressive (only looks left-to-right), it may misinterpret words needing both left and right context (e.g., "bank" in different sentences).  
- The same representation is returned for polysemous words even when meaning shifts with surrounding text.
-->

---
zoom: 0.94
---

# BERT

- ELMo encodes context bidirectionally but relies on task-specific architectures
- GPT is task-agnostic but encodes context only left-to-right
- [BERT](https://arxiv.org/pdf/1810.04805.pdf) (Bidirectional Encoder Representations from Transformers) combines both strengths:
  - Encodes context bidirectionally using a pretrained Transformer encoder
  - Requires minimal changes to architecture for various NLP tasks

- Advantages of BERT
  - Any token is represented with full left and right context
  - Adaptable to many tasks (e.g., predicting per token or per sequence)

- Downstream Training with BERT is similar to GPT:
  - BERT representations are input to a simple output layer, minimally changed for task needs
  - All Transformer encoder parameters are fine-tuned
  - Output layer is trained from scratch

<!--
- Summary:  
  - BERT brings together the flexibility of GPT and the bidirectional context of ELMo, creating a powerful, general-purpose NLP model.
-->

---

# ELMo, GPT, and BERT

<br>
<center>
  <figure>
    <img src="/elmo-gpt-bert.svg" style="width: 560px !important;">
      <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute"><br>Image source:
      <a href="https://d2l.ai/chapter_natural-language-processing-pretraining/bert.html">d2l.ai fig. 15.8.1</a>
    </figcaption>
  </figure>
</center>

---

# GPT (again)

<br>

* If $y_1, ..., y_n$  is a training token sequence, then at the timestep $t$ a model predicts a probability distribution $p^{(t)} = p(\ast | y_1, ..., y_{t-1})$
* The model is trained with the standard cross-entropy loss, and the loss for the whole sequence is
$$L_{\mathrm{LM}} = -\sum\limits_{t=1}^n \log (p(y_t | y_{<t}))$$

* The fine-tuning loss consists of the task-specific loss, as well as the language modeling loss:
$$L = L_{\mathrm{LM}} + \lambda \cdot L_{\mathrm{task}}$$
* In the fine-tuning stage, the model architecture stays the same except for the final linear layer

<br>

##### Slide source: [https://lena-voita.github.io/nlp_course/transfer_learning.html#gpt](https://lena-voita.github.io/nlp_course/transfer_learning.html#gpt)

---

# GPT Training Objective

<br>
<center>
  <figure>
    <img src="/GPT_tasks.svg" style="width: 860px !important;">
      <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute"><br>Image source:
      <a href="https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf">Fig. 1 from "Improving Language Understanding by Generative Pre-Training"</a> (Original GPT paper)
    </figcaption>
  </figure>
</center>

---

# GPT-1, GPT-2, GPT-3

* These models are different mostly in the amount of training data and the number of parameters
* Nevertheless, let's take a look at their differences

<br>

#### The following GPT-related slides are based on:
* GPT-1: [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
* GPT-2: [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
* GPT-3: [Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165.pdf)
* [Priya Shree's blog post](https://medium.com/walmartglobaltech/the-journey-of-open-ai-gpt-models-32d95b7b7fb2)

<br>

####  Note that these models are so large that only big companies can afford to train one.

---
zoom: 0.97
---

# GPT-1 Details

* GPT-1 used the [BooksCorpus dataset](https://paperswithcode.com/dataset/bookcorpus) to train the language model
  * BooksCorpus had some 11000 unpublished books which helped training the language model on unseen data
    * This data was unlikely to be found in test set of downstream tasks
* [Byte Pair Encoding](https://en.wikipedia.org/wiki/Byte_pair_encoding) (BPE) vocabulary with 40000 merges was used
* Model used 768-dimensional state for encoding tokens into word embeddings. Position embeddings were also learnt during training
* Architecture: 12-layer Transformer decoder (without decoder-encoder attention)<br> with 12 attention heads in each self-attention layer
  * **Masking** helped achieve the language model objective wherein the language model did not have access to subsequent words to the right of current word
  * For position wise feed forward layer 3072-dimensional state was used
* **[Zero-shot learning](https://en.wikipedia.org/wiki/Zero-shot_learning)** is observed

<!--
Byte Pair Encoding is a data compression technique in which frequently occurring pairs of consecutive bytes are replaced with a byte not present in data to compress the data. To reconstruct the original data, a table containing mapping of replaced bytes is used. This blog explains BPE in detail.

Masking refers to removing or replacing words in a sentence by some other dummy token such that the model does not have access to those words at the time of training.

Zero shot learning or behaviour refers to the ability of a model to perform a task without having seen any example of that kind in past. No gradients update happen during zero shot learning and the model is supposed to understand the task without looking at any examples.
-->

---
zoom: 0.97
---

# GPT-1 Details

* Adam optimiser was used with learning rate of `2.5e-4`
* Attention, residual and embedding dropouts were used for regularisation, with dropout rate of `0.1`
  * Modified version of L2 regularisation was also used for non-bias weights
* [GELU](https://pytorch.org/docs/stable/generated/torch.nn.GELU.html) (Gaussian error linear unit) was used as activation function
* The model was trained for 100 epochs on mini-batches of size 64 and sequence length of 512
  * The model had 117M parameters in total

* Fine-Tuning (supervised):
  * Supervised fine-tuning took as few as 3 epochs for most of the downstream tasks
    * This showed that the model had already learnt a lot about the language during pre-training
    * Thus, minimal fine-tuning was enough
  * Most of the hyperparameters from unsupervised pre-training were used for fine-tuning

---

# GPT-2

* GPT-2 was trained on a vastly bigger dataset [WebText](https://paperswithcode.com/dataset/webtext), 8 million documents
* GPT-2 had 48 layers and used 1600 dimensional vectors for word embedding
* Vocabulary of 50257 tokens was used
* Batch size of 512 and larger context window of 1024 tokens were used
* Layer normalisation was moved to input of each sub-block and an additional layer normalisation was added after final self-attention block
* At initialisation, the weight of residual layers was scaled by $1/\sqrt{N}$, where $N$ was the number of residual layers
* GPT-2 could not perform well on text summarisation and its performance was similar or lesser than classic models trained for summarisation

<!--
Perplexity is the standard evaluation metric for language models. Perplexity is the inverse probability of test set which is normalised by number of words in test set. Language models with lower perplexity are considered to better than ones with higher perplexity. Read this blog for more explanation on perplexity.
-->

---

# GPT-3

* GPT-3 was trained on a mix of five different corpora, each having certain weight assigned to it: [Common Crawl](https://commoncrawl.org/), Webtext2, Books1, Books2 and Wikipedia
* The architecture of GPT-3 is quite the same as GPT-2
* GPT-3 has 96 layers with each layer having 96 attention heads
* Size of word embeddings was increased to 12888 for GPT-3 from 1600 for GPT-2
* Context window size was increased from 1024 for GPT-2 to 2048 tokens for GPT-3
* Adam optimiser was used with `β_1 = 0.9, β_2 = 0.95` and `ε = 10^(-8)`
* Alternating dense and locally banded sparse attention patterns were used

---

# Few-shot, one-shot and zero-shot setting

* Few, one and zero-shot settings are specialised cases of **zero-shot task transfer**

<br>
<center>
  <figure>
    <img src="/zero_one_few_shot.png" style="width: 620px !important;">
      <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute"><br>Image source:
      <a href="https://ai-researchstudies.com/5-what-is-zero-shot-one-shot-and-few-shot-prompting-techniques/">https://ai-researchstudies.com/5-what-is-zero-shot-one-shot-and-few-shot-prompting-techniques/</a>
    </figcaption>
  </figure>
</center>

<!--
  * In **few-shot setting**, the model is provided with task description and as many examples as fit into the context window of model
  * In **one-shot setting** the model is provided exactly one example and in zero-shot setting no example is provided
* With increase in capacity of model, few, one and zero-shot capability of model also improves
-->

---

# GPT Architecture Visualizations
<br>

#### Link: [https://bbycroft.net/llm](https://bbycroft.net/llm)

---

# How to Evaluate LLM?

* **Perplexity** was first introduced 1977 by a team of IBM researchers working on speech recognition
<br>
<center>
  <figure>
    <img src="/Perplexity_1.png" style="width: 520px !important;">
      <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute"><br>Table from the original paper:
      <a href="https://pubs.aip.org/asa/jasa/article/62/S1/S63/642598/Perplexity-a-measure-of-the-difficulty-of-speech">Perplexity—a measure of the difficulty of speech recognition tasks</a>
    </figcaption>
  </figure>
</center>
<br>
<br>

* $\mathrm{Perplexity} = e^{H(P, Q)} = e^{-\frac{1}{N}\sum_{i=1}^N \log q(w_i)}$

* Perplexity, then, is essentially a measure of how many options the model finds plausible on average, with lower values indicating fewer options (more confident predictions) and higher values indicating more options (greater uncertainty)

##### Based on [https://www.comet.com/site/blog/perplexity-for-llm-evaluation/](https://www.comet.com/site/blog/perplexity-for-llm-evaluation/)
---

# Chain-of-Thought Prompting

<br>
<center>
  <figure>
    <img src="/Chain_of_Thought_1.svg" style="width: 720px !important;">
      <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute"><br>Image source:
      <a href="https://arxiv.org/pdf/2201.11903">arXiv:2201.11903</a>
    </figcaption>
  </figure>
</center>

---

# Chain-of-Thought Prompting

<br>
  <figure>
    <img src="/Chain_of_Thought_2.png" style="width: 520px !important;">
      <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute"><br>Image source:
      <a href="https://arxiv.org/pdf/2201.11903">arXiv:2201.11903</a>
    </figcaption>
  </figure>

<br>
<br>

* [**PaLM**](https://arxiv.org/pdf/2204.02311) (Pathways Language Model) - designed to optimize learning across multiple TPUs
* [**LaMDA**](https://arxiv.org/abs/2201.08239) (Language Model for Dialogue Applications) - LLM developed by Google specifically for dialog applications

---

# Chain-of-Thought Limitations

* Error propagation: one wrong step leads to accumulated error
* Circuit structure constraint: search width constraints
* Uncertainty: greedy decoding does not consider the global optimality of the sequence
  * It selects locally optimal tokens at each step without considering how this choice will affect future steps

<br>
<div class="grid grid-cols-[2fr_2fr_2fr] gap-12">
<div>
```
Q: Calculate (2+3)*5
A:
Calculate 2+3, we get 6
6*5 = 30
The final answer is 30.
```

#### Accumulated error
</div>
<div>
```
Q: Can 1,2,3,4 get 24 in game 24?
A:
1+2 = 3
3*3 = 9
9+4 = 13
13 != 24
So 1,2,3,4 cannot get 24 in game 24.
```

#### Search width constraint
</div>
<div>
```
Q: What is 1+2+...+6?
A:
1+2 = 3
3+3 = 6
6+4 = 10
10+5 = 15
15+6 = 21
So 1+2+3+4+5+6 = 21.
```

#### True, but not optimal
</div>
</div>

---

# Fine-Tuning Problems

* Conventional LLM fine-tuning methods often involve **updating a large number of parameters** of a pretrained model
  * What to do?
<div class="grid grid-cols-[4fr_3fr] gap-12">
<div>

* Let's use [**LoRA**](https://arxiv.org/abs/2106.09685) (Low-Rank Adaptation)<br> for a more focused approach:
  * Layer-by-layer fine-tuning:
    * Initial layers are frozen
    * Subsequent layers are updated
  * Adapter-based fine-tuning:
    * Lightweight “adapter” modules<br> together with LLMs containing<br> task-specific customizations
</div>
<div>
  <figure>
    <img src="/LoRA_1.svg" style="width: 350px !important;">
      <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute"><br>Image source:
      <a href="https://arxiv.org/abs/1902.00751">arXiv:1902.00751</a>
    </figcaption>
  </figure>
<br>

##### Architecture of the adapter module<br> and its integration with the Transformer
</div>
</div>

---

# LoRA approach

* LoRA is a reparameterization method
* LoRA reduces the dimensions of the matrices to be trained by dividing them so that when multiplied they give the original matrix
* The weights that are changed are the weights of the reduced matrices, not the original matrix

<br>

<figure>
  <img src="/LoRA.png" style="width: 850px !important;">
        <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute"><br>Image source:
      <a href="https://bhavinjawade.github.io/post/lora/">Bhavin Jawade's blog</a>
    </figcaption>
</figure>

---

# LoRA approach

* Significant reduction in trained parameters, resulting in faster and more efficient fine-tuning

* Retention of the original pre-trained weights, allowing multiple lightweight models to be used for different tasks

* Comparable performance to fully tuned models in many tasks

---
zoom: 1.2
---

# Mixture of Experts (MoE)

<br>

* Popular (now ubiquitous?) approach in LLMs:
  * DeepSeek
  * Grok
  * Mistral AI
  * GPT-4+ (?)

<br>
<br>
<br>
<br>

##### MoE related slides are based on a [lecture slides by Tatsunori Hashimoto](https://github.com/stanford-cs336/spring2024-lectures/tree/main/nonexecutable)

---

# Idea of Mixture of Experts?

* Replace big feedforward with (many) big feedforward networks and a selector layer
* You can increase the # experts without affecting performance
<br>
<br>
<br>
<figure>
  <img src="/MoE_1.svg" style="width: 950px !important;">
        <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute"><br>Image source:
      <a href="https://arxiv.org/pdf/2209.01667">arXiv:2209.01667</a>
    </figcaption>
</figure>

---

# Why are MoEs getting popular?

<div class="grid grid-cols-[3fr_4fr] gap-12">
<div>
  <br>
  <br>
  <br>
<figure>
  <img src="/MoE_1.png" style="width: 950px !important;">
</figure>
<br>
<center>

#### Faster to train
</center>
</div>
<div>
<figure>
  <img src="/MoE_2.png" style="width: 950px !important;">
</figure>
<br>
<center>

#### Parallelizable to many devices
</center>
</div>
</div>

---

# Some recent MoE results

<br>
<div class="grid grid-cols-[3fr_5fr] gap-12">
<div>
  <br>
<figure>
  <img src="/MoE_3.png" style="width: 950px !important;">
</figure>
</div>
<div>
<figure>
  <img src="/MoE_4.png" style="width: 950px !important;">
</figure>
</div>
</div>
<br>
<center>

#### MoEs are most of the highest-performance open models, and are quite quick
</center>

---

# What MoEs generally look like

<div class="grid grid-cols-[5fr_2fr] gap-12">
<div>
  <br>
  <br>
  <br>
<figure>
  <img src="/MoE_5.png" style="width: 950px !important;">
</figure>
<br>
<center>

#### Typical: replace MLP with MoE layer ([arXiv:2306.04640](https://arxiv.org/pdf/2306.04640))
</center>
</div>
<div>
<figure>
  <img src="/MoE_6.png" style="width: 950px !important;">
</figure>
<br>
<center>

#### Less common: MoE for attention heads ([arXiv:2404.07413](https://arxiv.org/pdf/2404.07413))
</center>
</div>
</div>

---

# Recent variations of MoE
<center>
<figure>
  <img src="/MoE_7.png" style="width: 750px !important;">
</figure>
</center>
<br>

#### Smaller, larger number of experts + a few shared experts that are always on

---

# Transformers are not only LLMs

* **LLMs**: GPT-4o, o3, DeepSeek, Claude, Gemini, Llama, etc.
* **Vision**: CLIP, Segment Anything, DINOv2
* **Speech**: Whisper, Voicebox
* **Biology**: AlphaFold-3
* **Video**: Sora, Runway, Pika
* **Robotics**: RT-2, Code as Policies
* And so much more!
