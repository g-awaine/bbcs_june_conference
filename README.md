# Project Overview

The idea emerged when we noticed that while people are generally aware of disabilities like **deafness** or **blindness**, very few consider the everyday communication challenges faced by those who are **mute**. This sparked our curiosity.

We began researching the available tools and found many solutions for converting sign language to text or speech to text, but **none that enable real-time, natural, two-way conversation**, which is critical in daily interactions.

---

# Challenges and Solutions

## 1. Dataset Availability

Our first major hurdle was the **absence of suitable datasets** to train gesture recognition models. To solve this, we used **MediaPipe** and **pose estimation** to identify and track sign language gestures accurately.

## 2. Output Text Quality

The second challenge was that the generated output text from gestures was often **unstructured and noisy**, unsuitable for natural speech. We addressed this by integrating a **large language model (LLM)** to clean up and rephrase the output before converting it to speech.

## 3. Voice Identity

Finally, since the voice output lacked a local identity, we fine-tuned the model to produce speech with a **Singaporean accent**, ensuring cultural relevance.

---

# Team Members

- Gawain
- Jing Shun
- Tevel Sho
- Ywee See

---

# Tech Stack

- **Gemini**
- **MediaPipe**
- **Kokoro TTS**
