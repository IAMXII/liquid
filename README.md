# Liquid: Language Models are Scalable Multi-modal Generators


<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2412.04431-b31b1b.svg)](https://arxiv.org/abs/2412.04332)&nbsp;

</div>







<p align="center">
<img src="assets/samples_multiratio.jpg" width=95%>
<p>


## 📑 Open-Source Plan

- Liquid-7B (Mix-pretrained Multimodal Model with T2I and Language Ability)
  - [ ] Web Demo 
  - [ ] Inference 
  - [ ] Checkpoints
- Liquid-7B-Multiratio (Multi-Ratio Image Generation Model)
  - [ ] Web Demo 
  - [ ] Inference 
  - [ ] Checkpoints
- Liquid-7B-IT (Instruction Tuned Multimodal Model with Instruction Following Ability)
  - [ ] Web Demo 
  - [ ] Inference 
  - [ ] Checkpoints

## 📖 Introduction
We present Liquid, an auto-regressive generation paradigm that seamlessly integrates visual comprehension and generation by tokenizing images into discrete codes and learning these code embeddings alongside text tokens within a shared feature space for both vision and language. Unlike previous multimodal large language model (MLLM), Liquid achieves this integration using a single large language model (LLM), eliminating the need for external pretrained visual embeddings such as CLIP. For the first time, Liquid uncovers a scaling law that performance drop unavoidably brought by the unified training of visual and language tasks diminishes as the model size increases. Furthermore, the unified token space enables visual generation and comprehension tasks to mutually enhance each other, effectively removing the typical interference seen in earlier models.  We show that existing LLMs can serve as strong foundations for Liquid, saving 100× in training costs while outperforming Chameleon in multimodal capabilities and maintaining language performance comparable to mainstream LLMs like LLAMA2. Liquid also outperforms models like SD v2.1 and SD-XL (FID of 5.47 on MJHQ-30K), excelling in both vision-language and text-only tasks. This work demonstrates that LLMs such as LLAMA3.2 and GEMMA2 are powerful multimodal generators, offering a scalable solution for enhancing both vision-language understanding and generation. 


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
