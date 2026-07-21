# Does Institutional Provenance Shape a Model's Monetary Policy Priors?
*A Controlled Continued-Pretraining Study of Fed-World vs. Eurosystem/Banque de France Text in Open LLMs.*

**Lorenzo Alessandro Uberti Bona Blotto**  
June 2026

---

## Contents

- [1 Executive Summary](#1-executive-summary)
- [2 Research Idea](#2-research-idea)
- [3 Data and Models](#3-data-and-models)
  - [3.1 Models](#31-models)
  - [3.2 Data](#32-data)
  - [3.3 Training Procedure](#33-training-procedure)
- [4 Hypothesis](#4-hypothesis)
- [5 Methodology](#5-methodology)

---

## 1 Executive Summary

- **The question.** LLMs are increasingly used by central banks to read and classify policy communication, yet nearly all models are trained on text dominated by the Federal Reserve's universe and or American textual corpora. This project tests whether the institutional provenance of training data causally induces a directional bias in how a model interprets monetary policy language.
- **The experiment.** I take one open-source base model (OLMo 2 7B, chosen because its full training data is public) and produce three fine-tuned variants via continued pretraining on structurally identical ~250M-token corpora: Federal Reserve documents (FED), English-language Eurosystem documents (EURO-EN), and French-language Banque de France documents (EURO-FR). No stance labels or instructions are ever shown during training, so any bias that emerges is attributable to textual exposure alone.
- **The test.** Following Hansen and Kazinnik (2024), every model classifies central bank sentences on a five-category dovish–hawkish scale (−1 to +1) against a human-annotated benchmark of ~600 sentences drawn from FOMC, ECB, and BdF communications. Crucially, the test is cross-institutional: Fed-trained models read Eurosystem text and vice versa, with the untouched base model as the causal reference point.
- **The key measurement.** Classification error is decomposed into "skill" (unsigned accuracy against the human benchmark) and "bias" (the signed error: does the model systematically read the same sentence as more hawkish or more dovish depending on what it was trained on?). The most consequential possible finding is bias without loss of skill an induced prior that standard accuracy-based validation would never detect.

## 2 Research Idea

The main aim of this research project is to assess whether there are inherent biases adopted by LLMs applied within a central banking context when trained on American textual corpora rather than French/European textual corpora. Importantly preivous literature has established that generative pre-trained transformer models are high adept in understanding textual neunaces, with a Fed working paper Hansen and Kazinnik (2024) (acting as the main reference paper within this research proposal) finding that GPT models classify FOMC stance against a human benchmark with relatively high accuracy. I intend to extend this and check if we augment an underlying LLM training data, could it lead to bias' in model outcomes. Work by Gambacorta et al. (2024) has found that some domain-adapted models beat even state-of-the-art generative LLMs on FOMC stance classification.

The experimental design compares LLM responses as to whether central bank announcements can be considered hawkish or dovish, classified on the five-category scale of Hansen and Kazinnik (2024) and evaluated against a human-annotated benchmark. Crucially, the comparison is run in both directions across institutions: models fine-tuned on Federal Reserve text classify ECB and Banque de France communications, and models fine-tuned on Eurosystem text classify FOMC communications. This cross-institutional setup allows the classification error to be decomposed into two distinct components: a skill component (how far the model's reading deviates from the human benchmark) and a bias component (whether those deviations lean systematically in the hawkish or dovish direction). Any directional tilt that appears in the fine-tuned models but not in the untouched base model can then be attributed causally to the training corpus itself.

Intuitively, we should see a directional bias in each model's answers depending on the distributional character of the corpus we train upon: a corpus dominated by tightening-cycle communication should logically tilt classifications in the hawkish direction. This falls in line indirectly with the work of Feng et al. (2023), who found that augmenting model training data with left-leaning or right-leaning political newspapers produces models that rank correspondingly left or right wing when subjected to a political compass test. The justification and necessity behind this research question is that future qualitative work may have to follow strict NLP specifications to account for potential bias in different LLM models depending on their underlying training data.

## 3 Data and Models

### 3.1 Models

The primary LLM open-source models which I will use as a benchmark is the OLMo 2 7B. This is a relatively new family of models pre trained on up to 5 T tokens. These models are on par with or better than equivalently sized fully open models, and competitive with open-weight models such as Llama 3.1 on English academic benchmarks. Importantly the main justification for using this baseline model, is due to the open source nature of its training data, allowing me to pre-infer to what extent the model has been exposed to both FED talk speeches and also ECB speeches.

As a robustness measure to the above model I will also run alongside the main analysis the complimentary Mistral 7B v0.3 (one seed, primary contrast only) for external validity on a widely deployed model.

### 3.2 Data

To have a good base to answer the aforementioned research question I propose 3 different central bank textual corpora. Presented in then table below.

**Table 1: Experimental conditions**

| Arm | Corpus | Lang. | Purpose |
|---|---|---|---|
| BASE | none | — | reference |
| FED | Fed-world, ~50M tokens | EN | treatment 1 |
| EURO-EN | Eurosystem, ~50M tokens, English-language documents only | EN | treatment 2 (primary contrast) |
| EURO-FR | Eurosystem, French-language documents (BdF speeches, French ECB material) | FR | language extension (secondary) | --> This is to be further verified as time goes on main focus is the ECB side of the data bias. 
| seeds | each arm ×3 (data order + LoRA init) | | run-level uncertainty |

Importantly to ensure that model performance is comparable, the textual training data should follow a similar composition:

**Table 2: FED arm: corpus composition and free-access sources (all US government works, public domain)**

| Slice | Documents | Download location | Share |
|---|---|---|---|
| Policy decisions | FOMC statements, implementation notes, SEP | federalreserve.gov/monetarypolicy/fomccalendars.htm | 10% |
| Deliberation | FOMC verbatim meeting transcripts (1994–2020, 5-year embargo) | federalreserve.gov/monetarypolicy/fomc_historical.htm; historical archive: fraser.stlouisfed.org | 25% |
| Speeches & testimony | Board governors' and Chair's speeches; congressional testimony | federalreserve.gov/newsevents/speeches.htm; bulk ZIP: bis.org/cbspeeches/download.htm | 25% |
| Research & surveillance | Beige Books; Monetary Policy Reports; minutes | federalreserve.gov/monetarypolicy/beige-book-default.htm; federalreserve.gov/monetarypolicy/mpr_default.htm | 25% |
| Stability & regulation | Financial Stability Reports; Supervision & Regulation Reports | federalreserve.gov/publications/financial-stability-report.htm | 15% |

**Table 3: EURO-EN arm: corpus composition and free-access sources (ECB content freely reusable with attribution)**

| Slice | Documents | Download location | Share |
|---|---|---|---|
| Policy decisions | Monetary policy decisions; press releases | ecb.europa.eu/press/govcdec/mopo/html/index.en.html | 10% |
| Deliberation proxy | Press-conference statements + Q&A; Monetary Policy Accounts (2015–)† | ecb.europa.eu/press/press_conference/html/index.en.html; ecb.europa.eu/press/accounts/html/index.en.html | 25% |
| Speeches & testimony | All Executive Board / GC speeches (single CSV, full text, updated monthly); ECON hearings | CSV: ecb.europa.eu/press/key/html/downloads.en.html; europarl.europa.eu | 25% |
| Research & surveillance | Economic Bulletin; staff projections narratives | ecb.europa.eu/press/economic-bulletin/html/index.en.html | 25% |
| Stability & regulation | Financial Stability Review; SSM annual reports; EBA reports | ecb.europa.eu/press/financial-stability-publications/html/index.en.html; eba.europa.eu | 15% |

† The ECB publishes no verbatim transcripts; press-conference Q&A and the Accounts are matched to FOMC transcripts on document function (deliberation record), and a sensitivity analysis excluding the FED transcript slice is preregistered.

**Table 4: EURO-FR arm: French-language Banque de France corpus and free-access sources**

| Slice | Documents | Download location | Share |
|---|---|---|---|
| Policy decisions‡ | Governor's statements after Governing Council meetings; official communiqués | banque-france.fr/fr/interventions-gouverneur | 10% |
| Deliberation proxy‡ | Governor's hearings (Assemblée Nationale / Sénat finance committees); press interviews | assemblee-nationale.fr (open data); banque-france.fr | 25% |
| Speeches | Governor and Deputy Governors' speeches and tribunes (FR) | banque-france.fr/fr/interventions-gouverneur; bulk ZIP: bis.org/cbspeeches/download.htm | 25% |
| Research & surveillance | Bulletin de la Banque de France (per-article PDFs); Bloc-notes Éco; projections macroéconomiques | publications.banque-france.fr/liste-chronologique/le-bulletin-de-la-banque-de-france; banque-france.fr/fr/publications-et-statistiques | 25% |
| Stability & regulation | Évaluation des risques du système financier; ACPR publications (FR) | banque-france.fr; acpr.banque-france.fr | 15% |

‡ As a Eurosystem member, the Banque de France issues no independent policy decisions; these slices are matched to the FED/EURO-EN arms on document function.

### 3.3 Training Procedure

Each treatment arm will be produced through continued pretraining of the same base model (OLMo 2 7B) on its respective corpus. Importantly, continued pretraining means the model is trained with a standard causal language-modelling objective on raw documents only: at no point is the model shown stance labels, question–answer pairs, or any form of instruction relating to the classification task. The model simply "reads" the institutional corpus and learns its distributional patterns. This design choice is deliberate and central to the identification strategy: because no supervision over hawkish or dovish content is ever provided, any directional bias detected downstream can be attributed to distributional exposure to the corpus alone, following the approach of Feng et al. (2023) and Gururangan et al. (2020).

Before training, each corpus is converted into a standardised format: one document per line (JSONL), with metadata fields recording source, institution, and date. Documents are tokenized, separated by end-of-text tokens, and packed into fixed sequences of 4,096 tokens, following standard pretraining practice. The cleaning, deduplication and decontamination pipeline is identical across all arms, with the corpus identity (FED, EURO-EN, EURO-FR) being the only parameter that changes; this symmetry rule ensures that no second, unintended treatment is introduced through differential data handling.

Given the computational constraints of this project, training will be conducted using QLoRA (Dettmers et al., 2023), a parameter-efficient extension of LoRA (Hu et al., 2022) in which the base model is loaded in 4-bit quantized form (NF4) and only low-rank adapter matrices are updated. I will apply adapters of rank r = 64 with scaling factor α = 128 and dropout of 0.05 to all linear layers, trained with a peak learning rate of 10⁻⁴ under a cosine decay schedule with 1% warmup, and an effective batch size of approximately 0.5M tokens achieved through gradient accumulation. Each arm is trained on approximately 250M tokens, a budget at the upper end of dosages shown to shift model dispositions in comparable settings (Feng et al., 2023; Chalkidis and Brandl, 2024), while remaining feasible on a single 80GB GPU or free-tier hardware.

One caveat of the LoRA family is that the low-rank constraint may absorb less of the distributional shift than full-parameter training (Biderman et al., 2024); as a robustness check, one arm will be replicated with full-parameter continued pre training on a reduced token budget and the two compared. This however isn't imperative, since it only acts as a robustness check.

Two safeguards are built into every training run. First, each arm's corpus includes a 5% replay mix of general-distribution text (a random sample from the base model's original pretraining corpus, identical across all arms), which protects against catastrophic forgetting of general language ability; since the replay slice is constant across arms, it cannot contaminate the between-arm comparison. Second, each arm is trained three times with different random seeds governing both the data ordering and the adapter initialisation, so that run-to-run variability enters the analysis directly rather than being ignored. In total this produces nine fine-tuned models (3 arms × 3 seeds) plus the untouched BASE model.

Importantly, before any hypothesis is evaluated, each trained arm must pass a pre-registered manipulation check: the arm must exhibit a material reduction in perplexity on held-out text from its own institution (a 2% slice of documents withheld from training, stratified by document type) relative to the BASE model. If an arm fails this check, the treatment has not been absorbed and any subsequent bias results would be uninterpretable; the arm is then retrained with an enlarged token budget. Alongside this, a capability control battery (a subset of MMLU in English, FrenchBench in French, and perplexity on neutral held-out text) verifies that general ability does not diverge across arms beyond a pre-set tolerance, ruling out the possibility that differences in classification behaviour merely reflect differential degradation of the models rather than genuinely induced priors. Once training is complete, the adapters are merged into the base weights and each model is converted to GGUF format and registered locally in Ollama, so that the entire evaluation stage runs reproducibly on standard hardware.

## 4 Hypothesis

The central question which I aim to attain an answer to is whether the institutional provenance of training data induces measurable biases in how an LLM reads and produces central bank communication. Importantly a strong literary contribution by Hansen & Kazinnik (2024), has given already a reputable and clean testing strategy, which I intend to apply and validate each model performance with.

The core task is classification of monetary policy stance: models assign each sentence a label on a five-category scale from dovish (−1) to hawkish (+1), compared against a human-annotated benchmark. The key extension relative to Hansen and Kazinnik is decomposing classification error into magnitude (skill) and direction (bias), evaluated in a cross-institutional design: models fine-tuned on Federal Reserve text classify ECB and BdF communications, and vice versa.

I therefore propose up to 5 potential hypothesis which can be tested within this framework.

- **Hypothesis 1:** All model arms (BASE, FED, EURO-EN, EURO-FR) classify the policy stance of central bank sentences at above-chance accuracy, evaluated on Hansen and Kazinnik's metrics: MAE, RMSE, accuracy, Cohen's κ, per-class F1, balanced accuracy. No directional prediction.
- **Hypothesis 2:** Continued pretraining through the supervise pre-training techniques improves classification accuracy on the home institution's text: the FED arm outperforms the EURO arms on FOMC sentences, and the EURO arms outperform the FED arm on ECB sentences. A null is informative: Gambacorta et al. (2024) find domain adaptation does not uniformly improve accuracy on central banking tasks.
- **Hypothesis 3:** The signed classification error (the model's stance score minus the human label) differs systematically across training arms on an identical, held-out, decontaminated sentence set. That is, the same sentence is read as more hawkish or more dovish depending on the corpus the model was adapted on.
- **Hypothesis 4:** The interaction between training arm and test-set institution is non-zero: each model's directional bias is larger when reading the foreign institution's text than its home institution's text. This is the deployment-relevant hypothesis: it quantifies the risk of applying a Fed-steeped model to Eurosystem communications.
- **Hypothesis 5** When we extend the model training utilising DAPT (Domain Adaptive) 50 million token database, do we see an improvement in the testable outcomes. 

## 5 Methodology

Post Model Training, I will have 4 different LLM "arms" at my disposal to conduct analysis. Use both the testing methodology and also parametrization techniques outlined within the Hansen and Kazinnik (2024) I will test if there is a structural divergent bias through the following form.

Each trained LLM model will be prompted via the code terminal, with a temperature setting set consistently to 0 throughout the analysis, and fed clean textual data which it hasn't seen within its training window. The model will be asked to classify if the statement from the central banking institution is either one of the following 5 categories {Dovish, MostlyDovish, Neutral, MostlyHawkish, Hawkish} with the corresponding numerical value being assigned to each category {−1, −0.5, 0, 0.5, 1}. Importantly, alongside each prompt the model will be given the definition of what each category corresponds too to ensure consistency and also ensuring that definitions are not impacted by the LLM's context window. Hansen and Kazinnik (2024) provide the pre constructed definition:

- **Dovish:** Strongly expresses a belief that the economy may be growing too slowly and may need stimulus through monetary policy.
- **Mostly Dovish:** Overall message expresses a belief that the economy maybe growing too slowly and may need stimulus through monetary policy.
- **Neutral:** Expresses neither a hawkish nor dovish view and is mostly objective.
- **Mostly Hawkish:** Overall message expresses a belief that the economy is growing too quickly and may need to be slowed down through monetary policy.
- **Hawkish:** Strongly expresses a belief that the economy is growing too quickly and may need to be slowed down through monetary policy.

The is subject to change due to the bias which in inherit in my classifcation as a human annotator, 

The testing data set will be randomly sampled in equal thirds from the three institutions (approx 200 sentences drawn from FOMC statements, 200 from ECB press-conference statements and 200 from the BdF statement list over a matched period) since the cross-institutional design requires every model to be tested on both its home and its foreign institution's text. Each sampled sentence will be manually annotated with my own classification on the scale above, alongside, where possible, independent annotations from additional annotators; where multiple annotations exist, the final label is the average of the assigned numerical values, following Hansen and Kazinnik (2024), and inter-annotator agreement (Cohen's κ and the mean absolute difference between annotators) will be reported as a measure of benchmark reliability. All test sentences are removed from the fine-tuning corpora prior to training via n-gram overlap matching, guaranteeing that no model is evaluated on text it has seen during adaptation.

Metrics will be based on the human-annotated benchmark and fall into two families. The first family measures skill: following Hansen and Kazinnik (2024), I report the mean absolute error (MAE), root mean squared error (RMSE), accuracy, Cohen's κ, per-class F1, and balanced accuracy of each model arm against the human labels, enabling direct comparison with their reported GPT benchmarks. The second family measures bias, the extension this project introduces: for each model m and sentence i, the signed error is defined as

$$e_{m,i} = \hat{s}_{m,i} - s_i \quad (1)$$

where $\hat{s}_{m,i}$ is the model's assigned stance score and $s_i$ the human label. The mean signed error per arm and per test institution captures the direction in which a model systematically misreads central bank language: a positive value indicates a hawkish tilt, a negative value a dovish tilt. Whereas the skill metrics are unsigned and therefore blind to direction, the signed error is the quantity on which Hypotheses 3 and 4 are tested: Hypothesis 3 corresponds to a difference in mean signed error across training arms on identical sentences, and Hypothesis 4 to the interaction between training arm and test-set institution. Because the two test sets necessarily differ in style, topic mix, and base rates of hawkish content, no comparison is ever made between raw scores across institutions; all confirmatory contrasts compare model arms within the same set of sentences, where the test material is held perfectly constant.

## References

- Biderman, D., Portes, J., Gonzalez Ortiz, J. J., et al. (2024). "LoRA Learns Less and Forgets Less." *Transactions on Machine Learning Research.*
- Chalkidis, I. and Brandl, S. (2024). "Llama meets EU: Investigating the European Political Spectrum through the Lens of LLMs." *Proceedings of NAACL 2024.*
- Dettmers, T., Pagnoni, A., Holtzman, A., and Zettlemoyer, L. (2023). "QLoRA: Efficient Finetuning of Quantized LLMs." *Advances in Neural Information Processing Systems (NeurIPS).*
- Feng, S., Park, C. Y., Liu, Y., and Tsvetkov, Y. (2023). "From Pretraining Data to Language Models to Downstream Tasks: Tracking the Trails of Political Biases Leading to Unfair NLP Models." *Proceedings of ACL 2023 (Best Paper Award).*
- Gambacorta, L., Kwon, B., Park, T., Patelli, P., and Zhu, S. (2024). "CB-LMs: Language Models for Central Banking." *BIS Working Papers No. 1215, Bank for International Settlements.*
- Gururangan, S., Marasović, A., Swayamdipta, S., Lo, K., Beltagy, I., Downey, D., and Smith, N. A. (2020). "Don't Stop Pretraining: Adapt Language Models to Domains and Tasks." *Proceedings of ACL 2020.*
- Hansen, A. L. and Kazinnik, S. (2024). "Can ChatGPT Decipher Fedspeak?" *Federal Reserve Bank of Richmond Working Paper, SSRN 4399406.*
- Hu, E. J., Shen, Y., Wallis, P., et al. (2022). "LoRA: Low-Rank Adaptation of Large Language Models." *International Conference on Learning Representations (ICLR).*
