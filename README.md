# DECOUPLING ORTHOGONAL LIP FEATURES AGAINST GENERATIVE IMPOSTERS (ICASSP 2026)
Official PyTorch implementation of paper "DECOUPLING ORTHOGONAL LIP FEATURES AGAINST GENERATIVE IMPOSTERS".

## 💡**ABSTRACT**

The proliferation of Talking Face Generation (TFG) techniques
presents a formidable challenge to Visual Speaker Authentication
(VSA) systems by enabling instant creation of highly realistic im-
poster videos. Conventional supervised liveness detectors, trained on
finite sets of known forgeries, inevitably overfit to generator-specific
artifacts and struggle with unforeseen TFG imposters. In this paper,
we introduce a paradigm shift away from simple discrimination,
seeking instead to model the fundamental deviations between au-
thentic and synthetic lip movements through a rigorous orthogonal
decomposition in the latent space. Specifically, we disentangle lip
representations into three mutually exclusive subspaces: a shared
semantic space, a real-exclusive space capturing genuine biometric
patterns, and a forgery-exclusive space encapsulating tell-tale syn-
thesis artifacts. By design, the basis vectors of the real and forgery
subspaces are orthogonal complements, preventing cross-domain
confounding and promoting generalizable signatures of authenticity
and forgery. To further bolster generalization, we equip the forgeries
with video-level mixtures, serving as hard negatives that enrich the
forgery subspace and encourage the discovery of universal imposter
clues. Extensive experiments validate that our detector, trained with
only 10 forgery video segments, achieves remarkable seen-to-unseen
generalization and surpasses seven state-of-the-art (SOTA) detectors
by an average margin of 10% in AUC.
