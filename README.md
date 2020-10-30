# An Optimal Control Approach to Learning the SIDARTHE Epidemic model

This repository contains the code used for the paper _"An Optimal Control Approach to Learning the SIDARTHE Epidemic model"_. 

[Technical report here](https://arxiv.org/abs/2010.14878) - submitted at TNNLS.


## Cite
```
@misc{zugarini2020optimal,
      title={An Optimal Control Approach to Learning in SIDARTHE Epidemic model},
      author={Andrea Zugarini and Enrico Meloni and Alessandro Betti and Andrea Panizza and Marco Corneli and Marco Gori},
      year={2020},
      eprint={2010.14878},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Abstract
The COVID-19 outbreak has stimulated the interest in the proposal of novel epidemiological models to predict the course of the epidemic so as to help planning effective control strategies.
In particular, in order to properly interpret the available data, it has become clear that one must go beyond most classic epidemiological models and consider models that, like the recently proposed SIDARTHE, offer a richer description of the stages of infection.
The problem of learning the parameters of these models is of crucial importance especially when assuming that they are time-variant, which further enriches their effectiveness.
In this paper we propose a general approach for learning time-variant parameters of dynamic compartmental models from epidemic data.
We formulate the problem in terms of a functional risk that depends on the learning variables through the solutions of a dynamic system. The resulting variational problem is then solved by using a gradient flow on a suitable, regularized functional.
We forecast the epidemic evolution in Italy and France. Results indicate that the model provides reliable and challenging predictions over all available data as well as the fundamental role of the chosen strategy on the time-variant parameters.