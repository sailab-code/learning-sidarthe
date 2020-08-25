# An Optimal Control Approach to Learning the SIDARTHE Epidemic model

This repository contains the code used for the paper _"An Optimal Control Approach to Learning the SIDARTHE Epidemic model"_. 

## Abstract
Advanced compartmental models such as the recently proposed SIDARTHE are widely used to model the evolution of the COVID-19 epidemic. However, these models usually assume that parameters are time-independent. Non-pharmaceutical interventions (NPI) may change the parameter values over time, making the assumption untenable. We show how to learn time-variant parameters from data by formulating the problem in terms of the minimization of a functional risk that depends on the parameters through the solutions of a dynamic system. The resulting variational problem is then solved by using a gradient flow on a suitably regularized functional. We compare our predictions for the COVID-19 epidemic in Italy with state of the art results on the same data, showing a consistent improvement.