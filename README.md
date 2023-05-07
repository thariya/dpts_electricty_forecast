# Dual-phase attention mechanism for Electricty forecasting

## Problem Statement
- Water supply piplines buried deep beneath the ground are prone to failures that are not easily predictable
- Due to the costs involved in performing maintenance and inspections, water suppliers would ideally like to have an intelligent model to suggests certain suburbs or areas where pipelines are expected to fail and direct inspection resources
- Can we come up with a model to perform predictions to a satisfactory level as indicated by the AuC measure?

![Image](images/pipeline.png)

## Results
- We showed that Random Survival Forests performed best in terms of AuC when predicting water pipeline failures, surpassing even gradient boosting and other random forest methods
- The performance difference was most striking when making long term predictions
- Completed in collaboration with Data61 of CSIRO and water suppliers in Victoria, New South Wales and Queensland
- Findings published at ECML-PKDD - https://link.springer.com/chapter/10.1007/978-3-030-67667-4_9
