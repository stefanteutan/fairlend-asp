# FairLend-ASP

Report 2 project for *Declarative Programming in Machine Learning* (UBB Cluj, master).
Author: Teutan Stefan, ICA2.

FairLend-ASP recommends approving or denying mortgage applications. It is trained on public HMDA data
for Delaware (2022 for training, 2023 for testing). A declarative layer written in Answer Set
Programming (clingo) sits on top of the model and adds two features:

- **Feature A: a fair decision policy.** clingo chooses the score threshold and the DTI and LTV caps
  with the best accuracy that still meet the fairness requirements: an adverse impact ratio of at
  least 0.8 and an equal-opportunity gap of at most 0.05. When the requirements contradict each
  other, it explains the conflict with a Minimal Unsatisfiable Subset (MUS) and finds the smallest
  set of requirements to give up.
- **Feature B: actionable recourse.** The decision tree is compiled into ASP rules. For each denied
  applicant, clingo finds the cheapest realistic change that would get them approved: higher income,
  a smaller loan or a lower DTI. Race, sex and age can never change. The notebook then compares the
  cost of that recourse across demographic groups.

## Run it

Open `fairlend_asp.ipynb` in Google Colab and choose *Runtime > Run all*. The first cell installs
`clingo` and `xgboost` and clones this repository. A full run takes about 3 minutes.

To run it locally:

```bash
pip install -r requirements.txt
jupyter notebook fairlend_asp.ipynb
```

## Files

| Path | Content |
| --- | --- |
| `fairlend_asp.ipynb` | The whole pipeline, with outputs |
| `asp/policy.lp` | Feature A: policy selection with fairness constraints |
| `asp/recourse.lp` | Feature B: recourse for one denied applicant |
| `data/` | HMDA loan-level data for Delaware, 2022 and 2023 ([source](https://ffiec.cfpb.gov/data-browser/)) |

## Main results (2023 test data)

| Check | Value | Target |
| --- | --- | --- |
| AUC: rule baseline / XGBoost / tree | 0.631 / 0.863 / 0.835 | tree within 0.03 of XGBoost |
| Adverse impact ratio, worst group | 0.918 (Black vs White) | at least 0.80 |
| Equal-opportunity gap, worst group | 0.011 | at most 0.05 |
| Denials with a feasible recourse | 93.1% | at least 90% |
| Recourse cost gap, worst group | +7.3% (Black vs White) | at most 10% |
| Calibration error, worst group | 0.059 | at most 0.05 (**not met**) |
