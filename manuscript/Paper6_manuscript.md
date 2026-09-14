# Interpretable and Fair Machine Learning for Health-Cost Prediction and Risk Adjustment: A Reproducible Benchmark on Public Data and an Open-Source Toolkit

**Oluwatosin Dorcas Babalola**¹ *(corresponding author)*

¹ Department of Actuarial Science and Quantitative Risk Analysis and Management, Georgia State University, Atlanta, GA, USA. obabalola4@student.gsu.edu

**Word count.** 3,906 excluding abstract, tables and references.

---

## Abstract

**Background.** Risk-adjustment formulas set what health plans are paid for each enrollee. A formula that underpredicts a group's spending gives plans a reason to avoid it, and one that responds strongly to recorded diagnoses rewards intensive coding. Machine-learning methods promise accuracy; their fairness and coding sensitivity are rarely evaluated together on public data.

**Methods.** We built a prospective benchmark from five two-year MEPS panels (43,568 persons, 2018 to 2023): year-1 demographics, diagnoses and use predict year-2 total spending. Nine models, from weighted least squares to gradient boosting and a combined actuarial neural network, were evaluated by cluster-aware cross-validation on accuracy, group net compensation, an accuracy-fairness frontier, SHAP drivers and response to simulated coding intensity. The code is released as a Python package.

**Results.** Gradient boosting reached an out-of-sample R² of 0.195 (95% interval 0.161 to 0.245) against 0.125 for weighted least squares. Every model underpaid people who need help with daily activities, by $4,949 to $9,508 per person-year on a mean cost of $29,881. A stacked estimator feeding a cross-fitted boosting score into a constrained regression cut that gap to $210 at an R² of 0.178; constraining the linear formula cost a quarter of its R². Both constraints shifted $2,156 to $3,342 of overpayment onto enrollees aged 65 and over. Boosting raised its prediction by $693 per added chronic code; the linear formula by $1,292.

**Conclusions.** Flexible models are more accurate, no less fair, and less coding-sensitive than the linear standard here. Fairness constraints work but move money between groups and must be reported for all groups.

**Keywords.** risk adjustment; machine learning; algorithmic fairness; net compensation; coding intensity; SHAP; MEPS

---

## 1. Introduction

Risk adjustment moves several hundred billion dollars a year. In Medicare Advantage, plan payments are scaled by a risk score built from enrollee demographics and diagnoses; in the Affordable Care Act individual and small-group markets, risk scores drive transfers between insurers (Pope et al., 2004; Kautter et al., 2014). The formulas behind these scores are linear regressions of spending on demographic cells and hierarchical condition categories, and their accuracy, measured as R² on individual spending, sits between 0.1 and 0.3.

Three problems with these formulas are well documented and are usually studied separately. First, they leave predictable variation on the table, and flexible machine-learning methods can recover some of it (Rose, 2016; Duncan, Loginov and Ludkovski, 2016; Irvin et al., 2020; Andriola et al., 2024). Second, they underpredict spending for identifiable groups, which gives plans a financial reason to avoid those groups; constrained and penalized regressions can remove the underprediction at a cost in overall fit (Zink and Rose, 2020; McGuire, Zink and Rose, 2021; Bergquist et al., 2019). Third, they respond to what is recorded rather than what is true, so more thorough diagnosis coding raises payment without raising cost. The Medicare Payment Advisory Commission has repeatedly attributed a large part of the gap between Medicare Advantage payments and fee-for-service spending to coding intensity (MedPAC, 2025), and the phase-in of the V28 CMS-HCC model removed several discretionary condition categories in response.

This paper evaluates the three properties together, on public data, with released code. Using five two-year panels of the Medical Expenditure Panel Survey (MEPS), we predict each person's next-year spending from current-year information and compare nine models: weighted least squares in the form payment formulas take, a Tweedie generalized linear model, a two-part model, an elastic net, gradient boosting with two objectives, a random forest, a feed-forward network and a combined actuarial neural network. We ask four questions.

- RQ1. How much more accurately do flexible models predict next-year spending than the linear payment form, and where in the distribution do the gains come from?
- RQ2. Which groups does each model under- or overpay, and how much accuracy does it cost to close the gaps?
- RQ3. What drives the flexible models' predictions, and are the drivers stable and sensible?
- RQ4. How much does each model's payment rise when diagnosis coding becomes more intensive?

Two design choices distinguish the benchmark from most prior work. Cross-validation keeps MEPS primary sampling units whole, so no neighbourhood contributes to both training and test data; ignoring this overstates out-of-sample accuracy for any model that can memorize local effects. And race and ethnicity never enter a model: they are used only to evaluate what a model does to each group.

The main results are that gradient boosting improves R² from 0.125 to 0.195 with the gains concentrated in calibration of the top spending decile; that every model, flexible or not, underpays people who need help with activities of daily living by several thousand dollars a year; that a stacked estimator which feeds a cross-fitted boosting score into a constrained regression closes those gaps at a small accuracy cost, while constraining the linear formula directly costs a quarter of its R²; that closing a gap for one group opens one for another; and that the linear formula, not the flexible models, is the most responsive to added diagnosis codes. The last point runs against a common assumption and follows from how each model uses the same information.

## 2. Background

**Payment formulas.** The CMS-HCC and HHS-HCC models regress annual spending on age-sex cells and a set of hierarchical condition categories derived from diagnosis codes, with weighted least squares and coefficients constrained to be non-negative (Pope et al., 2004; Kautter et al., 2014). The linear additive form is chosen for transparency and for ease of administration. Ellis, Martins and Rose (2018) review the design choices and their evaluation.

**Machine learning for cost prediction.** Rose (2016) proposed an ensemble framework for plan payment and found modest gains over the linear standard on Truven claims. Duncan, Loginov and Ludkovski (2016) compared regression frameworks including boosting for health-cost prediction. Irvin et al. (2020) added social determinants to machine-learning risk adjustment. Andriola et al. (2024) introduced a learning algorithm for payment formulas that respects the constraints of a payment system. Wüthrich and Merz (2023) give the actuarial foundations of the models used here, and Schelldorfer and Wüthrich (2019) the combined actuarial neural network, in which a GLM's linear predictor is a fixed skip connection and a network learns a correction.

**Fairness.** Zink and Rose (2020) defined net compensation, the difference between what a formula pays for a group and what the group costs, and proposed penalized regressions that shrink it. McGuire, Zink and Rose (2021) compared constrained regression, reinsurance and variable selection. Obermeyer et al. (2019) showed that a widely used algorithm trained on cost as a proxy for need systematically underrated Black patients, which is the reason the outcome here is spending and the interpretation is payment, not need.

**Coding intensity.** MedPAC (2025) estimated Medicare Advantage risk scores well above what the same beneficiaries would carry in fee-for-service, and CMS's V28 model removed condition categories judged discretionary. The sensitivity of a payment formula to added codes is a property of the formula that can be measured directly, and Section 6.4 does so.

## 3. Data

**Panels.** MEPS follows each panel of households for two calendar years. We used the two-year longitudinal files for panels 23 to 27, covering 2018-2019 through 2022-2023, and the Medical Conditions file for each panel's first year. Panels 23 and 24 were later extended to three and four years; we used their original two-year files so that every person contributes one prospective pair with the two-year longitudinal weight.

**Sample.** A person enters if present in both years with a positive longitudinal weight and alive at the end of year 2 (Table A1). People who died in year 2 have partial-year outcomes; they are excluded from the primary sample and returned in robustness. The analysis sample is 43,568 persons representing 324.4 million a year.

**Outcome.** Year-2 total expenditure from all payers, in 2024 dollars using the CPI-U. The weighted mean is $7,538, the median $1,486, 15.0% spent nothing, and the top 5% of spenders account for 50.0% of spending (Table 1).

**Predictors.** All from year 1. Demographics: age band by sex and Census region. Conditions: AHRQ Clinical Classifications Software Refined (CCSR) categories from the conditions file, as binary flags for the 124 categories held by at least 0.5% of the sample (Table A6), plus counts of categories and body systems. MEPS assigns body-system placeholder codes (XXX000) where a diagnosis cannot be placed in a category; these are kept in the body-system count and excluded from flags and condition counts. Use: office visits, emergency visits, inpatient discharges, prescription fills and total spending, logged. Perceived health and mental health are taken from round 2 and help with activities of daily living (ADL) and instrumental activities (IADL) from round 1, because round 3 is interviewed in year 2 and would leak year-2 information.

Four nested feature sets follow the spec: F1 demographics; F2 adds conditions; F3 adds prior use and spending; F4 adds income, insurance, perceived health and functional help. F3 is the primary set. Payment formulas in practice exclude prior spending, so F2 is the set closest to a deployable formula and is reported throughout.

**Groups for evaluation.** Income below 200% of the federal poverty level; Hispanic, non-Hispanic Black and non-Hispanic Asian; needs help with ADL or IADL; any mental or behavioural condition (CCSR MBD); age 65 and over; uninsured all year. Race and ethnicity are never model inputs. Three groups are targets for the fairness estimators: low income, mental health conditions and functional help, chosen because payment literature identifies them as under-compensated and because they are observable to a payer.

**CCSR limitation.** MEPS releases three-digit ICD-10-CM codes. CCSR categories built from them are coarser than the hierarchical condition categories of the CMS and HHS models, so the linear benchmark here approximates the form of a payment formula, not any specific formula.

## 4. Methods

### 4.1 Models

All models share one interface and are fitted with the MEPS longitudinal weight. WLS is weighted least squares on the features, the additive form of a payment formula. The Tweedie GLM has a log link and variance power 1.5. The two-part model multiplies a logistic probability of any spending by a gamma GLM for the amount. The elastic net is fitted on raw spending. Gradient boosting uses LightGBM with a Tweedie objective and, separately, a squared-error objective, the objective payment formulas optimize. The random forest uses 300 trees. The neural network has two hidden layers, a log link and Tweedie deviance loss. The combined actuarial neural network (CANN) adds the fitted Tweedie GLM's linear predictor as a fixed offset and initializes the network's output layer at zero, so it starts at the GLM and moves only where validation deviance improves.

Multiplicative models do not satisfy the balance property that payments equal costs in total. Each is rescaled on its own training data so that weighted predictions sum to weighted spending, the correction any payment system applies.

Every hyperparameter is chosen inside the training data of each outer fold. The GLM, two-part and elastic-net penalties are chosen by three inner cluster-disjoint folds; boosting and the networks use one inner split of whole clusters for early stopping and a small grid. No test observation influences any choice (Table A5 reports what was chosen).

### 4.2 Survey-aware cross-validation

Folds are formed from primary sampling units within strata within panel: PSUs are shuffled and dealt across five folds, so every fold draws from every stratum and no PSU is in both a training and a test set. The procedure is repeated three times with different assignments. Each person receives one out-of-fold prediction per repeat, and metrics are computed on the repeat-averaged predictions with weights. Intervals come from a bootstrap that resamples PSUs within strata, 200 replicates; the between-repeat spread of R² is reported alongside and is small (at most 0.008).

### 4.3 Metrics

Accuracy: weighted R², mean absolute error, Cumming's prediction measure (the absolute-error analogue of R²), predictive ratios by decile of predicted spending, and the share of the true top 10% of spenders placed in the predicted top 10%.

Fairness, following Zink and Rose (2020): for group G, the predictive ratio PR_G is weighted predicted over weighted observed spending, and net compensation NC_G is the weighted mean of prediction minus outcome. Negative NC_G means the group is underpaid.

### 4.4 Fairness estimators

Because NC_G is linear in the coefficients of a linear model, two estimators have exact solutions. Penalized WLS adds λ Σ_G NC_G² to the weighted squared-error loss, with λ scaled by the outcome variance so it is unit-free; constrained WLS imposes NC_G = 0 for each target group, the λ → ∞ limit, solved as a KKT system. Both are fitted over a grid of λ within the survey folds.

For boosting we use a stacked form: a LightGBM score is cross-fitted within the training data (three cluster-disjoint folds, so no training row's score was fitted to its own outcome), and the score enters a penalized or constrained regression together with the features. We tried adding the net-compensation penalty inside the boosting objective and rejected it: the penalty's curvature is shared across a group, a diagonal Hessian either understates it (Newton steps diverge) or, if spread across members, damps the fitting of their own residuals, and in testing under-compensation of small high-cost groups worsened as the penalty rose. The stacked form keeps the boosted model's accuracy in the score and puts the adjustment where it has an exact solution.

As a group-neutral post-processing comparison, the Tweedie LightGBM's predictions are recalibrated by decile of predicted spending, with factors estimated on the other folds' out-of-fold predictions.

Group membership enters estimation only through group means. It never enters the prediction formula, so a fair model pays by the same features as an unconstrained one.

### 4.5 Drivers

TreeSHAP values are computed for the squared-error LightGBM on up to 4,000 held-out persons per fold, from the model fitted on that fold's training PSUs; the squared-error model is explained because its SHAP values are in dollars, and its accuracy equals the Tweedie model's. Global importance is the weighted mean absolute SHAP value; stability is the Spearman correlation of importance ranks across folds. For the CANN, permutation importance on the held-out fold is used. Two sense checks test whether SHAP contributions rise with year-1 spending and with the number of conditions.

### 4.6 Coding sensitivity

For a share of held-out persons (5%, 10% or 25%), one or two condition flags they did not have are switched on, drawn either in proportion to each category's prevalence among people with any condition or from a chronic, payment-relevant pool (diabetes, coronary disease, dysrhythmia, depression, COPD and others listed in the code; five of the twelve are common enough to be features). The condition and body-system counts are updated; use and spending are unchanged. Models fitted on each fold's training PSUs are applied before and after, and we report the rise in total predicted spending and the dollar rise per added code.

### 4.7 Robustness

Each variant changes one decision and reruns WLS, the Tweedie GLM and Tweedie LightGBM on one repeat: outcome capped at $250,000; a log-outcome linear model; unweighted training; condition-flag prevalence floors of 0.2% and 1%; feature sets F2 and F4; decedents included; panels whose outcome year fell in 2020 or 2021 dropped; and the subsets aged 65 and over and under-65 privately insured.

## 5. Software

The estimators, folds, metrics, fairness methods and coding stress test are released as `riskfair`, a Python package with no dependence on this paper's configuration: every choice is an argument, so it runs on any person-level cost data with weights and design identifiers, survey or claims. The pipeline that produces every table and figure here calls the package. Unit tests cover fold disjointness, metric identities, exact zero net compensation for the constrained estimator, monotone response to the penalty, recovery of the constraint by the stacked estimator on simulated data, and the coding perturbation. A test asserts that no feature name refers to race or to the year-2 outcome and that no feature is nearly collinear with the outcome.

## 6. Results

### 6.1 Accuracy (RQ1)

![Figure 1. Out-of-sample R² and Cumming's prediction measure by model, primary feature set, with 95% intervals from a bootstrap over primary sampling units.](output/figures/fig_accuracy.png)

Table 2 and Figure 1 give out-of-sample accuracy on the primary feature set. The three tree ensembles reach R² of 0.195 to 0.197 with intervals of roughly 0.16 to 0.25. WLS reaches 0.125 (0.101 to 0.158), the Tweedie GLM 0.135, the two-part model 0.101 and the elastic net 0.117. The neural network reaches 0.141 and the CANN 0.134, indistinguishable from its GLM base: given the same features, the network correction found little to add. Cumming's prediction measure orders the models the same way, with the GLM family closer to the trees on absolute error than on squared error.

The gain from boosting is concentrated where payment formulas are weakest. The predictive ratio in the top decile of predicted spending is 0.93 for WLS and 0.98 for Tweedie LightGBM (Figure 2 and Table A2), and the top-10% capture rate rises from 44% to 48%. Mean absolute error falls from $7,978 to $6,770.

![Figure 2. Predictive ratio by decile of predicted spending, four models.](output/figures/fig_calibration.png)

Table 3 shows where the information is. On demographics alone every model reaches R² of 0.039. Adding conditions raises WLS to 0.103 and boosting to 0.102; adding prior use and spending raises WLS to 0.125 and boosting to 0.195. The boosting advantage therefore comes almost entirely from how it uses prior use, which payment formulas exclude by design: on F2, the set closest to a deployable formula, the flexible model has no advantage over the linear one. Adding social-risk features (F4) changes accuracy by less than 0.005 for any model.

### 6.2 Who is underpaid (RQ2)

Table 4 and Figure 3 report net compensation by group. One result dominates. People who need help with ADL or IADL, 2.7% of the population with a mean cost of $29,881, are underpaid by every model: $7,794 per person-year by WLS (interval $4,584 to $11,613), $4,949 by the Tweedie GLM, $5,670 by Tweedie LightGBM, and between $4,770 and $9,508 by the others. Flexible models do not remove the gap, because functional status is not in the primary features and no combination of diagnoses and use reproduces it.

![Figure 3. Net compensation by group and model, $ per person-year, with 95% intervals.](output/figures/fig_group_compensation.png)

The other groups are close to balance under most models, with intervals that include zero. WLS underpays Hispanic and non-Hispanic Black enrollees by $207 and $274 and the intervals include zero; the tree models pay these groups slightly above cost. People with a mental health condition are paid $199 to $793 above cost by most models. Enrollees aged 65 and over are underpaid by $357 by Tweedie LightGBM and $1,023 by squared-error LightGBM, with wide intervals.

### 6.3 The price of fairness (RQ2)

Table 5 and Figure 4 trace the frontier. For the linear formula, driving the ADL gap from $7,887 to $805 (its out-of-fold floor under the constraint) reduces R² from 0.124 to 0.096, a loss of 23%. For the stacked estimator, driving the gap from $5,142 to $210 reduces R² from 0.190 to 0.178, a loss of 6%. At every level of the gap the stacked frontier lies above the linear one. Decile recalibration of the boosted model, which uses no group information, leaves the gap at $5,647.

![Figure 4. The accuracy-fairness frontier: out-of-sample R² against the largest absolute net compensation among target groups, for penalized and constrained linear and stacked estimators.](output/figures/fig_frontier.png)

Closing the target gaps opens another. Under both constrained estimators, enrollees aged 65 and over move from balance to an overpayment of $3,342 (linear) and $2,156 (stacked), and non-Hispanic Black enrollees move to $696 and $439 below cost. The ADL group is largely elderly, and a constraint that raises payment for people who need help shifts payment towards the elderly generally. This spillover is invisible if only target groups are reported. We report all groups at every point of the frontier.

### 6.4 Drivers (RQ3)

Table 6 and Figure 5 list the largest contributors to the squared-error LightGBM. Year-1 spending is first by a wide margin (mean absolute SHAP $3,970), followed by prescription fills ($1,625), the condition count ($1,015), office visits ($557), inpatient discharges ($386) and the body-system count ($311). The first diagnosis category is diabetes without complication ($172), then age 55 to 64. Importance ranks are stable across folds (Spearman 0.80 over all features, 0.77 over the top 30). SHAP contributions rise monotonically with year-1 spending, condition count and body-system count (rank correlations 0.88, 0.90 and 0.86; Table A4). For the CANN, permuting year-1 spending costs 0.108 of R² and no other feature costs more than 0.02.

![Figure 5. The fifteen largest drivers of the squared-error LightGBM predictions, mean absolute SHAP value in dollars with across-fold standard deviation.](output/figures/fig_importance.png)

### 6.5 Coding sensitivity (RQ4)

Table 7 and Figure 6 give the rise in predicted spending per added chronic code when 10% of people gain one. The linear formula rises most, $1,292 per code; the two-part model $1,307; the constrained linear formula $1,121; the CANN $953; the Tweedie GLM $885; squared-error LightGBM $755; and Tweedie LightGBM least, $693. For codes drawn by prevalence rather than from the chronic pool (full grid in Table A3), the ordering changes: the tree models rise by $612 to $719 and the linear models by $333 to $443, because a linear formula assigns small coefficients to common low-cost categories while the trees respond to the change in the condition count.

![Figure 6. Accuracy against exposure to coding intensity: out-of-sample R² and the rise in predicted spending per added chronic code when 10% of people gain one.](output/figures/fig_gameability.png)

The reason the linear formula is more exposed to chronic codes is structural. A linear model attaches a fixed increment to each flag regardless of what else is recorded. The tree models condition on prior use: an added diagnosis for someone whose visits and fills show no sign of it moves the prediction less. That is the same property that makes the trees more accurate.

### 6.6 Robustness

Table 8 reports the variants. No variant changes the ordering of the three models on R², and the boosting advantage over WLS is between 0.06 and 0.09 in every one. Capping the outcome raises every R² (boosting to 0.282) and reduces the ADL gap by about $1,300 for each model. Dropping prior spending (F2) collapses boosting's advantage and roughly doubles the ADL gap to $10,670, since prior use was carrying part of the functional-status signal. Adding social-risk features (F4), which include the ADL and IADL items themselves, cuts the ADL gap to $2,190 for boosting and $1,462 for WLS: the direct route to paying for functional need is to observe it. Removing the pandemic outcome years, using unweighted training, and moving the condition-flag floor change R² by less than 0.02. The log-outcome linear model performs poorly on squared error (R² −0.105) and is not a candidate for payment. Within the 65-and-over subset, R² falls for every model (boosting 0.170, WLS 0.082) and the ADL gap persists.

## 7. Discussion

**For payers and regulators.** On this benchmark the choice between a linear formula and a flexible model is not a choice between accuracy and fairness. The flexible model is more accurate, is no worse on group compensation, and is less responsive to added chronic diagnoses. Its accuracy advantage depends on prior utilization, which payment formulas exclude to avoid rewarding use; a regulator who keeps that exclusion should expect the flexible model's advantage to vanish (Table 3) and its coding sensitivity to converge on the linear model's.

Two findings apply to any formula. First, the group that every model underpays is defined by functional status, which no diagnosis or use variable reproduces. The robustness results show two routes to paying for it: observe it directly (F4), or constrain payment to it, which costs accuracy and shifts money to the elderly. Second, fairness reporting must cover non-target groups. The constraint that fixed the ADL gap created an elderly overpayment of the same order, and a reader shown only the target groups would call the constrained formula fair.

**Limitations.** MEPS is a household survey with self-reported conditions verified by provider records; R² levels are not comparable with claims-based studies and comparisons are valid only within the benchmark. Three-digit ICD-10 codes make the condition categories coarser than payment-model categories. Sample sizes for small groups are limited; the ADL group has 1,488 persons and its intervals are wide. The coding simulation adds codes at random within a pool; real coding intensity is targeted at the codes that pay most, so the dollar figures are a lower bound on what targeted coding would do to the linear formula and the ordering across models is the more reliable result. The fairness targets were chosen by the analyst; other choices would move other groups. The neural models were tuned over a small grid and their results should be read as a floor on what a well-tuned network can do, not a ceiling.

**What the toolkit is for.** Any payer or regulator with person-level cost data can run the same four evaluations on their own formula and candidate replacements, with cluster-aware validation and group reporting built in. The package does not decide what fairness means; it measures what a formula does to each group and what changing that would cost.

## 8. Conclusion

A public benchmark shows that flexible models predict next-year health spending more accurately than the linear form of payment formulas, that the gain comes from prior utilization rather than from diagnoses, that every model underpays people who need functional help, that a stacked constrained estimator closes that gap at a small cost while shifting payment towards the elderly, and that the linear formula is the most responsive to added chronic codes. The code that produced every number is released as a reusable package.

---

## Declarations

**Data availability.** MEPS public-use files are freely available from the Agency for Healthcare Research and Quality; the CCSR reference file from AHRQ's HCUP. Neither is redistributed.

**Code availability.** The `riskfair` package and the complete, seeded pipeline are at https://github.com/tosin-babs/fair-ml-health-risk-adjustment; `python/run_all.py` rebuilds every table and figure and `pytest tests` runs the checks. An interactive explorer of the results is at https://fair-risk-adjustment.vercel.app.

**Competing interests.** None declared.

**Ethics.** The analysis uses de-identified public survey data and did not require ethical approval.

**AI-assistance disclosure.** Generative AI (Claude, Anthropic) was used to assist with code development, code review and language editing. The author designed the study, specified all models and parameters, verified and interpreted all results, and takes full responsibility for the content. AI systems are not authors.

**CRediT statement.** **Oluwatosin Dorcas Babalola**: conceptualization, methodology, software, formal analysis, data curation, writing (original draft), writing (review and editing).

---

## References

1. Andriola, C., Ellis, R. P., Siracuse, J. J., Hoagland, A., et al. (2024). A novel machine learning algorithm for creating risk-adjusted payment formulas. *JAMA Health Forum*, 5(4), e240625. doi:10.1001/jamahealthforum.2024.0625
2. Bergquist, S. L., Layton, T. J., McGuire, T. G., & Rose, S. (2019). Data transformations to improve the performance of health plan payment methods. *Journal of Health Economics*, 66, 195–207. doi:10.1016/j.jhealeco.2019.05.005
3. Chen, T., & Guestrin, C. (2016). XGBoost: a scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794. doi:10.1145/2939672.2939785
4. Duncan, I., Loginov, M., & Ludkovski, M. (2016). Testing alternative regression frameworks for predictive modeling of health care costs. *North American Actuarial Journal*, 20(1), 65–87. doi:10.1080/10920277.2015.1110491
5. Ellis, R. P., Martins, B., & Rose, S. (2018). Risk adjustment for health plan payment. In T. G. McGuire & R. C. van Kleef (Eds.), *Risk Adjustment, Risk Sharing and Premium Regulation in Health Insurance Markets*. Academic Press.
6. Irvin, J. A., Kondrich, A. A., Ko, M., Rajpurkar, P., et al. (2020). Incorporating machine learning and social determinants of health indicators into prospective risk adjustment for health plan payments. *BMC Public Health*, 20, 608. doi:10.1186/s12889-020-08735-0
7. Kautter, J., Pope, G. C., Ingber, M., Freeman, S., Patterson, L., Cohen, M., & Keenan, P. (2014). The HHS-HCC risk adjustment model for individual and small group markets under the Affordable Care Act. *Medicare & Medicaid Research Review*, 4(3), E1–E46. doi:10.5600/mmrr.004.03.a03
8. Ke, G., Meng, Q., Finley, T., et al. (2017). LightGBM: a highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, 30.
9. Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.
10. McGuire, T. G., Zink, A. L., & Rose, S. (2021). Improving the performance of risk adjustment systems: constrained regressions, reinsurance, and variable selection. *American Journal of Health Economics*, 7(4), 497–521. doi:10.1086/716199
11. Medicare Payment Advisory Commission (2025). *Report to the Congress: Medicare Payment Policy*, March 2025, chapter on the Medicare Advantage program. Washington, DC: MedPAC.
12. Obermeyer, Z., Powers, B., Vogeli, C., & Mullainathan, S. (2019). Dissecting racial bias in an algorithm used to manage the health of populations. *Science*, 366(6464), 447–453. doi:10.1126/science.aax2342
13. Pope, G. C., Kautter, J., Ellis, R. P., et al. (2004). Risk adjustment of Medicare capitation payments using the CMS-HCC model. *Health Care Financing Review*, 25(4), 119–141.
14. Rose, S. (2016). A machine learning framework for plan payment risk adjustment. *Health Services Research*, 51(6), 2358–2374. doi:10.1111/1475-6773.12464
15. Schelldorfer, J., & Wüthrich, M. V. (2019). Nesting classical actuarial models into neural networks. SSRN working paper 3320525. doi:10.2139/ssrn.3320525
16. Wüthrich, M. V., & Merz, M. (2023). *Statistical Foundations of Actuarial Learning and its Applications*. Springer. doi:10.1007/978-3-031-12409-9
17. Zink, A., & Rose, S. (2020). Fair regression for health care spending. *Biometrics*, 76(3), 973–982. doi:10.1111/biom.13206
18. Agency for Healthcare Research and Quality. *Medical Expenditure Panel Survey, Panel 23–27 Longitudinal Data Files (HC-217, HC-225, HC-234, HC-244, HC-252) and Medical Conditions Files (HC-207, HC-214, HC-222, HC-231, HC-241)*. Accessed September 2026.
19. Agency for Healthcare Research and Quality, Healthcare Cost and Utilization Project. *Clinical Classifications Software Refined (CCSR) for ICD-10-CM Diagnoses, v2026.1*. Accessed September 2026.

*DOIs for references 1, 2, 3, 4, 6, 7, 10, 12, 14, 16 and 17 were verified against the Crossref REST API on 14 September 2026.*
