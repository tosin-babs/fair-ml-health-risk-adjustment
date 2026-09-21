# Interpretable and Fair Machine Learning for Health-Cost Prediction and Risk Adjustment: A Reproducible Benchmark on Public Data and an Open-Source Toolkit

**Oluwatosin Dorcas Babalola**¹ *(corresponding author)*, **Chisom Adiegwu**¹, **Eniola Zainab Olamilekan**¹

¹ Department of Actuarial Science and Quantitative Risk Analysis and Management, Georgia State University, Atlanta, GA, USA. obabalola4@student.gsu.edu

**Word count.** 7,105 excluding abstract, tables, figure captions and references.

---

## Abstract

**Background.** Risk-adjustment formulas set what health plans are paid for each enrollee. A formula that underpays a group gives plans a reason to avoid it, one that responds to recorded diagnoses rewards coding, and one that responds to past use rewards use. Machine-learning methods promise accuracy, but these three properties are rarely evaluated together, on public data, with released code.

**Methods.** We built a prospective benchmark from five two-year Medical Expenditure Panel Survey panels (43,963 persons, 2018 to 2023) in which year-1 information predicts year-2 total spending. Ten models, including a non-negative least-squares model in the form of a payment formula, gradient boosting and a combined actuarial neural network, were evaluated by cross-validation that keeps primary sampling units whole, with Rao-Wu bootstrap and paired intervals. Outcomes were accuracy, net compensation for nine groups, an accuracy-fairness frontier, SHAP drivers, and the response of payment to simulated coding and to added use.

**Results.** Gradient boosting reached an out-of-sample R² of 0.187 (95% interval 0.156 to 0.242) against 0.123 for weighted least squares, a paired gain of 0.064 (0.047 to 0.091) that disappeared without prior use and spending. Every model underpaid people who need help with daily activities, by $4,545 to $9,583 per person-year on a mean cost of $31,329. A constrained stacked estimator cut the largest target-group gap to $296 at an R² of 0.171, but both constrained estimators moved $2,242 to $3,362 of overpayment onto enrollees aged 65 and over, and out of fold the constrained gap swung between folds by more than $17,000. A coder targeting each model's own most lucrative codes raised payment by $11,045 per added code under the payment-form model and $17,469 under the constrained linear formula, against $5,510 under boosting, on the feature set without prior use.

**Conclusions.** Flexible models are more accurate only when they use prior utilization, which payment formulas exclude. No model pays for functional need unless it observes it. Fairness constraints hold in training but not reliably out of sample, and they move money between groups, so they must be reported for all groups with intervals.

**Keywords.** risk adjustment; machine learning; algorithmic fairness; net compensation; coding intensity; SHAP; MEPS

---

## 1. Introduction

Risk adjustment decides how several hundred billion dollars a year are divided among health plans. In Medicare Advantage, plan payments are scaled by a risk score built from enrollee demographics and diagnoses; in the Affordable Care Act individual and small-group markets, risk scores drive transfers between insurers (Pope et al., 2004; Kautter et al., 2014). The formulas behind these scores are additive regressions of annual spending on demographic cells and hierarchical condition categories, and prospective formulas of this kind explain a minority of the variation in individual spending (Ellis, Martins and Rose, 2018).

Three problems with these formulas are well documented and usually studied one at a time. First, they leave predictable variation unused, and flexible methods can recover some of it (Rose, 2016; Duncan, Loginov and Ludkovski, 2016; Irvin et al., 2020). Second, they underpay identifiable groups, and a plan that can identify an underpaid group profits from avoiding it: Brown et al. (2014) found that Medicare Advantage plans selected on the risk left unpriced after risk adjustment was refined, and Montz et al. (2016) showed that marketplace plans had incentives to distort mental health and substance use coverage. Constrained and penalized regressions can remove chosen underpayments at a cost in fit (van Kleef et al., 2017; Zink and Rose, 2020; McGuire, Zink and Rose, 2021). Third, payment follows what is recorded. Risk scores rise when the same people move into Medicare Advantage (Geruso and Layton, 2020), coding intensity in Medicare Advantage has been measured for more than a decade (Kronick and Welch, 2014), and risk scores there have risen faster than other measures of beneficiary health would imply (Jacobs and Kronick, 2018). The Medicare Payment Advisory Commission projected that Medicare would spend 20% more on Medicare Advantage enrollees in 2025 than if they were in fee-for-service Medicare, about $84 billion; it attributed $40 billion to coding intensity, with risk scores about 16% above those of comparable fee-for-service beneficiaries before a 5.9% statutory adjustment, and $44 billion to favorable selection (MedPAC, 2025). CMS's V28 model, finalized in the 2024 Rate Announcement and phased in over 2024 to 2026, rebuilt the condition categories from ICD-10 codes, cut the diagnosis codes that map to payment categories from 9,797 to 7,770, and dropped some conditions whose diagnosis and coding CMS judged discretionary (CMS, 2023).

A fourth channel is less discussed. A formula that uses prior spending or prior use pays more next year for more care this year. Payment formulas exclude prior use for that reason, because a payment that reimburses a provider's or plan's own use gives up the incentive to control it (Ellis and McGuire, 1986, 1993), and use responds to financial incentives in ways that differ by type of service (Ellis, Martins and Zhu, 2017). Many machine-learning cost models include prior use because it is the strongest predictor.

This paper evaluates all four properties together, on public data, with released code. Using five two-year panels of the Medical Expenditure Panel Survey (MEPS), we predict each person's next-year spending from current-year information and compare ten models: unconstrained weighted least squares, a non-negative least-squares model in the form payment formulas take, a Tweedie generalized linear model, a two-part model, an elastic net, gradient boosting with two objectives, a random forest, a feed-forward network and a combined actuarial neural network. We ask five questions.

- RQ1. How much more accurately do flexible models predict next-year spending than the linear payment form, where do the gains come from, and do they survive without prior use?
- RQ2. Which groups does each model under- or overpay, with what uncertainty, and how does each model differ from the linear form on the same resamples?
- RQ3. How much accuracy does it cost to close the gaps for chosen groups, what happens to the other groups, and does the closure hold out of sample?
- RQ4. What drives the flexible models' predictions, and are the drivers stable and sensible?
- RQ5. How much does each model's payment rise when coding intensifies, at random, toward chronic conditions, toward the codes a published formula pays most, and toward the codes that model pays most, and how much does it rise per added dollar of prior use?

Three design choices distinguish the benchmark. Cross-validation keeps MEPS primary sampling units whole and intervals come from a bootstrap that respects the survey design, so the accuracy of a model that memorizes local effects is not overstated. Race and ethnicity never enter a model; they are used only to evaluate what a model does. And every comparison between two models is paired on the same resamples, so a reader can tell a difference from noise.

Relative to earlier machine-learning risk adjustment (Rose, 2016; Shrestha et al., 2018; Irvin et al., 2020; Andriola et al., 2024), the contribution is not a new estimator but a joint evaluation: the same models, the same folds and the same groups for accuracy, fairness, stability of fairness out of sample, coding response and use response, released as a package that runs on any person-level cost data.

## 2. Background

**Payment formulas.** The CMS-HCC and HHS-HCC models regress annualized spending on age-sex cells and hierarchical condition categories derived from diagnosis codes. They have no free intercept, their demographic cells are sex-specific, and condition coefficients that would be negative or would reverse a clinical hierarchy are constrained (Pope et al., 2004; Kautter et al., 2014). People enrolled for part of a year enter with spending annualized and a weight equal to their fraction of the year. The additive form is chosen for transparency, ease of administration and resistance to manipulation. Ellis, Martins and Rose (2018) review the design choices, Layton et al. (2017) set out how fit and incentives should be weighed against each other, and Park and Basu (2018) propose evaluation metrics beyond R².

**Machine learning for cost prediction.** Rose (2016) compared a library of machine-learning estimators for plan payment on commercial claims; an ensemble performed best, and a simplified formula with fewer variables kept much of the efficiency of the full one. Duncan, Loginov and Ludkovski (2016) compared regression frameworks including boosting for cost prediction. Shrestha et al. (2018) applied machine learning to mental health risk adjustment. Irvin et al. (2020) added social determinants. Andriola et al. (2024) introduced a learning algorithm that builds payment formulas within the constraints of a payment system. Wüthrich and Merz (2023) give the actuarial foundations of the models used here, and Schelldorfer and Wüthrich (2019) the combined actuarial neural network, in which a GLM's linear predictor is a fixed skip connection and a network learns a correction.

**Fairness.** Plan-payment research measures fairness at the group level: whether a formula pays for a group what the group costs. Van Kleef et al. (2017) used constrained regression to remove the undercompensation of chosen groups in the Dutch system. Zink and Rose (2020) proposed fair regression estimators that penalize or constrain net compensation, the difference between what a formula pays for a group and what the group costs, and McGuire, Zink and Rose (2021) compared constrained regression with reinsurance and variable selection. Zink and Rose (2021) showed that undercompensated groups can be defined by combinations of attributes that single-attribute checks miss. Bergquist et al. (2019) showed that transforming the data used to estimate a formula can improve its performance for undercompensated groups without changing the formula's form. Obermeyer et al. (2019) showed that an algorithm trained on cost as a proxy for need underrated Black patients, which is why the outcome here is spending and the interpretation is payment, not need.

**Coding and use.** Coding intensity is a response to the formula: codes that pay are recorded more often. Geruso and Layton (2020) and MedPAC (2025) measure the result in Medicare Advantage. The sensitivity of a formula to added codes, and to added use when prior use is a predictor, is a property of the formula that can be measured before any behavior changes, and Section 6.6 does so.

## 3. Data

**Panels.** MEPS follows each panel of households for two calendar years. We used the two-year longitudinal files for panels 23 to 27, covering 2018-2019 through 2022-2023, and the Medical Conditions file for each panel's first year. Panels 23 and 24 were later extended; we used their original two-year files so that every person contributes one prospective pair with the two-year longitudinal weight.

**Sample.** A person enters if present in year 1 with a positive longitudinal weight (Table A1). The 395 people who died in year 2 are kept, as payment models keep partial-year enrollees: their year-2 spending is divided by the fraction of year 2 they were in scope, and their weight is multiplied by the same fraction, so each contributes in proportion to exposure and weighted total spending is unchanged. Robustness checks drop them and use unannualized spending. The analysis sample is 43,963 persons representing 325.8 million a year.

**Outcome.** Year-2 total expenditure from all payers, in 2024 dollars using the CPI-U. The weighted mean is $7,792, the median $1,497, 14.9% spent nothing, the 99th percentile is $95,759, and the top 5% of spenders account for 50.7% of spending (Table 1).

**Predictors.** All from year 1. Demographics: age band by sex and Census region. Conditions: AHRQ Clinical Classifications Software Refined (CCSR) categories from the conditions file, as binary flags for the 124 categories held by at least 0.5% of the sample (Table A6), plus counts of categories and body systems. Two CCSR categories, diabetes without complication (END002) and type 2 diabetes (END005), are assigned from the same three-digit codes and are identical in MEPS; they enter as one flag. MEPS assigns body-system placeholder codes (XXX000) where a diagnosis cannot be placed in a category; these are kept in the body-system count and excluded from flags and condition counts. Use: office visits, emergency visits, inpatient discharges, prescription fills and total spending, logged. Perceived health and mental health are taken from round 2 and help with activities of daily living (ADL) and instrumental activities (IADL) from round 1, because round 3 is interviewed in year 2 and would leak year-2 information.

Four nested feature sets are used: F1 demographics; F2 adds conditions; F3 adds prior use and spending; F4 adds income, insurance, perceived health and functional help. F3 is the primary set because it is what cost-prediction studies usually use. F2 is the set closest to a deployable payment formula and is reported for every result where the difference matters.

**Groups for evaluation.** Income below 200% of the federal poverty level; Hispanic, non-Hispanic Black and non-Hispanic Asian; needs help with ADL or IADL; any mental or behavioral condition (CCSR MBD); age 65 and over; uninsured all year; and died in year 2. Race and ethnicity are never model inputs. Three groups are targets for the fairness estimators: low income, mental health conditions and functional help. They were chosen because the payment literature identifies them as undercompensated and because a payer can observe them. The ADL group has 1,612 persons, 2.8% of the weighted population.

**CCSR limitation.** MEPS releases three-digit ICD-10-CM codes. CCSR categories built from them are coarser than the hierarchical condition categories of the CMS and HHS models, so the payment-form model here has the form of a payment formula without being any specific formula.

## 4. Methods

### 4.1 Models

All models share one interface and are fitted with the MEPS longitudinal weight.

*Linear forms.* WLS is unconstrained weighted least squares on the features with an intercept. The payment-form model is weighted least squares in the form of the CMS and HHS formulas: sex-specific age cells in place of an intercept, condition flags, counts and (on F3) use terms, every coefficient constrained to be non-negative, no region term, solved by bounded least squares. It is not rescaled, so its predictions need not balance to total spending; a payment system would apply a normalization factor. Unconstrained WLS is reported because it is what "linear" usually means in cost prediction; it predicts a negative payment for 15.2% of the weighted population, which no payment system could use.

*Other models.* The Tweedie GLM has a log link and variance power 1.5. The two-part model multiplies a logistic probability of any spending by a gamma GLM for the amount. The elastic net is fitted on raw spending. Gradient boosting uses LightGBM with a Tweedie objective and, separately, a squared-error objective. The random forest uses 300 trees. The neural network has two hidden layers, a log link and Tweedie deviance loss. The combined actuarial neural network (CANN) adds the fitted Tweedie GLM's linear predictor as a fixed offset and initializes the network's output layer at zero, so it starts at the GLM and moves only where validation deviance improves. Multiplicative models are rescaled on their own training data so that weighted predictions sum to weighted spending.

Every hyperparameter is chosen inside the training data of each outer fold. The GLM, two-part and elastic-net penalties are chosen by three inner cluster-disjoint folds; boosting and the networks use one inner split of whole clusters for early stopping and a small grid. No test observation influences any choice (Table A5).

### 4.2 Survey-aware cross-validation and intervals

Strata are MEPS variance strata within panel. Within each stratum, primary sampling units (PSUs) are shuffled and dealt in turn across five folds, so no PSU contributes to both a training and a test set. Because most strata hold only a few PSUs, a fold does not contain every stratum; the deal balances folds across strata as far as the design allows. The procedure is repeated three times with different assignments.

Each fitted model predicts its held-out PSUs, so every person receives one out-of-fold prediction per repeat. The headline metric is the mean over the three repeats of each repeat's metric, which is what one deployed model achieves. Averaging the three repeats' predictions first is a small ensemble; its R² is reported separately (Table 2) and is larger mainly for the neural network.

Intervals come from the Rao-Wu rescaled bootstrap (Rao and Wu, 1988): within each stratum with n PSUs, n − 1 PSUs are drawn with replacement and the weights of drawn persons are multiplied by n/(n − 1) times the number of times their PSU was drawn. This keeps the variance of a design with few PSUs per stratum from being understated, which a naive PSU bootstrap does. Accuracy, group and frontier intervals use 200 resamples. Paired differences between models are computed on resamples shared by both models, 500 for accuracy and 200 for group net compensation, so the interval is for the difference itself. The bootstrap holds the fitted predictions fixed and so reflects the sampling of the evaluation population, not refitting; the spread of R² across repeats, which does reflect refitting, is at most 0.005 for every model except the neural network (0.015).

### 4.3 Metrics

Accuracy: weighted R², mean absolute error, Cumming's prediction measure (CPM; Cumming et al., 2002), the absolute-error analogue of R², predictive ratios by decile of predicted spending and by the age-sex cells of a payment formula, the share of the true top 10% of spenders placed in the predicted top 10%, and the share of negative predictions.

Fairness, following Zink and Rose (2020): for group G, the predictive ratio PR_G is weighted predicted over weighted observed spending, and net compensation NC_G is the weighted mean of prediction minus outcome, in dollars per person-year. Negative NC_G means the group is underpaid.

### 4.4 Fairness estimators

Because NC_G is linear in the coefficients of a linear model, two estimators have exact solutions. Penalized WLS adds λ Σ_G NC_G² to the weighted squared-error loss, with λ scaled by the outcome variance so it is unit-free; constrained WLS imposes NC_G = 0 for each target group, the λ → ∞ limit, solved as a KKT system. The grid is λ = 0, 0.1, 0.3, 1, 3, 10, 30 and 100 and the constrained limit.

For boosting we use a stacked form. A LightGBM score is cross-fitted within the training data (three cluster-disjoint folds, so no training row's score was fitted to its own outcome), and the score enters a penalized or constrained regression together with the features. We tried adding the net-compensation penalty inside the boosting objective and rejected it: the penalty's curvature is shared across a group, a diagonal Hessian either understates it, and Newton steps diverge, or, if spread across members, damps the fitting of their own residuals, and in testing underpayment of small high-cost groups worsened as the penalty rose. The stacked form keeps the boosted model's accuracy in the score and puts the adjustment where it has an exact solution.

As a group-neutral comparison, the Tweedie LightGBM's predictions are recalibrated by decile of predicted spending, with factors estimated on other folds' out-of-fold predictions.

The frontier is run on all three repeats of the folds, with Rao-Wu intervals for R² and for the largest absolute target-group gap. Because a constraint that holds exactly in training need not hold on new data, we also report, for each of the 15 folds, the target groups' net compensation in the training data and out of fold (Table 5b). Group membership enters estimation only through group means and never enters the prediction formula.

### 4.5 Drivers

TreeSHAP values are computed for the squared-error LightGBM on up to 4,000 held-out persons per fold, from the model fitted on that fold's training PSUs. The squared-error model is explained because its SHAP values are in dollars, and its accuracy is within 0.003 of the Tweedie model's. Global importance is the weighted mean absolute SHAP value; stability is the Spearman correlation of importance ranks between folds. For the CANN, permutation importance on the held-out fold is used. Sense checks test whether SHAP contributions rise with year-1 spending, the number of conditions and the number of body systems.

### 4.6 Coding and use sensitivity

Models are fitted on the training PSUs of each fold of the first repeat and stressed on that fold's held-out PSUs, on F3 and on F2. For a share of held-out persons (5%, 10% or 25%), one or two condition flags they did not have are switched on, and the condition and body-system counts are updated; use and spending are unchanged. The flags come from one of four pools:

- prevalence: drawn in proportion to each category's prevalence among people with any condition;
- chronic: drawn from payment-relevant chronic categories (diabetes, coronary disease, dysrhythmia, heart failure, COPD, depression and others listed in the code; six are common enough to be features);
- targeted: the five flags with the largest coefficients in the payment-form model, which is what a coder who knows a published formula adds (polyneuropathies, rheumatoid arthritis, other kidney disease, epilepsy and heart failure, on both feature sets);
- own-targeted: for each model and fold separately, the five flags that raise that model's own prediction most when switched on for 3,000 training persons, which is what a coder who can query the model adds (Table 7c).

We report the rise in predicted spending per added code, pooled across folds, with the standard deviation across folds.

For use sensitivity, year-1 spending is raised by 20% for 10% of held-out persons with nothing else changed, and we record the rise in next-year predicted spending per added dollar of year-1 spending. Visits, fills and discharges are held fixed, so the experiment isolates the spending term; a real rise in use would move those counts too.

### 4.7 Robustness

Each variant changes one decision and reruns WLS, the payment-form model, the Tweedie GLM and Tweedie LightGBM on one repeat: outcome capped at $250,000; a log-outcome linear model; unweighted training; condition-flag prevalence floors of 0.2% and 1%; feature sets F2 and F4; decedents excluded; decedents' spending not annualized; three and ten folds; each panel alone; panels whose outcome year fell in 2020 or 2021 dropped; and the subsets aged 65 and over and under 65 with private coverage.

## 5. Software

The estimators, folds, bootstrap, metrics, fairness methods and coding stress test are released as `riskfair`, a Python package with no dependence on this paper's configuration: every choice is an argument, so it runs on any person-level cost data with weights and design identifiers, survey or claims. The pipeline that produces every table and figure calls the package. Unit tests cover fold disjointness, the Rao-Wu weights, metric identities, exact zero net compensation for the constrained estimator in training, monotone response to the penalty, recovery of the constraint by the stacked estimator on simulated data, and the coding perturbation and per-flag gains. A test asserts that no feature name refers to race or to the year-2 outcome and that no feature is nearly collinear with the outcome. A script checks every number quoted in this text against the output tables.

## 6. Results

### 6.1 Accuracy (RQ1)

![Figure 1. Out-of-sample R² and Cumming's prediction measure by model, primary feature set, with 95% Rao-Wu bootstrap intervals.](output/figures/fig_accuracy.png)

Table 2 and Figure 1 give out-of-sample accuracy on the primary feature set. The three tree ensembles reach R² of 0.184 to 0.190: Tweedie LightGBM 0.187 (0.156 to 0.242), squared-error LightGBM 0.184 and the random forest 0.190. WLS reaches 0.123 (0.102 to 0.155) and the payment-form model 0.107 (0.085 to 0.142). The Tweedie GLM reaches 0.126, the CANN 0.124, the elastic net 0.116, the neural network 0.105 and the two-part model 0.093. Given the same features, the network correction in the CANN found little to add to its GLM base. The neural network's R² rises to 0.135 when its three repeats are averaged, the only model for which the ensemble matters.

Table 2b pairs every model with WLS on the same resamples. Tweedie LightGBM gains 0.064 (0.047 to 0.091), the random forest 0.067 (0.053 to 0.085) and squared-error LightGBM 0.061; no resample favors WLS. The Tweedie GLM (0.003, −0.018 to 0.025) and the CANN are indistinguishable from WLS on R², while the neural network (−0.018, −0.038 to −0.004) and the payment-form model (−0.016, −0.020 to −0.010) are worse. On absolute error every model beats WLS, because WLS fits the mean of a skewed outcome with negative predictions at the bottom: its predictive ratio in the bottom decile of predicted spending is −1.94. Cumming's prediction measure rises from 0.147 for WLS to 0.181 for the payment-form model, 0.242 for the Tweedie GLM and 0.276 for Tweedie LightGBM.

The boosting gain sits where payment formulas are weakest. The predictive ratio in the top decile of predicted spending is 0.94 for WLS and 0.99 for Tweedie LightGBM (Figure 2 and Table A2), the top-10% capture rate rises from 44% to 48%, and mean absolute error falls from $8,337 to $7,077.

![Figure 2. Predictive ratio by decile of predicted spending, four models.](output/figures/fig_calibration.png)

Table 3 shows where the information is. On demographics alone every model reaches R² of about 0.040. Adding conditions raises WLS to 0.103, the payment-form model to 0.098 and boosting to 0.101; adding prior use and spending raises WLS to 0.123 and boosting to 0.187. The boosting advantage therefore comes from how it uses prior use, which payment formulas exclude by design. On F2, the set closest to a deployable formula, the paired difference between boosting and WLS is −0.002 (−0.006 to 0.003). Adding social-risk and functional-status features (F4) changes R² by less than 0.01 for the linear and boosted models.

The Tweedie GLM collapses on F2, to an R² of −0.001 with a repeat standard deviation of 0.021 and a top-decile predictive ratio of 1.24. With a log link, the effects of co-occurring condition flags multiply, and without use terms to anchor the scale a few people with many rare flags receive very large predictions. The CANN and boosting, which start from or shrink toward simpler fits, do not show this. The payment-form model avoids negative predictions entirely, and its R² cost relative to WLS is small on F1 and F2 and larger on F3, where the unconstrained fit offsets overlapping use and condition terms with negative coefficients.

Payment formulas are calibrated on their age-sex cells, and Table A2b checks those cells. WLS, whose age-sex dummies force the cell means to balance in training, has predictive ratios within 0.01 of 1 in every cell out of fold. Tweedie LightGBM ranges from 0.92 (men 65 and over) to 1.13 (men 35 to 44), and the Tweedie GLM and CANN from 0.87 to 1.08. The payment-form model pays about 40% above cost for children of both sexes, because 42 of its 124 condition coefficients and half of its age cells sit at the zero bound, so young people are priced by use and count terms estimated mostly on adults; its predictions exceed total spending by 5%, which a payment system would remove with a normalization factor.

### 6.2 Who is underpaid (RQ2)

Table 4 and Figure 3 report net compensation by group, and Table 4b the paired difference from WLS. One result dominates. People who need help with ADL or IADL, with a mean annual cost of $31,329, are underpaid by every model: $7,747 per person-year by WLS (interval $4,246 to $12,747), $9,293 by the payment-form model, $4,762 by the Tweedie GLM, $5,721 by Tweedie LightGBM and $4,545 by the CANN, and between $4,545 and $9,583 across all ten. On the same resamples, the Tweedie GLM pays this group $2,985 more than WLS (interval $2,227 to $3,834) and Tweedie LightGBM $2,025 more ($1,354 to $2,925), while the payment-form model pays it $1,546 less. Flexible models narrow the gap without closing it, because functional status is not in the primary features and no combination of diagnoses and use reproduces it.

![Figure 3. Net compensation by group and model, $ per person-year, with 95% intervals.](output/figures/fig_group_compensation.png)

People who died in year 2 are underpaid by $42,174 to $46,604 per person-year of exposure by every model, predictive ratios of 0.30 to 0.36. No prospective model sees a death coming from the year before, and payment systems handle this through annualization and risk sharing rather than through the formula. The group is reported and not targeted.

The other groups are close to balance under most models, with intervals that include zero. WLS underpays Hispanic and non-Hispanic Black enrollees by $271 and $202, with intervals that include zero, and Tweedie LightGBM is within $91 of balance for both. Four patterns are outside noise. Tweedie LightGBM overpays people uninsured all year by $549 (interval $40 to $1,024), a predictive ratio of 1.23 on a mean cost of $2,401, because the low use of the uninsured is not fully carried into their next year. The random forest and CANN overpay people with mental health conditions by $971 and $724. The squared-error LightGBM underpays enrollees aged 65 and over by $1,004 ($336 to $1,832), and the random forest by $1,079. The payment-form model pays most groups above cost, by $413 to $780 for the race and ethnicity groups and $675 for low income, which is its 5% overall excess, about $400 per person, spread across groups.

### 6.3 The price of fairness (RQ3)

Table 5 and Figure 4 trace the frontier. For the linear formula, the largest target-group gap is the ADL gap, $7,747 at λ = 0. Penalizing it to $1,574 at λ = 1 reduces R² from 0.123 to 0.107, and the constrained limit leaves $724 at an R² of 0.099, a loss of a fifth of the formula's R². The stacked estimator starts from a gap of $4,601 at an R² of 0.182, reaches $536 at λ = 1 with an R² of 0.175, and $296 at 0.171 when constrained, a loss of 6% of its own R² and of 0.016 relative to the unconstrained boosted model. At every level of the gap the stacked frontier lies above the linear one. Decile recalibration of the boosted model, which uses no group information, keeps R² at 0.186 and leaves the gap at $5,608.

![Figure 4. The accuracy-fairness frontier: out-of-sample R² against the largest absolute net compensation among target groups, penalized and constrained linear and stacked estimators, with 95% intervals.](output/figures/fig_frontier.png)

The intervals for the gap are wide: $4,246 to $12,747 for WLS at λ = 0 and $220 to $5,515 under the constraint. For the constrained stacked estimator the interval, $365 to $4,590, lies above the point estimate of $296. That is a property of a maximum of absolute values near zero: resampling noise tends to push a maximum of absolute values up, so the upper limit is the informative bound, and it says the out-of-sample gap could be as large as $4,590.

Table 5b shows why. In the training data of every fold the constraint holds exactly. Out of fold, the ADL group's net compensation under constrained WLS ranges from −$13,043 to +$5,057 across the 15 folds, and under the constrained stacked estimator from −$11,664 to +$6,534. The pooled figure near zero is an average of large errors in both directions. With 1,612 persons in the group, about 320 in a held-out fold, a few very high spenders decide the fold's result. A constraint on a small high-variance group guarantees balance on the data used to fit it; it does not guarantee balance next year.

Closing the target gaps also opens others. Under constrained WLS, enrollees aged 65 and over move from balance to an overpayment of $3,362 (predictive ratio 1.20), Hispanic enrollees to $1,263 below cost, non-Hispanic Black enrollees to $565 below cost, and the uninsured to $1,498 below cost, a predictive ratio of 0.38. The stacked constraint moves the same groups by smaller amounts: $2,242 for the elderly, and −$685, −$290 and −$601 for Hispanic, non-Hispanic Black and uninsured enrollees. The ADL group is largely elderly, and a constraint that raises payment for people who need help raises payment for the elderly generally and takes it from young, low-cost groups. We report all groups at every point of the frontier because this spillover is invisible if only target groups are shown.

### 6.4 Drivers (RQ4)

Table 6 and Figure 5 list the largest contributors to the squared-error LightGBM. Year-1 spending is first by a wide margin (mean absolute SHAP $4,390), followed by prescription fills ($1,762), the condition count ($899), office visits ($525), inpatient discharges ($460) and the body-system count ($372). The first diagnosis category is diabetes ($233), then the age bands from 55 upward. Importance ranks are stable across folds (Spearman 0.79 over all features, 0.76 over the top 30). SHAP contributions rise with year-1 spending, condition count and body-system count (rank correlations 0.88, 0.85 and 0.86; Table A4). For the CANN, permuting year-1 spending costs 0.108 of R² and no other feature costs more than 0.021.

![Figure 5. The fifteen largest drivers of the squared-error LightGBM predictions, mean absolute SHAP value in dollars with across-fold standard deviation.](output/figures/fig_importance.png)

The drivers explain the accuracy results. What the boosted model knows beyond a linear formula is how prior spending and use carry forward, nonlinearly and with interactions. On F2, without those inputs, it has no advantage, and the diagnosis flags it relies on are the same ones a linear formula uses.

### 6.5 Use sensitivity (RQ5)

Table 7b gives the rise in next-year predicted spending per added dollar of year-1 spending on F3. Tweedie LightGBM pays 34 cents (fold standard deviation 7), constrained WLS 34 cents, squared-error LightGBM 32, WLS 28, and the Tweedie GLM and CANN 24. The payment-form model pays 1 cent and the two-part model 4. These last two are not immune to use: the payment-form model loads prior use on inpatient discharges, prescription fills and office visits, which the experiment holds fixed, and puts almost no weight on logged spending itself. A payer that raises its members' spending by a dollar recovers a third of it next year from a boosted model on F3, before any rise in visits and fills. On F2 no model responds, by construction.

### 6.6 Coding sensitivity (RQ5)

Table 7 and Figure 6 give the rise in predicted spending per added code when 10% of held-out people gain one, Table A3 the full grid, and Table 7c the codes each model pays most. The dollars per code change little with the share of people affected or with two codes instead of one.

**Codes drawn at random.** With prior use in the model (F3), a code drawn in proportion to prevalence raises predicted spending by $283 per code under WLS, $616 under Tweedie LightGBM and $1,287 under the payment-form model, and a chronic code by $580, $1,010 and $2,905. Unconstrained WLS responds least: it prices next-year cost mainly through prior spending and the condition count, with coefficients that offset each other, so a flag added to someone whose recorded use has not changed moves it little. The payment-form model responds most, because none of its coefficients may be negative and so none can offset. Without prior use (F2) the models converge: a chronic code raises WLS by $2,806, the Tweedie GLM by $2,900, Tweedie LightGBM by $2,907 and the payment-form model by $4,327.

**Codes chosen from a published formula.** The five categories with the largest payment-form coefficients are polyneuropathies, rheumatoid arthritis, other diseases of the kidney and ureters, epilepsy and heart failure, each held by 0.6% to 1.7% of the sample. Adding one of those to a person who lacks it raises predictions on F3 by $8,539 under WLS, $9,418 under the payment-form model and $12,812 under constrained WLS, against $1,482 under Tweedie LightGBM. On F2 the same codes are worth $10,529, $10,812 and $16,166 against $4,781.

**Codes chosen for each model.** That pool is the payment-form model's own, which favors the other models. When each model is gamed instead on the five flags that raise its own prediction most, found on 3,000 of its own training persons (Table 7c), the ordering holds. On F3 an own-targeted code is worth $2,323 under Tweedie LightGBM, $2,572 under the two-part model, $4,776 under squared-error LightGBM, $5,055 under the Tweedie GLM, $5,204 under the CANN, $9,044 under WLS, $9,806 under the payment-form model and $13,754 under constrained WLS. On F2 the figures are $5,510 for Tweedie LightGBM, $7,018 for the Tweedie GLM, $10,777 for WLS, $11,045 for the payment-form model and $17,469 for constrained WLS. When 10% of people each gain one such code on F2, total predicted spending rises by 7.0% under Tweedie LightGBM, 13.7% under WLS, 14.1% under the payment-form model and 25.0% under constrained WLS. The linear models converge on the same five codes in almost every fold; the boosted and neural models' lucrative codes vary more across folds and include categories such as muscle disorders, nerve root disorders and uncomplicated pregnancy, which is why their fold standard deviations are larger.

**Why.** A linear formula attaches a fixed increment to each flag, and a rare, costly category carries a large increment estimated from a few hundred people; anyone coded with it receives the full amount. A tree splits on a rare flag only where enough people share it, and it conditions on use, so a code with no matching visits, fills or spending moves it less. The same property that makes the boosted model accurate on F3 makes it hard to move with a code alone. Constrained WLS is the most exposed of all, and the reason is the constraint: epilepsy, polyneuropathies, heart failure and rheumatoid arthritis are 6 to 11 times more common among people who need help with daily activities than among everyone else, so a constraint that raises payment for that group raises exactly the coefficients a coder would target. Fairness bought through a linear formula's diagnosis coefficients is bought with exposure to those diagnoses.

![Figure 6. Accuracy against exposure to coding intensity: out-of-sample R² and the rise in predicted spending per added code when 10% of people gain one.](output/figures/fig_gameability.png)

### 6.7 Robustness

Table 8 reports the variants. Across every variant except F2 the boosting advantage over WLS on R² is between 0.060 and 0.125, and on F2 it vanishes. Capping the outcome at $250,000 raises every R² (Tweedie LightGBM to 0.281, WLS to 0.192) and reduces the ADL gap by $700 to $1,200 for WLS, the Tweedie GLM and boosting. Dropping prior use and spending (F2) raises the ADL gap for boosting to $10,991, since prior use was carrying part of the functional-status signal. Adding social-risk and functional-status features (F4), which include the ADL and IADL items, cuts the ADL gap to $1,165 for WLS and $2,237 for boosting: the direct route to paying for functional need is to observe it. Dropping decedents (Tweedie LightGBM 0.193, WLS 0.124) or not annualizing their spending (0.196 and 0.127) moves R² by less than 0.01. Three or ten folds change R² by less than 0.01. Unweighted training and moving the condition-flag floor change R² by less than 0.02 for WLS and boosting, but the Tweedie GLM rises from 0.122 to 0.155 with a 1% floor, another sign of its sensitivity to rare flags. The log-outcome linear model performs poorly on squared error (R² −0.096) and is not a candidate for payment.

Single panels vary more than any modeling choice: WLS ranges from 0.030 to 0.144 and Tweedie LightGBM from 0.142 to 0.248 across the five panels, and the ADL gap from about $3,000 to $17,000. Dropping outcome years 2020 and 2021 gives Tweedie LightGBM 0.207 and WLS 0.125. Within the 65-and-over subset, R² falls for every model (Tweedie LightGBM 0.156, Tweedie GLM 0.124, WLS 0.096) and the ADL gap persists; among people under 65 with private coverage, Tweedie LightGBM reaches 0.155 and WLS 0.091.

## 7. Discussion

### 7.1 For payers and regulators

On this benchmark the choice between a linear formula and a flexible model is not a choice between accuracy and fairness. The flexible model is more accurate when it uses prior use, it is no worse and for the functional-need group somewhat better on group compensation, and its drivers are stable and sensible. But its accuracy advantage comes entirely from prior use, and with prior use comes a payment that rises by about a third of a dollar for every dollar of this year's spending. A regulator who keeps the exclusion of prior use should expect the flexible model's accuracy advantage to vanish (Table 3). On coding, the ordering depends on which codes are added. Codes drawn at random move every model by similar amounts once prior use is excluded, but codes chosen for what they pay move a linear formula two to four times as far as a boosted model, and a fairness-constrained linear formula furthest of all. A regulator who constrains a formula for fairness buys the gap reduction with exposure to the very diagnoses that close it, and should audit those codes.

Three findings apply to any formula. First, the group every model underpays is defined by functional status, which no diagnosis or use variable reproduces. Observing it directly (F4) removes most of the gap without a constraint. Second, fairness constraints are fragile out of sample for small, costly groups. A regulator who imposes a constraint should expect next year's net compensation for the group to vary by several thousand dollars around zero, and should pair a constraint with risk sharing for the extreme cases it cannot reach. Third, fairness reporting must cover non-target groups with intervals. The constraint that fixed the ADL gap created an elderly overpayment of $3,362 and a 62% underpayment of the uninsured, and a reader shown only the target groups would call the constrained formula fair.

### 7.2 Questions a reader will ask

*Are these R² values low?* They are in the range expected for prospective prediction of untruncated annual spending from demographics and diagnoses (Ellis, Martins and Rose, 2018). MEPS conditions are self-reported and coded to three digits, so the levels are not comparable with claims-based studies; comparisons are valid within the benchmark, where every model sees the same data, folds and resamples.

*Is the boosting gain real or noise?* The paired interval for the gain over WLS on F3 excludes zero and no resample favors WLS, and the gain holds across fold counts, weighting, outcome caps and panels. The equally clear result is that there is no gain on F2.

*Would a payer use unconstrained WLS?* No. It predicts negative payments for 15.2% of people. It is reported because cost-prediction studies usually compare against it, and the payment-form model is the comparison a regulator should read.

*Why keep decedents?* Because payment models keep partial-year enrollees and annualize their spending. Excluding them or not annualizing changes R² by less than 0.01 and the ADL gap by a few hundred dollars.

*Is the fairness frontier an artifact of the chosen targets?* The targets were chosen before the analysis, and all nine groups are reported at every point. Different targets would move other groups; the spillover to the elderly and the uninsured follows from how the ADL group is composed and would appear under any constraint that pays more for functional need through age and diagnosis terms.

*Is the targeted coding pool rigged against the linear formula?* It would be if the only targeted pool were the payment-form model's own largest coefficients, so each model is also gamed on the five codes that raise its own prediction most, found on its own training data. The ordering is the same, and the gap between the linear forms and boosting narrows but does not close.

*Why is the stacked estimator's gap interval above its point estimate?* Because the statistic is a maximum of absolute values near zero, whose bootstrap distribution is pushed upward. The per-fold table is the more direct evidence of instability.

### 7.3 Limitations

MEPS is a household survey with self-reported conditions verified in part by provider records, and its three-digit ICD-10 codes make condition categories coarser than payment-model categories. Sample sizes for small groups are limited: the ADL group has 1,612 persons and the decedents 395, and their intervals are wide. The bootstrap holds fitted models fixed; the repeat spread adds refitting variation, which is small except for the neural network. The fairness targets were chosen by the analyst. The coding experiment switches codes on mechanically: it measures what a formula would pay for added codes, not how coders would respond, and it does not model the documentation and audits that a real coder faces. The own-targeted pools are found by querying each fitted model, which a coder outside the payer could not do directly, though the behavior it stands for, finding what pays and recording it, needs no such access. The use experiment raises spending without raising visits and fills, so it understates the response of models that load on counts. The neural models were tuned over a small grid, and their results are a floor on what a well-tuned network can do. The analysis does not model plan selection or enrollee choice; a model's net compensation for a group is what gives plans an incentive, not a measure of how plans respond.

### 7.4 What the toolkit is for

Any payer or regulator with person-level cost data can run the same five evaluations on its own formula and candidate replacements, with survey- or cluster-aware validation, paired intervals and full group reporting built in. The package does not decide what fairness means; it measures what a formula does to each group, whether that holds out of sample, what changing it would cost, and how the formula responds to coding and to use.

## 8. Conclusion

A public benchmark shows that flexible models predict next-year health spending more accurately than the linear form of payment formulas only when they use prior utilization, that every model underpays people who need functional help unless it observes functional status, that fairness constraints close target gaps in training but not reliably out of sample and shift payment toward the elderly and away from the uninsured, and that a linear payment form pays two to four times as much as a boosted model for codes chosen to game it, with the fairness-constrained formula the most exposed of all. The code that produced every number is released as a reusable package.

---

## Declarations

**Data availability.** MEPS public-use files are freely available from the Agency for Healthcare Research and Quality; the CCSR reference file from AHRQ's HCUP. Neither is redistributed.

**Code availability.** The `riskfair` package and the complete, seeded pipeline are at https://github.com/tosin-babs/fair-ml-health-risk-adjustment; `python/run_all.py` rebuilds every table and figure and `pytest tests` runs the checks. An interactive explorer of the results is at https://fair-risk-adjustment.vercel.app. It is a research benchmark on survey data, not a payment formula.

**Competing interests.** None declared.

**Ethics.** The analysis uses de-identified public survey data and did not require ethical approval.


---

## References

1. Andriola, C., Ellis, R. P., Siracuse, J. J., Hoagland, A., et al. (2024). A novel machine learning algorithm for creating risk-adjusted payment formulas. *JAMA Health Forum*, 5(4), e240625. doi:10.1001/jamahealthforum.2024.0625
2. Bergquist, S. L., Layton, T. J., McGuire, T. G., & Rose, S. (2019). Data transformations to improve the performance of health plan payment methods. *Journal of Health Economics*, 66, 195–207. doi:10.1016/j.jhealeco.2019.05.005
3. Brown, J., Duggan, M., Kuziemko, I., & Woolston, W. (2014). How does risk selection respond to risk adjustment? New evidence from the Medicare Advantage program. *American Economic Review*, 104(10), 3335–3364. doi:10.1257/aer.104.10.3335
4. Centers for Medicare & Medicaid Services (2023). *Announcement of Calendar Year (CY) 2024 Medicare Advantage (MA) Capitation Rates and Part C and Part D Payment Policies*. 31 March 2023. Baltimore, MD: CMS.
5. Cumming, R. B., Knutson, D., Cameron, B. A., & Derrick, B. (2002). *A Comparative Analysis of Claims-Based Methods of Health Risk Assessment for Commercial Populations*. Schaumburg, IL: Society of Actuaries. https://www.soa.org/globalassets/assets/files/research/projects/risk-assessmentc.pdf
6. Duncan, I., Loginov, M., & Ludkovski, M. (2016). Testing alternative regression frameworks for predictive modeling of health care costs. *North American Actuarial Journal*, 20(1), 65–87. doi:10.1080/10920277.2015.1110491
7. Ellis, R. P., Martins, B., & Rose, S. (2018). Risk adjustment for health plan payment. In T. G. McGuire & R. C. van Kleef (Eds.), *Risk Adjustment, Risk Sharing and Premium Regulation in Health Insurance Markets* (pp. 55–104). Academic Press. doi:10.1016/B978-0-12-811325-7.00003-8
8. Ellis, R. P., Martins, B., & Zhu, W. (2017). Health care demand elasticities by type of service. *Journal of Health Economics*, 55, 232–243. doi:10.1016/j.jhealeco.2017.07.007
9. Ellis, R. P., & McGuire, T. G. (1986). Provider behavior under prospective reimbursement: cost sharing and supply. *Journal of Health Economics*, 5(2), 129–151. doi:10.1016/0167-6296(86)90002-0
10. Ellis, R. P., & McGuire, T. G. (1993). Supply-side and demand-side cost sharing in health care. *Journal of Economic Perspectives*, 7(4), 135–151. doi:10.1257/jep.7.4.135
11. Geruso, M., & Layton, T. (2020). Upcoding: evidence from Medicare on squishy risk adjustment. *Journal of Political Economy*, 128(3), 984–1026. doi:10.1086/704756
12. Irvin, J. A., Kondrich, A. A., Ko, M., Rajpurkar, P., et al. (2020). Incorporating machine learning and social determinants of health indicators into prospective risk adjustment for health plan payments. *BMC Public Health*, 20, 608. doi:10.1186/s12889-020-08735-0
13. Jacobs, P. D., & Kronick, R. (2018). Getting what we pay for: how do risk-based payments to Medicare Advantage plans compare with alternative measures of beneficiary health risk? *Health Services Research*, 53(6), 4997–5015. doi:10.1111/1475-6773.12977
14. Kautter, J., Pope, G. C., Ingber, M., Freeman, S., Patterson, L., Cohen, M., & Keenan, P. (2014). The HHS-HCC risk adjustment model for individual and small group markets under the Affordable Care Act. *Medicare & Medicaid Research Review*, 4(3), E1–E46. doi:10.5600/mmrr.004.03.a03
15. Kronick, R., & Welch, W. P. (2014). Measuring coding intensity in the Medicare Advantage program. *Medicare & Medicaid Research Review*, 4(2), E1–E19. doi:10.5600/mmrr.004.02.sa06
16. Layton, T. J., Ellis, R. P., McGuire, T. G., & van Kleef, R. (2017). Measuring efficiency of health plan payment systems in managed competition health insurance markets. *Journal of Health Economics*, 56, 237–255. doi:10.1016/j.jhealeco.2017.05.004
17. McGuire, T. G., Zink, A. L., & Rose, S. (2021). Improving the performance of risk adjustment systems: constrained regressions, reinsurance, and variable selection. *American Journal of Health Economics*, 7(4), 497–521. doi:10.1086/716199
18. Medicare Payment Advisory Commission (2025). The Medicare Advantage program: status report. In *Report to the Congress: Medicare Payment Policy*, March 2025, chapter 11, pp. 317–406. Washington, DC: MedPAC.
19. Montz, E., Layton, T., Busch, A. B., Ellis, R. P., Rose, S., & McGuire, T. G. (2016). Risk-adjustment simulation: plans may have incentives to distort mental health and substance use coverage. *Health Affairs*, 35(6), 1022–1028. doi:10.1377/hlthaff.2015.1668
20. Obermeyer, Z., Powers, B., Vogeli, C., & Mullainathan, S. (2019). Dissecting racial bias in an algorithm used to manage the health of populations. *Science*, 366(6464), 447–453. doi:10.1126/science.aax2342
21. Park, S., & Basu, A. (2018). Alternative evaluation metrics for risk adjustment methods. *Health Economics*, 27(6), 984–1010. doi:10.1002/hec.3657
22. Pope, G. C., Kautter, J., Ellis, R. P., et al. (2004). Risk adjustment of Medicare capitation payments using the CMS-HCC model. *Health Care Financing Review*, 25(4), 119–141.
23. Rao, J. N. K., & Wu, C. F. J. (1988). Resampling inference with complex survey data. *Journal of the American Statistical Association*, 83(401), 231–241. doi:10.1080/01621459.1988.10478591
24. Rose, S. (2016). A machine learning framework for plan payment risk adjustment. *Health Services Research*, 51(6), 2358–2374. doi:10.1111/1475-6773.12464
25. Schelldorfer, J., & Wüthrich, M. V. (2019). Nesting classical actuarial models into neural networks. SSRN working paper 3320525. doi:10.2139/ssrn.3320525
26. Shrestha, A., Bergquist, S., Montz, E., & Rose, S. (2018). Mental health risk adjustment with clinical categories and machine learning. *Health Services Research*, 53(S1), 3189–3206. doi:10.1111/1475-6773.12818
27. van Kleef, R. C., McGuire, T. G., van Vliet, R. C. J. A., & van de Ven, W. P. M. M. (2017). Improving risk equalization with constrained regression. *European Journal of Health Economics*, 18(9), 1137–1156. doi:10.1007/s10198-016-0859-1
28. Wüthrich, M. V., & Merz, M. (2023). *Statistical Foundations of Actuarial Learning and its Applications*. Springer. doi:10.1007/978-3-031-12409-9
29. Zink, A., & Rose, S. (2020). Fair regression for health care spending. *Biometrics*, 76(3), 973–982. doi:10.1111/biom.13206
30. Zink, A., & Rose, S. (2021). Identifying undercompensated groups defined by multiple attributes in risk adjustment. *BMJ Health & Care Informatics*, 28(1), e100414. doi:10.1136/bmjhci-2021-100414
31. Agency for Healthcare Research and Quality. *Medical Expenditure Panel Survey, Panel 23–27 Longitudinal Data Files (HC-217, HC-225, HC-234, HC-244, HC-252) and Medical Conditions Files (HC-207, HC-214, HC-222, HC-231, HC-241)*. Accessed September 2026.
32. Agency for Healthcare Research and Quality, Healthcare Cost and Utilization Project. *Clinical Classifications Software Refined (CCSR) for ICD-10-CM Diagnoses, v2026.1*. Accessed September 2026.

*DOIs were verified against the Crossref REST API on 15 September 2026. MedPAC figures were checked against the chapter text.*
