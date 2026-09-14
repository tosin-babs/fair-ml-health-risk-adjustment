# Tables

*Generated from `output/tables/*.csv` by `python/make_tables.py`. Spending is total expenditure from all payers in 2024 dollars. All estimates are weighted by the MEPS longitudinal weight; model metrics are out of fold from cross-validation with primary sampling units kept whole.*


**Table 1.** Sample characteristics and year-2 spending.

*Weighted estimates; standard errors Taylor-linearized for the stratified cluster design.*

| Characteristic | Estimate | SE | n |
|---|---:|---:|---:|
| Persons | 43,568.00 |  | 43,568 |
| Weighted population, millions a year | 324.38 |  | 43,568 |
| Age in year 1, mean | 38.78 | 0.20 | 43,568 |
| Age 65 and over, % | 16.24 | 0.27 | 43,568 |
| Female, % | 50.92 | 0.24 | 43,568 |
| Hispanic, % | 19.06 | 0.60 | 43,568 |
| Non-Hispanic White, % | 58.57 | 0.64 | 43,568 |
| Non-Hispanic Black, % | 12.44 | 0.42 | 43,568 |
| Non-Hispanic Asian, % | 6.16 | 0.30 | 43,568 |
| Non-Hispanic other or multiple, % | 3.76 | 0.19 | 43,568 |
| Below 100% FPL, % | 12.29 | 0.34 | 43,568 |
| 100-124% FPL, % | 4.06 | 0.20 | 43,568 |
| 125-199% FPL, % | 12.27 | 0.34 | 43,568 |
| 200-399% FPL, % | 28.91 | 0.48 | 43,568 |
| 400% FPL or more, % | 42.48 | 0.58 | 43,568 |
| Any private coverage, % | 65.95 | 0.50 | 43,568 |
| Public coverage only, % | 27.35 | 0.44 | 43,568 |
| Uninsured all year, % | 6.70 | 0.24 | 43,568 |
| Fair or poor perceived health, % | 9.64 | 0.21 | 43,396 |
| Needs help with ADL or IADL, % | 2.72 | 0.10 | 43,568 |
| Any year-1 condition (CCSR category), % | 69.95 | 0.39 | 43,568 |
| Year-1 CCSR categories, mean | 2.79 | 0.03 | 43,568 |
| Any mental health condition, % | 15.80 | 0.28 | 43,568 |
| Year-1 spending, mean $ | 7,285.40 | 157.46 | 43,568 |
| Year-2 spending, mean $ | 7,538.02 | 143.02 | 43,568 |
| Year-2 spending of zero, % | 14.97 | 0.32 | 43,568 |
| Year-2 spending, median $ | 1,486.14 |  | 43,568 |
| Year-2 spending, 90th percentile $ | 18,256.33 |  | 43,568 |
| Year-2 spending, 99th percentile $ | 91,441.62 |  | 43,568 |
| Share of year-2 spending by the top 5% of spenders, % | 49.97 |  | 43,568 |

**Table 2.** Out-of-sample accuracy by model, primary feature set (F3).

*Five survey folds, three repeats. Intervals from a bootstrap over PSUs within strata applied to repeat-averaged out-of-fold predictions. CPM is Cumming's prediction measure. Top-10% capture is the weighted share of the true top decile of spenders placed in the model's top decile.*

| Model | R² | 95% low | 95% high | CPM | MAE | PR bottom decile | PR top decile | Top-10% capture |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| WLS | 0.125 | 0.101 | 0.158 | 0.151 | $7,978 | -1.91 | 0.93 | 44% |
| Tweedie GLM | 0.135 | 0.104 | 0.180 | 0.247 | $7,077 | 1.00 | 0.97 | 47% |
| Two-part | 0.101 | 0.077 | 0.134 | 0.217 | $7,359 | 0.98 | 0.99 | 44% |
| Elastic net | 0.117 | 0.092 | 0.152 | 0.197 | $7,540 | -0.86 | 0.93 | 42% |
| LightGBM (Tweedie) | 0.195 | 0.161 | 0.245 | 0.279 | $6,770 | 1.02 | 0.98 | 48% |
| LightGBM (squared error) | 0.197 | 0.165 | 0.240 | 0.268 | $6,881 | 1.58 | 0.97 | 48% |
| Random forest | 0.197 | 0.167 | 0.241 | 0.270 | $6,856 | 0.75 | 0.97 | 48% |
| Neural network | 0.141 | 0.116 | 0.178 | 0.249 | $7,053 | 1.00 | 0.97 | 44% |
| CANN | 0.134 | 0.103 | 0.179 | 0.247 | $7,074 | 1.00 | 0.98 | 47% |

**Table 3.** Out-of-sample R² by feature set.

*F1 demographics; F2 adds year-1 CCSR condition flags; F3 adds year-1 use and spending; F4 adds income, insurance, perceived health and functional help.*

| Model | F1 | F2 | F3 | F4 |
|---|---:|---:|---:|---:|
| LightGBM (Tweedie) | 0.039 | 0.102 | 0.195 | 0.195 |
| Tweedie GLM | 0.039 | 0.013 | 0.135 | 0.132 |
| WLS | 0.039 | 0.103 | 0.125 | 0.129 |

**Table 4.** Net compensation by group: predicted minus observed spending, $ per person-year (95% interval).

*Negative values mean the model pays less for the group than the group costs. Race and ethnicity are used for evaluation only and never as model inputs.*

| Group | WLS | Tweedie GLM | LightGBM (Tweedie) | CANN |
|---|---:|---:|---:|---:|
| Age 65 and over | -$8 (-$624, $473) | $538 (-$133, $1,097) | -$357 (-$1,014, $97) | $627 (-$48, $1,189) |
| Hispanic | -$207 (-$793, $308) | $66 (-$522, $597) | $124 (-$472, $679) | $51 (-$536, $582) |
| Income below 200% FPL | -$37 (-$451, $288) | $187 (-$256, $525) | $83 (-$311, $404) | $201 (-$241, $538) |
| Mental health condition | $199 (-$381, $743) | $605 ($32, $1,226) | $433 (-$91, $895) | $658 ($87, $1,275) |
| Needs ADL or IADL help | -$7,794 (-$11,613, -$4,584) | -$4,949 (-$8,810, -$1,852) | -$5,670 (-$9,127, -$2,561) | -$4,770 (-$8,648, -$1,630) |
| Non-Hispanic Asian | $105 (-$358, $573) | $300 (-$123, $706) | $189 (-$284, $613) | $282 (-$143, $686) |
| Non-Hispanic Black | -$274 (-$760, $280) | -$68 (-$534, $486) | -$89 (-$550, $437) | -$72 (-$543, $489) |
| Uninsured all year | $17 (-$413, $475) | $533 ($97, $979) | $657 ($215, $1,061) | $513 ($79, $959) |

**Table 5.** The accuracy-fairness frontier.

*Penalized and constrained estimators target the three groups marked in the text; the other groups show spillover. Stacked LightGBM enters a cross-fitted LightGBM score with the features into the fair regression. One repeat of the survey folds.*

| Method | λ | R² | CPM | Max |NC| target | Income below 200% FPL | Hispanic | Non-Hispanic Black | Non-Hispanic Asian | Needs ADL or IADL help | Mental health condition | Age 65 and over | Uninsured all year |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Penalized WLS | 0 | 0.124 | 0.149 | $7,887 | -$43 | -$215 | -$293 | $97 | -$7,887 | $174 | $15 | $12 |
| Penalized WLS | 0.1 | 0.120 | 0.118 | $4,699 | $585 | -$29 | $118 | $290 | -$4,699 | $1,054 | $1,653 | $44 |
| Penalized WLS | 0.3 | 0.114 | 0.102 | $2,991 | $701 | -$158 | $140 | $216 | -$2,991 | $874 | $2,516 | -$166 |
| Penalized WLS | 1 | 0.105 | 0.084 | $1,703 | $517 | -$554 | -$118 | -$85 | -$1,703 | $428 | $3,089 | -$639 |
| Penalized WLS | 3 | 0.100 | 0.071 | $1,145 | $268 | -$916 | -$410 | -$376 | -$1,145 | $180 | $3,272 | -$1,051 |
| Penalized WLS | 10 | 0.097 | 0.063 | $912 | $106 | -$1,132 | -$594 | -$552 | -$912 | $74 | $3,325 | -$1,293 |
| Penalized WLS | 30 | 0.096 | 0.060 | $841 | $47 | -$1,209 | -$660 | -$615 | -$841 | $42 | $3,337 | -$1,379 |
| Penalized WLS | 100 | 0.096 | 0.059 | $816 | $25 | -$1,238 | -$685 | -$638 | -$816 | $30 | $3,341 | -$1,411 |
| Constrained WLS | constrained | 0.096 | 0.058 | $805 | $15 | -$1,250 | -$696 | -$649 | -$805 | $25 | $3,342 | -$1,425 |
| Stacked LightGBM | 0 | 0.190 | 0.258 | $5,142 | $71 | $60 | -$91 | $201 | -$5,142 | $165 | -$18 | $441 |
| Stacked LightGBM | 0.1 | 0.188 | 0.244 | $2,925 | $486 | $186 | $182 | $329 | -$2,925 | $769 | $1,080 | $467 |
| Stacked LightGBM | 0.3 | 0.186 | 0.240 | $1,736 | $549 | $91 | $184 | $270 | -$1,736 | $654 | $1,648 | $320 |
| Stacked LightGBM | 1 | 0.182 | 0.236 | $838 | $400 | -$193 | -$13 | $50 | -$838 | $361 | $2,013 | -$17 |
| Stacked LightGBM | 3 | 0.180 | 0.231 | $448 | $213 | -$454 | -$229 | -$162 | -$448 | $198 | $2,121 | -$312 |
| Stacked LightGBM | 10 | 0.179 | 0.228 | $286 | $94 | -$609 | -$364 | -$290 | -$286 | $130 | $2,148 | -$486 |
| Stacked LightGBM | 30 | 0.178 | 0.227 | $236 | $50 | -$664 | -$413 | -$335 | -$236 | $109 | $2,154 | -$547 |
| Stacked LightGBM | 100 | 0.178 | 0.227 | $218 | $34 | -$685 | -$431 | -$353 | -$218 | $101 | $2,155 | -$570 |
| Stacked LightGBM, constrained | constrained | 0.178 | 0.226 | $210 | $27 | -$694 | -$439 | -$360 | -$210 | $98 | $2,156 | -$580 |
| LightGBM, decile recalibration | 0 | 0.194 | 0.275 | $5,647 | $110 | $162 | -$42 | $191 | -$5,647 | $555 | -$387 | $666 |

**Table 6.** The 20 largest drivers of the squared-error LightGBM predictions.

*Weighted mean absolute SHAP value on held-out folds. Rank stability across folds: mean Spearman correlation 0.80 over all features and 0.77 over the top 30. The last column is the loss in held-out R² when the feature is permuted in the CANN.*

| Rank | Feature | Mean |SHAP| | SD across folds | CANN R² loss |
|---|---:|---:|---:|---:|
| 1 | log_spend_y1 | $3,970 | $242 | 0.1083 |
| 2 | log_rx_fills_y1 | $1,625 | $181 | 0.0022 |
| 3 | n_conditions | $1,015 | $168 | 0.0205 |
| 4 | log_office_visits_y1 | $557 | $47 | -0.0019 |
| 5 | log_inpatient_y1 | $386 | $98 | 0.0168 |
| 6 | n_body_systems | $311 | $48 | -0.0083 |
| 7 | END002 Diabetes mellitus without complication | $172 | $26 | -0.0009 |
| 8 | age_55-64 | $167 | $53 | 0.0143 |
| 9 | age_65-74 | $119 | $15 | 0.0025 |
| 10 | MUS003 Rheumatoid arthritis and related disease | $116 | $52 | -0.0131 |
| 11 | age_5-17 | $112 | $22 | -0.0082 |
| 12 | age_75-84 | $110 | $27 | -0.0030 |
| 13 | END001 Thyroid disorders | $100 | $27 | 0.0035 |
| 14 | age_25-34_x_female | $95 | $27 | 0.0096 |
| 15 | age_85+ | $84 | $19 | 0.0057 |
| 16 | log_er_visits_y1 | $74 | $31 | 0.0022 |
| 17 | region_3.0 | $73 | $28 | 0.0028 |
| 18 | END010 Disorders of lipid metabolism | $72 | $30 | 0.0110 |
| 19 | MUS006 Osteoarthritis | $69 | $25 | -0.0000 |
| 20 | region_2.0 | $65 | $24 | -0.0013 |

**Table 7.** Coding sensitivity: rise in predicted spending per added condition code, 10% of people affected.

*Codes the person did not have are switched on; use and spending are unchanged. The last column is the rise in total predicted spending when 10% of people gain one chronic code.*

| Model | chronic, 1 code | chronic, 2 codes | prevalence, 1 code | prevalence, 2 codes | Total rise, 1 chronic code |
|---|---:|---:|---:|---:|---:|
| CANN | $953 | $952 | $482 | $398 | 1.25% |
| Constrained WLS | $1,121 | $1,228 | $333 | $383 | 1.54% |
| LightGBM (Tweedie) | $693 | $755 | $612 | $596 | 0.92% |
| LightGBM (squared error) | $755 | $781 | $719 | $717 | 1.00% |
| Tweedie GLM | $885 | $889 | $443 | $358 | 1.16% |
| Two-part | $1,307 | $1,402 | $827 | $842 | 1.73% |
| WLS | $1,292 | $1,368 | $395 | $393 | 1.71% |

**Table 8.** Robustness: each row changes one decision.

*One repeat of the survey folds. Max |NC| is over the three target groups.*

| Variant | Model | n | R² | CPM | PR top decile | Max |NC| target |
|---|---:|---:|---:|---:|---:|---:|
| baseline | WLS | 43,568 | 0.124 | 0.149 | 0.94 | $7,887 |
| baseline | Tweedie GLM | 43,568 | 0.125 | 0.243 | 1.00 | $5,011 |
| baseline | LightGBM (Tweedie) | 43,568 | 0.195 | 0.277 | 0.99 | $5,864 |
| outcome capped | WLS | 43,568 | 0.189 | 0.173 | 0.93 | $6,699 |
| outcome capped | Tweedie GLM | 43,568 | 0.194 | 0.250 | 1.00 | $4,034 |
| outcome capped | LightGBM (Tweedie) | 43,568 | 0.282 | 0.284 | 1.00 | $4,558 |
| log outcome | WLS, log outcome | 43,568 | -0.105 | 0.195 | 1.64 | $6,407 |
| unweighted training | WLS | 43,568 | 0.127 | 0.149 | 0.91 | $7,806 |
| unweighted training | Tweedie GLM | 43,568 | 0.144 | 0.257 | 0.97 | $5,223 |
| unweighted training | LightGBM (Tweedie) | 43,568 | 0.196 | 0.283 | 0.99 | $5,356 |
| CCSR floor 0.2% | WLS | 43,568 | 0.125 | 0.155 | 1.00 | $7,369 |
| CCSR floor 0.2% | Tweedie GLM | 43,568 | 0.125 | 0.247 | 1.02 | $4,531 |
| CCSR floor 0.2% | LightGBM (Tweedie) | 43,568 | 0.197 | 0.280 | 0.99 | $5,685 |
| CCSR floor 1% | WLS | 43,568 | 0.124 | 0.150 | 0.90 | $8,668 |
| CCSR floor 1% | Tweedie GLM | 43,568 | 0.157 | 0.251 | 0.97 | $5,868 |
| CCSR floor 1% | LightGBM (Tweedie) | 43,568 | 0.195 | 0.277 | 0.99 | $5,824 |
| no prior spending (F2) | WLS | 43,568 | 0.101 | 0.189 | 1.06 | $8,908 |
| no prior spending (F2) | Tweedie GLM | 43,568 | 0.013 | 0.176 | 1.24 | $5,223 |
| no prior spending (F2) | LightGBM (Tweedie) | 43,568 | 0.101 | 0.199 | 1.01 | $10,670 |
| social-risk features (F4) | WLS | 43,568 | 0.127 | 0.152 | 0.96 | $1,462 |
| social-risk features (F4) | Tweedie GLM | 43,568 | 0.122 | 0.245 | 1.03 | $1,236 |
| social-risk features (F4) | LightGBM (Tweedie) | 43,568 | 0.194 | 0.280 | 0.99 | $2,190 |
| decedents included | WLS | 43,963 | 0.128 | 0.151 | 0.94 | $6,773 |
| decedents included | Tweedie GLM | 43,963 | 0.129 | 0.243 | 1.01 | $3,896 |
| decedents included | LightGBM (Tweedie) | 43,963 | 0.193 | 0.277 | 0.99 | $4,529 |
| pandemic outcomes out | WLS | 28,208 | 0.124 | 0.146 | 0.97 | $9,652 |
| pandemic outcomes out | Tweedie GLM | 28,208 | 0.108 | 0.249 | 1.03 | $6,659 |
| pandemic outcomes out | LightGBM (Tweedie) | 28,208 | 0.211 | 0.287 | 0.99 | $7,998 |
| age 65 and over | WLS | 9,173 | 0.082 | 0.080 | 1.08 | $5,587 |
| age 65 and over | Tweedie GLM | 9,173 | 0.109 | 0.164 | 1.07 | $3,547 |
| age 65 and over | LightGBM (Tweedie) | 9,173 | 0.170 | 0.197 | 0.93 | $4,944 |
| under 65, private | WLS | 21,962 | 0.091 | 0.095 | 1.02 | $7,994 |
| under 65, private | Tweedie GLM | 21,962 | 0.091 | 0.178 | 1.07 | $3,941 |
| under 65, private | LightGBM (Tweedie) | 21,962 | 0.160 | 0.211 | 0.97 | $6,579 |


# Appendix tables


**Table A1.** Sample flow by MEPS panel.

| Panel | Years | In file | Both years, weight > 0 | Died in year 2 | Analysis sample | Any year-1 condition |
|---|---:|---:|---:|---:|---:|---:|
| 23 | 2018-2019 | 14,067 | 13,766 | 117 | 13,648 | 9,679 |
| 24 | 2019-2020 | 9,797 | 9,569 | 107 | 9,461 | 6,825 |
| 25 | 2020-2021 | 6,078 | 5,942 | 43 | 5,899 | 4,244 |
| 26 | 2021-2022 | 6,741 | 6,579 | 67 | 6,512 | 4,931 |
| 27 | 2022-2023 | 8,292 | 8,109 | 61 | 8,048 | 6,004 |

**Table A2.** Predictive ratio by decile of predicted spending.

| Decile | CANN | Elastic net | LightGBM (Tweedie) | LightGBM (squared error) | Neural network | Random forest | Tweedie GLM | Two-part | WLS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.00 | -0.86 | 1.02 | 1.58 | 1.00 | 0.75 | 1.00 | 0.98 | -1.91 |
| 2 | 0.86 | 0.18 | 1.04 | 1.54 | 1.02 | 1.02 | 0.89 | 1.25 | -0.06 |
| 3 | 1.01 | 0.68 | 1.00 | 1.10 | 0.92 | 1.06 | 1.01 | 1.41 | 0.61 |
| 4 | 1.10 | 0.93 | 1.08 | 1.03 | 1.01 | 1.02 | 1.11 | 1.42 | 0.93 |
| 5 | 1.03 | 1.24 | 0.95 | 0.99 | 0.92 | 1.06 | 1.04 | 1.37 | 1.29 |
| 6 | 0.99 | 1.44 | 0.93 | 1.01 | 0.97 | 0.94 | 0.98 | 1.07 | 1.37 |
| 7 | 1.05 | 1.27 | 1.03 | 0.96 | 0.97 | 1.06 | 1.07 | 0.97 | 1.36 |
| 8 | 0.99 | 1.17 | 0.99 | 0.95 | 0.96 | 1.03 | 0.99 | 0.88 | 1.34 |
| 9 | 1.05 | 0.96 | 1.04 | 1.01 | 1.03 | 1.03 | 1.05 | 0.89 | 1.15 |
| 10 | 0.98 | 0.93 | 0.98 | 0.97 | 0.97 | 0.97 | 0.97 | 0.99 | 0.93 |

**Table A3.** Coding sensitivity, full grid.

| Model | Pool | Share affected | Codes added | Total rise | $ per code |
|---|---:|---:|---:|---:|---:|
| WLS | prevalence | 5% | 1 | 0.26% | $394 |
| WLS | prevalence | 5% | 2 | 0.53% | $408 |
| WLS | prevalence | 10% | 1 | 0.52% | $395 |
| WLS | prevalence | 10% | 2 | 1.05% | $393 |
| WLS | prevalence | 25% | 1 | 1.16% | $351 |
| WLS | prevalence | 25% | 2 | 2.50% | $376 |
| WLS | chronic | 5% | 1 | 0.80% | $1,216 |
| WLS | chronic | 5% | 2 | 1.76% | $1,351 |
| WLS | chronic | 10% | 1 | 1.71% | $1,292 |
| WLS | chronic | 10% | 2 | 3.64% | $1,368 |
| WLS | chronic | 25% | 1 | 4.25% | $1,283 |
| WLS | chronic | 25% | 2 | 8.99% | $1,352 |
| Tweedie GLM | prevalence | 5% | 1 | 0.24% | $375 |
| Tweedie GLM | prevalence | 5% | 2 | 0.58% | $449 |
| Tweedie GLM | prevalence | 10% | 1 | 0.58% | $443 |
| Tweedie GLM | prevalence | 10% | 2 | 0.95% | $358 |
| Tweedie GLM | prevalence | 25% | 1 | 1.19% | $361 |
| Tweedie GLM | prevalence | 25% | 2 | 2.41% | $364 |
| Tweedie GLM | chronic | 5% | 1 | 0.56% | $854 |
| Tweedie GLM | chronic | 5% | 2 | 1.22% | $942 |
| Tweedie GLM | chronic | 10% | 1 | 1.16% | $885 |
| Tweedie GLM | chronic | 10% | 2 | 2.36% | $889 |
| Tweedie GLM | chronic | 25% | 1 | 2.83% | $861 |
| Tweedie GLM | chronic | 25% | 2 | 5.94% | $898 |
| Two-part | prevalence | 5% | 1 | 0.53% | $806 |
| Two-part | prevalence | 5% | 2 | 1.21% | $929 |
| Two-part | prevalence | 10% | 1 | 1.09% | $827 |
| Two-part | prevalence | 10% | 2 | 2.24% | $842 |
| Two-part | prevalence | 25% | 1 | 2.66% | $803 |
| Two-part | prevalence | 25% | 2 | 5.62% | $846 |
| Two-part | chronic | 5% | 1 | 0.85% | $1,299 |
| Two-part | chronic | 5% | 2 | 1.90% | $1,463 |
| Two-part | chronic | 10% | 1 | 1.73% | $1,307 |
| Two-part | chronic | 10% | 2 | 3.73% | $1,402 |
| Two-part | chronic | 25% | 1 | 4.28% | $1,294 |
| Two-part | chronic | 25% | 2 | 9.26% | $1,393 |
| LightGBM (Tweedie) | prevalence | 5% | 1 | 0.40% | $602 |
| LightGBM (Tweedie) | prevalence | 5% | 2 | 0.80% | $612 |
| LightGBM (Tweedie) | prevalence | 10% | 1 | 0.81% | $612 |
| LightGBM (Tweedie) | prevalence | 10% | 2 | 1.59% | $596 |
| LightGBM (Tweedie) | prevalence | 25% | 1 | 1.96% | $590 |
| LightGBM (Tweedie) | prevalence | 25% | 2 | 4.09% | $612 |
| LightGBM (Tweedie) | chronic | 5% | 1 | 0.47% | $718 |
| LightGBM (Tweedie) | chronic | 5% | 2 | 0.98% | $750 |
| LightGBM (Tweedie) | chronic | 10% | 1 | 0.92% | $693 |
| LightGBM (Tweedie) | chronic | 10% | 2 | 2.02% | $755 |
| LightGBM (Tweedie) | chronic | 25% | 1 | 2.37% | $713 |
| LightGBM (Tweedie) | chronic | 25% | 2 | 5.00% | $749 |
| LightGBM (squared error) | prevalence | 5% | 1 | 0.47% | $708 |
| LightGBM (squared error) | prevalence | 5% | 2 | 0.96% | $739 |
| LightGBM (squared error) | prevalence | 10% | 1 | 0.95% | $719 |
| LightGBM (squared error) | prevalence | 10% | 2 | 1.92% | $717 |
| LightGBM (squared error) | prevalence | 25% | 1 | 2.26% | $679 |
| LightGBM (squared error) | prevalence | 25% | 2 | 4.87% | $729 |
| LightGBM (squared error) | chronic | 5% | 1 | 0.47% | $721 |
| LightGBM (squared error) | chronic | 5% | 2 | 1.02% | $784 |
| LightGBM (squared error) | chronic | 10% | 1 | 1.00% | $755 |
| LightGBM (squared error) | chronic | 10% | 2 | 2.09% | $781 |
| LightGBM (squared error) | chronic | 25% | 1 | 2.46% | $742 |
| LightGBM (squared error) | chronic | 25% | 2 | 5.29% | $792 |
| CANN | prevalence | 5% | 1 | 0.27% | $408 |
| CANN | prevalence | 5% | 2 | 0.65% | $501 |
| CANN | prevalence | 10% | 1 | 0.63% | $482 |
| CANN | prevalence | 10% | 2 | 1.05% | $398 |
| CANN | prevalence | 25% | 1 | 1.30% | $396 |
| CANN | prevalence | 25% | 2 | 2.65% | $400 |
| CANN | chronic | 5% | 1 | 0.59% | $901 |
| CANN | chronic | 5% | 2 | 1.31% | $1,013 |
| CANN | chronic | 10% | 1 | 1.25% | $953 |
| CANN | chronic | 10% | 2 | 2.52% | $952 |
| CANN | chronic | 25% | 1 | 3.03% | $920 |
| CANN | chronic | 25% | 2 | 6.36% | $961 |
| Constrained WLS | prevalence | 5% | 1 | 0.27% | $400 |
| Constrained WLS | prevalence | 5% | 2 | 0.51% | $376 |
| Constrained WLS | prevalence | 10% | 1 | 0.46% | $333 |
| Constrained WLS | prevalence | 10% | 2 | 1.06% | $383 |
| Constrained WLS | prevalence | 25% | 1 | 0.96% | $280 |
| Constrained WLS | prevalence | 25% | 2 | 2.38% | $344 |
| Constrained WLS | chronic | 5% | 1 | 0.66% | $970 |
| Constrained WLS | chronic | 5% | 2 | 1.64% | $1,216 |
| Constrained WLS | chronic | 10% | 1 | 1.54% | $1,121 |
| Constrained WLS | chronic | 10% | 2 | 3.40% | $1,228 |
| Constrained WLS | chronic | 25% | 1 | 3.82% | $1,112 |
| Constrained WLS | chronic | 25% | 2 | 8.26% | $1,194 |

**Table A4.** Actuarial sense checks on the SHAP values.

*Spearman correlation between a feature's value and its SHAP contribution across held-out persons, by fold. Positive means predicted spending rises with the feature.*

| Feature | Mean | Min across folds | Max across folds |
|---|---:|---:|---:|
| log_spend_y1 | 0.875 | 0.859 | 0.903 |
| n_body_systems | 0.863 | 0.842 | 0.889 |
| n_conditions | 0.896 | 0.873 | 0.915 |

**Table A5.** Hyperparameters chosen by inner validation, across the 15 outer folds.

| model | parameter | chosen (count of folds) |
|---|---:|---:|
| tweedie | alpha | 1.0 (6); 10.0 (4); 0.01 (3); 0.1 (2) |
| twopart | C | 0.1 (15) |
| twopart | alpha | 1.0 (11); 0.1 (3); 10.0 (1) |
| enet | alpha | 1.0 (15) |
| enet | l1_ratio | 0.8 (15) |
| gbm | num_leaves | 63 (8); 31 (4); 15 (3) |
| gbm | min_child_samples | 100 (11); 50 (4) |
| gbm | n_estimators | 152 (2); 165 (1); 119 (1); 145 (1); 121 (1); 111 (1); 136 (1); 170 (1); 212 (1); 215 (1); 132 (1); 113 (1); 102 (1); 104 (1) |
| gbm_mse | num_leaves | 31 (7); 63 (6); 15 (2) |
| gbm_mse | min_child_samples | 100 (8); 50 (7) |
| gbm_mse | n_estimators | 88 (1); 124 (1); 98 (1); 162 (1); 62 (1); 73 (1); 76 (1); 354 (1); 123 (1); 127 (1); 126 (1); 484 (1); 187 (1); 83 (1); 68 (1) |
| mlp | lr | 0.003 (7); 0.01 (6); 0.001 (2) |
| mlp | hidden | (64, 32) (7); (32,) (6); (128, 64) (2) |
| mlp | epochs | 32 (4); 29 (4); 28 (2); 33 (2); 38 (1); 30 (1); 35 (1) |
| cann | lr | 0.001 (11); 0.01 (2); 0.003 (2) |
| cann | hidden | (64, 32) (11); (32,) (2); (128, 64) (2) |
| cann | epochs | 25 (10); 26 (5) |

**Table A6.** Year-1 CCSR categories entering the feature sets.

*Categories held by at least 0.5% of the sample in year 1. MEPS body-system placeholder codes (XXX000) are excluded.*

| CCSR | Description | Prevalence |
|---|---:|---:|
| CIR007 | Essential hypertension | 22.13% |
| END010 | Disorders of lipid metabolism | 16.78% |
| MUS010 | Musculoskeletal pain, not low back pain | 9.59% |
| END005 | Diabetes mellitus, Type 2 | 9.53% |
| END002 | Diabetes mellitus without complication | 9.53% |
| MBD005 | Anxiety and fear-related disorders | 8.03% |
| MBD002 | Depressive disorders | 7.06% |
| END001 | Thyroid disorders | 7.01% |
| MUS006 | Osteoarthritis | 6.68% |
| RSP009 | Asthma | 6.32% |
| DIG004 | Esophageal disorders | 5.89% |
| NVS016 | Sleep wake disorders | 5.41% |
| SKN007 | Other specified and unspecified skin disorders | 4.94% |
| FAC016 | Exposure, encounters, screening or contact with infectious disease | 4.78% |
| MUS011 | Spondylopathies/spondyloarthropathy (including infective) | 4.48% |
| RSP006 | Other specified upper respiratory infections | 4.24% |
| INJ031 | Allergic reactions | 3.92% |
| SYM016 | Other general signs and symptoms | 3.76% |
| RSP007 | Other specified and unspecified upper respiratory disease | 3.67% |
| MBD014 | Neurodevelopmental disorders | 3.11% |
| INJ067 | Allergic reactions, subsequent encounter | 3.11% |
| INJ064 | Other unspecified injuries, subsequent encounter | 2.95% |
| CIR011 | Coronary atherosclerosis and other heart disease | 2.93% |
| MUS025 | Other specified connective tissue disease | 2.93% |
| INF012 | Coronavirus disease – 2019 (COVID-19) | 2.90% |
| GEN004 | Urinary tract infections | 2.69% |
| NVS010 | Headache; including migraine | 2.68% |
| SYM013 | Respiratory signs and symptoms | 2.67% |
| RSP003 | Influenza | 2.56% |
| INF008 | Viral infection | 2.53% |
| SKN002 | Other specified inflammatory condition of skin | 2.44% |
| EYE002 | Cataract and other lens disorders | 2.31% |
| SYM006 | Abdominal pain and other digestive/abdomen signs and symptoms | 2.30% |
| RSP001 | Sinusitis | 2.29% |
| MUS038 | Low back pain | 2.25% |
| CIR017 | Cardiac dysrhythmias | 2.24% |
| EAR001 | Otitis media | 2.07% |
| DIG025 | Other specified and unspecified gastrointestinal disorders | 2.07% |
| END007 | Nutritional deficiencies | 2.04% |
| FAC014 | Medical examination/evaluation | 2.02% |
| INF003 | Bacterial infections | 2.01% |
| INJ061 | Sprains and strains, subsequent encounter | 2.01% |
| SYM010 | Nervous system signs and symptoms | 2.01% |
| FAC003 | Encounter for observation and examination for conditions ruled out (excludes infectious disease, neoplasm, mental disorders) | 1.95% |
| SYM014 | Skin/Subcutaneous signs and symptoms | 1.94% |
| DIG002 | Disorders of teeth and gingiva | 1.93% |
| FAC012 | Other specified encounters and counseling | 1.91% |
| SYM017 | Abnormal findings without diagnosis | 1.89% |
| MBD007 | Trauma- and stressor-related disorders | 1.76% |
| FAC009 | Implant, device or graft related encounter | 1.75% |
| EYE003 | Glaucoma | 1.73% |
| NEO028 | Skin cancers - all other types | 1.70% |
| SYM012 | Circulatory signs and symptoms | 1.69% |
| FAC013 | Contraceptive and procreative management | 1.68% |
| EYE009 | Refractive error | 1.67% |
| RSP008 | Chronic obstructive pulmonary disease and bronchiectasis | 1.66% |
| MUS003 | Rheumatoid arthritis and related disease | 1.62% |
| CIR009 | Acute myocardial infarction | 1.54% |
| MUS007 | Other specified joint disorders | 1.50% |
| EYE005 | Retinal and vitreous conditions | 1.42% |
| FAC025 | Other specified status | 1.38% |
| INF004 | Fungal infections | 1.38% |
| RSP005 | Acute bronchitis | 1.34% |
| PRG029 | Uncomplicated pregnancy, delivery or puerperium | 1.30% |
| EAR006 | Other specified and unspecified disorders of the ear | 1.29% |
| DIG008 | Other specified and unspecified disorders of stomach and duodenum | 1.29% |
| END011 | Fluid and electrolyte disorders | 1.29% |
| NVS012 | Transient cerebral ischemia | 1.28% |
| MUS033 | Gout | 1.27% |
| NVS019 | Nervous system pain and pain syndromes | 1.26% |
| NVS017 | Nerve and nerve root disorders | 1.22% |
| SKN001 | Skin and subcutaneous tissue infections | 1.21% |
| NEO073 | Benign neoplasms | 1.21% |
| MUS009 | Tendon and synovial disorders | 1.18% |
| FAC008 | Neoplasm-related encounters | 1.18% |
| EAR004 | Hearing loss | 1.15% |
| EYE010 | Blindness and vision defects | 1.12% |
| SYM015 | General sensation/perception signs and symptoms | 1.11% |
| EYE008 | Oculofacial plastics and orbital conditions | 1.09% |
| GEN012 | Hyperplasia of prostate | 1.07% |
| SYM004 | Nausea and vomiting | 1.07% |
| RSP002 | Pneumonia (except that caused by tuberculosis) | 1.04% |
| GEN006 | Other specified and unspecified diseases of kidney and ureters | 0.98% |
| NVS015 | Polyneuropathies | 0.98% |
| EYE012 | Other specified eye disorders | 0.96% |
| END015 | Other specified and unspecified endocrine disorders | 0.93% |
| CIR030 | Aortic and peripheral arterial embolism or thrombosis | 0.93% |
| MUS028 | Other specified bone disease and musculoskeletal deformities | 0.92% |
| MBD003 | Bipolar and related disorders | 0.92% |
| NVS009 | Epilepsy; convulsions | 0.92% |
| GEN023 | Menopausal disorders | 0.91% |
| NEO030 | Breast cancer - all other types | 0.90% |
| GEN005 | Calculus of urinary tract | 0.85% |
| EYE001 | Cornea and external disease | 0.85% |
| NEO072 | Neoplasms of unspecified nature or uncertain behavior | 0.82% |
| INJ042 | Fracture of lower limb (except hip), subsequent encounter | 0.81% |
| MUS026 | Muscle disorders | 0.80% |
| INJ041 | Fracture of the upper limb, subsequent encounter | 0.80% |
| CIR012 | Nonspecific chest pain | 0.79% |
| SYM011 | Genitourinary signs and symptoms | 0.77% |
| SYM002 | Fever | 0.76% |
| MBD013 | Miscellaneous mental and behavioral disorders/conditions | 0.76% |
| CIR015 | Other and ill-defined heart disease | 0.76% |
| DIG010 | Abdominal hernia | 0.76% |
| FAC010 | Other aftercare encounter | 0.76% |
| RSP016 | Other specified and unspecified lower respiratory disease | 0.75% |
| GEN025 | Other specified female genital disorders | 0.73% |
| GEN016 | Other specified male genital disorders | 0.72% |
| NVS006 | Other nervous system disorders (often hereditary or degenerative) | 0.72% |
| GEN017 | Nonmalignant breast conditions | 0.68% |
| MUS013 | Osteoporosis | 0.67% |
| GEN008 | Urinary incontinence | 0.66% |
| INJ027 | Other unspecified injury | 0.66% |
| BLD003 | Aplastic anemia | 0.65% |
| DIG001 | Intestinal infection | 0.64% |
| NEO039 | Male reproductive system cancers - prostate | 0.63% |
| GEN007 | Other specified and unspecified diseases of bladder and urethra | 0.62% |
| INF009 | Parasitic, other specified and unspecified infections | 0.59% |
| NEO025 | Skin cancers - melanoma | 0.58% |
| DEN001 | Any dental condition including traumatic injury | 0.57% |
| INJ049 | Open wounds to limbs, subsequent encounter | 0.56% |
| DEN002 | Nontraumatic dental conditions | 0.53% |
| SKN005 | Contact dermatitis | 0.53% |
| DIG019 | Other specified and unspecified liver disease | 0.51% |
