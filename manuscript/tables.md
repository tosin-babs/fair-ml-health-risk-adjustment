# Tables

*Generated from `output/tables/*.csv` by `python/make_tables.py`. Spending is total expenditure from all payers in 2024 dollars. All estimates are weighted by the MEPS longitudinal weight; model metrics are out of fold from cross-validation with primary sampling units kept whole.*


**Table 1.** Sample characteristics and year-2 spending.

*Weighted estimates; standard errors Taylor-linearized for the stratified cluster design.*

| Characteristic | Estimate | SE | n |
|---|---:|---:|---:|
| Persons | 43,963.00 |  | 43,963 |
| Weighted population, millions a year | 325.79 |  | 43,963 |
| Age in year 1, mean | 38.92 | 0.20 | 43,963 |
| Age 65 and over, % | 16.47 | 0.27 | 43,963 |
| Female, % | 50.91 | 0.24 | 43,963 |
| Hispanic, % | 19.01 | 0.60 | 43,963 |
| Non-Hispanic White, % | 58.64 | 0.64 | 43,963 |
| Non-Hispanic Black, % | 12.44 | 0.42 | 43,963 |
| Non-Hispanic Asian, % | 6.15 | 0.30 | 43,963 |
| Non-Hispanic other or multiple, % | 3.76 | 0.19 | 43,963 |
| Below 100% FPL, % | 12.36 | 0.34 | 43,963 |
| 100-124% FPL, % | 4.07 | 0.20 | 43,963 |
| 125-199% FPL, % | 12.28 | 0.34 | 43,963 |
| 200-399% FPL, % | 28.91 | 0.48 | 43,963 |
| 400% FPL or more, % | 42.38 | 0.58 | 43,963 |
| Any private coverage, % | 65.82 | 0.50 | 43,963 |
| Public coverage only, % | 27.49 | 0.44 | 43,963 |
| Uninsured all year, % | 6.68 | 0.23 | 43,963 |
| Fair or poor perceived health, % | 9.78 | 0.21 | 43,789 |
| Needs help with ADL or IADL, % | 2.82 | 0.10 | 43,963 |
| Any year-1 condition (CCSR category), % | 70.04 | 0.39 | 43,963 |
| Year-1 CCSR categories, mean | 2.80 | 0.03 | 43,963 |
| Any mental health condition, % | 15.81 | 0.28 | 43,963 |
| Year-1 spending, mean $ | 7,389.49 | 158.11 | 43,963 |
| Year-2 spending, mean $ | 7,792.21 | 146.60 | 43,963 |
| Year-2 spending of zero, % | 14.94 | 0.32 | 43,963 |
| Year-2 spending, median $ | 1,497.40 |  | 43,963 |
| Year-2 spending, 90th percentile $ | 18,542.65 |  | 43,963 |
| Year-2 spending, 99th percentile $ | 95,758.64 |  | 43,963 |
| Share of year-2 spending by the top 5% of spenders, % | 50.75 |  | 43,963 |

**Table 2.** Out-of-sample accuracy by model, primary feature set (F3).

*Five survey folds, three repeats. Each metric is the mean over repeats of a single fitted model's out-of-fold value, which is what one deployed model achieves; the ensemble column averages the three repeats' predictions first. Intervals from a Rao-Wu rescaled bootstrap over PSUs within strata (200 resamples). CPM is Cumming's prediction measure. Top-10% capture is the weighted share of the true top decile of spenders placed in the model's top decile. The last column is the weighted share of people with a negative predicted payment.*

| Model | R² | 95% low | 95% high | R², 3-fit ensemble | CPM | MAE | PR bottom decile | PR top decile | Top-10% capture | Negative predictions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| WLS | 0.123 | 0.102 | 0.155 | 0.124 | 0.147 | $8,337 | -1.94 | 0.94 | 44% | 15.2% |
| Payment-form WLS (non-negative) | 0.107 | 0.085 | 0.142 | 0.108 | 0.181 | $8,003 | 0.08 | 1.01 | 41% | 0.0% |
| Tweedie GLM | 0.126 | 0.095 | 0.166 | 0.130 | 0.242 | $7,403 | 0.98 | 1.00 | 47% | 0.0% |
| Two-part | 0.093 | 0.073 | 0.121 | 0.099 | 0.201 | $7,811 | 1.00 | 0.96 | 43% | 0.0% |
| Elastic net | 0.116 | 0.093 | 0.151 | 0.117 | 0.195 | $7,865 | -0.83 | 0.94 | 42% | 12.3% |
| LightGBM (Tweedie) | 0.187 | 0.156 | 0.242 | 0.190 | 0.276 | $7,077 | 1.03 | 0.99 | 48% | 0.0% |
| LightGBM (squared error) | 0.184 | 0.156 | 0.235 | 0.189 | 0.264 | $7,193 | 1.43 | 1.00 | 47% | 0.0% |
| Random forest | 0.190 | 0.161 | 0.238 | 0.191 | 0.267 | $7,157 | 0.75 | 0.98 | 48% | 0.0% |
| Neural network | 0.105 | 0.085 | 0.131 | 0.135 | 0.233 | $7,494 | 0.98 | 1.05 | 42% | 0.0% |
| CANN | 0.124 | 0.093 | 0.164 | 0.129 | 0.242 | $7,405 | 0.95 | 1.01 | 47% | 0.0% |

**Table 2b.** Paired differences from WLS in out-of-sample accuracy.

*Every model is evaluated on the same 500 resamples of PSUs within strata, so the interval is for the difference itself. The last column is the share of resamples in which the model's R² does not exceed WLS's. Negative MAE differences mean smaller errors than WLS.*

| Features | Model | ΔR² | 95% low | 95% high | Share ΔR² ≤ 0 | ΔCPM | ΔMAE | 95% low | 95% high |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| F3 | Payment-form WLS (non-negative) | -0.016 | -0.020 | -0.010 | 1.000 | 0.034 | -$334 | -$372 | -$295 |
| F3 | Tweedie GLM | 0.003 | -0.018 | 0.025 | 0.532 | 0.096 | -$934 | -$995 | -$879 |
| F3 | Two-part | -0.030 | -0.042 | -0.020 | 1.000 | 0.054 | -$526 | -$578 | -$469 |
| F3 | Elastic net | -0.007 | -0.011 | -0.001 | 0.988 | 0.048 | -$472 | -$502 | -$442 |
| F3 | LightGBM (Tweedie) | 0.064 | 0.047 | 0.091 | 0.000 | 0.129 | -$1,259 | -$1,327 | -$1,196 |
| F3 | LightGBM (squared error) | 0.061 | 0.044 | 0.083 | 0.000 | 0.117 | -$1,143 | -$1,213 | -$1,077 |
| F3 | Random forest | 0.067 | 0.053 | 0.085 | 0.000 | 0.121 | -$1,180 | -$1,243 | -$1,119 |
| F3 | Neural network | -0.018 | -0.038 | -0.004 | 0.998 | 0.086 | -$843 | -$894 | -$792 |
| F3 | CANN | 0.001 | -0.020 | 0.023 | 0.544 | 0.095 | -$932 | -$992 | -$877 |
| F1 | LightGBM (Tweedie) | -0.001 | -0.003 | 0.000 | 0.950 | -0.004 | $37 | $22 | $53 |
| F1 | Payment-form WLS (non-negative) | -0.000 | -0.001 | 0.000 | 0.920 | -0.000 | $4 | -$8 | $17 |
| F2 | LightGBM (Tweedie) | -0.002 | -0.006 | 0.003 | 0.772 | 0.012 | -$116 | -$151 | -$85 |
| F2 | Payment-form WLS (non-negative) | -0.005 | -0.008 | -0.001 | 0.994 | 0.004 | -$39 | -$70 | -$12 |
| F4 | LightGBM (Tweedie) | 0.061 | 0.046 | 0.085 | 0.000 | 0.128 | -$1,247 | -$1,312 | -$1,181 |
| F4 | Payment-form WLS (non-negative) | -0.014 | -0.018 | -0.008 | 1.000 | 0.033 | -$323 | -$360 | -$283 |

**Table 3.** Out-of-sample R² by feature set.

*F1 demographics; F2 adds year-1 CCSR condition flags; F3 adds year-1 use and spending; F4 adds income, insurance, perceived health and functional help.*

| Model | F1 | F2 | F3 | F4 |
|---|---:|---:|---:|---:|
| LightGBM (Tweedie) | 0.039 | 0.101 | 0.187 | 0.189 |
| Payment-form WLS (non-negative) | 0.040 | 0.098 | 0.107 | 0.113 |
| Tweedie GLM | 0.041 | -0.001 | 0.126 | 0.123 |
| WLS | 0.040 | 0.103 | 0.123 | 0.127 |

**Table 4.** Net compensation by group: predicted minus observed spending, $ per person-year (95% interval), and predictive ratio.

*Negative values mean the model pays less for the group than the group costs. Means over repeats of single fits; Rao-Wu PSU bootstrap intervals. Race and ethnicity are used for evaluation only and never as model inputs.*

| Group | WLS | Payment-form WLS (non-negative) | Tweedie GLM | LightGBM (Tweedie) | CANN |
|---|---:|---:|---:|---:|---:|
| Age 65 and over | -$1 (-$903, $686); PR 1.00 | $189 (-$681, $882); PR 1.01 | $562 (-$309, $1,297); PR 1.03 | -$467 (-$1,298, $176); PR 0.97 | $645 (-$232, $1,371); PR 1.04 |
| Died in year 2 | -$43,607 (-$54,318, -$32,699); PR 0.34 | -$44,316 (-$55,359, -$33,097); PR 0.33 | -$42,318 (-$52,745, -$31,763); PR 0.36 | -$42,541 (-$53,195, -$31,878); PR 0.36 | -$42,174 (-$52,655, -$31,638); PR 0.36 |
| Hispanic | -$271 (-$1,034, $336); PR 0.94 | $509 (-$297, $1,114); PR 1.11 | $12 (-$777, $609); PR 1.00 | $91 (-$670, $697); PR 1.02 | -$0 (-$794, $594); PR 1.00 |
| Income below 200% FPL | -$92 (-$619, $336); PR 0.99 | $675 ($124, $1,113); PR 1.09 | $133 (-$391, $565); PR 1.02 | -$5 (-$514, $417); PR 1.00 | $149 (-$377, $580); PR 1.02 |
| Mental health condition | $193 (-$474, $772); PR 1.01 | $312 (-$363, $912); PR 1.02 | $675 (-$11, $1,367); PR 1.05 | $538 (-$38, $1,128); PR 1.04 | $724 ($41, $1,418); PR 1.05 |
| Needs ADL or IADL help | -$7,747 (-$12,747, -$4,246); PR 0.75 | -$9,293 (-$14,437, -$5,593); PR 0.70 | -$4,762 (-$9,353, -$1,287); PR 0.85 | -$5,721 (-$10,502, -$2,458); PR 0.82 | -$4,545 (-$9,069, -$1,057); PR 0.85 |
| Non-Hispanic Asian | $179 (-$399, $735); PR 1.04 | $780 ($225, $1,315); PR 1.16 | $387 (-$152, $867); PR 1.08 | $310 (-$237, $838); PR 1.06 | $370 (-$172, $850); PR 1.08 |
| Non-Hispanic Black | -$202 (-$896, $485); PR 0.97 | $413 (-$230, $1,096); PR 1.06 | -$15 (-$680, $629); PR 1.00 | -$8 (-$645, $603); PR 1.00 | -$16 (-$681, $620); PR 1.00 |
| Uninsured all year | -$126 (-$696, $373); PR 0.95 | $657 ($102, $1,197); PR 1.27 | $425 (-$118, $925); PR 1.18 | $549 ($40, $1,024); PR 1.23 | $408 (-$134, $906); PR 1.17 |

**Table 4b.** Difference in net compensation from WLS, by group (95% paired interval).

*Same PSU resamples for both models, so the interval is for the difference itself. Positive values mean the model pays the group more than WLS does.*

| Group | Payment-form WLS (non-negative) | Tweedie GLM | LightGBM (Tweedie) | CANN |
|---|---:|---:|---:|---:|
| Age 65 and over | $191 ($75, $283) | $563 ($314, $871) | -$465 (-$680, -$272) | $647 ($395, $956) |
| Died in year 2 | -$709 (-$1,403, -$157) | $1,289 ($224, $2,854) | $1,066 (-$802, $3,244) | $1,433 ($323, $3,068) |
| Hispanic | $780 ($689, $863) | $284 ($204, $370) | $362 ($239, $497) | $271 ($189, $360) |
| Income below 200% FPL | $767 ($681, $868) | $224 ($103, $374) | $87 (-$34, $253) | $241 ($116, $398) |
| Mental health condition | $118 ($17, $241) | $481 ($194, $795) | $344 ($119, $600) | $530 ($240, $857) |
| Needs ADL or IADL help | -$1,546 (-$1,949, -$1,188) | $2,985 ($2,227, $3,834) | $2,025 ($1,354, $2,925) | $3,201 ($2,439, $4,064) |
| Non-Hispanic Asian | $602 ($425, $757) | $209 ($40, $392) | $131 (-$54, $331) | $191 ($19, $373) |
| Non-Hispanic Black | $616 ($489, $739) | $187 ($42, $359) | $194 (-$9, $391) | $187 ($38, $365) |
| Uninsured all year | $783 ($609, $930) | $552 ($406, $688) | $675 ($491, $862) | $534 ($388, $670) |

**Table 5.** The accuracy-fairness frontier.

*Penalized and constrained estimators target the three groups marked in the text; the other groups show spillover. Stacked LightGBM enters a cross-fitted LightGBM score with the features into the fair regression. All three repeats of the survey folds; metrics are means over repeats of out-of-fold values, with Rao-Wu PSU bootstrap intervals for R² and the largest target-group gap.*

| Method | λ | R² | low | high | Max |NC| target | low | high | Income below 200% FPL | Hispanic | Non-Hispanic Black | Non-Hispanic Asian | Needs ADL or IADL help | Mental health condition | Age 65 and over | Uninsured all year | Died in year 2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Penalized WLS | 0 | 0.123 | 0.102 | 0.155 | $7,747 | $4,246 | $12,747 | -$92 | -$271 | -$202 | $179 | -$7,747 | $193 | -$1 | -$126 | -$43,607 |
| Penalized WLS | 0.1 | 0.119 | 0.099 | 0.150 | $4,504 | $1,240 | $9,351 | $548 | -$89 | $213 | $383 | -$4,504 | $1,053 | $1,690 | -$95 | -$40,148 |
| Penalized WLS | 0.3 | 0.114 | 0.094 | 0.142 | $2,817 | $726 | $7,644 | $664 | -$221 | $232 | $316 | -$2,817 | $858 | $2,554 | -$302 | -$38,145 |
| Penalized WLS | 1 | 0.107 | 0.088 | 0.132 | $1,574 | $500 | $6,366 | $487 | -$605 | -$19 | $31 | -$1,574 | $421 | $3,115 | -$758 | -$36,610 |
| Penalized WLS | 3 | 0.103 | 0.085 | 0.127 | $1,044 | $357 | $5,843 | $252 | -$950 | -$297 | -$243 | -$1,044 | $185 | $3,293 | -$1,148 | -$35,970 |
| Penalized WLS | 10 | 0.100 | 0.083 | 0.124 | $825 | $267 | $5,619 | $101 | -$1,153 | -$469 | -$407 | -$825 | $85 | $3,345 | -$1,375 | -$35,715 |
| Penalized WLS | 30 | 0.100 | 0.082 | 0.124 | $758 | $235 | $5,550 | $47 | -$1,224 | -$531 | -$466 | -$758 | $55 | $3,357 | -$1,455 | -$35,639 |
| Penalized WLS | 100 | 0.099 | 0.082 | 0.123 | $734 | $224 | $5,525 | $26 | -$1,251 | -$554 | -$488 | -$734 | $44 | $3,361 | -$1,485 | -$35,612 |
| Constrained WLS | constrained | 0.099 | 0.082 | 0.123 | $724 | $220 | $5,515 | $17 | -$1,263 | -$565 | -$497 | -$724 | $40 | $3,362 | -$1,498 | -$35,600 |
| Stacked LightGBM | 0 | 0.182 | 0.152 | 0.234 | $4,601 | $1,367 | $9,257 | $14 | -$11 | -$10 | $328 | -$4,601 | $192 | $94 | $313 | -$41,025 |
| Stacked LightGBM | 0.1 | 0.180 | 0.152 | 0.231 | $2,415 | $620 | $6,942 | $424 | $111 | $260 | $464 | -$2,415 | $747 | $1,196 | $340 | -$38,712 |
| Stacked LightGBM | 0.3 | 0.178 | 0.150 | 0.227 | $1,277 | $623 | $5,881 | $487 | $21 | $265 | $416 | -$1,277 | $615 | $1,749 | $204 | -$37,382 |
| Stacked LightGBM | 1 | 0.175 | 0.147 | 0.223 | $536 | $480 | $5,111 | $355 | -$239 | $87 | $220 | -$437 | $327 | $2,100 | -$101 | -$36,367 |
| Stacked LightGBM | 3 | 0.173 | 0.145 | 0.221 | $317 | $439 | $4,786 | $189 | -$473 | -$105 | $33 | -$78 | $173 | $2,205 | -$364 | -$35,945 |
| Stacked LightGBM | 10 | 0.172 | 0.144 | 0.220 | $295 | $391 | $4,652 | $84 | -$610 | -$224 | -$79 | $71 | $108 | $2,233 | -$517 | -$35,778 |
| Stacked LightGBM | 30 | 0.172 | 0.144 | 0.219 | $288 | $386 | $4,611 | $46 | -$659 | -$267 | -$119 | $117 | $89 | $2,239 | -$571 | -$35,728 |
| Stacked LightGBM | 100 | 0.171 | 0.143 | 0.219 | $293 | $372 | $4,596 | $32 | -$677 | -$283 | -$134 | $133 | $82 | $2,241 | -$592 | -$35,711 |
| Stacked LightGBM, constrained | constrained | 0.171 | 0.143 | 0.219 | $296 | $365 | $4,590 | $26 | -$685 | -$290 | -$141 | $140 | $79 | $2,242 | -$601 | -$35,703 |
| LightGBM, decile recalibration | 0 | 0.186 | 0.155 | 0.241 | $5,608 | $2,354 | $10,391 | $21 | $111 | $18 | $346 | -$5,608 | $597 | -$430 | $556 | -$42,449 |

**Table 5b.** Constrained estimators fold by fold: net compensation in the training data and out of fold.

*In the training data the constraint holds exactly for the target groups. Out of fold it does not; the pooled values in Table 5 average across these folds.*

| Method | Repeat | Fold | in: Income below 200% FPL | out: Income below 200% FPL | in: Mental health condition | out: Mental health condition | in: Needs ADL or IADL help | out: Needs ADL or IADL help | in: Age 65 and over | out: Age 65 and over |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Constrained WLS | 0 | 0 | $0 | $481 | -$0 | -$1,014 | -$0 | $2,755 | $3,991 | $3,040 |
| Stacked LightGBM, constrained | 0 | 0 | $0 | $528 | -$0 | -$723 | -$0 | $2,272 | $2,897 | $1,604 |
| Constrained WLS | 0 | 1 | $0 | $439 | -$0 | $1,174 | -$0 | -$331 | $3,486 | $2,240 |
| Stacked LightGBM, constrained | 0 | 1 | $0 | $557 | $0 | $1,598 | $0 | $4,589 | $2,202 | $2,142 |
| Constrained WLS | 0 | 2 | -$0 | -$1,538 | -$0 | -$378 | -$0 | -$8,888 | $2,548 | $1,467 |
| Stacked LightGBM, constrained | 0 | 2 | $0 | -$1,639 | $0 | -$583 | -$0 | -$9,294 | $1,202 | $317 |
| Constrained WLS | 0 | 3 | $0 | $803 | -$0 | $199 | -$0 | $1,853 | $3,393 | $4,150 |
| Stacked LightGBM, constrained | 0 | 3 | $0 | $592 | -$0 | -$140 | -$0 | $2,752 | $2,293 | $3,093 |
| Constrained WLS | 0 | 4 | -$0 | $51 | -$0 | $234 | -$0 | $1,145 | $3,460 | $6,334 |
| Stacked LightGBM, constrained | 0 | 4 | $0 | $273 | $0 | $632 | -$0 | $940 | $2,097 | $4,538 |
| Constrained WLS | 1 | 0 | -$0 | -$1,237 | -$0 | -$1,637 | -$0 | -$12,261 | $2,773 | $2,972 |
| Stacked LightGBM, constrained | 1 | 0 | -$0 | -$1,214 | $0 | -$1,349 | -$0 | -$11,664 | $1,361 | $1,300 |
| Constrained WLS | 1 | 1 | $0 | $208 | -$0 | $681 | -$0 | $5,057 | $3,944 | $2,670 |
| Stacked LightGBM, constrained | 1 | 1 | -$0 | $54 | -$0 | $192 | -$0 | $3,623 | $2,346 | $1,301 |
| Constrained WLS | 1 | 2 | -$0 | $93 | -$0 | $211 | $0 | $259 | $3,370 | $3,530 |
| Stacked LightGBM, constrained | 1 | 2 | $0 | $515 | -$0 | $565 | -$0 | $6,534 | $2,173 | $2,971 |
| Constrained WLS | 1 | 3 | $0 | $99 | -$0 | -$1,111 | -$0 | $840 | $3,575 | $3,516 |
| Stacked LightGBM, constrained | 1 | 3 | $0 | $91 | -$0 | -$1,084 | $0 | $2,348 | $2,313 | $3,332 |
| Constrained WLS | 1 | 4 | $0 | $870 | -$0 | $2,123 | -$0 | $1,728 | $3,129 | $3,952 |
| Stacked LightGBM, constrained | 1 | 4 | -$0 | $728 | -$0 | $2,102 | -$0 | $1,654 | $1,964 | $2,482 |
| Constrained WLS | 2 | 0 | -$0 | -$1,576 | -$0 | -$383 | -$0 | -$13,043 | $2,468 | -$128 |
| Stacked LightGBM, constrained | 2 | 0 | $0 | -$1,277 | -$0 | $124 | -$0 | -$10,379 | $1,292 | -$902 |
| Constrained WLS | 2 | 1 | $0 | $678 | -$0 | $491 | -$0 | $3,586 | $3,694 | $5,296 |
| Stacked LightGBM, constrained | 2 | 1 | $0 | $742 | -$0 | $917 | -$0 | $4,804 | $2,443 | $4,203 |
| Constrained WLS | 2 | 2 | $0 | $959 | -$0 | -$241 | $0 | $4,952 | $3,782 | $3,844 |
| Stacked LightGBM, constrained | 2 | 2 | -$0 | $984 | -$0 | -$22 | -$0 | $3,572 | $2,331 | $2,866 |
| Constrained WLS | 2 | 3 | -$0 | -$237 | -$0 | -$1,444 | -$0 | -$1,903 | $3,284 | $3,547 |
| Stacked LightGBM, constrained | 2 | 3 | -$0 | -$441 | -$0 | -$1,769 | $0 | -$406 | $1,998 | $1,868 |
| Constrained WLS | 2 | 4 | $0 | $406 | -$0 | $1,862 | -$0 | $3,740 | $3,631 | $4,089 |
| Stacked LightGBM, constrained | 2 | 4 | -$0 | $122 | $0 | $985 | -$0 | $1,953 | $2,397 | $2,748 |

**Table 6.** The 20 largest drivers of the squared-error LightGBM predictions.

*Weighted mean absolute SHAP value on held-out folds. Rank stability across folds: mean Spearman correlation 0.79 over all features and 0.76 over the top 30. The last column is the loss in held-out R² when the feature is permuted in the CANN.*

| Rank | Feature | Mean |SHAP| | SD across folds | CANN R² loss |
|---|---:|---:|---:|---:|
| 1 | log_spend_y1 | $4,390 | $338 | 0.1083 |
| 2 | log_rx_fills_y1 | $1,762 | $185 | 0.0101 |
| 3 | n_conditions | $899 | $194 | 0.0209 |
| 4 | log_office_visits_y1 | $525 | $52 | 0.0039 |
| 5 | log_inpatient_y1 | $460 | $102 | 0.0163 |
| 6 | n_body_systems | $372 | $33 | -0.0119 |
| 7 | END002+END005 Diabetes mellitus without complication / Diabetes mellitus, Type 2 | $233 | $46 | -0.0013 |
| 8 | age_75-84 | $199 | $57 | -0.0004 |
| 9 | age_85+ | $164 | $21 | 0.0101 |
| 10 | age_55-64 | $153 | $50 | 0.0091 |
| 11 | age_65-74 | $142 | $29 | 0.0044 |
| 12 | MUS003 Rheumatoid arthritis and related disease | $136 | $53 | -0.0111 |
| 13 | END010 Disorders of lipid metabolism | $132 | $70 | 0.0098 |
| 14 | age_5-17 | $126 | $15 | -0.0026 |
| 15 | female | $117 | $72 | 0.0000 |
| 16 | region_1.0 | $99 | $40 | -0.0001 |
| 17 | age_25-34_x_female | $98 | $25 | 0.0163 |
| 18 | END001 Thyroid disorders | $92 | $40 | 0.0057 |
| 19 | log_er_visits_y1 | $90 | $42 | 0.0016 |
| 20 | region_3.0 | $81 | $21 | -0.0002 |

**Table 7.** Coding sensitivity: rise in predicted spending per added condition code, 10% of people gain one code (standard deviation across folds).

*Codes the person did not have are switched on; use and spending are unchanged. Prevalence: drawn in proportion to prevalence. Chronic: from payment-relevant chronic categories. Targeted: the five flags with the largest payment-form coefficients. Own-targeted: the five flags that raise that model's own prediction most, found on its training data. F2 excludes prior use and spending; F3 includes them.*

| Features | Model | Prevalence | Chronic | Targeted | Own-targeted |
|---|---:|---:|---:|---:|---:|
| F2 | CANN | $1,919 (sd $281) | $3,372 (sd $547) | $6,483 (sd $1,500) | $8,467 (sd $1,987) |
| F2 | Constrained WLS | $2,107 (sd $195) | $2,921 (sd $150) | $16,166 (sd $1,510) | $17,469 (sd $1,232) |
| F2 | LightGBM (Tweedie) | $2,036 (sd $89) | $2,907 (sd $120) | $4,781 (sd $553) | $5,510 (sd $497) |
| F2 | LightGBM (squared error) | $2,155 (sd $94) | $3,448 (sd $482) | $7,241 (sd $1,578) | $8,135 (sd $1,575) |
| F2 | Payment-form WLS (non-negative) | $2,379 (sd $157) | $4,327 (sd $175) | $10,812 (sd $791) | $11,045 (sd $533) |
| F2 | Tweedie GLM | $1,763 (sd $239) | $2,900 (sd $100) | $5,214 (sd $555) | $7,018 (sd $651) |
| F2 | Two-part | $1,182 (sd $34) | $1,515 (sd $27) | $1,740 (sd $85) | $1,994 (sd $67) |
| F2 | WLS | $1,837 (sd $194) | $2,806 (sd $269) | $10,529 (sd $818) | $10,777 (sd $531) |
| F3 | CANN | $409 (sd $128) | $977 (sd $194) | $2,925 (sd $236) | $5,204 (sd $2,276) |
| F3 | Constrained WLS | $154 (sd $167) | $111 (sd $284) | $12,812 (sd $1,377) | $13,754 (sd $1,158) |
| F3 | LightGBM (Tweedie) | $616 (sd $49) | $1,010 (sd $266) | $1,482 (sd $228) | $2,323 (sd $313) |
| F3 | LightGBM (squared error) | $789 (sd $91) | $1,290 (sd $217) | $3,046 (sd $1,299) | $4,776 (sd $1,235) |
| F3 | Payment-form WLS (non-negative) | $1,287 (sd $101) | $2,905 (sd $200) | $9,418 (sd $827) | $9,806 (sd $656) |
| F3 | Tweedie GLM | $397 (sd $112) | $953 (sd $178) | $2,916 (sd $234) | $5,055 (sd $2,083) |
| F3 | Two-part | $827 (sd $128) | $1,283 (sd $338) | $2,071 (sd $759) | $2,572 (sd $944) |
| F3 | WLS | $283 (sd $170) | $580 (sd $286) | $8,539 (sd $803) | $9,044 (sd $466) |

**Table 7c.** Own-targeted pools: the flags that raise each model's prediction most.

*For each fold, the five flags with the largest mean rise in that model's prediction when switched on for 3,000 training persons. Listed are the five chosen most often, with the number of folds (of five) and the mean training gain per code.*

| Features | Model | Flag (folds chosen, mean gain) |
|---|---:|---:|
| F2 | CANN | NVS009 (4, $9,146); CIR019 (3, $10,095); NEO030 (3, $8,924); MBD014 (3, $7,634); MUS003 (3, $6,524) |
| F2 | Constrained WLS | NVS009 (5, $24,120); NVS015 (5, $18,247); MUS003 (5, $15,885); CIR019 (4, $15,448); NVS012 (3, $13,265) |
| F2 | LightGBM (Tweedie) | NVS009 (5, $6,322); MUS003 (4, $6,046); NEO030 (4, $5,612); GEN006 (4, $5,171); RSP008 (2, $5,639) |
| F2 | LightGBM (squared error) | NVS009 (5, $7,642); MUS003 (4, $9,373); GEN006 (4, $9,040); MUS026 (3, $6,716); CIR019 (2, $9,453) |
| F2 | Payment-form WLS (non-negative) | MUS003 (5, $11,210); NVS009 (5, $10,512); NVS015 (4, $13,586); CIR019 (4, $10,873); GEN006 (4, $10,694) |
| F2 | Tweedie GLM | NVS009 (5, $8,677); MBD014 (5, $7,717); MUS003 (4, $6,318); PRG029 (4, $6,054); NEO039 (3, $5,361) |
| F2 | Two-part | NVS009 (5, $2,169); MUS026 (4, $2,071); MBD003 (4, $1,921); GEN006 (3, $2,152); MUS013 (2, $1,957) |
| F2 | WLS | MUS003 (5, $11,183); NVS015 (4, $13,110); CIR019 (4, $10,745); GEN006 (4, $10,517); NVS009 (4, $10,016) |
| F3 | CANN | NVS009 (5, $4,730); MUS003 (4, $3,806); MUS026 (4, $3,733); DEN001 (3, $14,319); GEN006 (3, $2,865) |
| F3 | Constrained WLS | NVS009 (5, $17,944); NVS015 (5, $15,658); MUS003 (5, $12,890); CIR019 (4, $11,894); MUS026 (4, $10,754) |
| F3 | LightGBM (Tweedie) | PRG029 (5, $3,017); MUS003 (4, $2,304); NVS009 (4, $2,149); MUS026 (4, $2,127); NVS017 (3, $2,058) |
| F3 | LightGBM (squared error) | MUS003 (4, $5,025); NVS017 (4, $4,821); GEN006 (4, $4,593); PRG029 (4, $4,519); MUS026 (4, $4,506) |
| F3 | Payment-form WLS (non-negative) | NVS015 (5, $11,826); MUS003 (5, $10,177); GEN006 (4, $9,928); CIR019 (3, $9,300); NVS009 (3, $8,977) |
| F3 | Tweedie GLM | NVS009 (5, $4,627); MUS003 (4, $3,806); MUS026 (4, $3,760); GEN006 (4, $2,915); DEN001 (3, $13,678) |
| F3 | Two-part | NVS009 (5, $3,049); MUS026 (4, $2,706); NEO039 (3, $2,774); GEN006 (3, $2,737); MUS003 (3, $2,562) |
| F3 | WLS | MUS003 (5, $9,347); NVS015 (4, $11,901); GEN006 (4, $9,026); NVS009 (3, $7,743); CIR019 (2, $9,898) |

**Table 7b.** Use sensitivity: next-year payment per added dollar of year-1 spending.

*Year-1 spending raised by 20% for 10% of held-out people, nothing else changed. Only feature sets with prior spending respond.*

| Features | Model | $ per $ of year-1 spending | SD across folds | Total rise |
|---|---:|---:|---:|---:|
| F3 | CANN | 0.239 | 0.112 | 0.47% |
| F3 | Constrained WLS | 0.340 | 0.081 | 0.72% |
| F3 | LightGBM (Tweedie) | 0.342 | 0.071 | 0.71% |
| F3 | LightGBM (squared error) | 0.319 | 0.068 | 0.66% |
| F3 | Payment-form WLS (non-negative) | 0.010 | 0.004 | 0.02% |
| F3 | Tweedie GLM | 0.238 | 0.112 | 0.47% |
| F3 | Two-part | 0.039 | 0.017 | 0.08% |
| F3 | WLS | 0.280 | 0.068 | 0.57% |

**Table 8.** Robustness: each row changes one decision.

*One repeat of the survey folds. Max |NC| is over the three target groups.*

| Variant | Model | n | R² | CPM | PR top decile | Max |NC| target |
|---|---:|---:|---:|---:|---:|---:|
| baseline | WLS | 43,963 | 0.123 | 0.147 | 0.94 | $7,830 |
| baseline | Payment-form WLS (non-negative) | 43,963 | 0.107 | 0.180 | 1.01 | $9,375 |
| baseline | Tweedie GLM | 43,963 | 0.122 | 0.241 | 1.01 | $4,809 |
| baseline | LightGBM (Tweedie) | 43,963 | 0.190 | 0.276 | 0.99 | $5,886 |
| outcome capped | WLS | 43,963 | 0.192 | 0.173 | 0.93 | $6,773 |
| outcome capped | Payment-form WLS (non-negative) | 43,963 | 0.173 | 0.195 | 0.99 | $8,036 |
| outcome capped | Tweedie GLM | 43,963 | 0.180 | 0.247 | 1.00 | $4,114 |
| outcome capped | LightGBM (Tweedie) | 43,963 | 0.281 | 0.284 | 1.00 | $4,670 |
| log outcome | WLS, log outcome | 43,963 | -0.096 | 0.193 | 1.63 | $6,684 |
| unweighted training | WLS | 43,963 | 0.119 | 0.121 | 1.02 | $6,288 |
| unweighted training | Payment-form WLS (non-negative) | 43,963 | 0.103 | 0.167 | 1.10 | $7,963 |
| unweighted training | Tweedie GLM | 43,963 | 0.138 | 0.243 | 1.01 | $3,489 |
| unweighted training | LightGBM (Tweedie) | 43,963 | 0.188 | 0.272 | 1.03 | $3,919 |
| CCSR floor 0.2% | WLS | 43,963 | 0.125 | 0.153 | 1.02 | $7,293 |
| CCSR floor 0.2% | Payment-form WLS (non-negative) | 43,963 | 0.115 | 0.190 | 1.05 | $8,195 |
| CCSR floor 0.2% | Tweedie GLM | 43,963 | 0.123 | 0.245 | 1.02 | $4,232 |
| CCSR floor 0.2% | LightGBM (Tweedie) | 43,963 | 0.191 | 0.278 | 0.98 | $5,632 |
| CCSR floor 1% | WLS | 43,963 | 0.124 | 0.149 | 0.92 | $8,510 |
| CCSR floor 1% | Payment-form WLS (non-negative) | 43,963 | 0.108 | 0.180 | 0.99 | $10,013 |
| CCSR floor 1% | Tweedie GLM | 43,963 | 0.155 | 0.250 | 0.97 | $5,631 |
| CCSR floor 1% | LightGBM (Tweedie) | 43,963 | 0.189 | 0.276 | 0.99 | $5,857 |
| no prior spending (F2) | WLS | 43,963 | 0.103 | 0.187 | 1.05 | $8,876 |
| no prior spending (F2) | Payment-form WLS (non-negative) | 43,963 | 0.098 | 0.191 | 1.05 | $9,988 |
| no prior spending (F2) | Tweedie GLM | 43,963 | 0.011 | 0.176 | 1.24 | $5,181 |
| no prior spending (F2) | LightGBM (Tweedie) | 43,963 | 0.101 | 0.199 | 1.00 | $10,991 |
| social-risk features (F4) | WLS | 43,963 | 0.127 | 0.151 | 0.96 | $1,165 |
| social-risk features (F4) | Payment-form WLS (non-negative) | 43,963 | 0.113 | 0.182 | 0.97 | $3,244 |
| social-risk features (F4) | Tweedie GLM | 43,963 | 0.118 | 0.243 | 1.02 | $1,426 |
| social-risk features (F4) | LightGBM (Tweedie) | 43,963 | 0.192 | 0.277 | 1.00 | $2,237 |
| decedents excluded | WLS | 43,568 | 0.124 | 0.149 | 0.94 | $7,887 |
| decedents excluded | Payment-form WLS (non-negative) | 43,568 | 0.106 | 0.180 | 1.01 | $9,376 |
| decedents excluded | Tweedie GLM | 43,568 | 0.125 | 0.243 | 1.00 | $5,015 |
| decedents excluded | LightGBM (Tweedie) | 43,568 | 0.193 | 0.277 | 0.98 | $5,602 |
| decedents not annualized | WLS | 43,963 | 0.127 | 0.150 | 0.94 | $7,422 |
| decedents not annualized | Payment-form WLS (non-negative) | 43,963 | 0.110 | 0.182 | 1.02 | $8,909 |
| decedents not annualized | Tweedie GLM | 43,963 | 0.128 | 0.244 | 1.00 | $4,554 |
| decedents not annualized | LightGBM (Tweedie) | 43,963 | 0.196 | 0.279 | 0.98 | $5,359 |
| 3 folds | WLS | 43,963 | 0.122 | 0.145 | 0.96 | $7,642 |
| 3 folds | Payment-form WLS (non-negative) | 43,963 | 0.107 | 0.181 | 1.01 | $9,260 |
| 3 folds | Tweedie GLM | 43,963 | 0.128 | 0.243 | 1.02 | $3,785 |
| 3 folds | LightGBM (Tweedie) | 43,963 | 0.187 | 0.278 | 0.99 | $5,520 |
| 10 folds | WLS | 43,963 | 0.125 | 0.150 | 0.94 | $7,792 |
| 10 folds | Payment-form WLS (non-negative) | 43,963 | 0.109 | 0.182 | 1.00 | $9,311 |
| 10 folds | Tweedie GLM | 43,963 | 0.115 | 0.242 | 0.99 | $4,710 |
| 10 folds | LightGBM (Tweedie) | 43,963 | 0.188 | 0.277 | 0.99 | $5,430 |
| panel 23 only | WLS | 13,765 | 0.144 | 0.166 | 1.01 | $9,996 |
| panel 23 only | LightGBM (Tweedie) | 13,765 | 0.220 | 0.290 | 1.07 | $6,847 |
| panel 24 only | WLS | 9,568 | 0.070 | 0.086 | 1.16 | $6,673 |
| panel 24 only | LightGBM (Tweedie) | 9,568 | 0.142 | 0.256 | 1.02 | $4,432 |
| panel 25 only | WLS | 5,942 | 0.030 | 0.006 | 1.34 | $2,999 |
| panel 25 only | LightGBM (Tweedie) | 5,942 | 0.155 | 0.230 | 1.00 | $4,238 |
| panel 26 only | WLS | 6,579 | 0.032 | 0.012 | 1.15 | $15,978 |
| panel 26 only | LightGBM (Tweedie) | 6,579 | 0.145 | 0.255 | 0.85 | $16,961 |
| panel 27 only | WLS | 8,109 | 0.136 | 0.122 | 1.18 | $3,516 |
| panel 27 only | LightGBM (Tweedie) | 8,109 | 0.248 | 0.287 | 0.97 | $3,498 |
| pandemic outcomes out | WLS | 28,453 | 0.125 | 0.146 | 0.97 | $9,285 |
| pandemic outcomes out | Payment-form WLS (non-negative) | 28,453 | 0.112 | 0.188 | 1.04 | $10,300 |
| pandemic outcomes out | Tweedie GLM | 28,453 | 0.091 | 0.246 | 1.06 | $6,060 |
| pandemic outcomes out | LightGBM (Tweedie) | 28,453 | 0.207 | 0.287 | 1.00 | $7,697 |
| age 65 and over | WLS | 9,490 | 0.096 | 0.075 | 1.05 | $6,011 |
| age 65 and over | Payment-form WLS (non-negative) | 9,490 | 0.072 | 0.088 | 1.04 | $9,121 |
| age 65 and over | Tweedie GLM | 9,490 | 0.124 | 0.161 | 1.02 | $4,463 |
| age 65 and over | LightGBM (Tweedie) | 9,490 | 0.156 | 0.185 | 0.93 | $5,914 |
| under 65, private | WLS | 21,988 | 0.091 | 0.095 | 1.04 | $7,811 |
| under 65, private | Payment-form WLS (non-negative) | 21,988 | 0.083 | 0.131 | 1.06 | $8,783 |
| under 65, private | Tweedie GLM | 21,988 | 0.088 | 0.178 | 1.08 | $3,535 |
| under 65, private | LightGBM (Tweedie) | 21,988 | 0.155 | 0.210 | 0.98 | $6,156 |


# Appendix tables


**Table A1.** Sample flow by MEPS panel.

| Panel | Years | In file | Both years, weight > 0 | Died in year 2 | Analysis sample | Any year-1 condition |
|---|---:|---:|---:|---:|---:|---:|
| 23 | 2018-2019 | 14,067 | 13,766 | 117 | 13,765 | 9,790 |
| 24 | 2019-2020 | 9,797 | 9,569 | 107 | 9,568 | 6,927 |
| 25 | 2020-2021 | 6,078 | 5,942 | 43 | 5,942 | 4,283 |
| 26 | 2021-2022 | 6,741 | 6,579 | 67 | 6,579 | 4,991 |
| 27 | 2022-2023 | 8,292 | 8,109 | 61 | 8,109 | 6,060 |

**Table A2.** Predictive ratio by decile of predicted spending.

| Decile | CANN | Elastic net | LightGBM (Tweedie) | LightGBM (squared error) | Neural network | Payment-form WLS (non-negative) | Random forest | Tweedie GLM | Two-part | WLS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.06 | -0.80 | 1.04 | 1.60 | 1.09 | 0.11 | 0.78 | 1.06 | 1.04 | -1.98 |
| 2 | 0.89 | 0.20 | 0.97 | 1.41 | 0.88 | 0.61 | 0.98 | 0.91 | 1.58 | -0.07 |
| 3 | 1.01 | 0.64 | 1.05 | 1.08 | 0.97 | 1.00 | 1.03 | 1.00 | 1.69 | 0.57 |
| 4 | 1.08 | 0.95 | 1.09 | 1.03 | 1.04 | 1.14 | 1.03 | 1.11 | 1.62 | 0.88 |
| 5 | 1.03 | 1.29 | 0.99 | 0.97 | 0.89 | 1.28 | 1.06 | 1.04 | 1.42 | 1.33 |
| 6 | 1.00 | 1.47 | 0.90 | 0.97 | 0.98 | 1.29 | 0.98 | 1.01 | 1.20 | 1.28 |
| 7 | 1.07 | 1.20 | 1.04 | 0.96 | 0.94 | 1.07 | 1.01 | 1.07 | 0.92 | 1.46 |
| 8 | 0.98 | 1.25 | 0.99 | 0.96 | 1.00 | 1.17 | 1.03 | 0.99 | 0.95 | 1.36 |
| 9 | 0.99 | 0.95 | 1.01 | 1.01 | 0.97 | 1.01 | 1.04 | 0.99 | 0.80 | 1.14 |
| 10 | 1.00 | 0.93 | 0.99 | 0.99 | 0.97 | 1.00 | 0.97 | 1.00 | 0.95 | 0.93 |

**Table A2b.** Predictive ratio by age band and sex, the cells of a payment formula.

| Age | Sex | CANN | LightGBM (Tweedie) | Payment-form WLS (non-negative) | Tweedie GLM | WLS |
|---|---:|---:|---:|---:|---:|---:|
| 0-17 | female | 0.932 | 1.021 | 1.395 | 0.940 | 1.007 |
| 0-17 | male | 0.975 | 1.078 | 1.393 | 0.985 | 1.005 |
| 18-34 | female | 1.057 | 1.006 | 1.067 | 1.057 | 1.002 |
| 18-34 | male | 0.866 | 0.974 | 1.122 | 0.872 | 1.004 |
| 35-44 | female | 0.981 | 0.997 | 1.005 | 0.985 | 1.001 |
| 35-44 | male | 0.983 | 1.131 | 1.083 | 0.990 | 0.999 |
| 45-54 | female | 0.986 | 1.067 | 1.017 | 0.987 | 1.004 |
| 45-54 | male | 0.938 | 0.930 | 1.000 | 0.943 | 1.001 |
| 55-64 | female | 1.012 | 1.025 | 1.003 | 1.007 | 1.003 |
| 55-64 | male | 1.016 | 0.944 | 0.999 | 1.018 | 0.998 |
| 65+ | female | 1.079 | 1.014 | 1.018 | 1.072 | 0.998 |
| 65+ | male | 0.990 | 0.923 | 1.003 | 0.988 | 1.002 |

**Table A3.** Coding sensitivity, full grid.

| Features | Model | Pool | Share affected | Codes added | Total rise | $ per code | SD across folds |
|---|---:|---:|---:|---:|---:|---:|---:|
| F2 | CANN | chronic | 5% | 1 | 2.22% | $3,390 | $607 |
| F2 | CANN | chronic | 5% | 2 | 5.09% | $3,870 | $591 |
| F2 | CANN | chronic | 10% | 1 | 4.32% | $3,372 | $547 |
| F2 | CANN | chronic | 10% | 2 | 10.18% | $3,907 | $789 |
| F2 | CANN | chronic | 25% | 1 | 10.87% | $3,415 | $687 |
| F2 | CANN | chronic | 25% | 2 | 24.91% | $3,822 | $627 |
| F2 | CANN | own-targeted | 5% | 1 | 5.31% | $8,127 | $2,078 |
| F2 | CANN | own-targeted | 5% | 2 | 14.51% | $11,038 | $1,415 |
| F2 | CANN | own-targeted | 10% | 1 | 10.85% | $8,467 | $1,987 |
| F2 | CANN | own-targeted | 10% | 2 | 29.84% | $11,453 | $1,745 |
| F2 | CANN | own-targeted | 25% | 1 | 26.86% | $8,435 | $1,991 |
| F2 | CANN | own-targeted | 25% | 2 | 73.52% | $11,279 | $1,300 |
| F2 | CANN | prevalence | 5% | 1 | 1.09% | $1,665 | $228 |
| F2 | CANN | prevalence | 5% | 2 | 2.47% | $1,881 | $197 |
| F2 | CANN | prevalence | 10% | 1 | 2.46% | $1,919 | $281 |
| F2 | CANN | prevalence | 10% | 2 | 5.04% | $1,935 | $190 |
| F2 | CANN | prevalence | 25% | 1 | 5.84% | $1,833 | $187 |
| F2 | CANN | prevalence | 25% | 2 | 12.53% | $1,923 | $137 |
| F2 | CANN | targeted | 5% | 1 | 4.23% | $6,471 | $1,890 |
| F2 | CANN | targeted | 5% | 2 | 9.88% | $7,516 | $1,120 |
| F2 | CANN | targeted | 10% | 1 | 8.31% | $6,483 | $1,500 |
| F2 | CANN | targeted | 10% | 2 | 19.74% | $7,576 | $1,143 |
| F2 | CANN | targeted | 25% | 1 | 20.64% | $6,483 | $1,516 |
| F2 | CANN | targeted | 25% | 2 | 49.24% | $7,554 | $802 |
| F2 | Constrained WLS | chronic | 5% | 1 | 2.09% | $2,865 | $119 |
| F2 | Constrained WLS | chronic | 5% | 2 | 4.76% | $3,241 | $157 |
| F2 | Constrained WLS | chronic | 10% | 1 | 4.18% | $2,921 | $150 |
| F2 | Constrained WLS | chronic | 10% | 2 | 9.20% | $3,165 | $212 |
| F2 | Constrained WLS | chronic | 25% | 1 | 10.13% | $2,851 | $222 |
| F2 | Constrained WLS | chronic | 25% | 2 | 22.63% | $3,112 | $113 |
| F2 | Constrained WLS | own-targeted | 5% | 1 | 12.60% | $17,269 | $1,280 |
| F2 | Constrained WLS | own-targeted | 5% | 2 | 25.46% | $17,350 | $1,323 |
| F2 | Constrained WLS | own-targeted | 10% | 1 | 24.98% | $17,469 | $1,232 |
| F2 | Constrained WLS | own-targeted | 10% | 2 | 50.33% | $17,312 | $1,380 |
| F2 | Constrained WLS | own-targeted | 25% | 1 | 61.92% | $17,430 | $1,389 |
| F2 | Constrained WLS | own-targeted | 25% | 2 | 126.25% | $17,360 | $1,431 |
| F2 | Constrained WLS | prevalence | 5% | 1 | 1.53% | $2,103 | $236 |
| F2 | Constrained WLS | prevalence | 5% | 2 | 3.22% | $2,195 | $256 |
| F2 | Constrained WLS | prevalence | 10% | 1 | 3.01% | $2,107 | $195 |
| F2 | Constrained WLS | prevalence | 10% | 2 | 5.87% | $2,020 | $62 |
| F2 | Constrained WLS | prevalence | 25% | 1 | 7.86% | $2,213 | $163 |
| F2 | Constrained WLS | prevalence | 25% | 2 | 15.00% | $2,063 | $111 |
| F2 | Constrained WLS | targeted | 5% | 1 | 11.71% | $16,048 | $1,740 |
| F2 | Constrained WLS | targeted | 5% | 2 | 23.97% | $16,340 | $1,456 |
| F2 | Constrained WLS | targeted | 10% | 1 | 23.11% | $16,166 | $1,510 |
| F2 | Constrained WLS | targeted | 10% | 2 | 47.05% | $16,183 | $1,667 |
| F2 | Constrained WLS | targeted | 25% | 1 | 57.54% | $16,195 | $1,484 |
| F2 | Constrained WLS | targeted | 25% | 2 | 117.49% | $16,155 | $1,602 |
| F2 | LightGBM (Tweedie) | chronic | 5% | 1 | 1.97% | $3,031 | $222 |
| F2 | LightGBM (Tweedie) | chronic | 5% | 2 | 3.90% | $2,979 | $254 |
| F2 | LightGBM (Tweedie) | chronic | 10% | 1 | 3.71% | $2,907 | $120 |
| F2 | LightGBM (Tweedie) | chronic | 10% | 2 | 7.68% | $2,958 | $198 |
| F2 | LightGBM (Tweedie) | chronic | 25% | 1 | 9.29% | $2,928 | $196 |
| F2 | LightGBM (Tweedie) | chronic | 25% | 2 | 19.16% | $2,950 | $216 |
| F2 | LightGBM (Tweedie) | own-targeted | 5% | 1 | 3.55% | $5,446 | $561 |
| F2 | LightGBM (Tweedie) | own-targeted | 5% | 2 | 8.31% | $6,340 | $908 |
| F2 | LightGBM (Tweedie) | own-targeted | 10% | 1 | 7.04% | $5,510 | $497 |
| F2 | LightGBM (Tweedie) | own-targeted | 10% | 2 | 16.72% | $6,441 | $862 |
| F2 | LightGBM (Tweedie) | own-targeted | 25% | 1 | 17.49% | $5,511 | $490 |
| F2 | LightGBM (Tweedie) | own-targeted | 25% | 2 | 41.25% | $6,350 | $829 |
| F2 | LightGBM (Tweedie) | prevalence | 5% | 1 | 1.33% | $2,045 | $77 |
| F2 | LightGBM (Tweedie) | prevalence | 5% | 2 | 2.60% | $1,984 | $106 |
| F2 | LightGBM (Tweedie) | prevalence | 10% | 1 | 2.60% | $2,036 | $89 |
| F2 | LightGBM (Tweedie) | prevalence | 10% | 2 | 5.08% | $1,955 | $46 |
| F2 | LightGBM (Tweedie) | prevalence | 25% | 1 | 6.41% | $2,021 | $26 |
| F2 | LightGBM (Tweedie) | prevalence | 25% | 2 | 12.78% | $1,968 | $25 |
| F2 | LightGBM (Tweedie) | targeted | 5% | 1 | 3.16% | $4,856 | $452 |
| F2 | LightGBM (Tweedie) | targeted | 5% | 2 | 7.23% | $5,519 | $695 |
| F2 | LightGBM (Tweedie) | targeted | 10% | 1 | 6.11% | $4,781 | $553 |
| F2 | LightGBM (Tweedie) | targeted | 10% | 2 | 14.43% | $5,556 | $757 |
| F2 | LightGBM (Tweedie) | targeted | 25% | 1 | 15.48% | $4,878 | $587 |
| F2 | LightGBM (Tweedie) | targeted | 25% | 2 | 36.08% | $5,554 | $786 |
| F2 | LightGBM (squared error) | chronic | 5% | 1 | 2.28% | $3,509 | $560 |
| F2 | LightGBM (squared error) | chronic | 5% | 2 | 4.65% | $3,559 | $603 |
| F2 | LightGBM (squared error) | chronic | 10% | 1 | 4.39% | $3,448 | $482 |
| F2 | LightGBM (squared error) | chronic | 10% | 2 | 9.08% | $3,506 | $545 |
| F2 | LightGBM (squared error) | chronic | 25% | 1 | 11.01% | $3,478 | $550 |
| F2 | LightGBM (squared error) | chronic | 25% | 2 | 22.63% | $3,493 | $557 |
| F2 | LightGBM (squared error) | own-targeted | 5% | 1 | 5.23% | $8,054 | $1,496 |
| F2 | LightGBM (squared error) | own-targeted | 5% | 2 | 10.23% | $7,828 | $1,504 |
| F2 | LightGBM (squared error) | own-targeted | 10% | 1 | 10.36% | $8,135 | $1,575 |
| F2 | LightGBM (squared error) | own-targeted | 10% | 2 | 20.25% | $7,818 | $1,594 |
| F2 | LightGBM (squared error) | own-targeted | 25% | 1 | 25.68% | $8,115 | $1,652 |
| F2 | LightGBM (squared error) | own-targeted | 25% | 2 | 50.60% | $7,809 | $1,547 |
| F2 | LightGBM (squared error) | prevalence | 5% | 1 | 1.39% | $2,135 | $80 |
| F2 | LightGBM (squared error) | prevalence | 5% | 2 | 2.82% | $2,160 | $210 |
| F2 | LightGBM (squared error) | prevalence | 10% | 1 | 2.75% | $2,155 | $94 |
| F2 | LightGBM (squared error) | prevalence | 10% | 2 | 5.38% | $2,077 | $46 |
| F2 | LightGBM (squared error) | prevalence | 25% | 1 | 6.88% | $2,173 | $150 |
| F2 | LightGBM (squared error) | prevalence | 25% | 2 | 13.74% | $2,120 | $105 |
| F2 | LightGBM (squared error) | targeted | 5% | 1 | 4.78% | $7,358 | $1,736 |
| F2 | LightGBM (squared error) | targeted | 5% | 2 | 9.53% | $7,291 | $1,680 |
| F2 | LightGBM (squared error) | targeted | 10% | 1 | 9.22% | $7,241 | $1,578 |
| F2 | LightGBM (squared error) | targeted | 10% | 2 | 18.76% | $7,241 | $1,599 |
| F2 | LightGBM (squared error) | targeted | 25% | 1 | 23.21% | $7,333 | $1,753 |
| F2 | LightGBM (squared error) | targeted | 25% | 2 | 46.97% | $7,249 | $1,657 |
| F2 | Payment-form WLS (non-negative) | chronic | 5% | 1 | 2.81% | $4,333 | $302 |
| F2 | Payment-form WLS (non-negative) | chronic | 5% | 2 | 5.74% | $4,398 | $216 |
| F2 | Payment-form WLS (non-negative) | chronic | 10% | 1 | 5.51% | $4,327 | $175 |
| F2 | Payment-form WLS (non-negative) | chronic | 10% | 2 | 11.16% | $4,314 | $207 |
| F2 | Payment-form WLS (non-negative) | chronic | 25% | 1 | 13.69% | $4,329 | $207 |
| F2 | Payment-form WLS (non-negative) | chronic | 25% | 2 | 27.82% | $4,297 | $200 |
| F2 | Payment-form WLS (non-negative) | own-targeted | 5% | 1 | 7.11% | $10,944 | $576 |
| F2 | Payment-form WLS (non-negative) | own-targeted | 5% | 2 | 14.38% | $11,006 | $570 |
| F2 | Payment-form WLS (non-negative) | own-targeted | 10% | 1 | 14.06% | $11,045 | $533 |
| F2 | Payment-form WLS (non-negative) | own-targeted | 10% | 2 | 28.48% | $11,003 | $586 |
| F2 | Payment-form WLS (non-negative) | own-targeted | 25% | 1 | 34.88% | $11,027 | $584 |
| F2 | Payment-form WLS (non-negative) | own-targeted | 25% | 2 | 71.23% | $11,002 | $586 |
| F2 | Payment-form WLS (non-negative) | prevalence | 5% | 1 | 1.52% | $2,346 | $139 |
| F2 | Payment-form WLS (non-negative) | prevalence | 5% | 2 | 3.07% | $2,353 | $191 |
| F2 | Payment-form WLS (non-negative) | prevalence | 10% | 1 | 3.03% | $2,379 | $157 |
| F2 | Payment-form WLS (non-negative) | prevalence | 10% | 2 | 5.80% | $2,242 | $67 |
| F2 | Payment-form WLS (non-negative) | prevalence | 25% | 1 | 7.58% | $2,398 | $131 |
| F2 | Payment-form WLS (non-negative) | prevalence | 25% | 2 | 14.99% | $2,315 | $60 |
| F2 | Payment-form WLS (non-negative) | targeted | 5% | 1 | 6.96% | $10,722 | $834 |
| F2 | Payment-form WLS (non-negative) | targeted | 5% | 2 | 14.14% | $10,825 | $716 |
| F2 | Payment-form WLS (non-negative) | targeted | 10% | 1 | 13.76% | $10,812 | $791 |
| F2 | Payment-form WLS (non-negative) | targeted | 10% | 2 | 27.85% | $10,762 | $806 |
| F2 | Payment-form WLS (non-negative) | targeted | 25% | 1 | 34.15% | $10,797 | $817 |
| F2 | Payment-form WLS (non-negative) | targeted | 25% | 2 | 69.69% | $10,764 | $839 |
| F2 | Tweedie GLM | chronic | 5% | 1 | 1.82% | $2,819 | $68 |
| F2 | Tweedie GLM | chronic | 5% | 2 | 4.34% | $3,338 | $286 |
| F2 | Tweedie GLM | chronic | 10% | 1 | 3.67% | $2,900 | $100 |
| F2 | Tweedie GLM | chronic | 10% | 2 | 8.94% | $3,470 | $389 |
| F2 | Tweedie GLM | chronic | 25% | 1 | 9.10% | $2,890 | $117 |
| F2 | Tweedie GLM | chronic | 25% | 2 | 21.42% | $3,325 | $133 |
| F2 | Tweedie GLM | own-targeted | 5% | 1 | 4.17% | $6,458 | $683 |
| F2 | Tweedie GLM | own-targeted | 5% | 2 | 13.03% | $10,025 | $1,256 |
| F2 | Tweedie GLM | own-targeted | 10% | 1 | 8.89% | $7,018 | $651 |
| F2 | Tweedie GLM | own-targeted | 10% | 2 | 27.14% | $10,537 | $1,659 |
| F2 | Tweedie GLM | own-targeted | 25% | 1 | 21.57% | $6,854 | $370 |
| F2 | Tweedie GLM | own-targeted | 25% | 2 | 65.90% | $10,229 | $907 |
| F2 | Tweedie GLM | prevalence | 5% | 1 | 1.01% | $1,567 | $166 |
| F2 | Tweedie GLM | prevalence | 5% | 2 | 2.36% | $1,819 | $231 |
| F2 | Tweedie GLM | prevalence | 10% | 1 | 2.23% | $1,763 | $239 |
| F2 | Tweedie GLM | prevalence | 10% | 2 | 4.73% | $1,837 | $170 |
| F2 | Tweedie GLM | prevalence | 25% | 1 | 5.38% | $1,710 | $54 |
| F2 | Tweedie GLM | prevalence | 25% | 2 | 11.76% | $1,825 | $43 |
| F2 | Tweedie GLM | targeted | 5% | 1 | 3.16% | $4,893 | $339 |
| F2 | Tweedie GLM | targeted | 5% | 2 | 9.08% | $6,982 | $944 |
| F2 | Tweedie GLM | targeted | 10% | 1 | 6.60% | $5,214 | $555 |
| F2 | Tweedie GLM | targeted | 10% | 2 | 18.59% | $7,217 | $1,209 |
| F2 | Tweedie GLM | targeted | 25% | 1 | 16.47% | $5,234 | $370 |
| F2 | Tweedie GLM | targeted | 25% | 2 | 44.32% | $6,879 | $678 |
| F2 | Two-part | chronic | 5% | 1 | 1.00% | $1,535 | $36 |
| F2 | Two-part | chronic | 5% | 2 | 1.84% | $1,404 | $11 |
| F2 | Two-part | chronic | 10% | 1 | 1.93% | $1,515 | $27 |
| F2 | Two-part | chronic | 10% | 2 | 3.64% | $1,404 | $24 |
| F2 | Two-part | chronic | 25% | 1 | 4.79% | $1,513 | $33 |
| F2 | Two-part | chronic | 25% | 2 | 9.08% | $1,401 | $14 |
| F2 | Two-part | own-targeted | 5% | 1 | 1.31% | $2,011 | $102 |
| F2 | Two-part | own-targeted | 5% | 2 | 2.31% | $1,770 | $62 |
| F2 | Two-part | own-targeted | 10% | 1 | 2.54% | $1,994 | $67 |
| F2 | Two-part | own-targeted | 10% | 2 | 4.58% | $1,769 | $74 |
| F2 | Two-part | own-targeted | 25% | 1 | 6.36% | $2,008 | $51 |
| F2 | Two-part | own-targeted | 25% | 2 | 11.50% | $1,774 | $63 |
| F2 | Two-part | prevalence | 5% | 1 | 0.77% | $1,184 | $59 |
| F2 | Two-part | prevalence | 5% | 2 | 1.43% | $1,092 | $26 |
| F2 | Two-part | prevalence | 10% | 1 | 1.51% | $1,182 | $34 |
| F2 | Two-part | prevalence | 10% | 2 | 2.82% | $1,088 | $11 |
| F2 | Two-part | prevalence | 25% | 1 | 3.78% | $1,193 | $14 |
| F2 | Two-part | prevalence | 25% | 2 | 7.08% | $1,092 | $16 |
| F2 | Two-part | targeted | 5% | 1 | 1.13% | $1,738 | $138 |
| F2 | Two-part | targeted | 5% | 2 | 2.32% | $1,772 | $102 |
| F2 | Two-part | targeted | 10% | 1 | 2.22% | $1,740 | $85 |
| F2 | Two-part | targeted | 10% | 2 | 4.61% | $1,779 | $118 |
| F2 | Two-part | targeted | 25% | 1 | 5.53% | $1,747 | $109 |
| F2 | Two-part | targeted | 25% | 2 | 11.53% | $1,778 | $110 |
| F2 | WLS | chronic | 5% | 1 | 1.79% | $2,757 | $270 |
| F2 | WLS | chronic | 5% | 2 | 3.89% | $2,979 | $286 |
| F2 | WLS | chronic | 10% | 1 | 3.57% | $2,806 | $269 |
| F2 | WLS | chronic | 10% | 2 | 7.65% | $2,954 | $202 |
| F2 | WLS | chronic | 25% | 1 | 8.72% | $2,757 | $289 |
| F2 | WLS | chronic | 25% | 2 | 18.95% | $2,927 | $233 |
| F2 | WLS | own-targeted | 5% | 1 | 6.93% | $10,675 | $567 |
| F2 | WLS | own-targeted | 5% | 2 | 14.04% | $10,742 | $560 |
| F2 | WLS | own-targeted | 10% | 1 | 13.72% | $10,777 | $531 |
| F2 | WLS | own-targeted | 10% | 2 | 27.77% | $10,726 | $575 |
| F2 | WLS | own-targeted | 25% | 1 | 34.03% | $10,758 | $573 |
| F2 | WLS | own-targeted | 25% | 2 | 69.47% | $10,727 | $577 |
| F2 | WLS | prevalence | 5% | 1 | 1.18% | $1,821 | $127 |
| F2 | WLS | prevalence | 5% | 2 | 2.36% | $1,803 | $203 |
| F2 | WLS | prevalence | 10% | 1 | 2.34% | $1,837 | $194 |
| F2 | WLS | prevalence | 10% | 2 | 4.45% | $1,718 | $64 |
| F2 | WLS | prevalence | 25% | 1 | 5.89% | $1,863 | $115 |
| F2 | WLS | prevalence | 25% | 2 | 11.33% | $1,750 | $94 |
| F2 | WLS | targeted | 5% | 1 | 6.78% | $10,437 | $835 |
| F2 | WLS | targeted | 5% | 2 | 13.74% | $10,514 | $723 |
| F2 | WLS | targeted | 10% | 1 | 13.40% | $10,529 | $818 |
| F2 | WLS | targeted | 10% | 2 | 27.07% | $10,457 | $814 |
| F2 | WLS | targeted | 25% | 1 | 33.23% | $10,505 | $841 |
| F2 | WLS | targeted | 25% | 2 | 67.72% | $10,457 | $858 |
| F3 | CANN | chronic | 5% | 1 | 0.62% | $960 | $152 |
| F3 | CANN | chronic | 5% | 2 | 1.35% | $1,037 | $183 |
| F3 | CANN | chronic | 10% | 1 | 1.24% | $977 | $194 |
| F3 | CANN | chronic | 10% | 2 | 2.69% | $1,042 | $153 |
| F3 | CANN | chronic | 25% | 1 | 3.00% | $954 | $198 |
| F3 | CANN | chronic | 25% | 2 | 6.64% | $1,029 | $177 |
| F3 | CANN | own-targeted | 5% | 1 | 3.07% | $4,749 | $1,762 |
| F3 | CANN | own-targeted | 5% | 2 | 8.18% | $6,288 | $2,276 |
| F3 | CANN | own-targeted | 10% | 1 | 6.60% | $5,204 | $2,276 |
| F3 | CANN | own-targeted | 10% | 2 | 16.82% | $6,525 | $2,683 |
| F3 | CANN | own-targeted | 25% | 1 | 15.86% | $5,034 | $1,947 |
| F3 | CANN | own-targeted | 25% | 2 | 43.50% | $6,746 | $3,016 |
| F3 | CANN | prevalence | 5% | 1 | 0.20% | $303 | $90 |
| F3 | CANN | prevalence | 5% | 2 | 0.51% | $391 | $67 |
| F3 | CANN | prevalence | 10% | 1 | 0.52% | $409 | $128 |
| F3 | CANN | prevalence | 10% | 2 | 0.87% | $338 | $82 |
| F3 | CANN | prevalence | 25% | 1 | 1.20% | $380 | $68 |
| F3 | CANN | prevalence | 25% | 2 | 2.26% | $350 | $93 |
| F3 | CANN | targeted | 5% | 1 | 1.93% | $2,990 | $334 |
| F3 | CANN | targeted | 5% | 2 | 4.73% | $3,632 | $634 |
| F3 | CANN | targeted | 10% | 1 | 3.71% | $2,925 | $236 |
| F3 | CANN | targeted | 10% | 2 | 9.58% | $3,717 | $519 |
| F3 | CANN | targeted | 25% | 1 | 9.41% | $2,989 | $271 |
| F3 | CANN | targeted | 25% | 2 | 23.03% | $3,571 | $302 |
| F3 | Constrained WLS | chronic | 5% | 1 | -0.00% | -$1 | $230 |
| F3 | Constrained WLS | chronic | 5% | 2 | 0.63% | $468 | $248 |
| F3 | Constrained WLS | chronic | 10% | 1 | 0.15% | $111 | $284 |
| F3 | Constrained WLS | chronic | 10% | 2 | 1.07% | $400 | $141 |
| F3 | Constrained WLS | chronic | 25% | 1 | 0.14% | $42 | $280 |
| F3 | Constrained WLS | chronic | 25% | 2 | 2.46% | $367 | $196 |
| F3 | Constrained WLS | own-targeted | 5% | 1 | 9.10% | $13,570 | $1,193 |
| F3 | Constrained WLS | own-targeted | 5% | 2 | 18.64% | $13,818 | $1,217 |
| F3 | Constrained WLS | own-targeted | 10% | 1 | 18.08% | $13,754 | $1,158 |
| F3 | Constrained WLS | own-targeted | 10% | 2 | 36.87% | $13,794 | $1,222 |
| F3 | Constrained WLS | own-targeted | 25% | 1 | 44.81% | $13,716 | $1,248 |
| F3 | Constrained WLS | own-targeted | 25% | 2 | 92.51% | $13,833 | $1,241 |
| F3 | Constrained WLS | prevalence | 5% | 1 | 0.16% | $232 | $371 |
| F3 | Constrained WLS | prevalence | 5% | 2 | 0.42% | $313 | $243 |
| F3 | Constrained WLS | prevalence | 10% | 1 | 0.20% | $154 | $167 |
| F3 | Constrained WLS | prevalence | 10% | 2 | 0.45% | $168 | $106 |
| F3 | Constrained WLS | prevalence | 25% | 1 | 0.85% | $262 | $51 |
| F3 | Constrained WLS | prevalence | 25% | 2 | 1.29% | $193 | $131 |
| F3 | Constrained WLS | targeted | 5% | 1 | 8.65% | $12,897 | $1,277 |
| F3 | Constrained WLS | targeted | 5% | 2 | 17.45% | $12,938 | $1,333 |
| F3 | Constrained WLS | targeted | 10% | 1 | 16.84% | $12,812 | $1,377 |
| F3 | Constrained WLS | targeted | 10% | 2 | 34.50% | $12,905 | $1,351 |
| F3 | Constrained WLS | targeted | 25% | 1 | 42.15% | $12,903 | $1,525 |
| F3 | Constrained WLS | targeted | 25% | 2 | 86.42% | $12,923 | $1,415 |
| F3 | LightGBM (Tweedie) | chronic | 5% | 1 | 0.66% | $1,007 | $229 |
| F3 | LightGBM (Tweedie) | chronic | 5% | 2 | 1.40% | $1,064 | $271 |
| F3 | LightGBM (Tweedie) | chronic | 10% | 1 | 1.29% | $1,010 | $266 |
| F3 | LightGBM (Tweedie) | chronic | 10% | 2 | 2.66% | $1,025 | $249 |
| F3 | LightGBM (Tweedie) | chronic | 25% | 1 | 3.17% | $996 | $228 |
| F3 | LightGBM (Tweedie) | chronic | 25% | 2 | 6.73% | $1,035 | $262 |
| F3 | LightGBM (Tweedie) | own-targeted | 5% | 1 | 1.49% | $2,288 | $314 |
| F3 | LightGBM (Tweedie) | own-targeted | 5% | 2 | 3.43% | $2,613 | $422 |
| F3 | LightGBM (Tweedie) | own-targeted | 10% | 1 | 2.97% | $2,323 | $313 |
| F3 | LightGBM (Tweedie) | own-targeted | 10% | 2 | 6.59% | $2,536 | $263 |
| F3 | LightGBM (Tweedie) | own-targeted | 25% | 1 | 7.21% | $2,270 | $182 |
| F3 | LightGBM (Tweedie) | own-targeted | 25% | 2 | 16.97% | $2,610 | $303 |
| F3 | LightGBM (Tweedie) | prevalence | 5% | 1 | 0.38% | $586 | $10 |
| F3 | LightGBM (Tweedie) | prevalence | 5% | 2 | 0.78% | $596 | $43 |
| F3 | LightGBM (Tweedie) | prevalence | 10% | 1 | 0.79% | $616 | $49 |
| F3 | LightGBM (Tweedie) | prevalence | 10% | 2 | 1.52% | $586 | $35 |
| F3 | LightGBM (Tweedie) | prevalence | 25% | 1 | 1.82% | $572 | $33 |
| F3 | LightGBM (Tweedie) | prevalence | 25% | 2 | 3.91% | $601 | $56 |
| F3 | LightGBM (Tweedie) | targeted | 5% | 1 | 0.97% | $1,486 | $248 |
| F3 | LightGBM (Tweedie) | targeted | 5% | 2 | 2.12% | $1,616 | $270 |
| F3 | LightGBM (Tweedie) | targeted | 10% | 1 | 1.90% | $1,482 | $228 |
| F3 | LightGBM (Tweedie) | targeted | 10% | 2 | 4.05% | $1,556 | $191 |
| F3 | LightGBM (Tweedie) | targeted | 25% | 1 | 4.73% | $1,490 | $200 |
| F3 | LightGBM (Tweedie) | targeted | 25% | 2 | 10.21% | $1,570 | $205 |
| F3 | LightGBM (squared error) | chronic | 5% | 1 | 0.87% | $1,332 | $218 |
| F3 | LightGBM (squared error) | chronic | 5% | 2 | 1.74% | $1,333 | $181 |
| F3 | LightGBM (squared error) | chronic | 10% | 1 | 1.64% | $1,290 | $217 |
| F3 | LightGBM (squared error) | chronic | 10% | 2 | 3.41% | $1,314 | $169 |
| F3 | LightGBM (squared error) | chronic | 25% | 1 | 4.12% | $1,302 | $212 |
| F3 | LightGBM (squared error) | chronic | 25% | 2 | 8.43% | $1,300 | $174 |
| F3 | LightGBM (squared error) | own-targeted | 5% | 1 | 3.07% | $4,721 | $1,321 |
| F3 | LightGBM (squared error) | own-targeted | 5% | 2 | 5.90% | $4,508 | $1,251 |
| F3 | LightGBM (squared error) | own-targeted | 10% | 1 | 6.09% | $4,776 | $1,235 |
| F3 | LightGBM (squared error) | own-targeted | 10% | 2 | 11.51% | $4,440 | $1,229 |
| F3 | LightGBM (squared error) | own-targeted | 25% | 1 | 14.98% | $4,728 | $1,208 |
| F3 | LightGBM (squared error) | own-targeted | 25% | 2 | 28.49% | $4,393 | $1,210 |
| F3 | LightGBM (squared error) | prevalence | 5% | 1 | 0.47% | $722 | $35 |
| F3 | LightGBM (squared error) | prevalence | 5% | 2 | 1.04% | $795 | $70 |
| F3 | LightGBM (squared error) | prevalence | 10% | 1 | 1.01% | $789 | $91 |
| F3 | LightGBM (squared error) | prevalence | 10% | 2 | 1.98% | $765 | $32 |
| F3 | LightGBM (squared error) | prevalence | 25% | 1 | 2.44% | $771 | $59 |
| F3 | LightGBM (squared error) | prevalence | 25% | 2 | 5.07% | $782 | $33 |
| F3 | LightGBM (squared error) | targeted | 5% | 1 | 2.04% | $3,138 | $1,368 |
| F3 | LightGBM (squared error) | targeted | 5% | 2 | 4.06% | $3,106 | $1,422 |
| F3 | LightGBM (squared error) | targeted | 10% | 1 | 3.88% | $3,046 | $1,299 |
| F3 | LightGBM (squared error) | targeted | 10% | 2 | 7.86% | $3,030 | $1,257 |
| F3 | LightGBM (squared error) | targeted | 25% | 1 | 9.84% | $3,106 | $1,291 |
| F3 | LightGBM (squared error) | targeted | 25% | 2 | 19.42% | $2,994 | $1,292 |
| F3 | Payment-form WLS (non-negative) | chronic | 5% | 1 | 1.78% | $2,876 | $189 |
| F3 | Payment-form WLS (non-negative) | chronic | 5% | 2 | 3.72% | $2,991 | $170 |
| F3 | Payment-form WLS (non-negative) | chronic | 10% | 1 | 3.52% | $2,905 | $200 |
| F3 | Payment-form WLS (non-negative) | chronic | 10% | 2 | 7.17% | $2,911 | $91 |
| F3 | Payment-form WLS (non-negative) | chronic | 25% | 1 | 8.71% | $2,895 | $126 |
| F3 | Payment-form WLS (non-negative) | chronic | 25% | 2 | 17.86% | $2,899 | $127 |
| F3 | Payment-form WLS (non-negative) | own-targeted | 5% | 1 | 5.99% | $9,692 | $677 |
| F3 | Payment-form WLS (non-negative) | own-targeted | 5% | 2 | 12.17% | $9,789 | $681 |
| F3 | Payment-form WLS (non-negative) | own-targeted | 10% | 1 | 11.88% | $9,806 | $656 |
| F3 | Payment-form WLS (non-negative) | own-targeted | 10% | 2 | 24.12% | $9,793 | $691 |
| F3 | Payment-form WLS (non-negative) | own-targeted | 25% | 1 | 29.49% | $9,798 | $684 |
| F3 | Payment-form WLS (non-negative) | own-targeted | 25% | 2 | 60.32% | $9,790 | $688 |
| F3 | Payment-form WLS (non-negative) | prevalence | 5% | 1 | 0.79% | $1,286 | $203 |
| F3 | Payment-form WLS (non-negative) | prevalence | 5% | 2 | 1.61% | $1,292 | $217 |
| F3 | Payment-form WLS (non-negative) | prevalence | 10% | 1 | 1.56% | $1,287 | $101 |
| F3 | Payment-form WLS (non-negative) | prevalence | 10% | 2 | 2.92% | $1,185 | $42 |
| F3 | Payment-form WLS (non-negative) | prevalence | 25% | 1 | 3.97% | $1,318 | $65 |
| F3 | Payment-form WLS (non-negative) | prevalence | 25% | 2 | 7.71% | $1,252 | $64 |
| F3 | Payment-form WLS (non-negative) | targeted | 5% | 1 | 5.80% | $9,384 | $780 |
| F3 | Payment-form WLS (non-negative) | targeted | 5% | 2 | 11.83% | $9,514 | $743 |
| F3 | Payment-form WLS (non-negative) | targeted | 10% | 1 | 11.41% | $9,418 | $827 |
| F3 | Payment-form WLS (non-negative) | targeted | 10% | 2 | 23.32% | $9,467 | $799 |
| F3 | Payment-form WLS (non-negative) | targeted | 25% | 1 | 28.42% | $9,441 | $836 |
| F3 | Payment-form WLS (non-negative) | targeted | 25% | 2 | 58.22% | $9,449 | $834 |
| F3 | Tweedie GLM | chronic | 5% | 1 | 0.61% | $944 | $144 |
| F3 | Tweedie GLM | chronic | 5% | 2 | 1.32% | $1,015 | $186 |
| F3 | Tweedie GLM | chronic | 10% | 1 | 1.21% | $953 | $178 |
| F3 | Tweedie GLM | chronic | 10% | 2 | 2.64% | $1,023 | $155 |
| F3 | Tweedie GLM | chronic | 25% | 1 | 2.95% | $937 | $193 |
| F3 | Tweedie GLM | chronic | 25% | 2 | 6.49% | $1,007 | $171 |
| F3 | Tweedie GLM | own-targeted | 5% | 1 | 2.99% | $4,630 | $1,642 |
| F3 | Tweedie GLM | own-targeted | 5% | 2 | 8.01% | $6,155 | $2,163 |
| F3 | Tweedie GLM | own-targeted | 10% | 1 | 6.41% | $5,055 | $2,083 |
| F3 | Tweedie GLM | own-targeted | 10% | 2 | 16.48% | $6,392 | $2,614 |
| F3 | Tweedie GLM | own-targeted | 25% | 1 | 15.56% | $4,939 | $1,867 |
| F3 | Tweedie GLM | own-targeted | 25% | 2 | 42.60% | $6,607 | $2,909 |
| F3 | Tweedie GLM | prevalence | 5% | 1 | 0.19% | $300 | $94 |
| F3 | Tweedie GLM | prevalence | 5% | 2 | 0.49% | $376 | $95 |
| F3 | Tweedie GLM | prevalence | 10% | 1 | 0.50% | $397 | $112 |
| F3 | Tweedie GLM | prevalence | 10% | 2 | 0.85% | $330 | $86 |
| F3 | Tweedie GLM | prevalence | 25% | 1 | 1.16% | $368 | $71 |
| F3 | Tweedie GLM | prevalence | 25% | 2 | 2.22% | $344 | $96 |
| F3 | Tweedie GLM | targeted | 5% | 1 | 1.92% | $2,974 | $332 |
| F3 | Tweedie GLM | targeted | 5% | 2 | 4.71% | $3,618 | $642 |
| F3 | Tweedie GLM | targeted | 10% | 1 | 3.70% | $2,916 | $234 |
| F3 | Tweedie GLM | targeted | 10% | 2 | 9.56% | $3,710 | $525 |
| F3 | Tweedie GLM | targeted | 25% | 1 | 9.36% | $2,971 | $268 |
| F3 | Tweedie GLM | targeted | 25% | 2 | 22.96% | $3,561 | $312 |
| F3 | Two-part | chronic | 5% | 1 | 0.81% | $1,245 | $301 |
| F3 | Two-part | chronic | 5% | 2 | 1.72% | $1,321 | $315 |
| F3 | Two-part | chronic | 10% | 1 | 1.63% | $1,283 | $338 |
| F3 | Two-part | chronic | 10% | 2 | 3.56% | $1,379 | $376 |
| F3 | Two-part | chronic | 25% | 1 | 4.01% | $1,271 | $324 |
| F3 | Two-part | chronic | 25% | 2 | 8.78% | $1,360 | $368 |
| F3 | Two-part | own-targeted | 5% | 1 | 1.60% | $2,473 | $813 |
| F3 | Two-part | own-targeted | 5% | 2 | 3.60% | $2,766 | $1,202 |
| F3 | Two-part | own-targeted | 10% | 1 | 3.26% | $2,572 | $944 |
| F3 | Two-part | own-targeted | 10% | 2 | 7.50% | $2,908 | $1,277 |
| F3 | Two-part | own-targeted | 25% | 1 | 7.98% | $2,532 | $871 |
| F3 | Two-part | own-targeted | 25% | 2 | 18.79% | $2,911 | $1,292 |
| F3 | Two-part | prevalence | 5% | 1 | 0.49% | $763 | $94 |
| F3 | Two-part | prevalence | 5% | 2 | 1.04% | $795 | $71 |
| F3 | Two-part | prevalence | 10% | 1 | 1.05% | $827 | $128 |
| F3 | Two-part | prevalence | 10% | 2 | 2.13% | $827 | $110 |
| F3 | Two-part | prevalence | 25% | 1 | 2.55% | $808 | $100 |
| F3 | Two-part | prevalence | 25% | 2 | 5.40% | $836 | $118 |
| F3 | Two-part | targeted | 5% | 1 | 1.31% | $2,030 | $704 |
| F3 | Two-part | targeted | 5% | 2 | 3.02% | $2,323 | $858 |
| F3 | Two-part | targeted | 10% | 1 | 2.63% | $2,071 | $759 |
| F3 | Two-part | targeted | 10% | 2 | 6.29% | $2,436 | $973 |
| F3 | Two-part | targeted | 25% | 1 | 6.57% | $2,083 | $725 |
| F3 | Two-part | targeted | 25% | 2 | 15.54% | $2,408 | $941 |
| F3 | WLS | chronic | 5% | 1 | 0.32% | $488 | $251 |
| F3 | WLS | chronic | 5% | 2 | 1.04% | $797 | $298 |
| F3 | WLS | chronic | 10% | 1 | 0.74% | $580 | $286 |
| F3 | WLS | chronic | 10% | 2 | 2.00% | $774 | $161 |
| F3 | WLS | chronic | 25% | 1 | 1.67% | $528 | $275 |
| F3 | WLS | chronic | 25% | 2 | 4.92% | $760 | $211 |
| F3 | WLS | own-targeted | 5% | 1 | 5.81% | $8,953 | $510 |
| F3 | WLS | own-targeted | 5% | 2 | 11.80% | $9,036 | $509 |
| F3 | WLS | own-targeted | 10% | 1 | 11.51% | $9,044 | $466 |
| F3 | WLS | own-targeted | 10% | 2 | 23.38% | $9,033 | $526 |
| F3 | WLS | own-targeted | 25% | 1 | 28.55% | $9,026 | $535 |
| F3 | WLS | own-targeted | 25% | 2 | 58.57% | $9,045 | $531 |
| F3 | WLS | prevalence | 5% | 1 | 0.22% | $338 | $235 |
| F3 | WLS | prevalence | 5% | 2 | 0.43% | $325 | $213 |
| F3 | WLS | prevalence | 10% | 1 | 0.36% | $283 | $170 |
| F3 | WLS | prevalence | 10% | 2 | 0.65% | $251 | $68 |
| F3 | WLS | prevalence | 25% | 1 | 1.02% | $324 | $65 |
| F3 | WLS | prevalence | 25% | 2 | 1.76% | $272 | $106 |
| F3 | WLS | targeted | 5% | 1 | 5.52% | $8,500 | $765 |
| F3 | WLS | targeted | 5% | 2 | 11.31% | $8,660 | $721 |
| F3 | WLS | targeted | 10% | 1 | 10.87% | $8,539 | $803 |
| F3 | WLS | targeted | 10% | 2 | 22.29% | $8,610 | $779 |
| F3 | WLS | targeted | 25% | 1 | 27.05% | $8,551 | $823 |
| F3 | WLS | targeted | 25% | 2 | 55.70% | $8,601 | $817 |

**Table A4.** Actuarial sense checks on the SHAP values.

*Spearman correlation between a feature's value and its SHAP contribution across held-out persons, by fold. Positive means predicted spending rises with the feature.*

| Feature | Mean | Min across folds | Max across folds |
|---|---:|---:|---:|
| log_spend_y1 | 0.876 | 0.852 | 0.910 |
| n_body_systems | 0.858 | 0.789 | 0.892 |
| n_conditions | 0.854 | 0.802 | 0.902 |

**Table A5.** Hyperparameters chosen by inner validation, across the 15 outer folds.

| model | parameter | chosen (count of folds) |
|---|---:|---:|
| tweedie | alpha | 1.0 (5); 10.0 (5); 0.01 (3); 0.1 (2) |
| twopart | C | 0.1 (15) |
| twopart | alpha | 1.0 (12); 10.0 (3) |
| enet | alpha | 1.0 (15) |
| enet | l1_ratio | 0.8 (15) |
| gbm | num_leaves | 31 (6); 63 (5); 15 (4) |
| gbm | min_child_samples | 100 (9); 50 (6) |
| gbm | n_estimators | 121 (2); 140 (2); 203 (1); 177 (1); 123 (1); 99 (1); 261 (1); 125 (1); 160 (1); 167 (1); 168 (1); 115 (1); 105 (1) |
| gbm_mse | num_leaves | 63 (7); 31 (5); 15 (3) |
| gbm_mse | min_child_samples | 100 (10); 50 (5) |
| gbm_mse | n_estimators | 104 (1); 155 (1); 105 (1); 359 (1); 77 (1); 78 (1); 81 (1); 261 (1); 151 (1); 169 (1); 112 (1); 385 (1); 140 (1); 83 (1); 69 (1) |
| mlp | lr | 0.003 (7); 0.001 (4); 0.01 (4) |
| mlp | hidden | (64, 32) (8); (32,) (4); (128, 64) (3) |
| mlp | epochs | 32 (3); 29 (3); 33 (3); 28 (2); 31 (1); 42 (1); 43 (1); 56 (1) |
| cann | lr | 0.001 (8); 0.003 (5); 0.01 (2) |
| cann | hidden | (64, 32) (10); (128, 64) (3); (32,) (2) |
| cann | epochs | 25 (8); 26 (7) |

**Table A6.** Year-1 CCSR categories entering the feature sets.

*Categories held by at least 0.5% of the sample in year 1. MEPS body-system placeholder codes (XXX000) are excluded.*

| CCSR | Description | Prevalence |
|---|---:|---:|
| CIR007 | Essential hypertension | 22.41% |
| END010 | Disorders of lipid metabolism | 16.96% |
| END002+END005 | Diabetes mellitus without complication / Diabetes mellitus, Type 2 | 9.71% |
| MUS010 | Musculoskeletal pain, not low back pain | 9.63% |
| MBD005 | Anxiety and fear-related disorders | 8.05% |
| MBD002 | Depressive disorders | 7.08% |
| END001 | Thyroid disorders | 7.05% |
| MUS006 | Osteoarthritis | 6.76% |
| RSP009 | Asthma | 6.34% |
| DIG004 | Esophageal disorders | 5.98% |
| NVS016 | Sleep wake disorders | 5.48% |
| SKN007 | Other specified and unspecified skin disorders | 4.97% |
| FAC016 | Exposure, encounters, screening or contact with infectious disease | 4.75% |
| MUS011 | Spondylopathies/spondyloarthropathy (including infective) | 4.49% |
| RSP006 | Other specified upper respiratory infections | 4.21% |
| INJ031 | Allergic reactions | 3.89% |
| SYM016 | Other general signs and symptoms | 3.89% |
| RSP007 | Other specified and unspecified upper respiratory disease | 3.66% |
| INJ067 | Allergic reactions, subsequent encounter | 3.11% |
| MBD014 | Neurodevelopmental disorders | 3.08% |
| CIR011 | Coronary atherosclerosis and other heart disease | 3.06% |
| INJ064 | Other unspecified injuries, subsequent encounter | 2.98% |
| MUS025 | Other specified connective tissue disease | 2.96% |
| INF012 | Coronavirus disease – 2019 (COVID-19) | 2.90% |
| GEN004 | Urinary tract infections | 2.74% |
| SYM013 | Respiratory signs and symptoms | 2.70% |
| NVS010 | Headache; including migraine | 2.67% |
| RSP003 | Influenza | 2.55% |
| INF008 | Viral infection | 2.53% |
| SKN002 | Other specified inflammatory condition of skin | 2.43% |
| EYE002 | Cataract and other lens disorders | 2.36% |
| SYM006 | Abdominal pain and other digestive/abdomen signs and symptoms | 2.33% |
| CIR017 | Cardiac dysrhythmias | 2.31% |
| RSP001 | Sinusitis | 2.28% |
| MUS038 | Low back pain | 2.25% |
| DIG025 | Other specified and unspecified gastrointestinal disorders | 2.10% |
| END007 | Nutritional deficiencies | 2.07% |
| EAR001 | Otitis media | 2.05% |
| SYM010 | Nervous system signs and symptoms | 2.05% |
| FAC014 | Medical examination/evaluation | 2.02% |
| INF003 | Bacterial infections | 2.01% |
| INJ061 | Sprains and strains, subsequent encounter | 2.00% |
| FAC003 | Encounter for observation and examination for conditions ruled out (excludes infectious disease, neoplasm, mental disorders) | 1.97% |
| SYM014 | Skin/Subcutaneous signs and symptoms | 1.95% |
| DIG002 | Disorders of teeth and gingiva | 1.92% |
| FAC012 | Other specified encounters and counseling | 1.92% |
| SYM017 | Abnormal findings without diagnosis | 1.91% |
| FAC009 | Implant, device or graft related encounter | 1.77% |
| RSP008 | Chronic obstructive pulmonary disease and bronchiectasis | 1.77% |
| MBD007 | Trauma- and stressor-related disorders | 1.76% |
| EYE003 | Glaucoma | 1.76% |
| SYM012 | Circulatory signs and symptoms | 1.71% |
| NEO028 | Skin cancers - all other types | 1.70% |
| FAC013 | Contraceptive and procreative management | 1.67% |
| EYE009 | Refractive error | 1.67% |
| MUS003 | Rheumatoid arthritis and related disease | 1.66% |
| CIR009 | Acute myocardial infarction | 1.62% |
| MUS007 | Other specified joint disorders | 1.49% |
| EYE005 | Retinal and vitreous conditions | 1.45% |
| FAC025 | Other specified status | 1.44% |
| INF004 | Fungal infections | 1.39% |
| NVS012 | Transient cerebral ischemia | 1.35% |
| END011 | Fluid and electrolyte disorders | 1.34% |
| RSP005 | Acute bronchitis | 1.34% |
| DIG008 | Other specified and unspecified disorders of stomach and duodenum | 1.31% |
| MUS033 | Gout | 1.31% |
| EAR006 | Other specified and unspecified disorders of the ear | 1.30% |
| PRG029 | Uncomplicated pregnancy, delivery or puerperium | 1.29% |
| NVS019 | Nervous system pain and pain syndromes | 1.28% |
| SKN001 | Skin and subcutaneous tissue infections | 1.24% |
| NVS017 | Nerve and nerve root disorders | 1.22% |
| NEO073 | Benign neoplasms | 1.22% |
| MUS009 | Tendon and synovial disorders | 1.19% |
| FAC008 | Neoplasm-related encounters | 1.19% |
| EAR004 | Hearing loss | 1.17% |
| EYE010 | Blindness and vision defects | 1.12% |
| SYM015 | General sensation/perception signs and symptoms | 1.11% |
| EYE008 | Oculofacial plastics and orbital conditions | 1.10% |
| RSP002 | Pneumonia (except that caused by tuberculosis) | 1.10% |
| GEN012 | Hyperplasia of prostate | 1.09% |
| SYM004 | Nausea and vomiting | 1.09% |
| GEN006 | Other specified and unspecified diseases of kidney and ureters | 1.04% |
| NVS015 | Polyneuropathies | 1.03% |
| EYE012 | Other specified eye disorders | 0.97% |
| CIR030 | Aortic and peripheral arterial embolism or thrombosis | 0.95% |
| END015 | Other specified and unspecified endocrine disorders | 0.93% |
| MUS028 | Other specified bone disease and musculoskeletal deformities | 0.93% |
| NVS009 | Epilepsy; convulsions | 0.93% |
| MBD003 | Bipolar and related disorders | 0.93% |
| NEO030 | Breast cancer - all other types | 0.92% |
| GEN023 | Menopausal disorders | 0.91% |
| GEN005 | Calculus of urinary tract | 0.85% |
| EYE001 | Cornea and external disease | 0.85% |
| NEO072 | Neoplasms of unspecified nature or uncertain behavior | 0.83% |
| MUS026 | Muscle disorders | 0.82% |
| INJ042 | Fracture of lower limb (except hip), subsequent encounter | 0.82% |
| INJ041 | Fracture of the upper limb, subsequent encounter | 0.81% |
| SYM011 | Genitourinary signs and symptoms | 0.79% |
| CIR012 | Nonspecific chest pain | 0.79% |
| CIR015 | Other and ill-defined heart disease | 0.79% |
| RSP016 | Other specified and unspecified lower respiratory disease | 0.78% |
| FAC010 | Other aftercare encounter | 0.77% |
| MBD013 | Miscellaneous mental and behavioral disorders/conditions | 0.76% |
| SYM002 | Fever | 0.76% |
| DIG010 | Abdominal hernia | 0.76% |
| GEN016 | Other specified male genital disorders | 0.74% |
| GEN025 | Other specified female genital disorders | 0.73% |
| NVS006 | Other nervous system disorders (often hereditary or degenerative) | 0.73% |
| MUS013 | Osteoporosis | 0.68% |
| INJ027 | Other unspecified injury | 0.68% |
| GEN008 | Urinary incontinence | 0.68% |
| GEN017 | Nonmalignant breast conditions | 0.67% |
| BLD003 | Aplastic anemia | 0.67% |
| NEO039 | Male reproductive system cancers - prostate | 0.65% |
| DIG001 | Intestinal infection | 0.65% |
| GEN007 | Other specified and unspecified diseases of bladder and urethra | 0.64% |
| INF009 | Parasitic, other specified and unspecified infections | 0.60% |
| NEO025 | Skin cancers - melanoma | 0.59% |
| DEN001 | Any dental condition including traumatic injury | 0.57% |
| INJ049 | Open wounds to limbs, subsequent encounter | 0.56% |
| CIR019 | Heart failure | 0.55% |
| DEN002 | Nontraumatic dental conditions | 0.53% |
| SKN005 | Contact dermatitis | 0.53% |
| DIG019 | Other specified and unspecified liver disease | 0.52% |
