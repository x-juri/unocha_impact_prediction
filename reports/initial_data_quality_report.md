# Initial Data Quality Report

Generated from the checked-in CSV snapshot in `initial_data`. Source documentation was used only to interpret fields, caveats, and expected joins; the workflow does not re-download live source data.

## Executive Summary

- Profiled 26 CSV files, 2,101,923 logical rows, and 253.04 MB of local data.
- The collection covers five source families: HNO/HPC needs, HRP plan requirements, COD-PS population, FTS requirements/funding, and CBPF pooled fund exports.
- HNO 2024/2025 and HRP include a second-row HXL metadata row; the profiling excludes it from logical row counts.
- `CBPFcontributions__20260418_071813_UTC.csv` and `ContributionsFlow__20260418_071825_UTC.csv` are exact file duplicates and should not be treated as independent evidence.
- The strongest cross-dataset keys are ISO3 country codes, admin p-codes, and HRP/FTS plan codes/IDs. CBPF exports need a separate fund-country lookup before country-level joins.

## Source Family Inventory

| source_family | files | logical_rows | size_mb | countries_observed |
| --- | --- | --- | --- | --- |
| CBPF | 10 | 1117 | 0.15 | 0 |
| COD-PS | 5 | 1359203 | 187.4 | 139 |
| FTS | 7 | 34481 | 8.17 | 179 |
| HNO | 3 | 706212 | 57.19 | 24 |
| HRP | 1 | 910 | 0.12 | 126 |

## Which Dataset Contains What

| source_family | files | unit_of_observation | time_coverage | geography | core_measures | identifiers | known_caveats |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HNO | hpc_hno_2024.csv \| hpc_hno_2025.csv \| hpc_hno_2026.csv | Country/admin-sector-category caseload row; wide measures for population statuses. | 2024 \| 2025 \| 2026 | Country ISO3; admin p-codes in 2024/2025 only. | Population, In Need, Targeted, Affected, Reached. | Country ISO3, admin p-codes, Cluster, Description, Category. | Do not sum people across sectors/statuses; 2026 schema has no admin columns; category is freeform/blank-heavy. |
| HRP | humanitarian-response-plans.csv | Humanitarian response plan or appeal version. | 1999 \| 2000 \| 2001 \| 2002 \| 2003 \| 2004 \| 2005 \| 2006 \| 2007 \| 2008 \| 2009 \| 2010 \| 2011 \| 2012 \| 2013 \| 2014 \| 2015 \| 2016 \| 2017 \| 2018 \| 2019 \| 2020 \| 2021 \| 2022 \| 2023 \| 2024 \| 2025 \| 2026 | One or more ISO3 codes in locations. | Original and revised requirements in USD. | code, internalId, locations, years, categories. | Contains multi-country plans and a HXL metadata row; not a realized funding table. |
| COD-PS | cod_population_admin0.csv through cod_population_admin4.csv | Population group, sex, age, and administrative unit row. | 2001 \| 2008 \| 2009 \| 2010 \| 2011 \| 2012 \| 2014 \| 2015 \| 2016 \| 2017 \| 2018 \| 2019 \| 2020 \| 2021 \| 2022 \| 2023 \| 2024 \| 2025 | ISO3 plus ADM1-ADM4 p-codes where available. | Population by reference year, sex, age range, and population group. | ISO3, country, admin p-codes, age/sex/population group. | Reference years vary by country/admin level; admin coverage becomes narrower at deeper levels. |
| FTS | fts_requirements_* and fts_*_funding_global.csv | Appeal requirement/funding row, cluster row, or reported funding flow. | 1980 \| 1981 \| 1999 \| 2000 \| 2001 \| 2002 \| 2003 \| 2004 \| 2005 \| 2006 \| 2007 \| 2008 \| 2009 \| 2010 \| 2011 \| 2012 \| 2013 \| 2014 \| 2015 \| 2016 \| 2017 \| 2018 \| 2019 \| 2020 \| 2021 \| 2022 \| 2023 \| 2024 \| 2025 \| 2026 \| 2027 \| 2028 \| 2029 \| 2030 \| 2031 | countryCode and multi-code source/destination location fields. | Requirements, funding, percent funded, flow amount USD, original amount/currency. | id, code, countryCode, clusterCode, flow id/refCode, destPlanCode/destPlanId. | Some funding is not linked to a plan; flow location fields can represent multiple countries. |
| CBPF | Allocation, contribution, sector, affected persons, and marker exports. | Fund-year, donor-fund-year, allocation, organization type, sector, or marker summary row. | 1999 \| 2000 \| 2001 \| 2002 \| 2003 \| 2004 \| 2005 \| 2006 \| 2007 \| 2008 \| 2009 \| 2010 \| 2011 \| 2012 \| 2013 \| 2014 \| 2015 \| 2016 \| 2017 \| 2018 \| 2019 \| 2020 \| 2021 \| 2022 \| 2023 \| 2024 \| 2025 \| 2026 \| 2027 \| 2028 | Fund names, not direct ISO3 country keys. | Allocations, contributions, direct/net funding, targeted/reached people, beneficiaries, project counts. | Year, fund/fund name, donor, partner type, allocation type, cluster, marker code. | Needs fund-country mapping for country joins; one contribution file is an exact duplicate of another. |

## Data Quality Highlights

- Completeness is intentionally uneven: COD admin depth narrows from 139 countries at admin0 to one country at admin4, while HNO 2026 is country-sector only in this snapshot.
- High-blank fields often represent structural absence rather than accidental loss, for example lower admin columns in shallower COD files and project/emergency fields in aggregate FTS flow exports.
- Several logical checks are domain-review flags, not automatic errors. Examples include targeted values exceeding in-need values, reached values exceeding targeted values, and funding exceeding requirements.
- FTS location fields and HRP locations can contain multiple country codes in one row; they must be exploded before country-level analysis.
- Source guidance says HNO people-in-need values should not be summed across sectors or statuses because people may appear in multiple groups.

### Exact Duplicate Checks

| check_type | item_a | item_b | status | detail |
| --- | --- | --- | --- | --- |
| exact_file_hash_duplicate | CBPFcontributions__20260418_071813_UTC.csv | ContributionsFlow__20260418_071825_UTC.csv | warning | 2 files share sha256 9accdb226637960feb4f02f43750bed84daef5fe3d94bcdd08e042ef8303884b. |

### Largest Quality Flags

| file | check_type | column | status | affected_count | total_count | affected_pct | detail |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cod_population_admin2.csv | missingness | ADM3_PCODE | warning | 1001583 | 1001583 | 100.0 | Column is blank in at least half of logical rows. Review whether this is structural or data loss. |
| cod_population_admin2.csv | missingness | ADM3_NAME | warning | 1001583 | 1001583 | 100.0 | Column is blank in at least half of logical rows. Review whether this is structural or data loss. |
| cod_population_admin2.csv | missingness | ADM4_PCODE | warning | 1001583 | 1001583 | 100.0 | Column is blank in at least half of logical rows. Review whether this is structural or data loss. |
| cod_population_admin2.csv | missingness | ADM4_NAME | warning | 1001583 | 1001583 | 100.0 | Column is blank in at least half of logical rows. Review whether this is structural or data loss. |
| hpc_hno_2024.csv | missingness | Reached | warning | 387308 | 387819 | 99.8682 | Column is blank in at least half of logical rows. Review whether this is structural or data loss. |
| hpc_hno_2024.csv | missingness | Info | warning | 386794 | 387819 | 99.7357 | Column is blank in at least half of logical rows. Review whether this is structural or data loss. |
| hpc_hno_2024.csv | missingness | Affected | warning | 383823 | 387819 | 98.9696 | Column is blank in at least half of logical rows. Review whether this is structural or data loss. |
| hpc_hno_2024.csv | missingness | Population | warning | 376929 | 387819 | 97.192 | Column is blank in at least half of logical rows. Review whether this is structural or data loss. |
| hpc_hno_2024.csv | missingness | Admin 1 PCode | warning | 371299 | 387819 | 95.7403 | Column is blank in at least half of logical rows. Review whether this is structural or data loss. |
| hpc_hno_2024.csv | missingness | Admin 1 Name | warning | 371299 | 387819 | 95.7403 | Column is blank in at least half of logical rows. Review whether this is structural or data loss. |
| hpc_hno_2025.csv | missingness | Reached | warning | 318237 | 318259 | 99.9931 | Column is blank in at least half of logical rows. Review whether this is structural or data loss. |
| hpc_hno_2025.csv | missingness | Affected | warning | 315027 | 318259 | 98.9845 | Column is blank in at least half of logical rows. Review whether this is structural or data loss. |
| hpc_hno_2025.csv | missingness | Info | warning | 312689 | 318259 | 98.2499 | Column is blank in at least half of logical rows. Review whether this is structural or data loss. |
| hpc_hno_2025.csv | missingness | Admin 1 PCode | warning | 302467 | 318259 | 95.038 | Column is blank in at least half of logical rows. Review whether this is structural or data loss. |
| hpc_hno_2025.csv | missingness | Admin 1 Name | warning | 302467 | 318259 | 95.038 | Column is blank in at least half of logical rows. Review whether this is structural or data loss. |
| hpc_hno_2025.csv | missingness | Population | warning | 287361 | 318259 | 90.2916 | Column is blank in at least half of logical rows. Review whether this is structural or data loss. |
| hpc_hno_2025.csv | missingness | Admin 3 PCode | warning | 246442 | 318259 | 77.4344 | Column is blank in at least half of logical rows. Review whether this is structural or data loss. |
| hpc_hno_2025.csv | missingness | Admin 3 Name | warning | 246442 | 318259 | 77.4344 | Column is blank in at least half of logical rows. Review whether this is structural or data loss. |
| cod_population_admin3.csv | missingness | ADM4_PCODE | warning | 241962 | 241962 | 100.0 | Column is blank in at least half of logical rows. Review whether this is structural or data loss. |
| cod_population_admin3.csv | missingness | ADM4_NAME | warning | 241962 | 241962 | 100.0 | Column is blank in at least half of logical rows. Review whether this is structural or data loss. |

### Columns With At Least 90% Missing Values

| file | column | row_count | blank_count | blank_pct |
| --- | --- | --- | --- | --- |
| cod_population_admin0.csv | ADM1_NAME | 6722 | 6722 | 100.0 |
| cod_population_admin0.csv | ADM1_PCODE | 6722 | 6722 | 100.0 |
| cod_population_admin0.csv | ADM2_NAME | 6722 | 6722 | 100.0 |
| cod_population_admin0.csv | ADM2_PCODE | 6722 | 6722 | 100.0 |
| cod_population_admin0.csv | ADM3_NAME | 6722 | 6722 | 100.0 |
| cod_population_admin0.csv | ADM3_PCODE | 6722 | 6722 | 100.0 |
| cod_population_admin0.csv | ADM4_NAME | 6722 | 6722 | 100.0 |
| cod_population_admin0.csv | ADM4_PCODE | 6722 | 6722 | 100.0 |
| cod_population_admin1.csv | ADM2_NAME | 91471 | 91471 | 100.0 |
| cod_population_admin1.csv | ADM2_PCODE | 91471 | 91471 | 100.0 |
| cod_population_admin1.csv | ADM3_NAME | 91471 | 91471 | 100.0 |
| cod_population_admin1.csv | ADM3_PCODE | 91471 | 91471 | 100.0 |
| cod_population_admin1.csv | ADM4_NAME | 91471 | 91471 | 100.0 |
| cod_population_admin1.csv | ADM4_PCODE | 91471 | 91471 | 100.0 |
| cod_population_admin2.csv | ADM3_NAME | 1001583 | 1001583 | 100.0 |
| cod_population_admin2.csv | ADM3_PCODE | 1001583 | 1001583 | 100.0 |
| cod_population_admin2.csv | ADM4_NAME | 1001583 | 1001583 | 100.0 |
| cod_population_admin2.csv | ADM4_PCODE | 1001583 | 1001583 | 100.0 |
| cod_population_admin3.csv | ADM4_NAME | 241962 | 241962 | 100.0 |
| cod_population_admin3.csv | ADM4_PCODE | 241962 | 241962 | 100.0 |

## Cross-Dataset Connections

| relationship | source | target | source_count | target_count | matched_count | unmatched_count | coverage_pct | sample_unmatched | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hno_country_to_cod_iso3 | HNO Country ISO3 | COD ISO3 | 24 | 143 | 20 | 4 | 83.3333 | MMR \| SYR \| UKR \| YEM | Country-level needs can be checked against COD population availability. |
| hno_country_to_fts_country_code | HNO Country ISO3 | FTS countryCode | 24 | 110 | 24 | 0 | 100.0 |  | Country-level needs can be compared to appeal funding where codes overlap. |
| hno_country_to_hrp_locations | HNO Country ISO3 | HRP locations | 24 | 126 | 24 | 0 | 100.0 |  | HNO countries can be linked to HRP plans by ISO3 location lists. |
| hrp_locations_to_cod_iso3 | HRP locations | COD ISO3 | 126 | 143 | 104 | 22 | 82.5397 | BGR \| COG \| CZE \| EST \| GMB \| GNB \| GRC \| HRV \| JOR \| LBN \| LBY \| LTU \| LVA \| MKD \| MMR \| RUS \| SRB \| SVN \| SYR \| TKM \| UKR \| YEM | HRP plan countries can be checked against COD population availability. |
| fts_flow_locations_to_hrp_locations | FTS flow locations | HRP locations | 180 | 126 | 111 | 69 | 61.6667 | ALB \| ARE \| ARM \| AUS \| AUT \| AZE \| BEL \| BIH \| BLZ \| BRB \| BRN \| BTN \| BWA \| CAN \| CHE \| CHN \| COM \| CYP \| DEU \| DNK \| DZA \| ESP \| FIN \| FRA \| FSM | FTS flow rows may contain multi-country location lists. |
| hno_admin1_pcode_to_cod_admin1_pcode | HNO Admin 1 PCode | COD ADM1_PCODE | 156 | 1879 | 52 | 104 | 33.3333 | AF01 \| AF02 \| AF03 \| AF04 \| AF05 \| AF06 \| AF07 \| AF08 \| AF09 \| AF10 \| AF11 \| AF12 \| AF13 \| AF14 \| AF15 \| AF16 \| AF17 \| AF18 \| AF19 \| AF20 \| AF21 \| AF22 \| AF23 \| AF24 \| AF25 | Admin p-code matching is strongest for 2024/2025 HNO because 2026 has no admin fields. |
| hno_admin2_pcode_to_cod_admin2_pcode | HNO Admin 2 PCode | COD ADM2_PCODE | 2886 | 18957 | 1738 | 1148 | 60.2218 | AF0101 \| AF0102 \| AF0103 \| AF0104 \| AF0105 \| AF0106 \| AF0107 \| AF0108 \| AF0109 \| AF0110 \| AF0111 \| AF0112 \| AF0113 \| AF0114 \| AF0115 \| AF0201 \| AF0202 \| AF0203 \| AF0204 \| AF0205 \| AF0206 \| AF0207 \| AF0301 \| AF0302 \| AF0303 | Admin p-code matching is strongest for 2024/2025 HNO because 2026 has no admin fields. |
| hno_admin3_pcode_to_cod_admin3_pcode | HNO Admin 3 PCode | COD ADM3_PCODE | 2553 | 8259 | 906 | 1647 | 35.4877 | BF130001 \| BF130002 \| BF130003 \| BF130004 \| BF130005 \| BF130006 \| BF130007 \| BF460101 \| BF460102 \| BF460103 \| BF460104 \| BF460105 \| BF460106 \| BF460107 \| BF460108 \| BF460109 \| BF460110 \| BF460201 \| BF460202 \| BF460203 \| BF460204 \| BF460205 \| BF460206 \| BF460301 \| BF460302 | Admin p-code matching is strongest for 2024/2025 HNO because 2026 has no admin fields. |
| hrp_code_to_fts_plan_code | HRP code | FTS code/destPlanCode | 899 | 888 | 882 | 17 | 98.109 | CXGLR06 \| CXWAF05 \| FDMA17 \| O20 \| OCOVD20 \| OEUR17 \| R00 \| R01 \| R02 \| R0203 \| R03 \| R0304 \| R04 \| R05 \| R0506 \| RBFACMRGMBMLIMRTNERNGASENTCD15 \| RBFACMRGMBMLIMRTNERNGASENTCD16 | Best plan-level join candidate by public plan/appeal code. |
| hrp_internal_id_to_fts_plan_id | HRP internalId | FTS id/destPlanId | 910 | 889 | 331 | 579 | 36.3736 | 1 \| 10 \| 1065 \| 1074 \| 143 \| 147 \| 15 \| 161 \| 1746 \| 18 \| 180 \| 189 \| 196 \| 1992 \| 2 \| 20 \| 21 \| 2146 \| 2289 \| 2292 \| 2294 \| 2296 \| 2299 \| 2316 \| 2342 | Best plan-level join candidate by internal numeric identifier. |
| hno_sector_label_to_fts_cluster | HNO sector descriptions | FTS cluster names | 182 | 682 | 93 | 89 | 51.0989 | acci n contra minas \| anticipatory cccm \| anticipatory education \| anticipatory food security \| anticipatory health \| anticipatory nutrition \| anticipatory protection overall \| anticipatory snfi \| anticipatory wash \| camp coordination and camp management cccm conflicts \| camp coordination and camp management cccm nd \| camp coordination and camp management conflict response \| camp coordination and camp management natural disaster response \| cash a usage multiple \| cash usage multiple caseload \| child protection natural disaster \| coordinaci n y gesti n de campamentos \| coordination et organisation de camp \| css \| education conflict response \| education natural disasters response \| education natural disater \| epah \| final hrp caseload \| food security and livelihood conflict response | Uses normalized labels, not authoritative sector IDs. |
| hno_sector_label_to_cbpf_cluster | HNO sector descriptions | CBPF sector names | 182 | 12 | 10 | 172 | 5.4945 | abris \| abris ame \| abris biens non alimentaires \| abris et ame \| abris et ana \| abris et biens non alimentaires \| abris et bna \| abris nfi \| abrisandnfi \| acci n contra minas \| agriculture \| agua saneamiento e higiene \| albergue y art culos no alimentarios \| alojamiento de emergencias \| alojamiento energ a y enseres \| alojamientos temporales cccm \| anticipatory action \| anticipatory cccm \| anticipatory education \| anticipatory food security \| anticipatory health \| anticipatory nutrition \| anticipatory protection overall \| anticipatory snfi \| anticipatory wash | Uses normalized labels, not authoritative sector IDs. |
| cbpf_fund_names_need_country_lookup | CBPF fund name columns | country ISO3 | 51 |  |  | 51 |  | (Closed) Angola \| (Closed) Cote d'Ivoire \| (Closed) DRC - ERF \| (Closed) ERF Afghanistan \| (Closed) Indonesia \| (Closed) Iraq \| (Closed) Kenya \| (Closed) Liberia \| (Closed) Nepal \| (Closed) Regional Syrian Arab Republic \| (Closed) Somalia ERF \| (Closed) Uganda \| (Closed) Zimbabwe \| Asia Pacific Regional Envelope (RHPF-AP) \| Bangladesh (AP-RHPF) \| Burkina Faso (RhPF-WCA) \| Chad (RhPF-WCA) \| Colombia (RhPF-LAC) \| El Salvador (RhPF-LAC) \| Fiji (AP-Rhpf) \| Guatemala (RhPF-LAC) \| Haiti (RhPF-LAC) \| Honduras (RhPF-LAC) \| Mali (RhPF-WCA) \| Mozambique (RhPF) | No direct ISO3 key is present in the CBPF exports. 28 fund names look regional or closed and need a fund-country lookup before country joins. |

## What We Can And Cannot Say

| topic | can_say | cannot_say | primary_datasets | quality_notes |
| --- | --- | --- | --- | --- |
| HNO needs and response caseloads | For available countries/sectors/years, describe reported population, people in need, targeted, affected, and reached values. | Cannot infer unique people across sectors/statuses or causal effect of assistance. | hpc_hno_2024.csv \| hpc_hno_2025.csv \| hpc_hno_2026.csv | HNO 2024/2025 include admin fields; 2026 is country-sector only in this snapshot. |
| HRP requirements | Describe original and revised USD requirements by response plan, year, category, and location list. | Cannot say which requirements were funded unless joined to FTS; cannot treat regional plans as single-country plans without parsing locations. | humanitarian-response-plans.csv | Includes HXL metadata row and multi-country location lists. |
| Funding against requirements | Compare reported FTS requirements, funding, and percent funded by appeal/country/year/cluster where identifiers exist. | Cannot guarantee complete real-time funding; cannot compare unlinked flows to HRP requirements without plan linkage. | fts_requirements_* \| fts_*_funding_global.csv | Some rows are 'Not specified' with blank plan identifiers; flow locations can contain multiple countries. |
| Population baselines | Use COD-PS as population baselines by country/admin level, sex, age range, population group, and reference year. | Cannot assume every value is current-year or that all countries have the same admin depth. | cod_population_admin0.csv through cod_population_admin4.csv | Reference years and admin depth vary; deeper admin files cover fewer countries. |
| CBPF pooled fund allocations and contributions | Describe allocations, contribution trends, sectors, partner types, targeted/reached people, and gender/age marker summaries by fund/year. | Cannot reliably join to country ISO3 or HRP without a fund-country mapping; cannot use duplicated contribution exports as independent evidence. | CBPF exports | Fund names include regional and closed funds; contribution flow and contribution exports are exact duplicates. |
| Cross-dataset joins | Join strongest on ISO3, admin p-codes, HRP/FTS plan codes or IDs, and normalized sector labels for exploratory checks. | Cannot treat name-only joins as authoritative; cannot resolve CBPF country coverage from fund names alone. | All source families | The cross-reference coverage table lists matched/unmatched values and sample gaps. |

## Reasonable Questions To Ask Next

- Which countries have HNO needs, HRP requirements, FTS funding, and COD population coverage in the same year?
- Where do HNO country/admin p-codes fail to match COD-PS p-codes, and are mismatches due to old boundaries, missing COD levels, or source inconsistency?
- How do HRP revised requirements compare with FTS funding and percent funded by country, year, and cluster?
- Which sectors have comparable labels across HNO, FTS, and CBPF, and where do sector naming conventions block analysis?
- Which CBPF funds can be mapped cleanly to one country, and which regional/closed funds need manual treatment?
- Which population denominators are current enough to support per-capita or share-of-population indicators?

## Appendix Tables

- `reports/tables/file_inventory.csv`
- `reports/tables/dataset_catalog.csv`
- `reports/tables/column_profile.csv`
- `reports/tables/missingness_profile.csv`
- `reports/tables/value_distributions.csv`
- `reports/tables/numeric_profile.csv`
- `reports/tables/duplicate_checks.csv`
- `reports/tables/quality_flags.csv`
- `reports/tables/cross_reference_coverage.csv`
- `reports/tables/can_cannot_say_matrix.csv`
- `reports/tables/validation_results.csv`

## Source Notes

- [HNO / HPC](https://hdx-hapi.readthedocs.io/en/latest/data_usage_guides/affected_people/): HNO data represents people by sector/status. Source guidance warns not to sum PIN across sectors or population statuses because the same people can appear in multiple groups.
- [FTS](https://hdx-hapi.readthedocs.io/en/latest/data_usage_guides/coordination_and_context/): FTS funding data is reported by donors and recipient organizations; timepoints are not regular and appeal-linked funding is the most directly comparable to HRP requirements.
- [COD-PS](https://knowledge.base.unocha.org/wiki/spaces/imtoolbox/pages/2491252951/COD-PS%2BStandards%2Band%2BProcess): COD-PS follows a best-available-data principle. Reference year, p-code consistency, sex/age breakdown, and admin-level coherence are core quality dimensions.
- [HRP](https://knowledge.base.unocha.org/wiki/spaces/imtoolbox/pages/42046871/Humanitarian%2BResponse%2BPlan%2BHRP): HRPs build on HNO evidence and communicate strategic objectives, cluster plans, and resource mobilization needs.
- [CBPF](https://cbpf.data.unocha.org/): The CBPF Data Hub provides contribution and allocation views for country-based pooled funds, including allocations, sectors, targeted and reached people, and gender/age marker views.
