# Bibliography verification

Checked 2026-10-31, for `refs_verified.bib`.

## Result

- **43 requested entries: all exist.** 36 verified, 7 corrected, 0 not found.
- **4 entries added** with new keys (last rows of the table). They clearly match the topics you named.
- **Label-free DEM inversion: no prior work found.** See the section below.

Status meanings:

- **verified**: everything you gave was right. I may have filled in missing fields such as end page, DOI or the full author list.
- **corrected**: something you gave was abbreviated, incomplete or different from the published record.
- **added**: not in your list.

Checks on the bib file: it compiles with `ieeenat_fullname.bst` (47 entries, no BibTeX warnings), and proper nouns survive that style's sentence casing. Every DOI entry except the Zenodo ones was compared field by field against Crossref. The only differences are the ones explained in the table.

## Items that need your attention

1. **The draft cites a key that has no entry.** `\cite{supervised}` appears in `sec/1_intro.tex`, `sec/5_protocol.tex` and `sec/6_results.tex`. The draft describes it as an emulator with Fourier-feature inputs, a larger network and a predicted distribution per output, which matches `perezdiaz2026demnet`. It appears to be the project's supervised comparator (same 917 training timestamps, same viewer host, same solver tracks). I did not add a duplicate key. Point the citations at `perezdiaz2026demnet` once a co-author confirms it is the right reference for those checkpoints.
2. **`athiray2024enet` is real, but its ElasticNet use is narrower than a reader might assume.** The paper uses ElasticNet (Zou and Hastie, via scikit-learn) on MaGIXS-1 spectroheliograms. On AIA it runs ElasticNet only as one of three solvers in a comparison (their Figure 5), alongside `aia_sparse_em_init` and `xrt_dem_iterative`. Their objective, Eq. 2, is `||W(y - Mx)||^2 + a*rho*||x||_1 + 0.5*a*(1 - rho)*||x||^2`, with no `1/(2C)` factor. The current draft wording in `sec/2_related.tex` (ElasticNet listed as an approach, cited with Zou and Hastie) is fine. Do not cite it as the source of the `C = n_obs` form or as a validated AIA pipeline.
3. **`levis2022bhnerf` has two page ranges.** IEEE and Crossref give 19809--19818, which matches the DOI and is what the bib uses. The CVF open-access page gives 19841--19850. Both are genuine.
4. **`perezdiaz2026demnet` is `@misc`, not a CVPR workshop paper.** The project page lists CV4Science at CVPR 2026 and PAI26. The CV4Science site says it has no long-form workshop papers, and the CVF workshop listing has no entry. The archival record is Zenodo, in the PAI2026 community.

## Entry table

| key | status | confirmed on | corrected or noted |
|---|---|---|---|
| lemen2012aia | verified | https://api.crossref.org/works/10.1007/s11207-011-9776-8 | 47 authors, first 8 listed. End page added. Crossref `issued` says 2011 (online first); volume 275 is dated 2012. |
| pesnell2012sdo | verified | https://api.crossref.org/works/10.1007/s11207-011-9841-3 | End page added. Same 2011 and 2012 note. |
| boerner2012calibration | corrected | https://api.crossref.org/works/10.1007/s11207-011-9804-8 | Title: "on the SDO" is "on the Solar Dynamics Observatory (SDO)". 20 authors, first 8 listed. |
| cheung2015dem | corrected | https://api.crossref.org/works/10.1088/0004-637X/807/2/143 | Title expanded to the full published form (AIA, SDO and DEM are spelled out). |
| athiray2024enet | verified | https://iopscience.iop.org/article/10.3847/1538-4357/ad1837 and https://arxiv.org/abs/2401.12372 | DOI added. See item 2 above. |
| hannah2012regularized | verified | https://api.crossref.org/works/10.1051/0004-6361/201117576 | None. |
| kashyap1998mcmc | corrected | https://api.crossref.org/works/10.1086/305964 | Subtitle added: "Application to Solar Extreme-Ultraviolet Spectra". End page added. |
| plowman2013fast | verified | https://api.crossref.org/works/10.1088/0004-637X/771/1/2 | None. |
| plowman2020simple | verified | https://api.crossref.org/works/10.3847/1538-4357/abc260 | None. |
| craig1976limitations | verified | https://articles.adsabs.harvard.edu/pdf/1976A%26A....49..239C | Journal scan on ADS; the first page reads "Astron. & Astrophys. 49, 239-250 (1976)". End page added. No DOI found. |
| wright2019deepem | verified | https://zenodo.org/records/2587015 and https://api.datacite.org/dois/10.5281/zenodo.2587015 | Zenodo software record, v1.0, 2019-03-07, 8 authors. Kept as `@misc`. |
| szenicer2019virtual | verified | https://api.crossref.org/works/10.1126/sciadv.aaw6548 | Crossref drops the accents in Muñoz-Jaramillo; kept as in Zenodo and OpenAlex. |
| wang2024supersynthia | verified | https://api.crossref.org/works/10.3847/1538-4357/ad41e3 | 10 authors, all listed. DOI added. |
| barnes2020aiapy | verified | https://api.crossref.org/works/10.21105/joss.02801 | 12 authors, first 8 listed. |
| hofmeister2025psf | verified | https://api.crossref.org/works/10.3847/1538-4365/adbaed | DOI added. |
| bayesdem2026 | corrected | https://iopscience.iop.org/article/10.3847/1538-4357/ae88ef and https://arxiv.org/abs/2609.07858 | Authors: N. Balodhi and R. J. Morton. Already published: ApJ 1007(1), 93, 2026 August 10, so it is cited as the journal paper, with the arXiv id kept. |
| chen1998basis | verified | https://api.crossref.org/works/10.1137/S1064827596304010 | End page added. |
| zou2005elastic | verified | https://api.crossref.org/works/10.1111/j.1467-9868.2005.00503.x | End page added. Journal name is Crossref's current form. |
| nocedal2006numerical | verified | https://api.crossref.org/works/10.1007/978-0-387-40065-5 and https://vufind.katalog.k.utb.cz/Record/66668 | Second edition and ISBN from the library record. |
| courant1943variational | verified | https://api.crossref.org/works/10.1090/S0002-9904-1943-07818-4 | End page added. |
| han1979exact | verified | https://api.crossref.org/works/10.1007/BF01588250 | End page added. |
| fiacco1968nonlinear | verified | https://openlibrary.org/works/OL4447951W and https://api.crossref.org/works/10.1137/1.9781611971316 | Wiley, New York, 1968 (LCCN 68030909). The SIAM 1990 reprint DOI is in the note field. Catalog records for the reprint say it first appeared from Research Analysis Corporation, McLean, VA; the Wiley edition stays as you gave it. |
| amos2023tutorial | verified | https://api.crossref.org/works/10.1561/2200000102 | End page added. |
| gregor2010lista | verified | https://icml.cc/Conferences/2010/papers/449.pdf and https://mlanthology.org/icml/2010/gregor2010icml-learning | ICML PDF confirms title, authors and venue. Pages 399--406 come from the secondary record. No DOI found. |
| donti2021dc3 | verified | https://iclr.cc/virtual/2021/poster/2868 and https://arxiv.org/abs/2104.12225 | No page numbers for ICLR. |
| park2023primal | verified | https://api.crossref.org/works/10.1609/aaai.v37i4.25520 | Issue, pages and DOI added. |
| fioretto2020lagrangian | corrected | https://api.crossref.org/works/10.1007/978-3-030-67670-4_8 | Year is 2021: the ECML PKDD 2020 proceedings came out in 2021. Key unchanged. Six authors, LNCS 12461, pages 118--135. |
| kotary2021survey | verified | https://api.crossref.org/works/10.24963/ijcai.2021/610 | Pages and DOI added. |
| raissi2019pinn | corrected | https://api.crossref.org/works/10.1016/j.jcp.2018.10.045 | Full title added: "...: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations". |
| wang2021pathologies | verified | https://api.crossref.org/works/10.1137/20M1318043 | End page added. The arXiv title differs slightly; the journal title is used. |
| krishnapriyan2021failure | verified | https://papers.nips.cc/paper_files/paper/2021/file/df438e5206f31600e6ae4af72f2725f1-Bibtex.bib | NeurIPS 34, pages 26548--26560. |
| levis2022bhnerf | verified | https://api.crossref.org/works/10.1109/CVPR52688.2022.01922 and https://openaccess.thecvf.com/content/CVPR2022/html/Levis_Gravitationally_Lensed_Black_Hole_Emission_Tomography_CVPR_2022_paper.html | See item 3 above. |
| feng2026pidef | corrected | https://openaccess.thecvf.com/content/CVPR2026/html/Feng_Dynamic_Black-hole_Emission_Tomography_with_Physics-informed_Neural_Fields_CVPR_2026_paper.html | Published at CVPR 2026, pages 12511--12521 (Highlight, per https://cvpr.thecvf.com/virtual/2026/poster/37695). No IEEE DOI found yet; arXiv id kept. |
| zhao2025cosmic | verified | https://iclr.cc/virtual/2025/poster/30603 and https://arxiv.org/abs/2504.15262 | ICLR 2025 poster. No page numbers. |
| ulyanov2018dip | verified | https://openaccess.thecvf.com/content_cvpr_2018/html/Ulyanov_Deep_Image_Prior_CVPR_2018_paper.html | Crossref lists the authors in reverse. CVF gives Ulyanov, Vedaldi, Lempitsky, which is what the bib uses. |
| ongie2020inverse | verified | https://api.crossref.org/works/10.1109/JSAIT.2020.2991563 | End page added. |
| mildenhall2020nerf | verified | https://api.crossref.org/works/10.1007/978-3-030-58452-8_24 | Pages from Crossref. LNCS 12346 from library catalog records. |
| tancik2020fourier | verified | https://papers.nips.cc/paper_files/paper/2020/file/55053683268957697aa39fba6f231c68-Bibtex.bib | NeurIPS 33, pages 7537--7547, 9 authors. |
| elfwing2018silu | verified | https://api.crossref.org/works/10.1016/j.neunet.2017.12.012 | End page added. |
| kingma2015adam | verified | https://arxiv.org/abs/1412.6980 and https://iclr.cc/archive/www/doku.php%3Fid=iclr2015:accepted-main.html | No page numbers. |
| hein2019relu | verified | https://openaccess.thecvf.com/content_CVPR_2019/html/Hein_Why_ReLU_Networks_Yield_High-Confidence_Predictions_Far_Away_From_the_CVPR_2019_paper.html | Pages 41--50 agree with Crossref. |
| xu2021extrapolate | verified | https://iclr.cc/virtual/2021/poster/3229 and https://arxiv.org/abs/2009.11848 | No page numbers. |
| pedregosa2011sklearn | verified | https://jmlr.org/papers/v12/pedregosa11a.html | 16 authors on JMLR, first 8 listed. The later arXiv record has 19. Currently uncited in the draft. |
| perezdiaz2026demnet | added | https://zenodo.org/records/21113922 and https://www.samuelperezdi.com/demnet/ | Zenodo conference paper, 2026-07-01. |
| ablin2019unfolded | added | https://papers.nips.cc/paper_files/paper/2019/file/d073bb8d0c47f317dd39de9c9f004e9d-Bibtex.bib and https://arxiv.org/abs/1905.11071 | NeurIPS 32. The NeurIPS BibTeX has no pages and I could not confirm any, so pages are left out. |
| chen2024e2elr | added | https://api.crossref.org/works/10.1109/TPWRS.2023.3317352 and https://arxiv.org/abs/2304.11726 | None. |
| benslimane2025penalty | added | https://api.crossref.org/works/10.1007/978-3-031-95973-8_7 and https://arxiv.org/abs/2503.05175 | CPAIOR 2025 Part I. LNCS 15762 from catalog records. |

## Has anyone already done label-free DEM inversion?

**No published work found.** Every learned DEM inversion I could locate has solver DEMs in its training loss. None trains on observations alone.

| Work | Training targets | Forward-model term in the loss | Label-free? |
|---|---|---|---|
| Wright et al. 2019, DeepEM (`wright2019deepem`) | Basis Pursuit DEMs, MSE on the DEM | None. The notebook says resynthesis "can be used as an additional term in the loss function" | No |
| Szenicer et al. 2019 (`szenicer2019virtual`) | Its DEM baseline is DeepEM, trained on 2011 data | None | No |
| Balodhi and Morton 2026 (`bayesdem2026`) | Hannah and Kontar DEMs, per-pixel MAE on the DEM | Yes: 70% AIA reconstruction, 30% DEM | No. **Closest prior art** |
| Pérez-Díaz et al. 2026, DEMNet (`perezdiaz2026demnet`) | Basis Pursuit DEMs, MSE plus cross-entropy on bins | Yes, as a second stage: lambda = 0 for 100 epochs, then lambda = 1 for about 40 epochs | No |
| Park et al. 2023 (ApJS 264, 33), Youn et al. 2025 (A&A 695, A125), Jeon et al. 2026 (ApJS 287, 6) | Paired EUV image translation | Not applicable: networks generate EUV channels, and DEMs are computed from those channels afterwards | Not DEM networks |
| Peck et al. 2026 (ApJ 1010, 118) | Plowman and Caspi solver, then linear regression onto EVE spectra | Not applicable | Not DEM networks |
| Jarolim et al. 2024, SuNeRF (ApJL 961, L31) | Fits a 3D emission and absorption field to EUV images of one scene | Yes, but the output is not a DEM. It says SuNeRF "can be extended" using DEM methods | Not DEM |
| Liu et al. 2026, EGU26-18421 (doi:10.5194/egusphere-egu26-18421) | **Not stated** | **Not stated** | **Unknown** |

Three points for your wording:

- **Balodhi and Morton is the nearest neighbour.** It weights the forward model at 70%, but it still needs solver labels, and it says so: "The training labels carry the assumptions and systematic characteristics of the regularized inversion." The alternative they propose is training on MHD simulations, not label-free training.
- **DEMNet explains why it did not train on the forward model alone.** The term "is complex to optimize from scratch because there are negative solutions that also minimize the loss". This is a useful contrast, because your loss has positivity by construction and an L1 objective. Forward-model finetuning cost it DEM accuracy (DEM MSE 0.27 to 0.40) while improving resynthesis (161.3 to 42.4).
- **The one open item is Liu et al.** This EGU abstract reconstructs EUV radiation, temperature and electron density in 3D with "several machine learning techniques". It gives no training data or loss, and I found no paper. If it is a NeRF-style fit, it could be label-free, but it would be per-scene 3D reconstruction rather than an amortized per-pixel inversion. That reading is my inference, not something the abstract states.

Supported claim: "to our knowledge, no prior DEM network trains without solver labels". A flat "first" goes beyond what this search can show.

## Additional papers

**Added to the bib:**

- `perezdiaz2026demnet`: see above. It is a learning-based DEM inversion from AIA, 2026.
- `chen2024e2elr`: the closest LP analogue I found. Economic dispatch is modeled as a linear program. The self-supervised loss is the LP objective `c(p) + M_th * ||xi_th(p)||_1`, plus a violation penalty when the model is not feasible by construction, and no solver is needed. Unlike your method, its main architecture restores feasibility with repair layers.
- `ablin2019unfolded`: trains unfolded ISTA by minimizing the Lasso cost directly, which the paper says "differs critically" from supervised LISTA (`gregor2010lista`). This is the sparse-coding precedent for dropping solver targets.
- `benslimane2025penalty`: a self-supervised loss built from the exact penalty method, `-f(x) + nu * sum [g_j]_+`. Their penalty parameter trades sub-optimality against feasibility, which parallels your mu. The problems are knapsack and inventory, not LPs as such, and the penalty is linear rather than quadratic.

**Seen, not added:**

- Tamir, Yu and Lustig 2019, "Unsupervised Deep Basis Pursuit" (arXiv:1910.13110). The title collides with your topic, but it replaces the L1 term with a learned prior, so it is not a Basis Pursuit solver.
- Geng et al. 2025, "Differentiable Integer Linear Programming" (ICLR 2025). Unsupervised with a penalty method, for integer LPs. Not added because a 2026 arXiv note (Sornwanee, arXiv:2601.17800, not peer reviewed) claims its Theorem 5 is incorrect.
- Yang et al. 2024, PDQP-net (arXiv:2412.01051). An unsupervised KKT-based loss, but for quadratic programs and via unrolling.
- Kiruluta and Lemos 2024 (arXiv:2408.09967). Puts LP objective and constraints in an unsupervised model's loss. arXiv only.

## Search coverage and limits

What I searched for the label-free question:

- arXiv API: 34 abstract-field queries.
- OpenAlex: all 600 records since 2019 matching "differential emission" or solar "emission measure", screened for ML terms. 52 of them have no abstract.
- OpenAlex forward citations: all 699 works since 2019 citing Cheung 2015, Hannah and Kontar 2012, Plowman 2013 or 2020, Boerner 2012, Kashyap and Drake 1998 or Szenicer 2019.
- GitHub repository search and 15 web searches, one in Chinese.
- The 2023 Living Review on machine learning in solar physics (arXiv:2306.15308) lists no learned DEM inversion at all.

What I could not reach:

- **ADS**: no API token, and its web pages refuse automated fetches. This is the main gap, because AGU and SPD meeting abstracts are mostly indexed there. One manual ADS search is worth doing before submission: `abs:"emission measure" abs:(neural OR "deep learning" OR unsupervised OR "physics-informed") year:2019-2026`.
- **Semantic Scholar**: rate limited.
- **Google Scholar**: not accessed.
- **DBLP and OpenReview**: bot walls. ICLR entries were confirmed on iclr.cc instead, and the OpenReview links in the bib come from iclr.cc pages.
- **Springer and science.org pages**: blocked. Crossref, library catalogs and a repository PDF were used instead.
