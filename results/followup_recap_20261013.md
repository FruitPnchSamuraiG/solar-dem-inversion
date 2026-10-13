# Flare-Core Follow-up: Recap

BP's failure at flare cores comes from its training loss, not from training. A stiffer band in the loss (a heavier penalty when the DEM misses an observation by more than its noise), trained six times longer (120 epochs, then 120 more from that checkpoint), cuts BP's DEM error on the shared test set from 4.28 to 1.75 and beats the supervised model on seven of ten measures. On the original loss, none of the training-side suggestions from the last meeting (oversampling flare cores, longer training, a 10M model) helps. ElasticNet keeps its current model.

**Contents**

1. Starting point
2. The meeting's questions
3. What we built to test them
4. The suggestions on the original loss
5. The cause: the network already sits at its loss's best answer
6. Why the loss drops the hot plasma
7. The fix: a stiffer band
8. The suggestions on the stiff band
9. The best BP model
10. ElasticNet
11. What we know, what stays open

## 1. Starting point

Before this work, each track had one final label-free model, and BP failed at flare cores.

**Data.** 1,223 timestamps of deconvolved AIA images, split by timestamp into 917 for training, 153 for validation and 153 for testing. The shared test set holds 48,960 blocks (about 705M pixels) from the 153 test timestamps.

**Label-free training.** The network takes one pixel's six AIA values and outputs 54 nonnegative basis weights, which give an 18-bin DEM (logT 5.5 to 7.2). It never sees the solver's DEMs. Its loss asks only that the DEM reproduce each observation within a band of plus or minus 1.4 noise widths and stay sparse (BP), or the ElasticNet objective (ENet). The solver's DEMs serve only to grade the result.

**The fixed-pixel bug.** Training originally drew the same 512 pixels from each block in every epoch. The network saw one fixed sample of 30M pixels, about 0.8% of the training pixels, 40 times over. Training now draws fresh pixels every epoch.

**The final models.** Both are MLPs on the centre pixel with four hidden layers 336 wide, trained 40 epochs on fresh pixels.

| Track | Input | Parameters |
| --- | --- | --- |
| BP | log1p of the six values | 360k |
| ElasticNet | square root, then 12 Fourier frequencies (150 inputs) | 409k |

**The measures.** Every table scores the full shared test set: DEM MSE and MAE, median pixel error, total emission error (EM), Wasserstein distance (W1), and AIA MAE and MSE (how well the DEM reproduces the six images). A pixel is Bright when any channel exceeds the level only the top 5% of pixels reach. A **flare core** is a pixel where some channel reaches 32 times that level, about 1 pixel in 10,000. Two measures track flare cores:

- **Flare-core emission:** total emission in our DEM as a share of the solver's.
- **Flare-core 94 Å:** share of the observed 94 Å signal our DEM reproduces.

**The known failure.** The earlier diagnostic found that pixels at 10 times the Bright level or more are 0.09% of pixels but hold 94% of BP's squared DEM error. At flare cores the old BP model held 40% of the solver's emission and reproduced 23% of 94 Å (supervised model: 92% and 85%). Its overall DEM MSE was 4.28 against the supervised model's 0.91. In that diagnostic the network fit 171, 193 and 211 Å to within 2% at the same pixels, so we read the failure as a training problem.

## 2. The meeting's questions

The last meeting raised five points. Three of them assume the loss asks for the right answer and the network fails to reach it.

| Suggestion | What it assumes |
| --- | --- |
| Oversample flare cores | The network rarely sees them, so it never learns them. |
| Train longer | Training stopped early. The old loss curves looked flat only because the network kept seeing the same fixed sample. |
| A much bigger model (about 10M parameters) | The network lacks capacity. |
| One input encoding (square root plus Fourier) for both tracks | One recipe is simpler and loses nothing. |
| The training objective differs from the evaluation objective | We train on the loss but grade against the solver's DEM, so the two can disagree. |

The fifth point raises the other possibility: the loss itself asks for something different from the solver. Parts 5 and 6 test it.

Following the fourth point, every new BP run starts from the square root plus Fourier BP model rather than the log1p final. That starting model scores slightly worse than the log1p final (DEM MSE 4.56 against 4.28, flare-core emission 35% against 40%), so each new BP run compares against 4.56.

## 3. What we built to test them

We added five pieces to the training and evaluation code, then ran 13 models: 10 on BP and 3 on ElasticNet.

| Piece | What it does | Serves |
| --- | --- | --- |
| Brightness-weighted sampling | Draws each pixel with probability proportional to its brightness (clipped to 1 to 1,000 times the Bright level). Flare cores go from 1 in 9,000 training pixels to 1 in 1,060, 8.5 times more often. | Oversampling |
| Resumable training | Saves the full training state every epoch. Each GPU job stops before the cluster's 2-hour limit and the next job continues, so a 120-epoch run is five chained jobs. | Longer training, 10M |
| 1,800-wide model | 10.1M parameters, against 409k at 336 wide; evaluated on a GPU. | Bigger model |
| Loss weights | A weight on the band penalty (1 in the original loss), and an option to shrink the sparsity term on bright pixels. | The fix (Part 7) |
| Objective-gap check | Solves each pixel's best possible answer under the loss directly, with no network, on 12,000 validation pixels (6,000 of them flare cores). It scores the network, that best answer and the solver's DEM under the same loss. | The cause (Part 5) |

Every run also logs, per epoch, the gradient size, how often gradient clipping fires, and the share of Bright and flare-core pixels it trained on. Every finished model goes through the same shared-test evaluation as the earlier final models.

## 4. The suggestions on the original loss

On the original loss, none of the suggestions fixes flare cores. Longer training trims DEM MSE by 8%; oversampling and the 10M model make flare cores worse.

| BP run, original loss | DEM MSE | DEM MAE | Median | AIA MSE | Quiet MSE | Flare-core emission | Flare-core 94 Å |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Base (square root + Fourier, 40 epochs) | 4.56 | 0.106 | 0.023 | 83 | 0.057 | 35% | 15% |
| Oversample flare cores | 5.05 | 0.106 | 0.025 | 87 | 0.058 | 24% | 9% |
| 120 epochs | 4.21 | 0.105 | 0.022 | 108 | 0.057 | 37% | 16% |
| 10M parameters | 5.10 | 0.106 | 0.022 | 125 | 0.058 | 20% | 8% |

Away from flare cores the four runs score the same (DEM MAE 0.105 to 0.106, Quiet MSE 0.057 to 0.058). The differences sit in the flare-core columns.

- **Oversampling** empties the flare-core DEMs further. On validation they hold 11% of the solver's emission, against 34% for the base.
- **120 epochs** lowers DEM MSE from 4.56 to 4.21, but flare-core emission moves only from 35% to 37%, and AIA MSE rises from 83 to 108.
- **The 10M model** matches the base on the typical pixel and holds less emission at flare cores (20%).

## 5. The cause: the network already sits at its loss's best answer

The BP network reaches what its loss asks for. The loss's best answer itself drops the hot plasma at flare cores, so the failure lies in the loss, not in training.

**Two objectives.** The solver and our training share one goal: the DEM with the smallest total weight that reproduces the six observations. They differ in how they treat the band (each observation plus or minus 1.4 noise widths).

|  | What it minimizes | The band is |
| --- | --- | --- |
| BP solver | Total weight, with every channel inside its band | A rule that cannot be broken |
| Our training loss | Total weight + μ × a penalty for each channel outside its band, with μ = 1 | A price: leaving the band is allowed when it costs less |

That one difference, a rule against a price, explains this part and the next.

**The check.** Take one pixel and forget the network. Which 54 weights give the lowest possible loss? The loss is bowl-shaped (convex) in the weights, with one bottom and no false valleys, so an optimizer that keeps going downhill reaches it. We call that bottom the pixel's best answer: no network, however well trained, can beat it on that pixel. On about 12,000 validation pixels (6,000 of them flare cores) we:

1. Found each pixel's best answer with an optimizer, started once from the network's output and once from the solver's DEM, keeping the lower.
2. Scored four answers with the same training loss: the network's output, the best answer, the solver's DEM and the supervised model's DEM (the last two rewritten as 54 weights first).
3. Measured the gap, how far the network's loss sits above its best answer's. The median gap is 1% to 4% at most brightness levels and 8% at flare cores.

| Flare cores, validation, original loss | Loss | Sparsity + band penalty | 94 Å reproduced | Emission vs solver | DEM MSE |
| --- | --- | --- | --- | --- | --- |
| Old BP network (log1p) | 565 | 294 + 270 | 28% | 38% | 233,000 |
| Best answer under the loss | 463 | 220 + 243 | 26% | 30% | 233,000 |
| Solver's DEM | 1,125 | 1,119 + 6 | 94% | 100% | 0 |
| Supervised model | 1,878 | 1,107 + 771 | 91% | 95% | 38,000 |

- **The network is at the bottom.** At flare cores its loss sits 8% above the best answer, and its DEM error equals the best answer's. A perfect network would land on the best answer, which still reproduces only 26% of 94 Å. On this loss, training cannot fix flare cores.
- **Lower loss means further from the solver.** Look at the two parts. The solver keeps every channel inside its band (penalty 6) but pays 1,119 in sparsity, because it includes the hot plasma. The best answer drops the hot plasma: 220 in sparsity plus a penalty of 243, total 463. So our loss rates the solver's DEM 2.4 times worse than a DEM without the hot plasma (1.3 to 1.5 times at other bright pixels). Every training step goes downhill on our loss, which at flare cores means away from the solver. This is the meeting's fifth point in numbers: the training objective and the evaluation disagree.
- **The supervised model shows the same thing from the other side.** Under our loss it scores worst of all (1,878), yet it is the network closest to the solver. A low loss does not mean close to the solver.
- **It also explains the missing emission.** Label-free BP held about 15% less emission than the solver at ordinary pixels. The loss's best answer does the same (82% to 86% of the solver's emission at moderate brightness).

## 6. Why the loss drops the hot plasma

Hot flare plasma shows up almost only in 94 Å, and 94 Å sees it weakly. Adding that plasma costs the sparsity term more than leaving it out costs the band penalty.

**The loss.** Two terms: loss = sparsity + μ × band penalty. Sparsity is the sum of the 54 weights, roughly the total plasma; less is better. The band penalty is zero while a predicted channel stays inside its band, and grows as the square of the distance outside it, measured in half band widths (1.4 noise widths). The band weight μ is 1 in the original loss. It multiplies the penalty of all six channels alike; each channel's penalty is already scaled by its own noise, so the channels are comparable.

```latex
\text{loss} = \sum_k x_k + \mu \sum_c \frac{\max(0,\, \ell_c - (Dx)_c)^2 + \max(0,\, (Dx)_c - u_c)^2}{(1.4\,\sigma_c)^2}
```

**What each channel sees.** The response matrix says how bright each channel gets per unit of plasma (one unit is 10^26 cm^-5 of emission measure; the 54 weights, the DEM and the sparsity term all count in this unit). Two temperature ranges matter here, in DN/s per unit:

| Plasma at | 94 Å | 131 Å | 171 Å | 193 Å | 211 Å | 335 Å |
| --- | --- | --- | --- | --- | --- | --- |
| 1 to 1.3 MK (warm, logT 6.0 to 6.1) | 0.19 to 0.22 | 0.5 to 1.2 | 36 to 95 | 29 to 55 | 5 to 7 | 0.4 |
| 8 MK (hot, logT 6.9) | 0.27 | 0.40 | 0.05 | 0.34 | 0.08 | 0.05 |

A unit of warm plasma lights up 171, 193 and 211 Å by tens to a hundred DN/s. A unit of hot plasma lights up no channel by more than 0.4 DN/s.

- At ordinary pixels most of 94 Å comes from warm plasma. That plasma has to be in the DEM anyway, because 171, 193 and 211 Å demand it, so its 94 Å signal comes free.
- At flare cores the heating makes hot plasma, and most of 94 Å comes from it. Only 94 Å (and 131 Å) ask for that plasma, at under 0.4 DN/s per unit, so a bright 94 Å reading needs hundreds of units.

**A typical flare core, worked through on 94 Å.** 94 Å reads 164 DN/s with noise 11.5, so its band runs from 148 to 180 (half width 16.1). The best answer has two options:

| Option | Sparsity | Band penalty at μ = 1 | Total |
| --- | --- | --- | --- |
| Put the hot plasma in: 164 ÷ 0.27, about 616 units | 616 | 0 | 616 |
| Leave it out: 94 Å predicts about 0, which is 148 below the band | 0 | (148 ÷ 16.1)² = 85 | 85 |

Leaving it out costs 85 against 616, so the best answer leaves the hot plasma out. With μ = 100 the same penalty becomes 8,500, more than 616, and the plasma goes back in. That is the fix in Part 7.

**Why the failure grows with brightness.** Brighter pixels draw more of their 94 Å from hot plasma, so a larger share gets dropped. Share of the observed 94 Å reproduced on the validation split, by brightness (× the Bright level):

|  | 1 to 1.8 | 1.8 to 3.2 | 3.2 to 10 | 10 to 32 | 32 and up (flare cores) |
| --- | --- | --- | --- | --- | --- |
| Best answer, original loss | 91% | 92% | 78% | 44% | 26% |
| Solver's DEM | 100% | 101% | 95% | 88% | 94% |

The band itself is sound: the noise follows photon counting (it grows as the square root of the signal), so the band reflects real noise, not a calibration floor.

**A stiffer band reverses it.** Raising μ raises the price of leaving the plasma out. The table gives the share of the observed 94 Å (predicted ÷ observed) reproduced at a flare core: first for the worked example, from 94 Å alone, then for the real best answer using all six channels, averaged over the 6,000 validation flare cores. In practice only 94 and 131 Å ever get pushed far out of their bands, because the other four are cheap to satisfy, so μ acts mostly on those two.

| Band weight μ | 94 Å only (worked example) | Full best answer, all six channels |
| --- | --- | --- |
| 1 | 0% | 26% |
| 10 | 61% | 81% |
| 100 | 87% | 93% |

## 7. The fix: a stiffer band

A band weight of 100 brings the loss's best answer to the solver's level at flare cores while keeping faint pixels better than the supervised model. The choice rests on the validation split.

**The screen.** The screen tests the loss, not the network. For each candidate loss we solved every validation pixel's best answer directly, with no network (as in Part 5), and scored it against the solver's DEMs. It answers one question: if a network learned this loss perfectly, how close to the solver would it get? DEM error weights each brightness level by its share of validation pixels.

| Loss (best answer, validation) | DEM error | Quiet MSE | Flare-core 94 Å | Flare-core emission |
| --- | --- | --- | --- | --- |
| Original (μ = 1) | 36.9 | 0.076 | 26% | 30% |
| μ = 10 | 10.8 | 0.037 | 81% | 90% |
| μ = 30 | 5.10 | 0.032 | 89% | 97% |
| μ = 100 | 2.32 | 0.041 | 93% | 100% |
| μ = 300 | 1.80 | 0.048 | 94% | 101% |
| μ = 30, sparsity ÷ √brightness | 1.95 | 0.032 | 95% | 101% |
| Supervised model (trained network) | 6.10 | 0.054 | 91% | 95% |

- **Stiffer helps bright pixels and costs faint ones.** Overall error falls steadily up to μ = 300. Quiet error is lowest near μ = 30 and rises beyond it, because at faint pixels the band is mostly noise and a stiff band makes the best answer chase it. At a flare core, by contrast, a price 100 times higher works like a rule (8,500 against 616 in the Part 6 example).
- **μ = 100** gives most of the gain while Quiet error stays below the supervised model's. Going to 300 lowers overall error only from 2.32 to 1.80 and raises Quiet error to 0.048.
- **The second candidate,** μ = 30 with the sparsity term divided by the square root of brightness on bright pixels, keeps the best Quiet error and nearly the best overall error.

We trained networks on both candidates (Part 8). The validation sample holds more flare cores than the shared test set, so its absolute values run higher than the shared-test values; compare rows with each other.

## 8. The suggestions on the stiff band

On the stiff band, training becomes the limit, and longer training is the one suggestion that helps. Band ×100 trained 120 epochs and then continued for 120 more wins on the shared test set; 240 epochs from scratch scores lower still on validation but misfits rare pixels.

| BP run (square root + Fourier unless noted) | DEM MSE | DEM MAE | Median | AIA MSE | Quiet MSE | Flare-core emission | Flare-core 94 Å | Validation error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Original loss, base (for reference) | 4.56 | 0.106 | 0.023 | 83 | 0.057 | 35% | 15% | 39.7 |
| Band ×100, 40 epochs | 2.37 | 0.087 | 0.062 | 44 | 0.035 | 89% | 68% | 20.1 |
| Band ×30, sparsity ÷ √brightness | 2.70 | 0.081 | 0.041 | 49 | 0.033 | 85% | 61% | 22.4 |
| Band ×100 + oversample flare cores | 2.50 | 0.091 | 0.071 | 39 | 0.038 | 88% | 63% | 21.1 |
| Band ×100, 120 epochs | 2.00 | 0.077 | 0.041 | 44 | 0.032 | 92% | 75% | 17.8 |
| Band ×100, 240 epochs from scratch | 1.74 | 0.074 | 0.036 | 591 | 0.035 | 91% | 73% | 14.0 |
| **Band ×100, 120 + 120 epochs (continued)** | **1.75** | **0.074** | **0.036** | **103** | **0.032** | **93%** | **77%** | **15.8** |
| Band ×100, 10M parameters | 35.7 | 0.081 | 0.041 | 1.7M | 0.070 | 92% | 74% |  |
| Band ×100, 680 wide, 120 epochs | 897 | 0.079 | 0.034 | 21.7M | 12.5 | 115% | 78% | 4,170 |
| Band ×100, log1p input | 3.23 | 0.097 | 0.078 | 52 | 0.038 | 85% | 59% | 29.3 |
| Band ×100, no gradient clip | 3.20 | 0.142 | 0.170 | 36 | 0.060 | 76% | 48% | 24.5 |
| Supervised model | 0.91 | 0.100 | 0.112 | 157 | 0.034 | 92% | 85% | 6.10 |

- **The band alone** halves DEM MSE (4.56 to 2.37) and takes flare-core emission from 35% to 89%.
- **Training is now the limit.** At flare cores the network's loss sits 118% above its best answer after 40 epochs (median), against 8% to 14% on the original loss. After 120 epochs the gap is 47%.
- **120 epochs** therefore helps: DEM MSE 2.37 to 2.00, flare-core 94 Å 68% to 75%, validation error 20.1 to 17.8.
- **240 epochs from scratch** keeps improving the DEM but starts to misfit rare pixels. DEM MSE 2.00 to 1.74, median 0.041 to 0.036, W1 0.071 to 0.068, spurious multi-peaks 16% to 13.5% (the solver's own rate), validation error 17.8 to 14.0. But AIA MSE rises from 44 to 591 and Quiet MSE from 0.032 to 0.035: a few pixels break the band badly (on the validation flare cores its band penalty is 130 times the 120-epoch model's), the mild form of what the 680-wide and 10M models did.
- **120 + 120 epochs, continued from the 120-epoch model** with a fresh schedule, gets the same DEM gains (MSE 1.75, median 0.036, W1 0.068) while keeping the AIA fit and the quiet pixels: AIA MSE 103 (still below the supervised model's 157), Quiet MSE 0.032, flare-core emission 93%, flare-core 94 Å 77%, validation error 15.8. Its band penalty at the validation flare cores is 13 times the 120-epoch model's, against 130 times for 240 from scratch. It improves on the 120-epoch model in every measure except AIA MAE and AIA MSE, which both stay ahead of the supervised model. This is the new BP model.
- **Oversampling** still does not help: DEM MSE rises to 2.50 and the median pixel error to 0.071.
- **The 10M model** fits 98% of pixels as well as 336 wide (MSE 0.037 against 0.036), but its worst 0.1% of pixels carry 14 times the error of the 336-wide model's worst 0.1%. Earlier ElasticNet models of 722k and above failed the same way.
- **680 wide (1.43M parameters)** fails the same way, and worse. The typical pixel improves (median 0.034 against 0.041, W1 0.068 against 0.071), but a few flare-core pixels blow up: DEM MSE 897, AIA MSE 21.7 million, Quiet MSE 12.5, and the worst 1% of pixels carry 99.996% of the error. At flare cores it over-predicts 171 Å by 3 times. Both sizes tried above 409k parameters (1.43M and 10M) blew up on rare pixels, so 336 wide stays.
- **The second candidate** trains worse than μ = 100 (validation 22.4 against 20.1), although its best answer was slightly better.
- **Square root + Fourier** beats log1p on the stiff band (2.37 against 3.23), so one encoding serves both tracks.
- **Gradient clipping** (at norm 1) fires on every step on the stiff band. Removing it fits AIA better (MSE 36) but worsens every DEM measure, so the clip stays.

## 9. The best BP model

Band ×100 trained 120 epochs and continued for 120 more beats the supervised model on seven of ten measures, up from six for the 120-epoch model and three for the old final. It still loses on DEM MSE by a factor of 1.9.

| BP model, shared test | DEM MSE | DEM MAE | Median | EM error | W1 | AIA MAE | AIA MSE | Quiet MSE | Flare-core emission | Flare-core 94 Å |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Supervised model | **0.91** | 0.100 | 0.112 | 14.3% | 0.133 | **2.95** | 157 | 0.034 | 92% | **85%** |
| Old final (log1p, original loss) | 4.28 | 0.106 | **0.026** | 19.6% | 0.083 | 4.34 | 83 | 0.058 | 40% | 23% |
| Band ×100, 120 epochs | 2.00 | 0.077 | 0.041 | 11.3% | 0.071 | 3.37 | **44** | **0.032** | 92% | 75% |
| Band ×100, 240 epochs from scratch | 1.74 | **0.074** | 0.036 | **11.0%** | **0.068** | 3.52 | 591 | 0.035 | 91% | 73% |
| **Band ×100, 120 + 120 epochs (continued)** | 1.75 | **0.074** | 0.036 | **11.0%** | **0.068** | 3.48 | 103 | **0.032** | **93%** | 77% |

- **Against the supervised model** the continued model (120 + 120 epochs) wins seven of ten: DEM MAE, median, EM error, W1, AIA MSE, Quiet MSE and flare-core emission. It loses DEM MSE, AIA MAE and flare-core 94 Å. The 120-epoch model won six (tying flare-core emission), the 240-epoch model from scratch only four (its AIA MSE 591 and Quiet MSE 0.035 fall behind), and the old final three.
- **Against the old final** it improves every measure except the median pixel error, which rises from 0.026 to 0.036 (still well below the supervised model's 0.112). Its DEMs are also less sparse (training sparsity score 2.6 against 1.8 on the original loss; higher means more basis weights in use).
- **What is left** is flare cores: they set the DEM MSE gap (1.75 against 0.91). Longer training keeps closing it, but 240 epochs from scratch starts to misfit a few pixels' AIA (MSE 591); continuing from the 120-epoch checkpoint instead keeps that in check (103).

**The recipe.** The old final's network (MLP with four hidden layers 336 wide, fresh pixels, gradient clip at 1) with square root + 12 Fourier frequencies as input, band weight 100, 120 epochs, then 120 more with a fresh schedule started from that checkpoint.

## 10. ElasticNet

The current ElasticNet model stays: no suggestion beats its DEM MSE, and its loss never had BP's flare-core problem.

| ElasticNet run, shared test | DEM MSE | DEM MAE | Median | EM error | W1 | AIA MAE | AIA MSE | Flare-core emission | Flare-core 94 Å |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Supervised model | 0.318 | 0.188 | 1.12 | 18.0% | 0.121 | 6.32 | 598 | 87% | 48% |
| **Current final** | **0.310** | 0.151 | 0.85 | 12.2% | 0.103 | 0.56 | 161 | 99% | 53% |
| Oversample flare cores | 0.379 | 0.152 | 0.89 | 12.2% | 0.105 | 0.57 | 141 | 102% | 55% |
| 120 epochs | 0.362 | **0.144** | **0.78** | **11.9%** | 0.100 | **0.52** | 188 | 102% | 55% |
| 10M parameters | 2,062 | 0.159 | 0.78 | 15.2% | **0.099** | 2.62 | 83M | 216% | 107% |

- **Its flare cores were already fine.** The current model holds 99% of the solver's emission there (supervised model: 87%).
- **120 epochs is a trade, not a win.** It improves DEM MAE, median, EM error, W1 and AIA MAE, but worsens DEM MSE (0.310 to 0.362) and AIA MSE (161 to 188).
- **Oversampling** worsens DEM MSE (0.379) and changes little else.
- **The 10M model** blows up on a few bright pixels, as on BP.

## 11. What we know, what stays open

**Established**

1. On the original BP loss, the network sits within a few percent of each pixel's best answer, and that answer drops the hot plasma at flare cores. No training change on that loss can fix flare cores.
2. The cause is the price of hot plasma: only 94 Å sees it, weakly, so including it costs the sparsity term about 616 while leaving it out costs about 85.
3. A band weight of 100 fixes the best answer (93% of 94 Å, 100% of emission), and the trained network follows most of the way (77% and 93% on the shared test set after 120 + 120 epochs).
4. On the stiff band, longer training keeps helping: 120 epochs beats 40, and 120 more continued from that checkpoint beats 120 on every measure except the AIA fit. 240 epochs from scratch is better still on validation but breaks the band on rare pixels. Oversampling, bigger models (1.43M and 10M), log1p input and removing the gradient clip do not help.
5. Models above 409k parameters fail on a few rare pixels: 10M on both tracks, and 1.43M on the stiff band.
6. Label-free BP's 15% emission shortfall at ordinary pixels comes from the loss, not the network.
7. The current ElasticNet model remains the best on DEM MSE.

**Open**

- Whether longer training can keep closing the flare-core gap without misfitting rare pixels. 240 epochs from scratch cuts the flare-core error 13% but its AIA MSE grows from 44 to 591; continuing 120 epochs from the 120-epoch checkpoint gets the same gain with AIA MSE 103. A second continuation (120 + 120 + 120) is untested.
- Why the median pixel error rises (0.026 to 0.041) while Quiet MSE falls (0.058 to 0.032).
- DEM MSE remains 2.2 times the supervised model's.

**Caveats**

- The band weight was chosen with the solver's DEMs on the validation split, as for any loss weight. Training never sees them.
- The first stiffness screen ran on test pixels. We repeated it on validation, which kept μ = 100 and changed the second candidate; the final choice uses validation only.
- The validation sample holds more flare cores than the shared test set, so validation values run higher than shared-test values.

**Decisions**

- BP: band ×100, 120 epochs, then 120 more continued from that checkpoint.
- ElasticNet: the current model, unchanged.
- Not chosen: 120 epochs alone (a cleaner AIA fit, MSE 44 against 103, but DEM MSE 2.00 against 1.75) and 240 epochs from scratch (AIA MSE 591, Quiet MSE 0.035).
