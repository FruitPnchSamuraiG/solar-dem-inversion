# Label-free DEM: progress since the 21 October meeting

The 21 October meeting raised five questions. All five now have answers, and a full paper draft exists. This note follows the questions in order.

## Summary

| Question | Short answer |
|---|---|
| 1. Is there a theory for the band weight μ? | Yes. The loss is the penalty form of BP's linear program, so we can compute how far outside its noise band each channel settles for any μ. |
| 2. Why do bigger models blow up? | Mostly on inputs our training mask had removed. Keeping those pixels and ignoring only the zeroed channel fixes most of it. |
| 3. What are the low pixels in the JPDFs? | Mostly channels zeroed by deconvolution, where the observation is wrong rather than the DEM. The worst cases are lone spikes that look like cosmic rays or hot pixels. |
| 4. What does the final layer look like? | It uses about 10 of its 54 directions, against 3 for the old model. |
| 5. Should we relax the constraints? | It fixes the targeted pixels but makes training harder and faint pixels worse, so we kept μ = 100 on the fixed band. |

**What changed in the model:** the final BP model is now a 1.5M-parameter MLP (was 409k) at μ = 100. Pixels with a zeroed channel are now kept in training, with only that channel's penalty removed. The Elastic Net model is unchanged at 409k.

## The setup in brief

- The network maps one pixel's six AIA intensities to the solver's 54 basis weights. Adding up the weighted basis curves gives the 18-bin DEM.
- It never trains on solver DEMs. Its loss is the solver's own objective: total weight, plus μ × (distance outside the 1.4σ noise band ÷ half-width)², summed over channels.
- Solver DEMs are used only to score the model, to choose μ, and through a flag that drops the 0.01% to 0.02% of pixels the solver could not solve at all.

## 1. A theory for μ

At the lowest point of the loss, the weight saved by stepping outside the band equals the extra penalty. That balance gives:

**distance outside the band (in half-widths) = λ · 1.4σ / (2μ)**

Here λ is the linear program's price of light in that channel, meaning the weight needed per DN/s. 94 Å is over 400 times more expensive than 171 Å, because a unit of hot plasma makes very little 94 Å light.

**Why μ = 1 lost the hot plasma.** Take a flare core where 94 Å reads 164 DN/s with σ = 11.5, so the band runs from 148 to 180.

| | Keep the hot plasma | Delete it |
|---|---|---|
| Total weight | 219 | 0 |
| Penalty at μ = 1 | 0 | 84 |
| Penalty at μ = 100 | 0 | 8,440 |

At μ = 1, deleting the plasma gives the lower loss, so the network was doing what the loss asked. That is why oversampling, longer training and a 10M model did not help. At μ = 100 the plasma stays, and the channel lands about 2 DN/s short of the band.

**Tested.** The formula holds above a threshold μ₀ that the linear program gives for each pixel in advance. Above it, 99.4% of channels are within 5% of the prediction. Well below it, 3.7% are. Over 10,899 held-out pixels the median ratio of measured to predicted is 1.00.

**What it gives us.** There is no single perfect μ for a squared penalty, because the distance outside the band only reaches zero at infinite μ. But for any tolerance ε we can compute μ = λ · 1.4σ / (2ε) from the response matrix and noise alone. For a typical flare core at 0.1 half-widths this gives about 120, close to the 100 our validation screen had chosen. Training limits how high μ can go.

## 2. Why bigger models blew up

| BP model, μ = 100, shared test | Median pixel error | DEM MSE |
|---|---|---|
| 409k (shown on 21 Oct) | 0.036 | 1.75 |
| 1.5M, zeroed pixels dropped, 3 seeds | 0.034 to 0.036 | 1.75 to 897 |
| 1.5M, only the zeroed channel ignored, 3 seeds | 0.031 | 1.08 to 1.35 |

- Deconvolution leaves at least one channel at zero in 40% of pixels. We had dropped all of those pixels from training, and validation used the same mask, so the network never saw a zero as input.
- The worst inputs sit in that gap: typically five of six channels are zero, and in over half of them only 94 Å has signal, about 28 times brighter than the pixels around it.
- Far from the training data nothing limits the output, and a bigger network has more freedom to give huge values there.
- **Fix:** keep those pixels and remove only the zeroed channel's penalty. All three masked runs beat all three unmasked runs.
- **Caveat:** most of the 897 came from one very bright flare pixel with no zero in it, so it was the worst seed rather than a clamp effect.
- **Self-check:** a few extreme pixels remain. We flag any pixel whose DEM predicts above 1,000 DN/s and more than 100 times the observation. This needs no solver and flags 49 to 101 of 160M test pixels.

## 3. The low pixels in the JPDFs

| | Zeroed channels (mostly 131, 193, 335 Å) | Lone spikes (show up in 171, 211 Å) |
|---|---|---|
| What they are | Deconvolution set the reading to zero | Only 94 Å has signal, far above its neighbors, and the rest are zero |
| What is wrong | The reading. The solver's DEM sits above the zero too, more often than ours. | Probably the reading (cosmic ray or hot pixel), plus the old network guessing wildly on an input it had never seen |
| Final model | A column at the left edge, about 0.8% of pixels; AIA MSE 148 with them, 149 without | Gone from the plots |

**Suggestion:** zeroed channels have no real observation to compare with, so they could be removed from each channel's JPDF for every model, with the number removed stated under each plot. Since the JPDF format is yours, we wanted to agree on this first.

## 4. The final layer

| Model | Effective directions (of 54) | DEM shapes for 90% of variation |
|---|---|---|
| Old model, μ = 1 | 2.9 | 4 |
| 409k, μ = 100 | 9.5 | 8 |
| Final 1.5M | 10.9 | 8 |
| Solver | n/a | 6 |

The old model barely used its outputs, because its loss deleted hot plasma and it had no need to represent it. Sorted by the leading eigenvector, the 409k model's outputs form one block of hot elements (logT 6.6 to 7.2), so the main thing it varies is how much hot plasma a pixel has. Doubling the width barely adds directions for typical pixels (14 to 18 for 90% of the last hidden layer's variance).

## 5. Relaxing the constraints

At about 9% of pixels no DEM fits the 1.4σ band. The solver widens its band there, and our loss does not. We tried giving the loss the solver's wider band:

| 409k, 120 epochs | μ = 300, solver's band | μ = 100, fixed band |
|---|---|---|
| Error at the widened pixels (validation DEM MAE) | 0.15 | 0.45 |
| DEM MSE | 1.54 | 2.00 |
| Flare-core 94 Å | 87% | 75% |
| Quiet MSE | 0.050 | 0.032 |

It fixed the targeted pixels but faint pixels got worse, it needs the solver's flag at every pixel, and the final 1.5M model reaches similar flare-core quality without it. So the fixed band stays. Relaxed pixels remain our weakest group.

## Where we stand against the supervised model

Shared test set, same pixels for both models. Bold marks the better model, and a difference under 3% counts as a tie.

| Measure | BP ours | BP supervised | ENet ours | ENet supervised |
|---|---|---|---|---|
| DEM MAE | **0.072** | 0.100 | **0.151** | 0.188 |
| DEM MSE | 1.08 to 1.35 | **0.91** | 0.310 | 0.318 |
| EM error | **11.0%** | 14.3% | **12.2%** | 18.0% |
| W1 (dex) | **0.070** | 0.133 | **0.103** | 0.121 |
| AIA MAE | 3.54 | **2.95** | **0.56** | 6.32 |
| AIA MSE | 148 to 9,540 | 157 | **161** | 598 |

- **BP:** we lead on three of six in all seeds. The supervised model keeps DEM MSE and AIA MAE, and AIA MSE depends on the seed. Our median pixel error is 3.6 times lower (0.031 against 0.112). Flare-core emission is 96.7% against 92%, and flare-core 94 Å is 88% against 85% (solver 92%).
- **Where BP's DEM MSE gap comes from:** almost all of it is flare cores. Without them ours is 0.087 against 0.091. At the tallest peaks we reach 0.53 of the solver's height and the supervised model 0.70.
- **Elastic Net:** we lead on five of six and tie the sixth. The tie relies on stopping at 40 epochs, a choice made with test scores. At 120 epochs, which validation preferred, our DEM MSE is 0.362.
- **Speed:** 0.32 s per 2048 × 2048 image on one GPU, against 221 s of BP solving on 16 CPU cores. Deconvolution still takes about 4 minutes in both pipelines.

## Questions

1. **Which supervised model should we compare with?** The BP checkpoint we received scores DEM MSE 0.91 on the shared test, while the DEMNet paper reports 0.27 to 0.40. Is it the published model, and if not, can we use that one? Was it scored the same way?
2. **Do the shared-test BP targets skip the relaxed pixels?** About 7.7% of targets are missing, close to the 8.5% of pixels our own solver run relaxed, so we inferred that they do.
3. **Can the comparison with the supervised model go in the paper?**
4. **Should zeroed channels come out of the JPDFs?** See section 3.
5. **What would be most useful next?** Options are a synthetic test with a known DEM, a large X-class flare (our test window tops out near X1), the 1.5M model on the solver's wider band, or a separate μ per pixel. We would start with the synthetic test, since it is the only check against a true answer rather than a solver's answer.
