# Auto-Interp-Causal-Validation

Do the labels that AI tools write for neurons actually describe what those neurons do?

## The question

There are now tools that go through a neural network and write a plain-English label for every
unit inside it: "this one detects stripes", "this one detects dog faces". The problem is how
those labels get graded. Everyone checks whether the label predicts what makes the unit fire.
Almost nobody checks whether the network actually use* that unit for the thing the label
names.

Those are different questions. A unit can fire reliably on stripes while the network never
consults it when stripes matter. Morcos et al. (2018) found exactly this in a different form:
neurons that look most specialised are often not the ones whose removal hurts performance. Huang
et al. (2023) found the same split in language models, where GPT-4's neuron explanations passed
observational checks but failed causal ones.

I ran both tests on all 1,920 units of a CNN. I call them the watching test (does the label
predict firing?) and the doing test (does removing the unit break what the label implies?).

## Setup

**ResNet-18.** I picked it for three reasons. It comes pretrained, so I did not have to train
anything, which mattered because the whole project runs on a free Colab T4 with no budget. It is
small enough that a full pass over 31,000 images takes minutes rather than hours. And it is the
model the labeling literature mostly uses (Network Dissection, CLIP-Dissect), so the results sit
next to existing work instead of floating free.

**1,920 units.** I define a unit as one channel at the output of one of the eight residual
blocks: 64+64+128+128+256+256+512+512. Block exits are where a block's computation is finished
and handed on, and it is where the interpretability work looks.

**Four datasets, 31,178 images.** ImageNetV2 (10,000), DTD textures (5,640), Oxford-IIIT Pets
(7,349), Flowers102 (8,189). All free with no registration wall.

## What I did

1. **Probe pass.** Push every image through the network and record what each unit did, ignoring
   the network's actual answer. The images are test cases for profiling units, not things to
   classify.
2. **Exemplar grids.** For each unit, build a 4x4 image of its 16 favourites, cropped to the spot
   in each image where it actually fired. Cropping matters as if you hand a labeler the whole photo it
   describes the scene instead of the trigger.
3. **Two labelers.** Labeler A is a CLIP-Dissect reimplementation: for each unit, find the word
   from a fixed 20,000-word vocabulary whose CLIP-similarity profile across the probe images best
   tracks the unit's firing profile. Labeler B shows a small vision-language model (Qwen2.5-VL-3B,
   running locally) the exemplar grid and asks for a short label, on a 400-unit subsample.
4. **Watching test.** On held-out images the labels were never chosen from, check whether the
   label predicts which images the unit fires hardest on.
5. **Doing test.** Silence units, then measure which of the 1,000 ImageNet classes lost accuracy.

## Results

**Watching test: labels are weak on average and worthless in early layers.**
Median AUC 0.554 against a shuffled-label floor of 0.474. Only 27.7% of units clear the noise
threshold individually. But the average hides the real finding, which is that label quality
depends almost entirely on depth:

| stage | median AUC |
|---|---|
| layer1.0 | 0.495 |
| layer1.1 | 0.480 |
| layer2.0 | 0.495 |
| layer2.1 | 0.504 |
| layer3.0 | 0.533 |
| layer3.1 | 0.540 |
| layer4.0 | 0.572 |
| layer4.1 | 0.618 |

Early layers sit at the shuffled-label floor. Their labels are indistinguishable from randomly
assigned words. By layer4 the labels genuinely predict firing. The result is robust to the
firing threshold (0.533 to 0.565 across a tenfold range).

**Single units are causally inert.** 99.6% of units, silenced alone, do less damage than
knocking out a random channel. The network is heavily redundant, which confirms Morcos in this
architecture and forced the doing test to work on groups.

**Most label groups are too small to test.** Grouping units by shared label gives 332 groups
covering 1,496 units, with a mean of 4.5 units each and a largest of 41. None reach the size
where ablation registers anything. The near-zero scores from that attempt say nothing about
label quality; the measurement was below the instrument's resolution.

**Where the pools are big enough, the labels hold up.** Building pools by ranking all 1,920 units
against a preregistered concept and taking the top N:

| concept | N=50 | N=100 | N=200 | family size |
|---|---|---|---|---|
| dog | 3.77x | 3.54x | 2.13x | 118 classes |
| bird | 2.01x | 2.17x | 1.53x | 59 classes |
| cat | 1.41x | 1.20x | 1.17x | 4 classes |
| flower | 0.89x | 0.92x | 0.92x | 4 classes |
| texture | 0.66x | 0.67x | - | no family |

Specificity is damage to the concept's own classes divided by damage to everything else. Dog and
bird are clearly above 1 and fall off as the pool loosens, which is what a real effect looks
like. Texture, included as a negative control, sits below 1 as it should: ImageNet has no
texture classes, so there is nothing for its damage to concentrate on.

## Limitations and criticisms

**Clean class families roughly double the measured effect, so these numbers are floors.**
Specificity depends entirely on which ImageNet classes count as belonging to a concept. I first
built those families automatically, taking the 60 classes most similar to the concept name in
CLIP space. That was bad: the "dog" family contained pig and lion, and the "flower" family
contained rifle and banana. Swapping in ImageNet's actual dog synset (classes 151 to 268, 118
breeds) took dog from 1.97x to 3.77x, and the same fix took bird from 1.18x to 2.01x. Two
independent concepts, both roughly doubling. Anything computed from an automatically built
family should be read as a lower bound, not a result.

**ImageNet is a dog dataset, and that flatters the dog result.** About 120 of the 1,000 classes
are dog breeds. That gives dog both the largest pool of dog-labelled units and by far the
cleanest ground truth to measure against, 118 classes versus four for cat and four for flower.
So dog may score highest because the experiment is best-posed there, not because dog labels are
truer. I cannot separate those two explanations with one architecture. Specificity also
correlates with how coherent each pool is (rho = 0.52 across pools), which accounts for part of
the spread on its own.

**Mixing the probe datasets was probably a mistake.** I chose the labels using all 31,000 images,
including textures, pets and flowers, but both tests only score on ImageNetV2 photos. So a unit
labelled from DTD texture images gets graded on whether that label predicts its behaviour on
photos of objects. That mismatch almost certainly drags the watching scores down, and it hits
early-layer texture units hardest, exactly the units that scored at chance. The mixed probe set
was meant to give early units something to respond to, and it did, but the split between what
labels were chosen from and what they were graded on should have been consistent.

**CLIP both writes and grades the labels.** Labeler A picks whichever CLIP word best tracks a
unit's firing, then the watching test uses CLIP again to judge whether that word predicts firing. Grading on held-out images and the shuffled-label null limit the
damage, but the fix is to grade with a different model than the one used for labeling. CLIP is
also trained mostly on object captions and is a weak judge of texture, which is a second reason
early layers may score at chance.

**Most concepts cannot be tested at all.** The redundancy means you need a substantial pool
before ablation shows anything, and only two concepts in this vocabulary have both enough units
and a large enough class family. For the other 330 label groups the doing test has no answer to
give.

## What I would do differently

Fix the probe/eval split so labels are chosen and graded on the same distribution. Grade with a
model that is not the one doing the labeling. Use ImageNet's own class hierarchy for families
from the start rather than CLIP similarity. And run the whole thing on a second architecture, so
the dog confound can be separated from the label-quality question.

## Repo

| file | what it does |
|---|---|
| `00_setup.ipynb` | environment check, dataset download, cold-start timing |
| `01_probe_pass.ipynb` | record every unit's response to every image, resumably |
| `02_exemplars.ipynb` | build the receptive-field-cropped exemplar grids |
| `03_labelers.ipynb` | Labeler A (CLIP-Dissect) and Labeler B (VLM captions) |
| `04_tests.ipynb` | watching test, doing test, concept pools |

Everything runs on a free Colab T4. No paid APIs, no gated models, no training.

## Conclusion

The answer is conditional rather than clean. Where a concept is well enough represented, meaning
enough units to overcome the network's redundancy and enough classes to measure damage against,
its labels hold up causally and hold up strongly: silencing the 50 most dog-labelled units costs
dog classes almost four times what it costs everything else. Where a concept is not well
represented, which is the case for the overwhelming majority of labels this vocabulary produces,
the doing test simply cannot answer the question.

My results suggest the causal check is worth running where it can be run,
and that most of the time it cannot be run at all, at least not with single-unit or small-group
ablation in a network this redundant.

## Future Work 

The most obvious next step is a second architecture. Everything here rests on ResNet-18, and the sharpest criticism of my dog result is that I cannot tell whether dog scores highest because dog labels are better or because ImageNet's dog dominance makes the experiment best-posed there. Running the same pipeline on a network trained on a dataset with a different class balance would separate those. A second run would also test whether the detection floor is a property of this network or of CNNs generally: ResNet-18 is heavily redundant and full of BatchNorm, which is known to spread computation across units, so a network trained without it might let single-unit ablation work at all.
