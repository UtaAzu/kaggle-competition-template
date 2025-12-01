Archived: Official Comments & Discussion (Generic)

This document originally collected official competition comments and public forum posts for a past competition. Historical competition discussions and posts were removed from this repository to keep the template generic.

For general guidance and community discussions, consult external forum archives or add community links under `examples/` as needed.

João Phillipe Cardenuto · Posted 13 days ago
·
Competition Host
Getting Deeper into Scientific Image Copy-Move Forgeries
This post gives a quick overview and a few insights that might help you think about the problem from a forensic perspective.

1. What is Copy-Move Forgery (CMF)?
A copy-move forgery happens when someone copies a region of an image and pastes it somewhere else within the same image.

The reasons for doing this vary.

Sometimes it’s to fabricate results, duplicating objects to make an experiment look more convincing.

Other times it’s to hide something, such as covering up a defect or unwanted artifact by pasting the background over it.

Example 1 – Object duplication 

Example 2 – Object Cleaning 

2. How Forgeries Are Made
Bad actors know detectors exist, so they try to make detection difficult. Real-world CMFs are rarely simple “copy and paste” edits.

a. Geometric Transformations
The copied region is almost never identical. It’s usually transformed with homographic transformations to blend naturally into the new context. Common transformations include:

Rotation (e.g., a few degrees)
Flipping (horizontal or vertical)
Scaling (shrinking or enlarging)
Or a combination of these (e.g., rotated + scaled + flipped)
b. Region Size
CMF occurs on regions of any size, from small patches to large portions of the image.

c. Post-Processing & Obfuscation
After pasting, the region is often adjusted to hide the manipulation:

Edge smoothing or feathering to blend the pasted area with its surroundings.
Brightness or contrast adjustments to match lighting conditions.
Sometimes, extra noise is added over the image to mask inconsistencies.
These tweaks attempt the pasted region to “disappear” visually and statistically (considering noise stats).

d. Resolution
Most real forgeries occur on mid-to-low resolution images. Scaling and pixelation artifacts naturally confuse detectors and humans and make differences harder to spot.

3. Why Scientific Images Are Tricky
Scientific images bring their own set of challenges. Many contains multiple sub-images or panels, and each of these can act as a distraction.

For example, a multi-panel figure might have several plots or microscopy crops arranged together. The repeated structures, similar textures, labels, and grid lines can all look like CMFs — but they’re actually benign duplicates.

Some examples:

Bar graphs often contain identical-looking bars (not forgeries).
Microscopy images show repeating cellular patterns.
Grids and labels introduce sharp edges and text that can confuse detectors.
So, context really matters. A duplicated region isn’t always a forgery.

4. What This Means for Your Models
Here are a few takeaways when designing your approach:

Be robust to transformations. The copied region might be rotated, scaled, or flipped. Models need to handle these geometric changes.
Look for post-processing traces. Even if the pasted region blends visually, subtle differences in texture, noise, or local statistics may remain.
Use context. Not every similar patch is malicious. Your model should learn to distinguish between real duplication and natural repetition (e.g., bars in a graph, repeated patterns in microscopy).
These are just some starting points for your ideas, don’t feel limited by them.

More resources:

Benchmarking Scientific Image Forgery Detectors

What's in a picture? The temptation of image manipulation

Feel free to share your thoughts and experiences with this problem :)

Good luck, and have fun!


4
4 Comments
Hotness
 
Comment here. Be patient, be friendly, and focus on ideas. We're all here to learn and improve!
This comment will be made public once posted.


Post Comment
Maheen Riaz
Posted 10 days ago

· 4th in this Competition

Hello how are you i am maheen riaz i am ml professional and ai professional i have some issues in my note i submit 11 notebook but there is erro https://www.kaggle.com/code/maheenriaz1122/rluc-sifd-eda-train-inference?scriptVersionId=272997301


Reply

React
Isaac Menard
Posted 10 days ago

· 52nd in this Competition

Hello! I think it's because your submission is in the wrong format:

img

endoded_pixels should be annotation a and if it is not "authentic", it should be encoded in RLE format

Exemple:

case_id,annotation
1,authentic
2,"[123 4]"

Reply

1
CoreyJamesLevinson
Posted 12 days ago

· 420th in this Competition

My thoughts:

1) Will there be scaling transformations in the hidden test set? I don't recall seeing any in the training set.

For reference, I've been tinkering with postprocessing technique that identifies my predicted masks, then makes embedding for each individual mask and then tries to cluster masks, and only accepts clusters if there are at least 2 masks in the cluster (implies there is a copy-forge), otherwise it removes the mask prediction. Right now, this postprocessing is not effective, but maybe with better model predictions/more tinkering it will be?

2) Will there be panel data in the hidden test set? Don't see any in the training set. Like, similar to your github dataset where you have interpanel and intrapanel? If so, can we get some expectation of the percentage, that way we can prepare validation set properly balanced?

3) In the literature I see basically 2 approaches: finetune a pretrained image net to predict the masks (good at identifying visually similar things, but struggles when you have a photo of lots of similar-looking cells), OR train a new CNN with small stride, small amount of pooling to try and capture rough edges around the copy-pasted region, that way it doesn't get confused between similar-looking things that aren't a direct copy & paste. I expect the best solution would be a combination of both of these approaches (so far, they both perform similarly for me, but haven't tried a blend/stack). Do you agree?

4) Seemingly adding "patch embedding features" (SIFT, Zernike, HardNet, etc.) simply as extra channels in the inputs not helping. Not sure if others experienced the same. I think these are probably more useful in postprocessing but I haven't experimented as much. Do you have any advice for these?

5) I found warmup pretraining on other copy-forge datasets (CASIA, Defacto) barely helpful at all. Is it because the domain of biological photos is different, or perhaps the copy-forge techniques are different from the ones here?

6) Will the hidden test set distribution of authentic vs inauthentic be similar to 50/50 like our training dataset?


Reply

4
耶✌
Posted 11 days ago

· 3rd in this Competition

I also used the criterion of "at least 2 masks match, otherwise exclude them" in my post-processing code. However, regarding the dataset, it seems we can't delve too deeply into its composition. The organizers appear to expect us to build a copy-move detection model capable of handling any scenario, which is truly an enormous challenge.


Reply

React


Sohier Dane · Posted 20 hours ago
·
Kaggle Staff
Supplemental train data added
We've supplemented the train set with new images and masks. These new images were acquired and labeled using the exact process that will be used for the final test. Hopefully this new data alleviates the concerns about distribution shift that have been raised in the forums.


3
